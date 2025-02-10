from dataclasses import dataclass
import itertools
from typing import Any, Optional
import torch
import numpy as np
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, AutoModelForMaskedLM
from transformers.modeling_outputs import ModelOutput
from .models import dit, dimamba, autoregressive, ema
from .noise_schedule import get_noise


def _sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))

@dataclass
class Output(ModelOutput):
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    nlls: Optional[torch.FloatTensor] = None
    token_mask: Optional[torch.FloatTensor] = None


def tokenization_params(tokenizer: PreTrainedTokenizer):
    vocab_size = tokenizer.vocab_size
    if (
        not hasattr(tokenizer, "mask_token")
        or tokenizer.mask_token is None
    ):
        mask_token_id = vocab_size
        vocab_size += 1
    else:
        mask_token_id = tokenizer.mask_token_id

    return dict(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

class Diffusion(torch.nn.Module):

    def __init__(self, config: Any, tokenizer: Optional[PreTrainedTokenizer] = None, dtype: Optional[Any] = None):
        super().__init__()
        self.config = config
        self.dtype = dtype

        self.tokenizer = tokenizer
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.importance_sampling = self.config.training.importance_sampling
        self.change_of_variables = self.config.training.change_of_variables

        if tokenizer is not None:
            tokenization_params = tokenization_params(tokenizer)
        else:
            tokenization_params = dict(
                vocab_size=self.config.vocab_size,
                mask_token_id=self.config.mask_token_id,
                pad_token_id=self.config.pad_token_id,
                bos_token_id=self.config.bos_token_id,
                eos_token_id=self.config.eos_token_id,
            )
        for k, v in tokenization_params.items():
            setattr(self, k, v)

        self.parameterization = self.config.parameterization
        if self.config.backbone == "dit":
            self.backbone = dit.DIT(self.config, vocab_size=self.vocab_size)
        elif self.config.backbone == "dimamba":
            self.backbone = dimamba.DiMamba(
                self.config,
                vocab_size=self.vocab_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elif self.config.backbone == "ar":
            self.backbone = autoregressive.AR(
                self.config, vocab_size=self.vocab_size, mask_index=self.mask_token_id
            )
        elif self.config.backbone == "hf_dit":
            self.backbone = AutoModelForMaskedLM.from_pretrained(
                config.eval.checkpoint_path, trust_remote_code=True
            )
        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone}")

        self.T = self.config.T
        self.subs_masking = self.config.subs_masking

        self.noise = get_noise(self.config, dtype=self.dtype)
        if self.config.training.ema > 0:
            self.ema = ema.ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(), self.noise.parameters()),
                decay=self.config.training.ema,
            )
        else:
            self.ema = None

        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self._validate_configuration()

    def _validate_configuration(self):
        assert not (self.change_of_variables and self.importance_sampling)
        if self.parameterization == "sedd":
            assert not self.importance_sampling
            assert not self.change_of_variables
        if self.parameterization == "d3pm":
            assert self.T > 0
        if self.T > 0:
            assert self.parameterization in {"d3pm", "subs"}
        if self.subs_masking:
            assert self.parameterization == "d3pm"


    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_token_id] += self.neg_infinity

        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_token_id
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _d3pm_parameterization(self, logits):
        if self.subs_masking:
            logits[:, :, self.mask_token_id] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return logits

    def _sedd_parameterization(self, logits, xt, sigma):
        esigm1_log = (
            torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
            .log()
            .to(logits.dtype)
        )
        # logits shape
        # (batch_size, diffusion_model_input_length, vocab_size)
        logits = logits - esigm1_log[:, None, None] - np.log(logits.shape[-1] - 1)
        # The below scatter operation sets the log score
        # for the input word to 0.
        logits = torch.scatter(
            logits, -1, xt[..., None], torch.zeros_like(logits[..., :1])
        )
        return logits

    def _process_sigma(self, sigma):
        if sigma is None:
            assert self.parameterization == "ar"
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def forward(self, x, sigma):
        """Returns log score."""
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.backbone(x, sigma)

        if self.parameterization == "subs":
            return self._subs_parameterization(logits=logits, xt=x)
        elif self.parameterization == "sedd":
            return self._sedd_parameterization(logits=logits, xt=x, sigma=sigma)
        elif self.parameterization == "d3pm":
            return self._d3pm_parameterization(logits=logits)
        return logits

    def _d3pm_loss(self, model_output, xt, x0, t):
        dt = 1 / self.T

        if torch.is_tensor(t):
            t = t[:, None]
            assert t.ndim == 2
            t = t.clamp(0.0, 1.0 - 1e-4)
        alpha_t = 1 - t + torch.zeros_like(xt)
        alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

        log_x_theta_at_x0 = torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = model_output[:, :, self.mask_token_id]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

        L_vb_masked = term_1_coef * (term_1_log_nr - term_1_log_dr) + term_2_coef * (
            term_2_log_nr - term_2_log_dr
        )

        L_vb = L_vb_masked * (xt == self.mask_token_id)

        return self.T * L_vb


    def q_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_token_id, x)
        return xt

    def _sample_prior(self, *batch_dims):
        return self.mask_token_id * torch.ones(*batch_dims, dtype=torch.int64)

    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        assert self.config.noise.type == "loglinear"
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            p_x0 = self.forward(x, sigma_t).exp()

        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_token_id] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_token_id).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _ddpm_update(self, x, t, dt):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_token_id] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_token_id).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x

    def _ar_sampler(self, bsz):
        # precompute token buffer
        num_pred_tokens = self.config.model.length - 1
        x = torch.zeros(
            (bsz, num_pred_tokens + 1), dtype=torch.long, device=self.device
        )
        x[:, 0] = self.tokenizer.bos_token_id
        # precompute noise
        noise = (
            torch.distributions.Gumbel(0, 1)
            .sample((bsz, num_pred_tokens, self.vocab_size))
            .to(self.device)
        )
        for i in range(num_pred_tokens):
            next_logits = self.forward(x[:, : i + 1], None)[:, -1]
            y = (next_logits + noise[:, i]).argmax(-1)
            x[:, i + 1] = y
        return x

    def get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        if self.parameterization == "subs":
            # score(x, t) = p_t(y) / p_t(x)
            # => log score(x, t) = log p_t(y) - log p_t(x)

            # case 1: x = masked
            #   (i) y = unmasked
            #     log score(x, t) = log p_\theta(x)|_y + log k
            #     where k = exp(- sigma) / (1 - exp(- sigma))
            #   (ii) y = masked
            #     log score(x, t) = 0

            # case 2: x = unmasked
            #   (i) y != masked, y != x
            #     log score(x_i, t) = - inf
            #   (ii) y = x
            #     log score(x_i, t) = 0
            #   (iii) y = masked token
            #     log score(x_i, t) = - log k
            #     where k = exp(- sigma) / (1 - exp(- sigma))

            log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
            assert log_k.ndim == 1

            masked_score = model_output + log_k[:, None, None]
            masked_score[:, :, self.mask_token_id] = 0

            unmasked_score = self.neg_infinity * torch.ones_like(model_output)
            unmasked_score = torch.scatter(
                unmasked_score,
                -1,
                x[..., None],
                torch.zeros_like(unmasked_score[..., :1]),
            )
            unmasked_score[:, :, self.mask_token_id] = -(
                log_k[:, None] * torch.ones_like(x)
            )

            masked_indices = (x == self.mask_token_id).to(model_output.dtype)[:, :, None]
            model_output = masked_score * masked_indices + unmasked_score * (
                1 - masked_indices
            )
        return model_output.exp()

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_token_id] += extra_const
        return score

    def _analytic_update(self, x, t, step_size):
        curr_sigma, _ = self.noise(t)
        next_sigma, _ = self.noise(t - step_size)
        dsigma = curr_sigma - next_sigma
        score = self.get_score(x, curr_sigma)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return _sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma, _ = self.noise(t)
        score = self.get_score(x, sigma)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_token_id] = 0
        samples = _sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_token_id, 1 - torch.exp(-sigma).squeeze(-1), 0)[
            ..., None
        ]
        return edge

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def _maybe_sub_sample(self, x0, attention_mask):
        seqlen = x0.shape[1]
        if seqlen > self.config.model.length:
            assert seqlen == 2 * self.config.model.length
            # cropping is needed for text8-crop dataset
            # try the same starting point for now
            start = np.random.choice(self.config.model.length)
            end = start + self.config.model.length
            input_tokens = x0[:, start:end]
            output_tokens = x0[:, start + 1 : end + 1]
            new_attention_mask = attention_mask[:, start:end]

            # Helps with validation PPL, since the val
            # examples will all start and end with BOS/EOS
            input_tokens[:, 0] = self.tokenizer.bos_token_id
            output_tokens[:, -1] = self.tokenizer.eos_token_id
        elif self.parameterization == "ar":
            input_tokens = x0[:, :-1]
            output_tokens = x0[:, 1:]
            new_attention_mask = attention_mask[:, 1:]
        else:
            input_tokens = x0
            output_tokens = None
            new_attention_mask = attention_mask
        return input_tokens, output_tokens, new_attention_mask

    def _reconstruction_loss(self, x0):
        t0 = torch.zeros(x0.shape[0], dtype=self.dtype, device=self.device)
        assert self.config.noise.type == "loglinear"
        # The above assert is for d3pm parameterization
        unet_conditioning = self.noise(t0)[0][:, None]
        model_output_t0 = self.forward(x0, unet_conditioning)
        return -torch.gather(
            input=model_output_t0, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

    def _forward_diffusion(self, x0, compute_loss=True):
        t = self._sample_t(x0.shape[0], x0.device)
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        if self.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance)
        logits = self.forward(xt, unet_conditioning)
        if not compute_loss:
            return (logits, None)

        if self.parameterization == "sedd":
            return dsigma[:, None] * self._score_entropy(
                logits, sigma[:, None], xt, x0
            )

        if self.T > 0:
            diffusion_loss = self._d3pm_loss(
                model_output=logits, xt=xt, x0=x0, t=t
            )
            if self.parameterization == "d3pm":
                reconstruction_loss = self._reconstruction_loss(x0)
            elif self.parameterization == "subs":
                reconstruction_loss = 0
            return reconstruction_loss + diffusion_loss

        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=logits, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

        if self.change_of_variables or self.importance_sampling:
            return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))

        loss = -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
        return logits, loss

    def forward_diffusion(self, x0, attention_mask, compute_loss=True) -> Output:
        (input_tokens, output_tokens, attention_mask) = self._maybe_sub_sample(
            x0, attention_mask
        )

        if self.parameterization == "ar":
            logits = self.backbone(input_tokens, None)
            loss = -logits.gather(-1, output_tokens[:, :, None])[:, :, 0]
        else:
            logits, loss = self._forward_diffusion(input_tokens, compute_loss=compute_loss)

        if not compute_loss:
            return Output(logits=logits, token_mask=attention_mask)

        nlls = loss * attention_mask
        count = attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return Output(logits=logits, loss=token_nll, nlls=nlls, token_mask=attention_mask)

    def _score_entropy(self, log_score, sigma, xt, x0):
        """Computes the SEDD loss.

        Args:
          log_score: float torch.Tensor with shape (batch_size,
              diffusion_model_input_length, vocab_size),
              log score, output of the denoising network.
          xt: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          x0: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          sigma: float torch.Tensor with shape (batch_size, 1).

        Returns:
          loss with shape (batch_size, diffusion_model_input_length)
        """
        masked_indices = xt == self.mask_token_id

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_score[masked_indices], -1, words_that_were_masked[..., None]
        ).squeeze(-1)
        score = log_score[masked_indices].exp()
        if self.mask_token_id == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_token_id].sum(dim=-1) + score[
                :, self.mask_token_id + 1 :
            ].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(*xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return entropy

