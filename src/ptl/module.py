"""Lightning module implementation for MDLM."""

from dataclasses import dataclass
import itertools
import math
import os
from typing import List, Optional

import lightning as L
import torch
import transformers
import torch.nn.functional as F
import torchmetrics
import numpy as np
import hydra
import hydra.utils
from ..diffusion import Diffusion
from . import dataloader

LOG2 = math.log(2)


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> torch.Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> torch.Tensor:
        """Computes the Perplexity.

        Returns:
         Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class LightningMDLM(L.LightningModule):
    """Lightning wrapper for MDLM model."""
    
    def __init__(
        self,
        config,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        gen_ppl_eval_model_name_or_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = Diffusion(config, tokenizer, self.dtype)
        
        # Setup metrics
        metrics = {
            "nll": NLL(),
            "bpd": BPD(),
            "ppl": Perplexity(),
        }
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/") 
        self.test_metrics = metrics.clone(prefix="test/")
        
        # Generative perplexity
        self.gen_ppl_metric = Perplexity()
        self.eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(
            gen_ppl_eval_model_name_or_path
        )
        if self.eval_model_tokenizer.pad_token is None:
            self.eval_model_tokenizer.pad_token = self.eval_model_tokenizer.eos_token
            self.eval_model_tokenizer.pad_token_id = (
                self.eval_model_tokenizer.eos_token_id
            )
        self.fast_forward_epochs = None
        self.fast_forward_batches = None

        self.lr = self.config.optim.lr

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint["ema"])
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"][
            "current"
        ]["completed"]
        self.fast_forward_batches = checkpoint["loops"]["fit_loop"][
            "epoch_loop.batch_progress"
        ]["current"]["completed"]

    def on_save_checkpoint(self, checkpoint, trainer):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"][
            "completed"
        ] = (
            checkpoint["loops"]["fit_loop"][
                "epoch_loop.automatic_optimization.optim_progress"
            ]["optimizer"]["step"]["total"]["completed"]
            * trainer.accumulate_grad_batches
        )
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
            "completed"
        ] = (
            checkpoint["loops"]["fit_loop"][
                "epoch_loop.automatic_optimization.optim_progress"
            ]["optimizer"]["step"]["current"]["completed"]
            * trainer.accumulate_grad_batches
        )
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"][
            "_batches_that_stepped"
        ] = checkpoint["loops"]["fit_loop"][
            "epoch_loop.automatic_optimization.optim_progress"
        ]["optimizer"]["step"]["total"]["completed"]
        if "sampler" not in checkpoint.keys():
            checkpoint["sampler"] = {}
        if hasattr(trainer.train_dataloader.sampler, "state_dict"):
            sampler_state_dict = trainer.train_dataloader.sampler.state_dict()
            checkpoint["sampler"]["random_state"] = sampler_state_dict.get(
                "random_state", None
            )
        else:
            checkpoint["sampler"]["random_state"] = None

    def forward(self, x, sigma):
        return self.model.forward(x, sigma)
    

    # --------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed
        )
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, "shuffle"):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if (
                distributed
                and self.fast_forward_epochs is not None
                and self.fast_forward_batches is not None
            ):
                dl_sampler.load_state_dict(
                    {
                        "epoch": self.fast_forward_epochs,
                        "counter": (
                            self.fast_forward_batches * self.config.loader.batch_size
                        ),
                    }
                )
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=True,
                )
            )
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()


    def _compute_loss(self, batch, prefix):
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"]
        else:
            attention_mask = None
        output = self.model.forward_diffusion(batch["input_ids"], attention_mask)
        loss = output.loss

        if prefix == "train":
            self.train_metrics.update(output.nlls, output.token_mask)
            metrics = self.train_metrics
        elif prefix == "val":
            self.valid_metrics.update(output.nlls, output.token_mask)
            metrics = self.valid_metrics
        elif prefix == "test":
            self.test_metrics.update(output.nlls, output.token_mask)
            metrics = self.test_metrics
        else:
            raise ValueError(f"Invalid prefix: {prefix}")

        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.model._compute_loss(batch, prefix="train")
        self.log(
            name="trainer/loss",
            value=loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    def on_validation_epoch_start(self):
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
        assert self.valid_metrics.nll.mean_value == 0
        assert self.valid_metrics.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        return self.model._compute_loss(batch, prefix="val")

    def on_validation_epoch_end(self):
        if (
            (
                self.config.eval.compute_perplexity_on_sanity
                or not self.trainer.sanity_checking
            )
            and self.config.eval.generate_samples
            and not self.parameterization == "ar"
        ):
            # TODO(justin): implement sampling and kv cache for AR
            samples, text_samples = None, None
            for _ in range(self.config.sampling.num_sample_batches):
                samples = self.model._sample()
                # Decode the samples to be re-tokenized by eval model
                text_samples = self.tokenizer.batch_decode(samples)
                if self.config.eval.compute_generative_perplexity:
                    self.compute_generative_perplexity(text_samples)
            if self.trainer.global_rank == 0 and hasattr(
                self.trainer.logger, "log_table"
            ):
                # Log the last generated samples
                text_samples = text_samples[: self.config.sampling.num_sample_log]
                self.trainer.logger.log_table(
                    key=f"samples@global_step{self.global_step}",
                    columns=["Generated Samples"],
                    data=[[s] for s in text_samples],
                )
            if self.config.eval.compute_generative_perplexity:
                self.log(
                    "val/gen_ppl",
                    self.gen_ppl_metric,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                )
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )

    # --------------------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------------------

    def configure_optimizers(self):
        # TODO(yair): Lightning currently giving this warning when using `fp16`:
        #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        #  Not clear if this is a problem or not.
        #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters(), self.noise.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "val/loss",
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler_dict]

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(
                itertools.chain(self.model.backbone.parameters(), self.model.noise.parameters())
            )

    # --------------------------------------------------------------------------
    # Sampling
    # --------------------------------------------------------------------------

    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
        samples = self._sample(num_steps=num_steps, eps=eps)
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.train()
        self.noise.train()
        return samples
    
    def restore_model_and_semi_ar_sample(self, stride_length, num_strides, dt=0.001):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
        (sampling_steps, samples, sequence_lengths) = self.sample_subs_guidance(
            n_samples=self.config.loader.eval_batch_size,
            stride_length=stride_length,
            num_strides=num_strides,
            dt=dt,
        )
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.train()
        self.noise.train()
        return sampling_steps, samples, sequence_lengths

    @torch.no_grad()
    def sample_subs_guidance(self, n_samples, stride_length, num_strides, dt=0.001):
        ones = torch.ones(n_samples, dtype=self.dtype, device=self.device)

        num_steps = int(1 / dt)
        sampling_steps = 0
        intermediate_tokens = []
        target = None
        for _ in range(num_strides + 1):
            p_x0_cache = None
            x = self._sample_prior(n_samples, self.config.model.length).to(self.device)
            if target is not None:
                x[:, :-stride_length] = target
            for i in range(num_steps + 1):
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    p_x0_cache = None
                    sampling_steps += 1
                x = x_next
            x = self.forward(x, 0 * ones).argmax(dim=-1)
            intermediate_tokens.append(x[:, :stride_length].cpu().numpy())
            target = x[:, stride_length:]

        intermediate_tokens.append(target.cpu().numpy())
        intermediate_text_samples = []
        sequence_lengths = (
            (
                np.concatenate(intermediate_tokens, axis=1)[:, 1:]
                == self.tokenizer.eos_token_id
            ).cumsum(-1)
            == 0
        ).sum(-1)
        for i in range(2, len(intermediate_tokens) + 1):
            intermediate_text_samples.append(
                self.tokenizer.batch_decode(
                    np.concatenate(intermediate_tokens[:i], axis=1)
                )
            )
        return (sampling_steps, intermediate_text_samples, sequence_lengths)

    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5):
        """Generate samples from the model."""
        batch_size_per_gpu = self.config.loader.eval_batch_size
        if self.parameterization == "ar":
            return self._ar_sampler(batch_size_per_gpu)
        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self._sample_prior(batch_size_per_gpu, self.config.model.length).to(
            self.device
        )
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "ddpm":
                x = self._ddpm_update(x, t, dt)
            elif self.sampler == "ddpm_cache":
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x, t, dt)

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "analytic":
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                x = self.forward(x, unet_conditioning).argmax(dim=-1)
        return x
    

    @torch.no_grad()
    def eval_retokenize(self, text_samples, max_length):
        """Retokenizes samples for the eval model.

        Args:
            text_samples: List of sentences generated by the model.
        Returns:
            samples: Samples re-tokenized for the eval model
            attn_mask: Attention mask for the eval model
            eval_context_size: Size of the context for the eval model
        """
        if "llama2" in self.gen_ppl_eval_model_name_or_path:
            tokenizer_kwargs = {
                "text_samples": text_samples,
                "return_tensors": "pt",
                "return_token_type_ids": False,
                "return_attention_mask": True,
                "truncation": True,
                "padding": True,
                "max_length": max_length,
            }
            eval_context_size = 4096
        else:
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "return_token_type_ids": False,
                "return_attention_mask": True,
                "truncation": True,
                "padding": True,
                "max_length": max_length,
            }
            eval_context_size = 1024
        samples = self.eval_model_tokenizer(text_samples, **tokenizer_kwargs)
        attn_mask = samples["attention_mask"]
        samples = samples["input_ids"]
        if "llama2" not in self.gen_ppl_eval_model_name_or_path:
            attn_mask = attn_mask.to(self.device)
            samples = samples.to(self.device)
        return samples, attn_mask, eval_context_size

    @torch.no_grad()
    def compute_generative_perplexity(
        self,
        text_samples: List[str],
        retokenize: bool = True,
        max_length: Optional[int] = None,
    ) -> None:
        """Compute the generative perplexity of the model.

        Args:
            text_samples: List of sentences generated by the model.

        Returns:
            Perplexity of the generated text under a different
            pre-trained AR model (e.g., GPT2).
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        eval_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.gen_ppl_eval_model_name_or_path
        ).eval()
        if max_length is None:
            max_length = self.config.model.length
        if "llama2" not in self.gen_ppl_eval_model_name_or_path:
            eval_model = eval_model.to(self.device)
        # Re-tokenize using eval model's tokenizer
        if retokenize:
            (samples, attn_mask, eval_context_size) = self.eval_retokenize(
                text_samples, max_length=max_length
            )
        else:
            samples = text_samples
            attn_mask = torch.ones(samples.shape).to(self.device)
            eval_context_size = samples.shape[-1]
        batch_size = min(self.config.eval.perplexity_batch_size, samples.shape[0])
        num_batches = samples.shape[0] // batch_size
        for i in range(num_batches):
            _samples = torch.split(
                samples[i * batch_size : (i + 1) * batch_size],
                eval_context_size,
                dim=-1,
            )
            _attn_mask = torch.split(
                attn_mask[i * batch_size : (i + 1) * batch_size],
                eval_context_size,
                dim=-1,
            )
            for sample_chunk, attn_mask_chunk in zip(_samples, _attn_mask):
                logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)[0]
                logits = logits.transpose(-1, -2)

                nlls = F.cross_entropy(
                    logits[..., :-1], sample_chunk[..., 1:], reduction="none"
                )
                first_eos = (
                    sample_chunk == self.eval_model_tokenizer.eos_token_id
                ).cumsum(-1) == 1
                token_mask = sample_chunk != self.eval_model_tokenizer.eos_token_id
                self.gen_ppl_metric.update(
                    nlls, first_eos[..., 1:] + token_mask[..., 1:]
                )