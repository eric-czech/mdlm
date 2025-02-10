from transformers import PretrainedConfig

class NoiseConfig(PretrainedConfig):
    """Nested config for noise parameters."""
    def __init__(
        self,
        type: str = "loglinear",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = type


class ModelConfig(PretrainedConfig):
    """Nested config for model architecture."""
    def __init__(
        self,
        type: str = "ddit",
        hidden_size: int = 768,
        cond_dim: int = 128,
        length: int = 1024,
        n_blocks: int = 12,
        n_heads: int = 12,
        scale_by_sigma: bool = True,
        dropout: float = 0.1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = type
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.length = length
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.scale_by_sigma = scale_by_sigma
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings


class TrainingConfig(PretrainedConfig):
    """Nested config for training parameters."""
    def __init__(
        self,
        ema: float = 0.9999,
        antithetic_sampling: bool = True,
        importance_sampling: bool = False,
        sampling_eps: float = 1e-3,
        change_of_variables: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ema = ema
        self.antithetic_sampling = antithetic_sampling
        self.importance_sampling = importance_sampling
        self.sampling_eps = sampling_eps
        self.change_of_variables = change_of_variables

class MDLMConfig(PretrainedConfig):
    model_type = "mdlm"

    def __init__(
        self,
        backbone: str = "dit",
        parameterization: str = "subs",
        time_conditioning: bool = False,
        T: int = 0,
        subs_masking: bool = False,
        # Nested configs
        model=None,
        noise=None,
        training=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.parameterization = parameterization
        self.time_conditioning = time_conditioning
        self.T = T
        self.subs_masking = subs_masking

        if model is None:
            model = {}
        if isinstance(model, dict):
            model = ModelConfig(**model)
        self.model = model

        if noise is None:
            noise = {}
        if isinstance(noise, dict):
            noise = NoiseConfig(**noise)
        self.noise = noise

        if training is None:
            training = {}
        if isinstance(training, dict):
            training = TrainingConfig(**training)
        self.training = training

