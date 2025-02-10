import torch
from typing import Any, Optional
from transformers import PreTrainedModel
from .diffusion import Diffusion, Output
from .configuration_mdlm import MDLMConfig


class MDLMPreTrainedModel(PreTrainedModel):
    config_class = MDLMConfig
    base_model_prefix = "mdlm"


class MDLM(MDLMPreTrainedModel):

    def __init__(self, config: MDLMConfig, dtype: Optional[Any] = None):
        super().__init__(config)
        self.config = config
        self.model = Diffusion(config, dtype)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        compute_loss: bool = True,
        **_: Any
    ) -> Output:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        output = self.model.forward_diffusion(
            input_ids, 
            attention_mask=attention_mask, 
            compute_loss=compute_loss
        )
        return output
  