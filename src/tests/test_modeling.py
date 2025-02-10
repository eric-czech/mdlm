
from typing import Dict
import warnings
import pytest
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
from composer import Trainer as ComposerTrainer
from composer.models import HuggingFaceModel
from composer.optim import DecoupledAdamW

from src.diffusion import Diffusion, tokenization_params
from src.configuration_mdlm import MDLMConfig
from src.modeling_mdlm import MDLM

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")

@pytest.fixture
def config(tokenizer):
    return MDLMConfig(**tokenization_params(tokenizer))

@pytest.fixture
def device():
    return torch.device("cuda")

def seed_everything(seed: int=0):
    # Adapted from: 
    # https://pytorch-lightning.readthedocs.io/en/1.7.7/_modules/pytorch_lightning/utilities/seed.html#seed_everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_inputs(config, batch_size=2, seq_length=16, device=None):
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones((batch_size, seq_length), device=device)
    return input_ids, attention_mask

def test_diffusion_forward(config, device):
    # Initialize model
    model = Diffusion(config).to(device)
    model.eval()

    # Create sample input
    input_ids, _ = create_inputs(config, device=device)
    batch_size, seq_length = input_ids.shape
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    noise, _ = model.noise(timesteps)

    # Run forward pass
    with torch.no_grad():
        output = model(input_ids, sigma=noise)

    # Basic output checks
    assert output.shape == (batch_size, seq_length, config.vocab_size)
    assert not torch.isnan(output).any()

def test_hf_model_forward(config, device):
    # Initialize model
    model = MDLM(config).to(device)
    model.eval()

    # Create sample input
    input_ids, attention_mask = create_inputs(config, device=device)
    batch_size, seq_length = input_ids.shape

    # Run forward pass
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    # Basic output checks
    assert output.loss is not None
    assert output.logits is not None
    assert output.nlls is not None
    assert output.token_mask is not None
    assert not torch.isnan(output.logits).any()
    assert not torch.isnan(output.nlls).any()
    assert not torch.isnan(output.token_mask).any()

    # Check output shapes
    assert output.logits.shape == (batch_size, seq_length, config.vocab_size)
    assert output.nlls.shape == (batch_size, seq_length)
    assert output.token_mask.shape == (batch_size, seq_length) 


def assert_config_equal(expected: MDLMConfig, actual: MDLMConfig) -> None:
    assert isinstance(expected, MDLMConfig)
    assert isinstance(actual, MDLMConfig)
    expected = expected.to_dict()
    actual = actual.to_dict()
    assert expected.keys() == actual.keys()
    for key in expected.keys():
        if key.startswith("_"):
            continue
        assert expected[key] == actual[key], (
            f"Config values for {key=!r} do not match, "
            f"got {actual[key]=!r} instead of {expected[key]=!r}"
        )

def test_automodel_roundtrip(tmp_path, config, device):
    AutoConfig.register("mdlm", MDLMConfig)
    AutoModel.register(MDLMConfig, MDLM)

    # Initialize model
    model = MDLM(config).to(device)

    # Create input
    input_ids, attention_mask = create_inputs(config, device=device)
    with torch.no_grad():
        seed_everything()
        expected = model(input_ids, attention_mask=attention_mask)

    # Save model
    model_path = tmp_path / "mdlm-model"
    model.save_pretrained(model_path)

    # Reload and validate
    model = AutoModel.from_pretrained(model_path).to(device)
    assert_config_equal(config, model.config)
    with torch.no_grad():
        seed_everything()
        actual = model(input_ids, attention_mask=attention_mask)
    assert torch.equal(actual.loss, expected.loss)
    assert torch.equal(actual.logits, expected.logits)
    assert torch.equal(actual.nlls, expected.nlls)
    assert torch.equal(actual.token_mask, expected.token_mask)


class SimpleDataset(Dataset):

    def __init__(self, inputs: Dict[str, torch.Tensor]):
        self.inputs = inputs

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        return self.inputs

def test_mosaic_training(config, tmp_path, device):
    # Initialize model
    model = MDLM(config).to(device)

    # Create sample input
    input_ids, _ = create_inputs(config, device=device)
    inputs = {"input_ids": input_ids}

    # Configure trainer
    composer_model = HuggingFaceModel(model)
    optimizer = DecoupledAdamW(model.parameters())
    trainer = ComposerTrainer(
        optimizers=optimizer,
        model=composer_model,
        log_to_console=True,
        progress_bar=False,
        train_dataloader=SimpleDataset(inputs),
        precision="amp_fp16",
        max_duration="3ba",
        device_train_microbatch_size=4,
        save_folder=str(tmp_path),
        save_interval="1ep",
        save_overwrite=True,
    )

    # Run training
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict .*",
        )
        trainer.fit()