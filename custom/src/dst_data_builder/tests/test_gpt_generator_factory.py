"""
Tests for GPTGeneratorFactory

Factory creates appropriate generator instances based on configuration.
Critical for ensuring correct generator type and initialization.
"""

import pytest
from dst_data_builder.gpt_generators.gpt_generator_factory import GPTGeneratorFactory
from dst_data_builder.gpt_generators.single_gpt_generator import SingleGPTGenerator
from dst_data_builder.gpt_generators.batch_gpt_generator import BatchGPTGenerator
from dst_data_builder.validators.structure_validator import StructureValidator
from dst_data_builder.validators.timestamps_validator import TimestampsValidator


# ============================================================================
# Test Data
# ============================================================================

BASE_PARAMS = {
    "model_name": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 1000,
    "max_retries": 2,
}


# ============================================================================
# Generator Creation Tests
# ============================================================================

@pytest.mark.parametrize("gen_type,expected_class,env_var,env_value", [
    ("single", SingleGPTGenerator, "OPENROUTER_API_KEY", "test_key_single"),
    ("batch", BatchGPTGenerator, "OPENAI_API_KEY", "test_key_batch"),
    ("SINGLE", SingleGPTGenerator, "OPENROUTER_API_KEY", "test_key"),  # case insensitive
    ("Single", SingleGPTGenerator, "OPENROUTER_API_KEY", "test_key"),  # mixed case
])
def test_factory_creates_generators(monkeypatch, gen_type, expected_class, env_var, env_value):
    """Test factory creates correct generator types"""
    monkeypatch.setenv(env_var, env_value)
    
    gen = GPTGeneratorFactory.create_generator(generator_type=gen_type, **BASE_PARAMS)
    
    assert isinstance(gen, expected_class)
    assert gen.model_name == BASE_PARAMS["model_name"]
    assert gen.temperature == BASE_PARAMS["temperature"]
    assert gen.max_tokens == BASE_PARAMS["max_tokens"]


def test_factory_batch_with_config(monkeypatch):
    """Test factory creates BatchGPTGenerator with custom config"""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    
    gen = GPTGeneratorFactory.create_generator(
        generator_type="batch",
        generator_cfg={"batch_size": 10},
        **BASE_PARAMS
    )
    
    assert isinstance(gen, BatchGPTGenerator)


def test_factory_invalid_generator_type():
    """Test factory raises error for invalid generator type"""
    with pytest.raises(ValueError, match="Unsupported generator type"):
        GPTGeneratorFactory.create_generator(generator_type="invalid", **BASE_PARAMS)


# ============================================================================
# Validator Tests
# ============================================================================

def test_factory_with_custom_validators(monkeypatch):
    """Test factory accepts custom validators in config"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    
    custom_validators = [StructureValidator(), TimestampsValidator()]
    gen = GPTGeneratorFactory.create_generator(
        generator_type="single",
        generator_cfg={"validators": custom_validators},
        **BASE_PARAMS
    )
    
    assert len(gen.validators) == 2
    assert isinstance(gen.validators[0], StructureValidator)
    assert isinstance(gen.validators[1], TimestampsValidator)


def test_factory_default_validators(monkeypatch):
    """Test factory creates default validators when none specified"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    
    gen = GPTGeneratorFactory.create_generator(generator_type="single", **BASE_PARAMS)
    
    # Should have default validators (Structure, Timestamps, Id)
    assert len(gen.validators) == 3


# ============================================================================
# API Key Tests
# ============================================================================

@pytest.mark.parametrize("gen_type,env_var", [
    ("single", "OPENROUTER_API_KEY"),
    ("batch", "OPENAI_API_KEY"),
])
def test_factory_missing_api_key(monkeypatch, gen_type, env_var):
    """Test factory handles missing API keys"""
    monkeypatch.delenv(env_var, raising=False)
    
    gen = GPTGeneratorFactory.create_generator(generator_type=gen_type, **BASE_PARAMS)
    
    # Should still create generator, but api_key will be None
    assert gen is not None


def test_factory_respects_generator_cfg_parameters(monkeypatch):
    """Test factory passes through generator-specific config"""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    
    gen = GPTGeneratorFactory.create_generator(
        generator_type="batch",
        model_name="gpt-4o",
        temperature=0.5,
        max_tokens=2000,
        max_retries=5,
        generator_cfg={"batch_size": 20, "check_interval": 30},
    )
    
    assert isinstance(gen, BatchGPTGenerator)
    assert gen.batch_size == 20
    assert gen.check_interval == 30


def test_factory_default_parameters(monkeypatch):
    """Test factory uses sensible defaults"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    
    gen = GPTGeneratorFactory.create_generator(generator_type="single")
    
    assert isinstance(gen, SingleGPTGenerator)
    assert gen.model_name == "gpt-4o"
    assert gen.temperature == 0.1
    assert gen.max_tokens == 4000
