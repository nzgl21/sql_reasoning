from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import model_config

base_model = AutoModelForCausalLM.from_pretrained(model_config.model.name)
tokenizer = AutoTokenizer.from_pretrained(model_config.model.name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = (
    LoraConfig(**model_config.lora.model_dump())
    if model_config.lora is not None
    else None
)
