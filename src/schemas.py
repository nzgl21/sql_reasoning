from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Конфигурация базовой модели и токенизатора."""
    
    name: str = Field()
    cache_dir: Path | None = Field()

    
    @classmethod
    def from_yaml(cls, file_path: Path | str):
        """Загружает конфиг из yaml

        Args:
            file_path (Path | str): Путь к yaml
        """ 
        file_path = Path(file_path)
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.model_validate(data)


class LoRaConfig(BaseModel):
    """Конфигурация параметров LoRA (Low-Rank Adaptation) для PEFT."""
    
    r: int = Field(
        default=16,
        ge=1,
        le=256,
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
    )
    task_type: str = Field(
        default="CAUSAL_LM",
    )
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    
    @classmethod
    def from_yaml(cls, file_path: Path | str):
        """Загружает конфиг из yaml

        Args:
            file_path (Path | str): Путь к yaml
        """ 
        file_path = Path(file_path)
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.model_validate(data)
    
    
class FullModelConfig(BaseModel):
    """Общий конфиг модели и адаптера."""
    model: ModelConfig
    lora: LoRaConfig | None = None
    @classmethod
    def from_yaml(cls, file_path: Path | str) -> "FullModelConfig":
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)