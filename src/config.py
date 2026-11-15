from pathlib import Path

from schemas import FullModelConfig

PROJECT_PATH = Path(__file__).parent.parent
TRAIN_CONFIG_PATH = PROJECT_PATH / "configs"

model_config = FullModelConfig.from_yaml(TRAIN_CONFIG_PATH / "model.yaml")
sft_config = None
