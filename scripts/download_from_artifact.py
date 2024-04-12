import wandb
from dataclasses import dataclass
import simple_parsing

@dataclass
class Config:
    artifact_name: str = "llm_surgery/shearllama/mistral_0-7:v1"
    artifact_type: str = "model"
    output_folder: str = "artifacts"


config = simple_parsing.parse(Config)

# Download the artifact
run = wandb.init()
artifact = run.use_artifact(config.artifact_name, type=config.artifact_type)
artifact_dir = artifact.download(root=config.output_folder)
