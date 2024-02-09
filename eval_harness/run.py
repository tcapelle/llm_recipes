import wandb, os, simple_parsing, logging, json, sys
import subprocess
from typing import Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

WANDB_PROJECT = "prune_experiments"  # launch will override this
WANDB_ENTITY = "llm_surgery"

@dataclass
class LaunchOverride(simple_parsing.Serializable):
    "A small dataclass to hold the arguments for the eval harness, you can pass any eval_lm argument here."
    model_path: str = "mistralai/Mistral-7B-v0.1" # The path to the model 
    model_args: str = "dtype=\"bfloat16\"" # The model arguments
    tasks: str = "arc_easy" # The tasks to run, comma separated
    device: str = "cuda:0" # The device to use
    batch_size: int = 1 # The batch size to use, can affect results
    output_path: str = "./output/mistral7b" # The path to save the output to
    wandb_args: str = f"project={WANDB_PROJECT},entity={WANDB_ENTITY}" # The arguments passed to wandb.init, comma separated

def exec(cmd):
    env = os.environ.copy()
    conda_pre_cmd = f"{sys.executable} -m"  # trick to make pick up the right python
    cmd = conda_pre_cmd + cmd
    logging.info(f"\nRunning: {cmd}\n")
    result = subprocess.run(cmd, text=True, shell=True, env=env)
    if result.returncode != 0:
        print("Error")

def maybe_from_artifact(model_at_address: str):
    "Download the model from wandb if it's an artifact, otherwise return the path."
    try:
        model_dir = wandb.use_artifact(model_at_address).download()
        logging.info(f"Downloading model from wandb: {model_at_address}")
        return model_dir
    except:
        logging.info(f"Using model from local path: {model_at_address}")
        return model_at_address


def lm_eval(**kwargs):
    model_path = maybe_from_artifact(kwargs.pop("model_path"))
    model_args = f"pretrained={model_path}" + "," + kwargs.pop("model_args")
    wandb_args = kwargs.pop("wandb_args")
    cmd = (
        "lm_eval --model hf "
        f"--model_args {model_args} "
        f"--wandb_args {wandb_args} "
    )
    cmd += " ".join([f"--{k} {v}" for k, v in kwargs.items()])
    exec(cmd)

def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

def leftover_args_to_dict(leftover_args):
    "Convert leftover args to a dictionary."
    return {k[2:]: v for k, v in zip(leftover_args[::2], leftover_args[1::2])}

if __name__ == "__main__":
    args, leftover_args = simple_parsing.parse_known_args(LaunchOverride)
    if len(leftover_args)==0:
        leftover_args = {}
    else:
        leftover_args = leftover_args_to_dict(leftover_args)
    
    logging.info(f"Using arguments: {args}")
    lm_eval(**args.to_dict(), **leftover_args)
