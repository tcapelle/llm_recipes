import wandb, os, simple_parsing, logging, json, sys
import subprocess
from typing import Optional
from dataclasses import dataclass
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

WANDB_PROJECT = "prune_experiments"  # launch will override this
WANDB_ENTITY = "llm_surgery"

@dataclass
class LaunchOverride(simple_parsing.Serializable):
    "A small dataclass to hold the arguments for the eval harness, you can pass any eval_lm argument here."
    model_path: str = "mistralai/Mistral-7B-v0.1" # The path to the model 
    model_args: str = "dtype=\"bfloat16\"" # The model arguments
    tasks: str = "arc_easy" # The tasks to run, comma separated
    log_samples: bool = True # Whether to log samples
    device: str = "cuda:0" # The device to use
    batch_size: int = 1 # The batch size to use, can affect results
    output_path: str = "./output/mistral7b" # The path to save the output to
    wandb_args: str = f"project={WANDB_PROJECT},entity={WANDB_ENTITY},job_type=lm_eval" # The arguments passed to wandb.init, comma separated

def exec(cmd):
    env = os.environ.copy()
    conda_pre_cmd = f"{sys.executable} -m "  # trick to make pick up the right python
    cmd = conda_pre_cmd + cmd
    logging.info(f"\nRunning: {cmd}\n")
    result = subprocess.run(cmd, text=True, shell=True, env=env)
    if result.returncode != 0:
        print("Error")

def maybe_from_artifact(model_at_address: str):
    "Download the model from wandb if it's an artifact, otherwise return the path."
    try:
        api = wandb.Api()
        model_dir = api.artifact(model_at_address).download()
        logging.info(f"Downloading model from wandb: {model_at_address}")
        return True, model_dir
    except:
        logging.info(f"Using model from local path: {model_at_address}")
        return False, model_at_address


def lm_eval(kwargs, leftover_args):
    logging.info(f"Running lm_eval with kwargs: {kwargs}")
    model_at_address = kwargs.pop("model_path")
    is_at, model_path = maybe_from_artifact(model_at_address)
    if is_at:
        original_model_name, alias = model_at_address.split("/")[-1].split(":")  # let's remove the artifact alias
    else:
        original_model_name = model_path.split("/")[-1]
        alias = "hf"
    # create the wandb run
    run = wandb.init(
        project=WANDB_PROJECT, 
        entity=WANDB_ENTITY, 
        name=f"{original_model_name}_{alias}_eval", 
        group=original_model_name, 
        job_type="lm_eval")
    if is_at:
        at = wandb.use_artifact(model_at_address)
    model_args = f"pretrained={model_path}" + "," + kwargs.pop("model_args")
    wandb_args = kwargs.pop("wandb_args")
    wandb_args = f"project={run.project},entity={run.entity},job_type=lm_eval" # The arguments passed to wandb.init, comma separated

    log_samples = kwargs.pop("log_samples")
    cmd = (
        "lm_eval --model=hf "
        f"--model_args={model_args} "
        f"--wandb_args={wandb_args},id={run.id},resume=allow " # resume the run
    )
    run.finish()
    cmd += " ".join([f"--{k}={v}" for k, v in kwargs.items()])
    if log_samples:
        cmd += " --log_samples "

    # add the leftover args
    cmd += " " + " ".join(leftover_args)
    exec(cmd)

if __name__ == "__main__":
    args, leftover_args = simple_parsing.parse_known_args(LaunchOverride)
    logging.info(f"args: {args}\nleftover_args: {leftover_args}")
    lm_eval(args.to_dict(), leftover_args)
