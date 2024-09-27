import os
import json
from pathlib import Path
import time
import openai
from pydantic import BaseModel, ConfigDict, model_validator, field_validator,Field
import logging
import simple_parsing
from dataclasses import dataclass


logger = logging.getLogger(__name__)

SERVER_URL = os.environ.get("SERVER_URL")
SERVER_API_KEY = os.environ.get("SERVER_API_KEY")

client = openai.OpenAI(
    api_key=SERVER_API_KEY,
    base_url=SERVER_URL,
)


class Body(BaseModel):
    model: str = Field(
        ..., description="Must be 'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo'"
    )
    model_config = ConfigDict(extra="allow")

    # Using field_validator to validate the model field
    @field_validator('model')
    def validate_model(cls, v):
        if not "Llama-3.2" in v:
            raise ValueError(
                "model must be 'meta-llama/Llama-3.2-*'"
            )
        return v


class RequestPayload(BaseModel):
    headers: dict = Field(
        ...,
        description="Must include 'Authorization' header with a Bearer token of exactly 40 alphanumeric characters",
    )
    body: Body
    wandb_api_key: str = Field(..., description="Automatically extracted from the Authorization header")

    # Using model_validator with mode='before' to set wandb_api_key
    @model_validator(mode='before')
    def set_wandb_api_key(cls, values):
        headers = values.get('headers', {})
        auth_token = headers.get('authorization', '').split('Bearer ')[-1]
        if auth_token and len(auth_token) == 40:
            values['wandb_api_key'] = auth_token
        else:
            raise ValueError("Invalid Authorization token")
        return values


def clean_wandb_api_key():
    if "WANDB_API_KEY" in os.environ:
        os.unsetenv("WANDB_API_KEY")
        del os.environ["WANDB_API_KEY"]

    # Delete the ~/.netrc file if it exists
    netrc_path = os.path.expanduser("~/.netrc")
    if os.path.exists(netrc_path):
        try:
            os.remove(netrc_path)
            logger.info(f"Deleted {netrc_path}")
        except OSError as e:
            logger.error(f"Error deleting {netrc_path}: {e}")
    else:
        logger.info(f"{netrc_path} does not exist")


def setup_wandb(wandb_api_key: str, weave_project: str):
    clean_wandb_api_key()
    os.environ["WANDB_API_KEY"] = wandb_api_key
    import wandb

    wandb.login(key=wandb_api_key, relogin=True, verify=True)
    time.sleep(1)
    print("Attempt login to Weave")
    import weave

    weave.init(weave_project)


def call_model(payload: RequestPayload, weave_project: str):
    setup_wandb(payload.wandb_api_key, weave_project)
    response = client.chat.completions.create(**payload.body.model_dump())
    clean_wandb_api_key()
    return response


def load_json(file: str | Path):
    if isinstance(file, str):
        with open(file, "r") as f:
            return json.load(f)
    else:
        with open(file, "r") as f:
            return json.load(f)


@dataclass
class Args:
    payload_file: str | Path
    weave_project: str = "prompt-eng/llama-90B-wandb"


if __name__ == "__main__":
    args = simple_parsing.parse(Args)
    payload = load_json(args.payload_file)
    payload = RequestPayload.model_validate(payload)

    print(payload)

    result = call_model(payload, args.weave_project)
    json_payload = result.model_dump_json(indent=2)

    print("<response>")
    print(json_payload)
    print("</response>")

    # from openai.types.chat.chat_completion import ChatCompletion

    # reconstructed_result = ChatCompletion.model_validate_json(json_payload)
    # print(reconstructed_result, type(reconstructed_result))
