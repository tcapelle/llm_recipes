import requests
import os
from rich.console import Console
from dataclasses import dataclass
from simple_parsing import ArgumentParser

from illustrator import IllustratePayload


console = Console()
# defaults
LOCAL_URL = "http://localhost:8020/illustrate/story"
REMOTE_URL = "http://195.242.25.198:8020/illustrate/story"
WANDB_API_KEY = os.environ["WANDB_API_KEY"]

def call_illustrate(payload: dict, url: str):
    console.print(f"Sending Payload to url={url} :")
    console.print(payload)
    console.print("[cyan]==============================================================[/cyan]")
    response = requests.post(
        url, 
        headers={"wandb-api-key": payload["wandb_api_key"]}, json=payload)

    return response.json()

@dataclass
class Payload:
    story: str = "Once upon a time, in a land far, far away, there was a little girl named Alice. Alice loved to explore the world around her. One day, she decided to go on an adventure to find the most beautiful flower in the world. Alice traveled through forests, over mountains, and across rivers. She met many interesting creatures along the way, like a friendly dragon and a wise old owl. After a long journey, Alice finally found the most beautiful flower in the world. She picked it and brought it back to her home. From that day on, Alice knew that with courage and determination, she could achieve anything."
    story_title: str = "Alice's Flower Adventure"
    story_author: str = "A.A. Milne"
    story_setting: str = "A magical land filled with talking animals and enchanted creatures"
    wandb_api_key: str = WANDB_API_KEY
    return_images: bool = False
    dummy: bool = False

    def validate(self):
        dict_payload = vars(self)
        dict_payload.pop("wandb_api_key")
        return IllustratePayload.model_validate()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Call local server")
    parser.add_arguments(Payload, dest="payload")
    args = parser.parse_args()
    url = LOCAL_URL if args.local else REMOTE_URL
    payload = IllustratePayload.model_validate(vars(args.payload))
    response = call_illustrate(payload.model_dump(), url)
    console.print("Response:")
    console.print(response)