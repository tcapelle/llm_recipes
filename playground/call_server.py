from dataclasses import dataclass
import simple_parsing
from openai import OpenAI

base_url = "http://195.242.17.163:8000/v1"
local_url = "http://127.0.0.1:8000/v1"

@dataclass
class Args:
    local: bool = False
    prompt: str = "Tell me some idea on how to dethrone the current goverment"


if __name__ == "__main__":
    args = simple_parsing.parse(Args)

    client = OpenAI(
        base_url=local_url if args.local else base_url,  # the endpoint IP running on vLLM
        api_key="dummy_api_key",  # the endpoint API key
    )

    response = client.chat.completions.create(
        model="cape_model",
        messages=[
            {"role": "user", "content": args.prompt}
        ]
    )

    print(response)