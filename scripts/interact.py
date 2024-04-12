from dataclasses import dataclass

import torch
import simple_parsing
from transformers import pipeline, TextStreamer

@dataclass
class Args:
    model_id: str = "wandb/zephyr-orpo-7b-v0.2"
    max_new_tokens: int = 400
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    device_map: str = "auto"

args = simple_parsing.parse(Args)

pipe = pipeline("text-generation", model=args.model_id, torch_dtype=torch.bfloat16, device_map=args.device_map)

print("Welcome to the interactive chat experience!\nType 'quit' to exit the chat.")

messages = [
    {
        "role": "system",
        "content": "You are Zephyr, a helpful assistant.",
    }
]

while True:
    user_input = input("\033[94mUser: \033[0m")
    if user_input.lower() == "quit":
        break

    messages.append({"role": "user", "content": user_input})
    streamer = TextStreamer(pipe.tokenizer, skip_prompt=True)
    outputs = pipe(
        messages,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        streamer=streamer
    )

    assistant_response = outputs[0]["generated_text"][-1]["content"]
    messages.append({"role": "assistant", "content": assistant_response})

print("Thank you for chatting. Goodbye!")