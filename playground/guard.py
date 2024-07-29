import re
import torch
import weave
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Any
from pydantic import PrivateAttr

class PromptGuard(weave.Model):    
    device: str = "cpu"
    _classifier: Any = PrivateAttr()

    def model_post_init(self, __context):
        self._classifier = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M", device_map=self.device)

    @weave.op
    async def predict(self, messages):
        text = messages[0]["content"]   
        return self._classifier(text)[0]

class LlamaGuard(weave.Model):
    device: str = "cuda"
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def model_post_init(self, __context):
        self._model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-8B", torch_dtype=torch.bfloat16, device_map=self.device)
        self._tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-8B")

    @weave.op
    async def predict(self, chat):
        input_ids = self._tokenizer.apply_chat_template(chat, return_tensors="pt").to(self._model.device)
        output = self._model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        llama_guard_out = self._tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        # Extract the error number from llama_guard_out
        error_match = re.search(r'S(\d+)', llama_guard_out)
        error_number = error_match.group(0) if error_match else 0
        return {"safe": "unsafe" not in llama_guard_out, "categories": error_number}