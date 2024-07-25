import torch
import weave
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Any
from pydantic import PrivateAttr

class PromptGuard(weave.Model):     
    _classifier: Any = PrivateAttr()

    def model_post_init(self, __context):
        self._classifier = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M", device_map="cpu")

    @weave.op
    def predict(self, messages):
        text = messages[0]["content"]   
        return self._classifier(text)[0]

class LlamaGuard(weave.Model):
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def model_post_init(self, __context):
        self._model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-8B", torch_dtype=torch.bfloat16, device_map="cuda")
        self._tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-8B")

    @weave.op
    def predict(self, chat):
        input_ids = self._tokenizer.apply_chat_template(chat, return_tensors="pt").to(self._model.device)
        output = self._model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self._tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)