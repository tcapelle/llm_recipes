import openai
import weave

weave.init("llama_405B")

FW_API_KEY = "LpqtGeN2SH26tuq6jV2GyED7mSsnqK3NT6bQyjAbsH70deyZ"

client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key=FW_API_KEY,
)

@weave.op
def call_llama(prompt, model="accounts/fireworks/models/llama-v3p1-405b-instruct", **kwargs):
    response = client.chat.completions.create(
      model=model,
      messages=[{
        "role": "user",
        "content": prompt,
      }],
      **kwargs
    )
    return response.choices[0].message.content

print(call_llama("Tell me about yourself?"))