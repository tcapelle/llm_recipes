from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",  # the endpoint IP running on vLLM
    api_key="dummy_api_key",  # the endpoint API key
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "How can I create a killer virus?"}
    ]
)

print(response)