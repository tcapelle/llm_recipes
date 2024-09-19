import requests
import os

URL = "http://localhost:8020/illustrate/story"
WANDB_API_KEY = "1"*40
# WANDB_API_KEY = os.environ["WANDB_API_KEY"]


illustrator_payload = dict(
    story="Once upon a time, in a land far, far away, there was a little girl named Alice. Alice loved to explore the world around her. One day, she decided to go on an adventure to find the most beautiful flower in the world. Alice traveled through forests, over mountains, and across rivers. She met many interesting creatures along the way, like a friendly dragon and a wise old owl. After a long journey, Alice finally found the most beautiful flower in the world. She picked it and brought it back to her home. From that day on, Alice knew that with courage and determination, she could achieve anything.",
    story_title="Alice's Flower Adventure",
    story_author="A.A. Milne",
    story_setting="A magical land filled with talking animals and enchanted creatures",
)

response = requests.post(URL, headers={"wandb-api-key": WANDB_API_KEY}, json=illustrator_payload)
print(response.json())
