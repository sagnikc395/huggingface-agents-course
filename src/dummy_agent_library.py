# fundamentals of building a dummy agent from scratch

import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")


client = InferenceClient(model="moonshotai/Kimi-K2.5")

# chat method to compose and apply chat templates
output = client.chat.completions.create(
    messages=[{"role": "user", "content": "The Capital of France is"}],
    stream=False,
    max_tokens=1024,
    extra_body={"thinking": {"type": "disabled"}},
)

print(output.choices[0].message.content)
