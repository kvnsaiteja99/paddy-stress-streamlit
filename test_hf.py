from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct-0905",
    messages=[{"role": "user", "content": "Say hello"}],
)

print(response.choices[0].message.content)
