from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

MODEL = 'gpt-4o-mini'
prompt = 'write something short but funny'

chat_completion = client.chat.completions.create(
    model=MODEL,
    messages=[{'role': 'user', 'content': prompt}]
)

print(chat_completion.choices[0].message.content)
