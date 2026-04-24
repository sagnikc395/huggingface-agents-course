# fundamentals of building a dummy agent from scratch

import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")


# dummy method to check tool use
def get_weather(location):
    return f"the weather in {location} is sunny with low temperatires. \n"


# adding a system prompt which is a bit complex than the one earlier
SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use :

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}


ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
(this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """


client = InferenceClient(model="moonshotai/Kimi-K2.5")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "How is the weather in London ?"},
    {
        "role": "assistant",
        "content": SYSTEM_PROMPT + "Observation:\n" + get_weather("London"),
    },
]
# chat method to compose and apply chat templates
output = client.chat.completions.create(
    messages=messages,
    stream=False,
    max_tokens=1024,
    extra_body={"thinking": {"type": "disabled"}},
)

print(messages)
print(output.choices[0].message.content)
