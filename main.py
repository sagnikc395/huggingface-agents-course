from smolagents import LiteLLMModel, CodeAgent
from transformers import AutoTokenizer

from src.simple_calculator_tool import calculator

messages = [
    {
        "role": "system",
        "content": "You are an AI assistant with access to various tools.",
    },
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]


def tokenize_data(messages=messages):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    rendered_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print(rendered_prompt)

    # use the tool
    print(calculator.to_string())


def main():
    model = LiteLLMModel(
        model_id="ollama_chat/qwen2:7b",
        api_base="http://127.0.0.1:11434",
        num_ctx=8192,
    )

    agent = CodeAgent(tools=[], model=model)

    result = agent.run(
        "Calculate the sum of numbers from 1 to 10 and write a python program for it"
    )
    print(result)


if __name__ == "__main__":
    # main()
    tokenize_data(messages=messages)
