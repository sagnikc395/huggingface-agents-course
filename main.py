from smolagents import LiteLLMModel, CodeAgent


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
    main()
