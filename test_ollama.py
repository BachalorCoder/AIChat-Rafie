import ollama

response = ollama.chat(
    model="qwen3.6:27b",
    messages=[
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
)

print(response["message"]["content"])