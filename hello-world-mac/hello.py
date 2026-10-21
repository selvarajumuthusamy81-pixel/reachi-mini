from openai import OpenAI

client = OpenAI()

print("Hello! Chat with OpenAI. Type 'quit' to exit.")

while True:
    message = input("\nYou: ")

    if message.lower() == "quit":
        break

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=message,
    )

    print("AI:", response.output_text)
