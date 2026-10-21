from openai import OpenAI
from reachy_mini import ReachyMini

# Connect to OpenAI
client = OpenAI()

# Connect to Reachy
robot = ReachyMini()

print("Connected to Reachy Mini!")
print("OpenAI conversation started.")
print("Type 'quit' to exit.")

while True:
    message = input("\nYou: ")

    if message.lower() == "quit":
        break

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=message,
    )

    answer = response.output_text

    print("AI:", answer)
