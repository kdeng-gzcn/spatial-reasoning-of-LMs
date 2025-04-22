import argparse
from src.models.utils import load_model

parser = argparse.ArgumentParser(description="Chatbot arguments")
parser.add_argument("--model_id", type=str, required=True, help="ID of the model to use")
args = parser.parse_args()
model_id = args.model_id

def interactive_chat():
    model = load_model(model_id)
    print("Interactive LLM Chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        try:
            reply = model.pipeline(user_input)
            print(f"LLM: {reply}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_chat()