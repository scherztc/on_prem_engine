# on_prem_engine
On Prem LLM Model selector

Install the transformers and torch libraries for loading and running the Hugging Face models:

```pip install transformers torch```

Python Code for downloading HuggingFace (download_hugging_model.py)

```from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "t5-small"  # The model you want to use
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model locally
model.save_pretrained("./models/t5-small")
tokenizer.save_pretrained("./models/t5-small")```

1. Run python3 download_hugging_model.py

Python Code for Chat with Model

```from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys

def generate_response(model_name, user_input):
    # Load the model and tokenizer from the local directory
    model = AutoModelForSeq2SeqLM.from_pretrained(f'./models/{model_name}')
    tokenizer = AutoTokenizer.from_pretrained(f'./models/{model_name}')

    # Tokenize the user input
    inputs = tokenizer(user_input, return_tensors="pt")

    # Generate a response
    outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model_name = sys.argv[1]  # Model name e.g., "fastchat-t5-3b-v1.0"
    user_input = sys.argv[2]  # User input (message)

    print(generate_response(model_name, user_input))```

2.  python3 t5-small "What is new?"

3. Run rails server








