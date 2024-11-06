from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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
    
    print(generate_response(model_name, user_input))
