from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
import sys

def generate_response(model_name, user_input):
    # Load the configuration to check model type
    config = AutoConfig.from_pretrained(f'./models/{model_name}')
    
    # Choose model class based on configuration
    if config.model_type == "llama":
        model = AutoModelForCausalLM.from_pretrained(f'./models/{model_name}')
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(f'./models/{model_name}')
    
    # Load the tokenizer (the same for both model types)
    tokenizer = AutoTokenizer.from_pretrained(f'./models/{model_name}')

    # Tokenize the user input
    inputs = tokenizer(user_input, return_tensors="pt")

    # Generate a response
    outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model_name = sys.argv[1]  # Model name e.g., "meta-llama/Llama-3.2-1B" or "fastchat-t5-3b-v1.0"
    user_input = sys.argv[2]  # User input (message)

    print(generate_response(model_name, user_input))

