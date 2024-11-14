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

    # Set pad_token to eos_token if padding token is not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Tokenize the user input, specifying padding to make sure attention_mask is generated
    inputs = tokenizer(user_input, return_tensors="pt", padding=True)

    # Generate a response with attention_mask and pad_token_id set
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Explicitly set attention mask
        pad_token_id=tokenizer.pad_token_id,      # Set pad_token_id if defined
        max_length=150,
        num_return_sequences=1
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model_name = sys.argv[1]  # Model name e.g., "meta-llama/Llama-3.2-1B" or "fastchat-t5-3b-v1.0"
    user_input = sys.argv[2]  # User input (message)

    print(generate_response(model_name, user_input))

