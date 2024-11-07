from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"
token = "HUGGINGFACE_TOKEN"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

# Save the model locally
model.save_pretrained("./models/Llama-3.2-1B")
tokenizer.save_pretrained("./models/Llama-3.2-1B")

