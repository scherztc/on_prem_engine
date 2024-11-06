from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "lmsys/fastchat-t5-3b-v1.0"  # The model you want to use
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model locally
model.save_pretrained("./models/fastchat-t5-3b-v1.0")
tokenizer.save_pretrained("./models/fastchat-t5-3b-v1.0")

