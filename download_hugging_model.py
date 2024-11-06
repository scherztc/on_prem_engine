from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "t5-small"  # The model you want to use
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model locally
model.save_pretrained("./models/t5-small")
tokenizer.save_pretrained("./models/t5-small")

