
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments from datasets import load_dataset
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) # Prepare the dataset
dataset = load_dataset("imdb")
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True) # Set up training arguments and Trainer
training_args = TrainingArguments(
output_dir="./results", evaluation_strategy="epoch",
learning_rate=2e-5, per_device_train_batch_size=16,
num_train_epochs=3
)
trainer = Trainer(
model=model, args=training_args,
train_dataset=tokenized_dataset["train"],
eval_dataset=tokenized_dataset["test"]
) # Train the model trainer.train()
