#1.
import torch

#set up apple M1 chip gpu usage
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Set device to MPS
    print(f"Using device: {device}")
else:
    print("MPS not available, falling back to CPU")

#Sample API Integration - Here’s how you can use OpenAI’s API with Hugging Face to enhance your LLM workflows:
import openai
openai.api_key = 'your-api-key'
response = openai.Completion.create( model="gpt-4", prompt="Write a Python function to calculate the factorial of a number.", max_tokens=100 )
print(response.choices[0].text.strip()) 

#2. Hands-On Fine-Tuning of a Small-Scale LLM
#We’ll fine-tune DistilBERT for a sentiment classification task (e.g., customer reviews).
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name) 

#Prepare the Dataset
#Use the Hugging Face datasets library to load and preprocess your data.
from datasets import load_dataset
dataset = load_dataset("imdb")
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)
tokenized_dataset = dataset.map(preprocess_function, batched=True)


#Set Up the Trainer
#The Hugging Face Trainer simplifies the fine-tuning process.
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
output_dir="./results", evaluation_strategy="epoch",
learning_rate=2e-5, per_device_train_batch_size=16,
num_train_epochs=3, weight_decay=0.01,
)
trainer = Trainer(
model=model, args=training_args,
train_dataset=tokenized_dataset["train"],
eval_dataset=tokenized_dataset["test"],
)


#Fine-tune the model with a single command:
trainer.train()

#Save your Fine-Tuned model and tokenizer for later use:
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
#3. Evaluating Fine-Tuned Models Using Relevant Metrics
#Metrics to Use
    #Accuracy: Percentage of correct predictions.
    #F1-Score: Balances precision and recall for overall performance.
    #Perplexity: Useful for generative tasks.

#Evaluate the Model
#Using Trainer:
results = trainer.evaluate()
print(results)

#Example output:
#{'eval_loss': 0.2, 'eval_accuracy': 0.91}

      
#For Detailed Metrics:
#Use sklearn for a detailed report:
from sklearn.metrics import classification_report
predictions = trainer.predict(tokenized_dataset["test"])
y_pred = predictions.predictions.argmax(axis=1)
y_true = tokenized_dataset["test"]["label"]
print(classification_report(y_true, y_pred))


