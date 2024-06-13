# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import nltk
from tqdm import tqdm


# Step 2: Set the path to your model in Google Drive
model_path = '/content/drive/MyDrive/trained_medical_chatbot'  # Replace with your model's path in Google Drive

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Load the dataset
dataset = load_dataset('ruslanmv/ai-medical-chatbot')
print(dataset)

# Split the 'train' dataset into 'train' and 'validation' subsets
split = dataset['train'].train_test_split(test_size=0.1)
train_dataset_full = split['train']
eval_dataset = split['test']

# Prepare data function for evaluation
def prepare_data(examples):
    concatenated_texts = [
        f"Patient: {pat} Doctor: {doc}"
        for pat, doc in zip(examples['Patient'], examples['Doctor'])
    ]
    model_inputs = tokenizer(
        concatenated_texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return {
        'input_ids': model_inputs['input_ids'],
        'attention_mask': model_inputs['attention_mask'],
        'labels': model_inputs['input_ids']  # Use input_ids as labels for evaluation
    }

# Map the data preparation function to the eval_dataset
eval_dataset = eval_dataset.map(prepare_data, batched=True, remove_columns=['Description', 'Patient', 'Doctor'])

# Set the dataset format to PyTorch tensors
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create DataLoader for evaluation dataset
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# Set the device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluation function
def evaluate_model(model, tokenizer, eval_loader):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Calculate loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            total_samples += 1

    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss))

    print(f'Average Loss: {avg_loss}')
    print(f'Perplexity: {perplexity.item()}')

# Evaluate the model
evaluate_model(model, tokenizer, eval_loader)
