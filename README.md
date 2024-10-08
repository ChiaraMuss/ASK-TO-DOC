
## Overview

**ASK TO DOC** is an advanced AI-driven chatbot designed to provide reliable and accurate medical advice. The chatbot leverages a comprehensive training process and a robust dataset of over 250,000 patient-doctor dialogues, utilizing the state-of-the-art Microsoft/DialoGPT-small model to handle complex, multiturn conversations effectively. 

## Key Features

- **Advanced AI Model**: Uses Microsoft/DialoGPT-small for generating accurate responses.
- **Extensive Training Dataset**: Trained on the RUSLANMV/AI-MEDICAL-CHATBOT dataset with over 250,000 dialogues.
- **Optimized Training Techniques**: Employs gradient accumulation, mixed precision training, and subset sampling.
- **Seamless Telegram Integration**: Provides easy access and interaction through the Telegram API.
- **Ideas for Security Measures**: Ensures data protection, secure communication, and compliance with regulations.

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- NLTK
- Rouge Score
- Flask
- Flask-JWT-Extended
- Cryptography
- Google Colab (for training)


## Usage

### Training the Model

1. **Prepare the Dataset**

   Ensure your dataset is in the correct format and split into training and evaluation sets.

2. **Run the Training Script in Google Colab**
  ```python
   import os
   import torch
   from torch.utils.data import DataLoader, Subset
   from datasets import load_dataset
   from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
   from google.colab import drive
   from tqdm import tqdm
   from random import sample
   from torch.cuda.amp import GradScaler, autocast
   import time
   from torch.utils.tensorboard import SummaryWriter

   # Set environment variable for memory management
   os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

   # Mount Google Drive
   drive.mount('/content/drive', force_remount=True)
   os.makedirs('/content/drive/MyDrive/new_folder', exist_ok=True)

   # Set device to GPU if available
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   print(f'Using device: {device}')

   # Load dataset
   dataset = load_dataset("ruslanmv/ai-medical-chatbot")
   print(dataset['train'].column_names)  # Print column names to verify
   split = dataset['train'].train_test_split(test_size=0.1)
   train_dataset_full = split['train']
   eval_dataset = split['test']

   # Load tokenizer and model
   tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')

   # Explicitly set pad_token_id to eos_token_id to avoid warning
   tokenizer.pad_token = tokenizer.eos_token
   tokenizer.pad_token_id = tokenizer.eos_token_id

   model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small').to(device)

   # Prepare data function for training
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
       }

   # Map the data preparation function to both datasets
   train_dataset_full = train_dataset_full.map(prepare_data, batched=True, remove_columns=['Description', 'Patient', 'Doctor'])
   eval_dataset = eval_dataset.map(prepare_data, batched=True, remove_columns=['Description', 'Patient', 'Doctor'])

   # Set the datasets' format to PyTorch tensors
   train_dataset_full.set_format(type='torch', columns=['input_ids', 'attention_mask'])
   eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

   # Create DataLoader for evaluation dataset
   eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

   # Optimizer setup
   optimizer = AdamW(model.parameters(), lr=5e-5)

   # Learning rate scheduler setup
   num_epochs = 3  # Number of epochs
   train_loader = DataLoader(train_dataset_full, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
   num_training_steps = num_epochs * len(train_loader)
   lr_scheduler = get_scheduler(
       name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
   )

   # Subset Sampling Function
   def get_subset(data, percentage=10):
       subset_size = int(len(data) * (percentage / 100))
       indices = sample(range(len(data)), subset_size)
       return Subset(data, indices)

   # Evaluation function with relevant metrics
   def compute_metrics():
       model.eval()
       all_predictions = []
       with torch.no_grad():
           for eval_batch in tqdm(eval_loader, desc="Evaluating"):
               input_ids = eval_batch['input_ids'].to(device)
               attention_mask = eval_batch['attention_mask'].to(device)

               outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=50)
               decoded_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

               all_predictions.extend(decoded_responses)

       # Sample response for quick verification
       return {'sample_responses': all_predictions[:5]}

   # Custom training loop with mixed precision training and gradient accumulation
   writer = SummaryWriter('/content/drive/MyDrive/logs')
   model.train()
   evaluation_interval = 1000  # Evaluation interval
   accumulation_steps = 32  # Gradient accumulation steps
   scaler = GradScaler()  # GradScaler for mixed precision

   global_step = 0
   for epoch in range(num_epochs):
       train_subset = get_subset(train_dataset_full, percentage=10)
       train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

       optimizer.zero_grad()
       start_time = time.time()
       for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1}"):
           input_ids = batch['input_ids'].to(device)
           attention_mask = batch['attention_mask'].to(device)

           with autocast():
               outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
               loss = outputs.loss
               loss = loss / accumulation_steps  # Scale loss by accumulation steps

           scaler.scale(loss).backward()

           if (batch_idx + 1) % accumulation_steps == 0:
               scaler.step(optimizer)
               scaler.update()
               optimizer.zero_grad()
               lr_scheduler.step()  # Step the scheduler

           if batch_idx % evaluation_interval == 0:
               metrics = compute_metrics()
               print(f'Evaluation Metrics at batch {batch_idx}: {metrics}')

               # Log the count of sample responses
               writer.add_scalar(f'Metrics/sample_responses_count', len(metrics['sample_responses']), global_step)

               # Optionally print the sample responses for inspection
               print("Sample Responses:", metrics['sample_responses'])

               model.train()

           writer.add_scalar('Loss/train', loss.item(), global_step)
           global_step += 1

       end_time = time.time()
       print(f'Epoch {epoch + 1} completed in {(end_time - start_time) / 60:.2f} minutes')

   # Save the trained model and tokenizer
   model.save_pretrained('/content/drive/MyDrive/trained_medical_chatbot')
   tokenizer.save_pretrained('/content/drive/MyDrive/trained_medical_chatbot')
   ```


### Integrating with Telegram

1. **Set Up the Telegram Bot**

   - Create a new bot using the BotFather on Telegram and get the API token.

2. **Run the Flask App**
```python
   import logging
   import os
   from telegram import Update
   from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM

   # Set device to GPU if available
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   print(f'Using device: {device}')

   # Load tokenizer and model from the saved directory
   tokenizer = AutoTokenizer.from_pretrained('./trained_medical_chatbot')
   model = AutoModelForCausalLM.from_pretrained('./trained_medical_chatbot').to(device)

   # Set up logging
   logging.basicConfig(
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       level=logging.INFO
   )

   logger = logging.getLogger(__name__)

   def start(update: Update, context: CallbackContext) -> None:
       """Send a message when the command /start is issued."""
       update.message.reply_text('Hi! I am your medical chatbot. How can I help you today?')

   def generate_response(patient_text):
       # Tokenize the input text
       input_text = f"Patient: {patient_text} Doctor:"
       input_ids = tokenizer.encode(input_text, return

_tensors='pt').to(device)

       # Generate a response from the model
       with torch.no_grad():
           output_ids = model.generate(
               input_ids,
               max_length=150,
               num_beams=5,
               no_repeat_ngram_size=3,
               early_stopping=True,
               pad_token_id=tokenizer.eos_token_id,
               temperature=0.7,
               top_k=50,
               top_p=0.95,
               do_sample=True  # Enable sampling
           )

       # Decode the generated tokens into text
       response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
       # Extract only the doctor's response part
       response = response.split("Doctor:")[-1].strip()
       return response

   def respond(update: Update, context: CallbackContext) -> None:
       """Generate a response to the user's message."""
       patient_input = update.message.text

       # Check for gratitude messages
       if any(gratitude in patient_input.lower() for gratitude in ['thanks', 'thank you', 'thank', 'thx']):
           update.message.reply_text("You're welcome! If you have any other questions, feel free to ask.")
       else:
           response = generate_response(patient_input)
           # basic answers
           if "headache" in patient_input.lower():
               response = "I'm sorry to hear that you're experiencing headaches. Headaches can be caused by various factors, including stress, dehydration, or lack of sleep. If your headaches persist, I recommend consulting a healthcare professional."
           elif "tired" in patient_input.lower():
               response = "Feeling tired can be due to many reasons, such as lack of sleep, stress, or even an underlying medical condition. Make sure you're getting enough rest, eating well, and staying hydrated. If your fatigue continues, consider visiting a doctor."
           elif "sore throat" in patient_input.lower() or "cough" in patient_input.lower():
               response = "A sore throat and cough can often be symptoms of a viral infection like the common cold. Make sure to stay hydrated, rest, and consider over-the-counter remedies. If your symptoms worsen or persist for more than a week, please see a doctor."
           update.message.reply_text(response)

   def main():
       """Start the bot."""
       # Get bot token from environment variable for security
       updater = Updater('your-telegram-bot-token', use_context=True)

       # Get the dispatcher to register handlers
       dp = updater.dispatcher

       # Register the handlers
       dp.add_handler(CommandHandler("start", start))
       dp.add_handler(MessageHandler(Filters.text & ~Filters.command, respond))

       # Start the Bot
       updater.start_polling()

       # Run the bot until you press Ctrl-C or the process receives SIGINT,
       # SIGTERM or SIGABRT
       updater.idle()

   if __name__ == '__main__':
       main()
   ```

### Evaluating the Model

1. **Evaluation Metrics**

   The model is evaluated using Average Loss, Perplexity, BLEU, and ROUGE scores to assess the quality of generated responses.

2. **Evaluation Script for Perplexity and Average Loss**

   ```python
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
   ```

3. **Evaluation Script for BLEU and ROUGE Scores**

   ```python
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from datasets import load_dataset
   import nltk
   from rouge_score import rouge_scorer
   from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
   from tqdm import tqdm
   import os
   from google.colab import drive

   # Mount Google Drive
   drive.mount('/content/drive')

   # Verify the files in the model directory
   model_path = '/content/drive/MyDrive/trained_medical_chatbot'
   print("Files in model directory:", os.listdir(model_path))

   # Download NLTK data
   nltk.download('punkt')

   # Load the model and tokenizer from the saved local path
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   model = AutoModelForCausalLM.from_pretrained(model_path)

   # Load the dataset
   dataset = load_dataset('ruslanmv/ai-medical-chatbot')
   split = dataset['train'].train_test_split(test_size=0.1)
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
           'labels': model_inputs['input_ids']
       }

   # Map the data preparation function to the eval_dataset
   eval_dataset = eval_dataset.map(prepare_data, batched=True, remove_columns=['Description', 'Patient', 'Doctor'])
   eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

   # Create DataLoader for evaluation dataset
   eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

   # Set the device
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model.to(device)

   # Function to generate response using beam search
   def generate_response(input_ids):
       with torch.no_grad():
           output_ids = model.generate(
               input_ids,
               max_length=150,
               num_beams=5,
               no_repeat_ngram_size=2,
               early_stopping=True,
               pad_token_id=tokenizer.eos_token_id,
               temperature=0.7,
               top_k=50,
               top_p=0.95,
               do_sample=True
           )
       response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
       response = [res.split("Doctor:")[-1].strip() for res in response]
       return response

   # Evaluate function with manual inspection
   def evaluate_model(model, tokenizer, eval_loader):
       model.eval()
       all_references = []
       all_hypotheses = []
       sample_outputs = []
       rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
       bleu_scores = []
       rouge1_scores = []
       rouge2_scores = []
       rougeL_scores = []

       # Generate responses for BLEU and ROUGE scores
       for batch in tqdm(eval_loader, desc="Evaluating BLEU and ROUGE"):
           input_ids = batch['input_ids'].to(device)
           attention_mask = batch['attention_mask'].to(device)
           labels = batch['labels'].to(device)
           batch_generated_texts = generate_response(input_ids)
           references = tokenizer.batch_decode(labels, skip_special_tokens=True)
           all_references.extend(references)
           all_hypotheses.extend(batch_generated_texts)

           if len(sample_outputs) < 20:
               sample_outputs.extend(zip(references, batch_generated_texts, tokenizer.batch_decode(input_ids, skip_special_tokens=True)))

           for ref, hyp in zip(references, batch_generated_texts):
               reference_tokens = nltk.word_tokenize(ref)
               hypothesis_tokens = nltk.word_tokenize(hyp)
               bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=SmoothingFunction().method1)
               bleu_scores.append(bleu_score)

               rouge_score = rouge.score(ref, hyp)
               rouge1_scores.append(rouge_score['rouge1'].fmeasure)
               rouge2_scores.append(rouge_score['rouge2'].fmeasure)
               rougeL_scores.append(rouge_score['rougeL'].fmeasure)

       avg_bleu = sum(bleu_scores) / len(bleu_scores)
       avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
       avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
       avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

       print(f'Average BLEU Score: {avg_bleu}')
       print(f'Average ROUGE-1 F1 Score: {avg_rouge1}')
       print(f'Average ROUGE-2 F1 Score: {avg_rouge2}')
       print(f'Average ROUGE-L F1 Score: {avg_rougeL}')

       for ref, hyp in sample_outputs:
           print(f'Reference: {ref}')
           print(f'Hypothesis: {hyp}')
           print('---')

   # Evaluate the model with manual inspection
   evaluate_model(model, tokenizer, eval_loader)
   ```




2. **Results**

   - **Average Loss**: 3.41
   - **Perplexity**: 30.31
   - **Average BLEU Score**: 0.270
   - **Average ROUGE-1 F1 Score**: 0.516
   - **Average ROUGE-2 F1 Score**: 0.498
   - **Average ROUGE-L F1 Score**: 0.510

## Security Measures

Comprehensive security measures that can be implemented to protect user data and ensure safe interactions. These include:

- Data Protection and Privacy
- Authentication and Authorization
- Secure Communication
- Session Management
- Input Validation
- Compliance with Regulations
- Threat Detection and Response
- API Security
- User and Developer Awareness

## Future Development

Future plans for the **ASK TO DOC** chatbot include:

- Exploring advanced AI techniques
- Expanding the dataset
- Incorporating user feedback
- Providing multilingual support
- Enhancing security measures
- Integrating with additional platforms
- Utilizing advanced GPU resources
- Continuous performance monitoring and improvement
- Collaborating with medical professionals

## Acknowledgements

This research received support during the Applied Natural Language Processing course, instructed by Professors Staiano Jacopo and Penzo Nicolò.

## References

- **Microsoft/DialoGPT-small**: [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536)
- **Hugging Face Transformers**: [Transformers Documentation](https://huggingface.co/transformers/)
- **Microsoft/DialoGPT-small**: [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536)
- **TensorBoard**: [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- **BLEU Score**: [BLEU: A Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040/)
- **ROUGE Score**: [ROUGE: A Package for Automatic Evaluation of Summaries](https://www.aclweb.org/anthology/W04-1013/)
- **JWT Authentication**: [JSON Web Tokens Introduction](https://jwt.io/introduction/)
- **Cryptography**: [Cryptography Documentation](https://cryptography.io/en/latest/)


Feel free to contribute to the project or report any issues you encounter. Your feedback is invaluable for the continuous improvement of **ASK TO DOC**.
