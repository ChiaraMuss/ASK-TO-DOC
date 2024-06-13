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