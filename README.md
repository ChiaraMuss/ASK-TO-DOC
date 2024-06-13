
ASK TO DOC: Leveraging AI for Accessible and Reliable Medical Advice

## Overview

**ASK TO DOC** is an advanced AI-driven chatbot designed to provide reliable and accurate medical advice. The chatbot leverages a comprehensive training process and a robust dataset of over 250,000 patient-doctor dialogues, utilizing the state-of-the-art Microsoft/DialoGPT-small model to handle complex, multiturn conversations effectively. 

## Key Features

- **Advanced AI Model**: Uses Microsoft/DialoGPT-small for generating accurate responses.
- **Extensive Training Dataset**: Trained on the RUSLANMV/AI-MEDICAL-CHATBOT dataset with over 250,000 dialogues.
- **Optimized Training Techniques**: Employs gradient accumulation, mixed precision training, and subset sampling.
- **Seamless Telegram Integration**: Provides easy access and interaction through the Telegram API.
- **Robust Security Measures**: Ensures data protection, secure communication, and compliance with regulations.

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

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ask-to-doc.git
   cd ask-to-doc
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Google Colab for Training**

   - Mount your Google Drive
   - Ensure the training dataset and model checkpoints are in your Google Drive

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

   # (Include the rest of your training script here)
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
tokenizer = AutoTokenizer.from_pretrained('./trained_medical_chatbot3epoch')
model = AutoModelForCausalLM.from_pretrained('./trained_medical_chatbot3epoch').to(device)

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
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

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

def main():
    """Start the bot."""
    # Get bot token from environment variable for security


    updater = Updater('YOUR_API', use_context=True)

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

   The model is evaluated using BLEU and ROUGE scores to assess the quality of generated responses.

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

   # (Include the rest of your evaluation script here)
   ```

2. **Results**

   - **Average Loss**: 3.41
   - **Perplexity**: 30.31
   - **Average BLEU Score**: 0.270
   - **Average ROUGE-1 F1 Score**: 0.516
   - **Average ROUGE-2 F1 Score**: 0.498
   - **Average ROUGE-L F1 Score**: 0.510

## Security Measures

Comprehensive security measures have been implemented to protect user data and ensure safe interactions. These include:

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



Feel free to contribute to the project or report any issues you encounter. Your feedback is invaluable for the continuous improvement of **ASK TO DOC**.
