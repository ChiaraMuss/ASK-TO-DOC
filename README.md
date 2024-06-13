
##ASK TO DOC: Leveraging AI for Accessible and Reliable Medical Advice

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
   from flask import Flask, request, jsonify
   from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
   from cryptography.fernet import Fernet
   import os

   app = Flask(__name__)

   # Configuration for JWT
   app.config['JWT_SECRET_KEY'] = 'super-secret-key'  # Change this in production
   jwt = JWTManager(app)

   # Generate a key for encryption
   encryption_key = Fernet.generate_key()
   cipher_suite = Fernet(encryption_key)

   # In-memory user data
   users = {
       "user1": {"password": "password1"},
   }

   # Endpoint to authenticate users and return a JWT
   @app.route('/login', methods=['POST'])
   def login():
       username = request.json.get('username', None)
       password = request.json.get('password', None)
       if username not in users or users[username]['password'] != password:
           return jsonify({"msg": "Bad username or password"}), 401

       access_token = create_access_token(identity=username)
       return jsonify(access_token=access_token)

   # Secure endpoint that requires authentication
   @app.route('/chat', methods=['POST'])
   @jwt_required()
   def chat():
       current_user = get_jwt_identity()
       message = request.json.get('message', '')
       
       # Encrypt the message
       encrypted_message = cipher_suite.encrypt(message.encode())
       
       # Decrypt the message (for demonstration)
       decrypted_message = cipher_suite.decrypt(encrypted_message).decode()
       
       response = f"Received your message, {current_user}. You said: {decrypted_message}"
       return jsonify({"response": response})

   # Endpoint for users to get their data (example of data minimization)
   @app.route('/userdata', methods=['GET'])
   @jwt_required()
   def get_userdata():
       current_user = get_jwt_identity()
       # Only return minimal necessary data
       return jsonify({"username": current_user})

   if __name__ == '__main__':
       app.run(ssl_context='adhoc')
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
