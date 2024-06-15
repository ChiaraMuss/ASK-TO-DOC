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


    updater = Updater('API_KEY', use_context=True)

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
