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