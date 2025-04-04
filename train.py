import os
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import argparse
from tqdm import tqdm

# Set up argument parser for command line options
parser = argparse.ArgumentParser(description="Fine-tune Phi-2 for essay evaluation")
parser.add_argument("--dataset_path", type=str, default="DREsS_New.tsv", help="Path to the dataset TSV file")
parser.add_argument("--output_dir", type=str, default="./phi2-finetuned", help="Directory to save the model")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")
args = parser.parse_args()

# Check for Apple Silicon GPU support
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Apple Silicon GPU not available, using CPU")

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Load the dataset
def load_dataset(file_path):
    print(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    return df

# Function to create prompt-response pairs formatted for phi-2
def format_for_phi2(row):
    prompt = f"""Instruction: You are an expert essay evaluator. Evaluate the following essay based on the DREsS rubric.

DREsS Rubric (score range: 1 to 5, with 0.5 increments):

1. Content: 
   - Paragraph is well-developed and relevant to the argument
   - Supported with strong reasons and examples

2. Organization:
   - The argument is effectively structured and developed
   - Easy for the reader to follow the ideas and understand the argument
   - Paragraphs use coherence devices effectively while focusing on a single main idea

3. Language:
   - The writing displays sophisticated control of vocabulary and collocations
   - The essay follows grammar and usage rules
   - Spelling and punctuation are correct throughout

Essay prompt: {row['prompt']}

Essay to evaluate:
{row['essay']}

Response:"""
    
    # Create the target output format
    response = f"""Content: {row['content']}/5
Organization: {row['organization']}/5
Language: {row['language']}/5
Overall Score: {row['total']}/15

Justification for Content: The essay demonstrates {get_justification_for_score(row['content'], 'content')}.
Justification for Organization: The essay {get_justification_for_score(row['organization'], 'organization')}.
Justification for Language: The writing {get_justification_for_score(row['language'], 'language')}."""
    
    return {
        "instruction": prompt,
        "output": response,
        "text": f"{prompt}\n{response}"
    }

# Helper function to generate justifications based on scores
def get_justification_for_score(score, category):
    score = float(score)
    
    if category == 'content':
        if score >= 4.5:
            return "excellent development with strong and relevant arguments supported by detailed examples"
        elif score >= 3.5:
            return "good development with relevant arguments and some supporting examples"
        elif score >= 2.5:
            return "adequate development with basic arguments but limited supporting examples"
        else:
            return "limited development with weak arguments and insufficient supporting examples"
    
    elif category == 'organization':
        if score >= 4.5:
            return "is exceptionally well-structured with clear progression of ideas and effective transitions"
        elif score >= 3.5:
            return "is well-organized with a logical flow and appropriate transitions between ideas"
        elif score >= 2.5:
            return "has a basic structure but lacks some coherence in the progression of ideas"
        else:
            return "lacks clear organization and coherence between ideas"
    
    elif category == 'language':
        if score >= 4.5:
            return "shows sophisticated control of vocabulary and grammar with minimal errors"
        elif score >= 3.5:
            return "demonstrates good command of language with some minor grammatical errors"
        elif score >= 2.5:
            return "exhibits adequate language use but contains noticeable grammatical errors"
        else:
            return "contains frequent language errors that interfere with comprehension"
    
    return "meets some of the criteria but needs improvement"

def main():
    print("Starting fine-tuning process for Phi-2 model")
    
    # Load and prepare the dataset
    df = load_dataset(args.dataset_path)
    print(f"Loaded dataset with {len(df)} examples")
    
    # Format data for fine-tuning
    formatted_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Formatting data"):
        formatted_data.append(format_for_phi2(row))
    
    # Convert to HF Dataset
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer and model
    print("Loading Phi-2 model and tokenizer")
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA (Low-Rank Adaptation)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "dense"
        ],
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    print_trainable_parameters(model)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=2048)
    
    print("Tokenizing dataset")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_steps=200,
        fp16=True,  # Use mixed precision training
        report_to="none",  # Disable reporting to Weights & Biases
        optim="adamw_torch",  # Use AdamW optimizer
        remove_unused_columns=False,
    )
    
    # Set up data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training")
    trainer.train()
    
    # Save the fine-tuned model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Fine-tuning complete!")

# Utility function to print trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} | All params: {all_params} | Trainable%: {100 * trainable_params / all_params:.2f}%"
    )

if __name__ == "__main__":
    main() 