import os
import pandas as pd
import torch
import json
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
parser = argparse.ArgumentParser(description="Fine-tune Phi-2 for essay evaluation on 100-point scale")
parser.add_argument("--processed_jsonl", type=str, default="finetune_dataset.jsonl", 
                    help="Path to the processed dataset JSONL file")
parser.add_argument("--output_dir", type=str, default="./phi2-finetuned-100scale", 
                    help="Directory to save the model")
parser.add_argument("--num_train_epochs", type=int, default=3, 
                    help="Number of training epochs")
parser.add_argument("--per_device_train_batch_size", type=int, default=1, 
                    help="Batch size per device during training")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                    help="Number of gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=2e-4, 
                    help="Learning rate")
parser.add_argument("--lora_r", type=int, default=16, 
                    help="LoRA attention dimension")
parser.add_argument("--lora_alpha", type=int, default=32, 
                    help="LoRA alpha parameter")
parser.add_argument("--lora_dropout", type=float, default=0.05, 
                    help="LoRA dropout probability")
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

# Load the processed dataset in JSONL format
def load_jsonl_dataset(file_path):
    print(f"Loading dataset from {file_path}")
    
    # Read JSONL file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Format data for HuggingFace dataset
    formatted_data = []
    for item in data:
        user_message = item['messages'][0]['content']
        assistant_message = item['messages'][1]['content']
        
        formatted_data.append({
            "instruction": user_message,
            "output": assistant_message,
            "text": f"{user_message}\n{assistant_message}"
        })
    
    return formatted_data

def main():
    print("Starting fine-tuning process for Phi-2 model with 100-point scale")
    
    # Load and prepare the dataset
    formatted_data = load_jsonl_dataset(args.processed_jsonl)
    print(f"Loaded dataset with {len(formatted_data)} examples")
    
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