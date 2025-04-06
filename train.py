import os
import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import time

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
# Configuration
MODEL_NAME = "microsoft/phi-2"
DATASET_PATH = "finetune_dataset.jsonl"
OUTPUT_DIR = "./phi2-finetuned"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRADIENT_NORM = 0.3

# Progress tracking class
class ProgressCallback:
    def __init__(self, total_steps, epochs):
        self.total_steps = total_steps
        self.epochs = epochs
        self.current_step = 0
        self.current_epoch = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"\nStarting training...")
        return control
    
    def on_step_begin(self, args, state, control, **kwargs):
        # We don't need to do anything here, but the method must exist
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        self.current_step += 1
        elapsed = time.time() - self.start_time
        
        if time.time() - self.last_log_time > 5 or self.current_step % 10 == 0:
            progress = self.current_step / self.total_steps * 100
            steps_per_sec = self.current_step / elapsed if elapsed > 0 else 0
            remaining = (self.total_steps - self.current_step) / steps_per_sec if steps_per_sec > 0 else 0
            
            bar_length = 30
            filled_length = int(bar_length * self.current_step // self.total_steps)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"\rEpoch {self.current_epoch+1}/{self.epochs} |{bar}| {progress:.1f}% | "
                  f"Step {self.current_step}/{self.total_steps} | "
                  f"Loss: {state.log_history[-1].get('loss', 0):.4f} | "
                  f"Speed: {steps_per_sec:.2f} steps/s | "
                  f"ETA: {remaining/60:.1f}min", end="")
            
            self.last_log_time = time.time()
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch += 1
        print(f"\n{'='*50}")
        print(f"Starting Epoch {self.current_epoch}/{self.epochs}")
        print(f"{'='*50}")

# Function to load and prepare the dataset
def load_and_prepare_dataset(jsonl_file):
    # Load JSONL file
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            
            # Convert the conversation to a simple text format
            text = ""
            for msg in example["messages"]:
                if msg["role"] == "user":
                    text += f"USER: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    text += f"ASSISTANT: {msg['content']}\n\n"
            
            data.append({"text": text})
    
    print(f"Loaded {len(data)} examples from {jsonl_file}")
    return Dataset.from_list(data)

def main():
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # For Apple Silicon
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load and prepare dataset
    print(f"Loading dataset from {DATASET_PATH}")
    dataset = load_and_prepare_dataset(DATASET_PATH)
    
    # Load tokenizer
    print(f"Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use FP32 for MPS compatibility
        trust_remote_code=True,
    )
    
    # Move model to device after loading
    model = model.to(device)
    
    # Prepare model for training 
    print("Preparing model for training")
    model.config.use_cache = False  # Disable cache for training
    
    # LoRA config for Phi-2
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        # Target modules for Phi-2
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of {total_params:,} total)")
    
    # Training arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=MAX_GRADIENT_NORM,
        learning_rate=LEARNING_RATE,
        fp16=False,  # Disabled fp16 - not supported on MPS
        bf16=False,  # Disabled bf16 - not supported on MPS
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",  # Using standard AdamW
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="none",
    )
    
    # SFT trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=args,
        tokenizer=tokenizer,
        peft_config=peft_config,
        # Simple text field
        dataset_text_field="text",
        max_seq_length=2048,
    )
    
    # Calculate steps
    steps_per_epoch = len(dataset) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) + 1
    total_steps = steps_per_epoch * NUM_TRAIN_EPOCHS
    
    # Add callback
    progress_callback = ProgressCallback(total_steps=total_steps, epochs=NUM_TRAIN_EPOCHS)
    trainer.add_callback(progress_callback)
    
    # Training summary
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION:")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_PATH} ({len(dataset)} examples)")
    print(f"Device: {device}")
    print(f"Training Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"Batch Size: {PER_DEVICE_TRAIN_BATCH_SIZE} (x{GRADIENT_ACCUMULATION_STEPS} gradient accumulation)")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"LoRA Parameters: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Expected Steps: {total_steps} (≈ {steps_per_epoch} steps/epoch)")
    print("="*70 + "\n")
    
    # Train
    print("Starting fine-tuning")
    print("Progress: (each █ represents approximately 3.3% completion)")
    trainer.train()
    print("\n")
    
    # Save
    print(f"Saving the model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    
    # Training time
    training_time = time.time() - progress_callback.start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Fine-tuning complete! Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    main()