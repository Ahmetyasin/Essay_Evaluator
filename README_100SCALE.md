# Fine-tuning Phi-2 for Essay Evaluation (100-Point Scale)

This guide explains how to prepare a dataset using GPT-4o mini via the OpenAI API and fine-tune the Microsoft Phi-2 model for essay evaluation using a 100-point scale.

## Process Overview

1. Prepare the dataset using GPT-4o mini to generate detailed justifications
2. Convert the 5-point scale scores to a 100-point scale
3. Fine-tune the Phi-2 model on the processed dataset
4. Update the app to use the fine-tuned model with the 100-point scale

## Requirements

First, install the required dependencies:

```bash
pip install -r requirements-train.txt openai
```

You'll need an OpenAI API key to use GPT-4o mini for dataset preparation.

## 1. Dataset Preparation

The `prepare_dataset.py` script uses the OpenAI API to generate detailed justifications for each essay, with examples from the text:

```bash
python prepare_dataset.py --dataset_path DREsS_New.tsv --api_key YOUR_OPENAI_API_KEY
```

This will:
- Load the first 500 rows from the DREsS dataset
- Convert scores from the 5-point scale to a 100-point scale (multiplying by 20)
- Use GPT-4o mini to generate one-sentence justifications with examples from the essays
- Create a processed TSV file and a JSONL file for fine-tuning

### Command Line Arguments:

- `--dataset_path`: Path to the original DREsS dataset (default: "DREsS_New.tsv")
- `--output_path`: Path to save the processed TSV file (default: "DREsS_Processed.tsv")
- `--api_key`: Your OpenAI API key (required)
- `--max_rows`: Maximum number of rows to process (default: 500)
- `--batch_size`: Batch size for API requests (default: 10)
- `--output_jsonl`: Path to save the fine-tuning JSONL file (default: "finetune_dataset.jsonl")

## 2. Fine-tuning on 100-Point Scale

After preparing the dataset, use the `train_100scale.py` script to fine-tune the model:

```bash
python train_100scale.py --processed_jsonl finetune_dataset.jsonl
```

This script is optimized for Mac M3 Pro with:
- 4-bit quantization
- LoRA (Low-Rank Adaptation)
- Apple MPS backend support
- Gradient accumulation
- Mixed precision training

### Command Line Arguments:

- `--processed_jsonl`: Path to the processed JSONL file (default: "finetune_dataset.jsonl")
- `--output_dir`: Directory to save the model (default: "./phi2-finetuned-100scale")
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per device (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--lora_r`: LoRA attention dimension (default: 16)
- `--lora_alpha`: LoRA alpha parameter (default: 32)
- `--lora_dropout`: LoRA dropout probability (default: 0.05)

## 3. Update the App for 100-Point Scale

After fine-tuning, use the `update_app_100scale.py` script to update your app.py to use the new model:

```bash
python update_app_100scale.py --backup
```

This will:
- Update the app to use the fine-tuned model
- Convert the rubric and scoring to use the 100-point scale
- Create a backup of your original app.py file

## Example Workflow

```bash
# 1. Prepare the dataset using GPT-4o mini
python prepare_dataset.py --api_key sk-your-openai-api-key --max_rows 500

# 2. Fine-tune the model (this will take some time)
python train_100scale.py --num_train_epochs 3

# 3. Update the app to use the fine-tuned model
python update_app_100scale.py --backup

# 4. Run the updated app
streamlit run app.py
```

## Performance Tips

1. The OpenAI API calls in dataset preparation have rate limiting - adjust the batch_size parameter if needed
2. Reduce max_rows for faster dataset preparation if you're just testing
3. For Mac M3 Pro, MPS acceleration works best with smaller models and lower precision
4. If you experience out-of-memory errors during training, try:
   - Decreasing the LoRA rank parameter (`--lora_r`)
   - Increasing gradient accumulation steps (`--gradient_accumulation_steps`)
   - Reducing batch size (`--per_device_train_batch_size`)
5. Close other applications during training to maximize available memory 