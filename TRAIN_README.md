# Fine-tuning Phi-2 for Essay Evaluation

This guide explains how to fine-tune the Microsoft Phi-2 model for essay evaluation using the DREsS dataset.

## Requirements

First, install the required dependencies:

```bash
pip install -r requirements-train.txt
```

## Dataset Format

The training dataset (`DREsS_New.tsv`) should have the following format:

| id | prompt | essay | content | organization | language | total |
|----|--------|-------|---------|--------------|----------|-------|
| 1  | Essay prompt text | Full essay text | 3.5 | 4 | 3.5 | 11 |

- `content`, `organization`, and `language` are scores from 1-5 (with 0.5 increments)
- `total` is the sum of the three scores (out of 15)

## Running the Fine-tuning

To fine-tune the model with default parameters:

```bash
python train.py --dataset_path DREsS_New.tsv
```

### Command Line Arguments

You can customize the training with these arguments:

- `--dataset_path`: Path to the dataset TSV file (default: "DREsS_New.tsv")
- `--output_dir`: Directory to save the model (default: "./phi2-finetuned")
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per device (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--lora_r`: LoRA attention dimension (default: 16)
- `--lora_alpha`: LoRA alpha parameter (default: 32)
- `--lora_dropout`: LoRA dropout probability (default: 0.05)

Example with custom parameters:

```bash
python train.py --dataset_path DREsS_New.tsv --num_train_epochs 5 --learning_rate 1e-4 --output_dir ./my-finetuned-model
```

## Optimization for Mac M3 Pro

The script is optimized for Apple Silicon using:

1. **4-bit Quantization**: Reduces memory usage while maintaining performance
2. **LoRA (Low-Rank Adaptation)**: Only trains a small subset of parameters
3. **Apple MPS Backend**: Utilizes Metal Performance Shaders when available
4. **Gradient Accumulation**: Simulates larger batch sizes on limited memory
5. **Mixed Precision Training**: Uses FP16 for faster training when possible

## Using the Fine-tuned Model

After training, update your app.py to use the fine-tuned model:

```python
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./phi2-finetuned")
model = AutoModelForCausalLM.from_pretrained("./phi2-finetuned")
```

## Performance Tips

1. For larger datasets, increase gradient accumulation steps instead of batch size
2. If you experience out-of-memory errors, try decreasing the LoRA rank (r) parameter
3. Fine-tuning is memory-intensive - close other applications during training
4. For Apple Silicon M3 Pro, MPS acceleration works best with smaller models
5. The first epoch may be slower due to compilation overhead 