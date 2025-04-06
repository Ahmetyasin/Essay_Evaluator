# Essay Evaluator

An improved essay evaluation system that leverages fine-tuned language models and optimized prompting techniques to provide accurate, rubric-based essay scoring.

## Project Overview

This project aims to build an enhanced essay evaluation system using the DREsS rubric. It incorporates:

1. **Fine-tuned Phi-2 Model**: A Microsoft Phi-2 model fine-tuned on academic essay data
2. **Optimized Prompting**: Enhanced prompt engineering with emotional stimuli and reward elements
3. **Streamlit Interface**: Easy-to-use web interface for essay evaluation

## Dataset

The project uses the **DREsS_New** dataset from [DREsS: Dataset for Rubric-based Essay Scoring](https://haneul-yoo.github.io/dress/), a high-quality academic dataset:

> "DREsS is a large-scale, standard dataset for rubric-based automated essay scoring. DREsS comprises three sub-datasets: DREsS_New,. We collect DREsS_New, a real-classroom dataset with 1.7K essays authored by EFL undergraduate students and scored by English education experts."

### Fine-tuning Process

Due to computational constraints (running on Mac M3 Pro with CPU), we:
1. Selected the first 500 rows from DREsS_New.tsv
2. Augmented the dataset using GPT-4o-mini to add score justifications
3. Formatted the data as instruction-response pairs in JSONL format
4. Applied LoRA fine-tuning to the Phi-2 model

### Prompt Optimization

In addition to fine-tuning, we optimized the inference prompt using techniques from research on emotional stimuli and reward-based prompting:

```
"You are the best expert ever for essay evaluation. I will tip you 1000 dollars for a perfect response. Evaluate the following essay based on the DREsS rubric."
```

This modified prompt, which includes elements of motivation and reward, was inspired by studies indicating that emotional stimuli and tip-offering can enhance the quality of LLM-generated responses (Salinas2024, Li2023).

## Files & Components

- `app.py`: Original Streamlit app using base Phi-2 model
- `improved_app.py`: Enhanced Streamlit app with fine-tuned model
- `train.py`: Script for fine-tuning Phi-2 with LoRA
- `prepare_dataset.py`: Script for processing DREsS data and generating fine-tuning dataset
- `finetune_dataset.jsonl`: The processed dataset used for fine-tuning
- `phi2-finetuned/`: Directory containing the fine-tuned model weights

## Setup & Usage

1. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Run the application**:
   
   To use the base Phi-2 model:
   ```
   streamlit run app.py
   ```
   
   To use the fine-tuned model:
   ```
   streamlit run improved_app.py
   ```

3. **Model activation**:
   The model will load automatically when you open the app. This may take a few minutes the first time.

4. **Evaluation process**:
   - Enter the essay prompt in the first text area
   - Paste the student's essay in the second text area
   - Click "Evaluate Essay"
   - View the detailed evaluation results

## DREsS Rubric

Essays are scored on a range of 1 to 5 (with 0.5 increments) based on these criteria:

| Criteria      | Description |
|---------------|-------------|
| Content       | Paragraph is well-developed and relevant to the argument, supported with strong reasons and examples. |
| Organization  | The argument is effectively structured and developed, making it easy for the reader to follow the ideas and understand how the writer is building the argument. Paragraphs use coherence devices effectively while focusing on a single main idea. |
| Language      | The writing displays sophisticated control of a wide range of vocabulary and collocations. The essay follows grammar and usage rules throughout the paper. Spelling and punctuation are correct throughout the paper. |

## Troubleshooting

If you encounter issues:

1. Make sure you have enough disk space for the model (at least 2GB)
2. The model requires around 4GB of RAM minimum to run
3. If the app crashes, try restarting it
4. The model is stored in a local cache directory (model_cache) for faster loading on subsequent runs
5. If you see errors related to Half precision or LayerNormKernelImpl, the app is now configured to use the appropriate precision based on your device

## Requirements

- Python 3.8+
- At least 8GB RAM recommended
- GPU acceleration recommended for faster inference 