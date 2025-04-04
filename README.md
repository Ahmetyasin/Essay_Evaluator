# Essay Evaluator

A simple Streamlit app that uses the Phi-2 language model to evaluate essays based on the DREsS rubric.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run app.py
   ```

3. The model will load automatically when you open the app. This may take a few minutes the first time.

## Usage

1. Once the model is loaded, enter an essay prompt in the first text area
2. Paste a student's essay in the second text area
3. Click the "Evaluate Essay" button
4. View the evaluation results, including:
   - Individual scores for Content, Organization, and Language (1-5 with 0.5 increments)
   - Overall score (average of the three criteria)
   - Justification for each score
   - Key strengths
   - Areas for improvement

## DREsS Rubric

The essays are scored on a range of 1 to 5, with increments of 0.5, based on three criteria:

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