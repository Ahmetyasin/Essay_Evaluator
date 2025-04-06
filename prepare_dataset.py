import os
import pandas as pd
import time
from tqdm import tqdm
import json
from openai import OpenAI

# Hardcoded configuration
DATASET_PATH = "DREsS_New.tsv"
OUTPUT_PATH = "DREsS_Processed.tsv"
API_KEY = "API_KEY"  # Replace this with your actual API key
MAX_ROWS = 500
BATCH_SIZE = 10
OUTPUT_JSONL = "finetune_dataset.jsonl"

# Set up OpenAI client
client = OpenAI(api_key=API_KEY)

def main():
    print(f"Loading dataset from {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, sep='\t')
    
    # Limit to the first max_rows
    if len(df) > MAX_ROWS:
        print(f"Limiting dataset to first {MAX_ROWS} rows")
        df = df.head(MAX_ROWS)
    
    print(f"Processing {len(df)} rows")
    
    # Add columns for the processed data
    df['content_100'] = df['content'].apply(lambda x: float(x) * 20)  # Convert to 100 scale
    df['organization_100'] = df['organization'].apply(lambda x: float(x) * 20)
    df['language_100'] = df['language'].apply(lambda x: float(x) * 20)
    df['overall_100'] = (df['content_100'] + df['organization_100'] + df['language_100']) / 3
    
    # Create empty columns for the justifications
    df['content_justification'] = ""
    df['organization_justification'] = ""
    df['language_justification'] = ""
    
    # Process in batches to avoid API rate limits
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing batches"):
        batch = df.iloc[i:min(i+BATCH_SIZE, len(df))]
        
        for j, row in batch.iterrows():
            try:
                # Generate justifications for each criterion
                justifications = generate_justifications(
                    row['essay'], 
                    row['content'], 
                    row['organization'], 
                    row['language']
                )
                
                # Update the dataframe with the generated justifications
                df.at[j, 'content_justification'] = justifications['content_justification']
                df.at[j, 'organization_justification'] = justifications['organization_justification']
                df.at[j, 'language_justification'] = justifications['language_justification']
                
            except Exception as e:
                print(f"Error processing row {j}: {e}")
                # Add default values in case of error
                df.at[j, 'content_justification'] = f"The essay demonstrates limited development as seen in the argument \"{row['essay'][:50]}...\", which lacks sufficient supporting evidence."
                df.at[j, 'organization_justification'] = f"The organization is problematic as shown in transitions like \"{row['essay'][100:150]}...\", which create abrupt shifts between ideas."
                df.at[j, 'language_justification'] = f"The language use shows inconsistencies, such as \"{row['essay'][200:250]}...\", where grammar errors affect clarity."
        
        # Sleep to avoid rate limits
        time.sleep(2)
    
    # Save the processed dataset
    print(f"Saving processed dataset to {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, sep='\t', index=False)
    
    # Create fine-tuning dataset in JSONL format
    print(f"Creating fine-tuning dataset in JSONL format: {OUTPUT_JSONL}")
    create_finetune_dataset(df, OUTPUT_JSONL)
    
    print("Dataset preparation complete!")

def generate_justifications(essay, content_score, organization_score, language_score):
    """Generate justifications using GPT-4o mini with examples from the essay"""
    
    prompt = f"""
You are an expert essay evaluator. Analyze this essay that has been scored:
- Content: {content_score}/5
- Organization: {organization_score}/5
- Language: {language_score}/5

Essay:
{essay}

Please provide:
1. ONE sentence justification for the content score that includes a specific example from the essay highlighting a weakness or strength based on the score
2. ONE sentence justification for the organization score that includes a specific example from the essay highlighting a weakness or strength based on the score  
3. ONE sentence justification for the language score that includes a specific example from the essay highlighting a weakness or strength based on the score

Each justification MUST include at least one specific example from the essay that illustrates a weakness, especially for scores below 4.

Format your response as a JSON object with these fields: 
content_justification, organization_justification, language_justification
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        print(f"API Error: {e}")
        # Return default values in case of error
        return {
            "content_justification": f"The essay demonstrates limited development as seen in the argument \"{essay[:50]}...\", which lacks sufficient supporting evidence.",
            "organization_justification": f"The organization is problematic as shown in transitions like \"{essay[100:150]}...\", which create abrupt shifts between ideas.",
            "language_justification": f"The language use shows inconsistencies, such as \"{essay[200:250]}...\", where grammar errors affect clarity."
        }

def create_finetune_dataset(df, output_path):
    """Create a JSONL file for fine-tuning with the processed data"""
    
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            # Format input prompt
            prompt = f"""Instruction: You are an expert essay evaluator. Evaluate the following essay based on the DREsS rubric.

DREsS Rubric (score range: 0 to 100):

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

            # Format expected output
            response = f"""Content: {row['content_100']:.1f}/100
Organization: {row['organization_100']:.1f}/100
Language: {row['language_100']:.1f}/100
Overall Score: {row['overall_100']:.1f}/100

Justification for Content: {row['content_justification']}
Justification for Organization: {row['organization_justification']}
Justification for Language: {row['language_justification']}"""
            
            # Create the json object for fine-tuning
            json_obj = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            }
            
            # Write to the file
            f.write(json.dumps(json_obj) + '\n')

if __name__ == "__main__":
    main() 