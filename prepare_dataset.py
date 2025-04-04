import os
import pandas as pd
import time
from tqdm import tqdm
import argparse
import openai
from openai import OpenAI
import json

# Set up argument parser for command line options
parser = argparse.ArgumentParser(description="Prepare dataset for fine-tuning using GPT-4o mini")
parser.add_argument("--dataset_path", type=str, default="DREsS_New.tsv", 
                    help="Path to the dataset TSV file")
parser.add_argument("--output_path", type=str, default="DREsS_Processed.tsv", 
                    help="Path to save the processed dataset")
parser.add_argument("--api_key", type=str, required=True, 
                    help="OpenAI API key")
parser.add_argument("--max_rows", type=int, default=500, 
                    help="Maximum number of rows to process")
parser.add_argument("--batch_size", type=int, default=10, 
                    help="Batch size for API requests to avoid rate limits")
parser.add_argument("--output_jsonl", type=str, default="finetune_dataset.jsonl", 
                    help="Path to save the fine-tuning dataset in JSONL format")
args = parser.parse_args()

# Set up OpenAI client
openai.api_key = args.api_key
client = OpenAI(api_key=args.api_key)

def main():
    print(f"Loading dataset from {args.dataset_path}")
    df = pd.read_csv(args.dataset_path, sep='\t')
    
    # Limit to the first max_rows
    if len(df) > args.max_rows:
        print(f"Limiting dataset to first {args.max_rows} rows")
        df = df.head(args.max_rows)
    
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
    df['key_strengths'] = ""
    df['areas_for_improvement'] = ""
    
    # Process in batches to avoid API rate limits
    for i in tqdm(range(0, len(df), args.batch_size), desc="Processing batches"):
        batch = df.iloc[i:min(i+args.batch_size, len(df))]
        
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
                df.at[j, 'key_strengths'] = json.dumps(justifications['key_strengths'])
                df.at[j, 'areas_for_improvement'] = json.dumps(justifications['areas_for_improvement'])
                
            except Exception as e:
                print(f"Error processing row {j}: {e}")
                # Add default values in case of error
                df.at[j, 'content_justification'] = f"The essay demonstrates development relative to the {row['content']}/5 score."
                df.at[j, 'organization_justification'] = f"The organization is structured appropriately for a {row['organization']}/5 rating."
                df.at[j, 'language_justification'] = f"The language use is consistent with a {row['language']}/5 score."
                df.at[j, 'key_strengths'] = json.dumps(["Clear thesis statement", "Appropriate use of examples", "Logical structure"])
                df.at[j, 'areas_for_improvement'] = json.dumps(["Enhance argument development", "Improve transitions", "Reduce grammar errors"])
        
        # Sleep to avoid rate limits
        time.sleep(2)
    
    # Save the processed dataset
    print(f"Saving processed dataset to {args.output_path}")
    df.to_csv(args.output_path, sep='\t', index=False)
    
    # Create fine-tuning dataset in JSONL format
    print(f"Creating fine-tuning dataset in JSONL format: {args.output_jsonl}")
    create_finetune_dataset(df, args.output_jsonl)
    
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
1. ONE sentence justification for the content score with a specific example from the essay
2. ONE sentence justification for the organization score with a specific example from the essay
3. ONE sentence justification for the language score with a specific example from the essay
4. Three key strengths (bullet points)
5. Three areas for improvement (bullet points)

Format your response as a JSON object with these fields: 
content_justification, organization_justification, language_justification, key_strengths (array), areas_for_improvement (array)
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
            "content_justification": f"The essay demonstrates development relative to the {content_score}/5 score.",
            "organization_justification": f"The organization is structured appropriately for a {organization_score}/5 rating.",
            "language_justification": f"The language use is consistent with a {language_score}/5 score.",
            "key_strengths": ["Clear thesis statement", "Appropriate use of examples", "Logical structure"],
            "areas_for_improvement": ["Enhance argument development", "Improve transitions", "Reduce grammar errors"]
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

            # Convert strengths and areas for improvement to lists if they're stored as JSON strings
            key_strengths = row['key_strengths']
            areas_for_improvement = row['areas_for_improvement']
            
            if isinstance(key_strengths, str):
                key_strengths = json.loads(key_strengths)
            
            if isinstance(areas_for_improvement, str):
                areas_for_improvement = json.loads(areas_for_improvement)
            
            # Format expected output
            response = f"""Content: {row['content_100']:.1f}/100
Organization: {row['organization_100']:.1f}/100
Language: {row['language_100']:.1f}/100
Overall Score: {row['overall_100']:.1f}/100

Justification for Content: {row['content_justification']}
Justification for Organization: {row['organization_justification']}
Justification for Language: {row['language_justification']}

Key strengths:
- {key_strengths[0] if len(key_strengths) > 0 else "Clear thesis statement"}
- {key_strengths[1] if len(key_strengths) > 1 else "Appropriate use of examples"}
- {key_strengths[2] if len(key_strengths) > 2 else "Logical structure"}

Areas for improvement:
- {areas_for_improvement[0] if len(areas_for_improvement) > 0 else "Enhance argument development"}
- {areas_for_improvement[1] if len(areas_for_improvement) > 1 else "Improve transitions"}
- {areas_for_improvement[2] if len(areas_for_improvement) > 2 else "Reduce grammar errors"}"""
            
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