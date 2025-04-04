#!/usr/bin/env python3
"""
This script updates the app.py file to use the fine-tuned model with 100-point scale.
Run it after successful fine-tuning.
"""
import os
import re
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser(description="Update app.py to use fine-tuned model with 100-point scale")
    parser.add_argument("--model_path", type=str, default="./phi2-finetuned-100scale", 
                       help="Path to the fine-tuned model (default: ./phi2-finetuned-100scale)")
    parser.add_argument("--app_file", type=str, default="app.py",
                       help="Path to the app.py file (default: app.py)")
    parser.add_argument("--backup", action="store_true",
                       help="Create a backup of the original app.py file")
    args = parser.parse_args()
    
    # Create backup if requested
    if args.backup and os.path.exists(args.app_file):
        backup_file = f"{args.app_file}.backup"
        print(f"Creating backup of {args.app_file} to {backup_file}")
        shutil.copy2(args.app_file, backup_file)
    
    # Ensure the model directory exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model directory {args.model_path} does not exist.")
        print("Please run the fine-tuning script first or provide the correct path.")
        return
    
    # Read the app.py file
    try:
        with open(args.app_file, 'r') as f:
            app_content = f.read()
    except FileNotFoundError:
        print(f"Error: {args.app_file} not found.")
        return
    
    # Update the model path in load_model function
    model_loading_pattern = r'(tokenizer = AutoTokenizer\.from_pretrained\([\s\n]*)"[^"]+?"'
    if not re.search(model_loading_pattern, app_content):
        print("Warning: Could not find tokenizer loading pattern in app.py")
    
    updated_content = re.sub(
        model_loading_pattern,
        f'\\1"{args.model_path}"',
        app_content
    )
    
    model_loading_pattern = r'(model = AutoModelForCausalLM\.from_pretrained\([\s\n]*)"[^"]+?"'
    if not re.search(model_loading_pattern, updated_content):
        print("Warning: Could not find model loading pattern in app.py")
    
    updated_content = re.sub(
        model_loading_pattern,
        f'\\1"{args.model_path}"',
        updated_content
    )
    
    # Remove trust_remote_code if the model is local
    updated_content = updated_content.replace(
        'trust_remote_code=True,',
        '# trust_remote_code not needed for local model'
    )
    
    # Update the rubric to use 100-point scale instead of 5-point scale
    updated_content = updated_content.replace(
        'DREsS Rubric (score range: 1 to 5, with 0.5 increments)',
        'DREsS Rubric (score range: 0 to 100)'
    )
    
    # Update the score format in the generate_evaluation function
    prompt_pattern = r'Content: \[Score\]/5'
    if prompt_pattern in updated_content:
        updated_content = updated_content.replace(
            'Content: [Score]/5',
            'Content: [Score]/100'
        )
        updated_content = updated_content.replace(
            'Organization: [Score]/5',
            'Organization: [Score]/100'
        )
        updated_content = updated_content.replace(
            'Language: [Score]/5',
            'Language: [Score]/100'
        )
        updated_content = updated_content.replace(
            'Overall Score: [Sum of three scores]/15',
            'Overall Score: [Average of three scores]/100'
        )
    
    # Write the updated content back to app.py
    with open(args.app_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully updated {args.app_file} to use the fine-tuned model at {args.model_path}")
    print("The app now uses a 100-point scale for evaluation")
    print("You can now run your app with: streamlit run app.py")

if __name__ == "__main__":
    main() 