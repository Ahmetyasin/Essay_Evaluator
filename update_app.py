#!/usr/bin/env python3
"""
This script updates the app.py file to use the fine-tuned model.
Run it after successful fine-tuning.
"""
import os
import re
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser(description="Update app.py to use fine-tuned model")
    parser.add_argument("--model_path", type=str, default="./phi2-finetuned", 
                       help="Path to the fine-tuned model (default: ./phi2-finetuned)")
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
    
    # Update the model path
    # Find and replace the model loading in the load_model function
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
    
    # Write the updated content back to app.py
    with open(args.app_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully updated {args.app_file} to use the fine-tuned model at {args.model_path}")
    print("You can now run your app with: streamlit run app.py")

if __name__ == "__main__":
    main() 