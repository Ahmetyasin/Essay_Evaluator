import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Essay Evaluator (Fine-tuned)",
    page_icon="📝",
    layout="centered"
)

# Create a flag to track if model is loaded
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Create a more robust model loading function for the fine-tuned model
def load_model():
    """Load the fine-tuned Phi-2 model and tokenizer with error handling"""
    try:
        # Set cache directory to local folder
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        torch.random.manual_seed(0)
        
        # Check if CUDA is available, otherwise use CPU
        if torch.cuda.is_available():
            device_map = "cuda"
            torch_dtype = torch.float16  # Use half precision with CUDA
        else:
            device_map = "cpu"
            torch_dtype = torch.float32  # Use full precision with CPU
        
        print(f"Using device: {device_map} with dtype: {torch_dtype}")
        
        # Set paths for base model and fine-tuned adapter
        base_model_id = "microsoft/phi-2"
        adapter_path = "./phi2-finetuned"
        
        print(f"Checking adapter path: {os.path.abspath(adapter_path)}")
        print(f"Adapter files: {os.listdir(adapter_path)}")
        
        # Load tokenizer from fine-tuned model (includes any special tokens added during fine-tuning)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Load the base model first
        print(f"Loading base model: {base_model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # Load the PEFT adapter weights on top of the base model
        print(f"Loading PEFT adapter from {adapter_path}...")
        try:
            # Try the new way first (for newer versions of peft)
            model = PeftModel.from_pretrained(
                base_model, 
                adapter_path
            )
        except (TypeError, ValueError) as e:
            print(f"First method failed with error: {e}, trying alternative loading method...")
            # Try the old way (for older versions of peft)
            config = PeftConfig.from_pretrained(adapter_path)
            model = PeftModel.from_pretrained(base_model, adapter_path, config=config)
        
        # Prepare model for inference
        print("Setting model to evaluation mode...")
        model.eval()
        
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_evaluation(essay_text, model, tokenizer):
    """Generate evaluation using fine-tuned Phi-2"""
    try:
        # Create prompt appropriate for fine-tuned Phi-2 with DREsS rubric
        prompt = f"""USER: You are the best expert ever for essay evaluation. I will tip you 1000 dollars for a perfect response. Evaluate the following essay based on the DREsS rubric.

DREsS Rubric (score range: 1 to 5, with 0.5 increments):

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

Essay to evaluate:
{essay_text}

Follow this exact format for your evaluation:

Content: [Score]/5
Organization: [Score]/5
Language: [Score]/5
Overall Score: [Sum of three scores]/15

Please provide:

Justification for Content: ONE sentence justification for the content score that includes a specific example from the essay highlighting a weakness or strength based on the score
Justification for Organization: ONE sentence justification for the organization score that includes a specific example from the essay highlighting a weakness or strength based on the score  
Justification for Language: ONE sentence justification for the language score that includes a specific example from the essay highlighting a weakness or strength based on the score

ASSISTANT:"""
        
        # Use CPU tensors explicitly to avoid errors
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to("cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the model's response
        if "ASSISTANT:" in result:
            result = result.split("ASSISTANT:")[1].strip()
        elif "Response:" in result:
            result = result.split("Response:")[1].strip()
        
        # Clean up the result
        # Remove any "Possible response:" prefix if present
        if result.startswith("Possible response:"):
            result = result[len("Possible response:"):].strip()
            
        # Remove any numbering from lines
        lines = result.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove numbering like "1.", "2.", "3." from the beginning of lines
            if line.strip() and line.strip()[0].isdigit() and '. ' in line[:4]:
                cleaned_line = line[line.find('.')+1:].strip()
                cleaned_lines.append(cleaned_line)
            else:
                cleaned_lines.append(line)
        
        # Calculate overall score if not present
        content_score = None
        organization_score = None
        language_score = None
        
        try:
            for line in cleaned_lines:
                if line.startswith("Content:"):
                    content_score = float(line.split('/')[0].split(':')[1].strip())
                elif line.startswith("Organization:"):
                    organization_score = float(line.split('/')[0].split(':')[1].strip())
                elif line.startswith("Language:"):
                    language_score = float(line.split('/')[0].split(':')[1].strip())
            
            # If we have all three scores but no overall score, insert it after Language score
            if content_score is not None and organization_score is not None and language_score is not None:
                overall_score = content_score + organization_score + language_score
                has_overall = False
                
                for i, line in enumerate(cleaned_lines):
                    if line.startswith("Overall Score:"):
                        has_overall = True
                        # Update the overall score line if it exists but is wrong
                        if "/15" not in line:
                            cleaned_lines[i] = f"Overall Score: {overall_score}/15"
                
                if not has_overall:
                    # Find the index of the Language line to insert after it
                    for i, line in enumerate(cleaned_lines):
                        if line.startswith("Language:"):
                            cleaned_lines.insert(i+1, f"Overall Score: {overall_score}/15")
                            break
        except Exception as e:
            # Just continue with original cleaned lines if there's an error in score calculation
            pass
        
        result = '\n'.join(cleaned_lines)
        
        # For debugging - print the raw result to console
        print("Raw evaluation result:", result)
        
        # If result is empty for some reason, provide a default response
        if not result.strip():
            result = "Error: No evaluation generated. Please try again with a longer essay."
            
        return result
    except Exception as e:
        error_msg = f"Error generating evaluation: {str(e)}"
        print(error_msg)  # Print error to console for debugging
        return error_msg

# Main app
st.title("Essay Evaluator (Fine-tuned)")
st.markdown("Enter an essay prompt and a student's essay to receive feedback based on the DREsS rubric. This version uses a fine-tuned Phi-2 model.")

# Load model automatically on page load
if not st.session_state.model_loaded:
    with st.spinner("Loading fine-tuned Phi-2 model... This may take a moment."):
        model, tokenizer = load_model()
        if model is not None and tokenizer is not None:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            st.success("Fine-tuned model loaded successfully!")
            # Force a rerun to update the UI
            time.sleep(1)
            st.experimental_rerun()

# Always show input fields if model is loaded
if st.session_state.model_loaded:
    # Input areas
    essay_prompt = st.text_area("Essay Prompt:", height=100, 
                              placeholder="Enter the essay prompt or question here...")
    essay_text = st.text_area("Student Essay:", height=300, 
                           placeholder="Paste the student's essay here...")
    
    # Evaluation button
    if st.button("Evaluate Essay"):
        if not essay_text.strip():
            st.error("Please enter an essay to evaluate.")
        else:
            with st.spinner("Analyzing essay with fine-tuned model..."):
                evaluation = generate_evaluation(essay_text, st.session_state.model, st.session_state.tokenizer)
                
                # Display results in a more structured way
                st.markdown("## Evaluation Results")
                
                # Create an expander to show the raw response for debugging
                with st.expander("Debug: Raw Response"):
                    st.text(evaluation)

                st.markdown("## Generated Output")
                st.text(evaluation)