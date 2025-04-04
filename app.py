import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Essay Evaluator",
    page_icon="📝",
    layout="centered"
)

# Create a flag to track if model is loaded
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Create a more robust model loading function
def load_model():
    """Load the Phi-2 model and tokenizer with error handling"""
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
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            cache_dir=cache_dir
        )
        
        # Load model with user's approach
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            device_map=device_map,
            torch_dtype=torch_dtype,  # Use appropriate dtype based on device
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_evaluation(essay_text, model, tokenizer):
    """Generate evaluation using Phi-2"""
    try:
        # Create prompt appropriate for Phi-2 with DREsS rubric
        prompt = f"""Instruction: You are an expert essay evaluator. Evaluate the following essay based on the DREsS rubric.

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

Follow this exact format for your evaluation (include no numbering and no "Possible response:"):

Content: [Score]/5
Organization: [Score]/5
Language: [Score]/5
Overall Score: [Sum of three scores]/15

Justification for Content: [ONE short sentence only]
Justification for Organization: [ONE short sentence only]
Justification for Language: [ONE short sentence only]

Key strengths:
- [Strength 1]
- [Strength 2]
- [Strength 3]

Areas for improvement:
- [Improvement 1]
- [Improvement 2]
- [Improvement 3]

Response:"""
        
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
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the model's response
        if "Response:" in result:
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
st.title("Essay Evaluator")
st.markdown("Enter an essay prompt and a student's essay to receive feedback based on the DREsS rubric.")

# Load model automatically on page load
if not st.session_state.model_loaded:
    with st.spinner("Loading Phi-2 model... This may take a moment."):
        model, tokenizer = load_model()
        if model is not None and tokenizer is not None:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")
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
            with st.spinner("Analyzing essay..."):
                evaluation = generate_evaluation(essay_text, st.session_state.model, st.session_state.tokenizer)
                
                # Display results in a more structured way
                st.markdown("## Evaluation Results")
                
                # Create an expander to show the raw response for debugging
                with st.expander("Debug: Raw Response"):
                    st.text(evaluation)
                
                # Display the formatted results
                try:
                    # Split the evaluation into sections
                    lines = evaluation.split('\n')
                    
                    # Display scores
                    for line in lines:
                        if any(line.startswith(prefix) for prefix in ["Content:", "Organization:", "Language:", "Overall Score:"]):
                            st.markdown(f"**{line}**")
                    
                    # Display justifications
                    st.markdown("### Justifications")
                    for line in lines:
                        if line.startswith("Justification for"):
                            st.markdown(line)
                    
                    # Display strengths and improvements if they exist
                    if "Key strengths:" in evaluation:
                        st.markdown("### Key Strengths")
                        strengths_section = False
                        for line in lines:
                            if line.strip() == "Key strengths:":
                                strengths_section = True
                                continue
                            if strengths_section and line.strip().startswith("-"):
                                st.markdown(line)
                            if strengths_section and line.strip() == "Areas for improvement:":
                                strengths_section = False
                    
                    if "Areas for improvement:" in evaluation:
                        st.markdown("### Areas for Improvement")
                        improvements_section = False
                        for line in lines:
                            if line.strip() == "Areas for improvement:":
                                improvements_section = True
                                continue
                            if improvements_section and line.strip().startswith("-"):
                                st.markdown(line)
                except Exception as e:
                    # If there's an error in parsing, fall back to displaying the raw output
                    st.markdown(evaluation) 