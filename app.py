import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import re

# Set page configuration
st.set_page_config(
    page_title="Essay Evaluator",
    page_icon="📝",
    layout="centered"
)

# Create a flag to track if model is loaded
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

def load_model():
    """Load the Phi-2 model and tokenizer with error handling"""
    try:
        # Set cache directory to local folder
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        torch.random.manual_seed(0)
        
        # Check if CUDA is available, otherwise use CPU
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            cache_dir=cache_dir
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_evaluation(essay_text, model, tokenizer):
    """Generate evaluation using Phi-2 with improved format enforcement"""
    try:
        # Use a more structured few-shot prompt to guide the model
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

Follow this exact format for your evaluation:

Content: [Score]/5
Organization: [Score]/5
Language: [Score]/5
Overall Score: [Sum of three scores]/15

Please provide:

Justification for Content: [ONE short sentence only]
Justification for Organization: [ONE short sentence only]
Justification for Language: [ONE short sentence only]

Response:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response part
        if "Response:" in result:
            result = result.split("Response:")[1].strip()
        
        # Basic cleanup
        result = result.replace("Possible response:", "").strip()
        
        # Post-process to fix format if needed
        try:
            return format_evaluation_output(result)
        except Exception as e:
            return f"Error processing evaluation: {str(e)}\n\nRaw response:\n{result}"
            
    except Exception as e:
        return f"Error generating evaluation: {str(e)}"

def format_evaluation_output(raw_output):
    """Format or fix the model output to match expected format"""
    
    # Initialize sections
    formatted_output = {}
    
    # Try to extract scores using regex patterns
    content_match = re.search(r"Content:\s*(\d+\.?\d*)/5", raw_output)
    org_match = re.search(r"Organization:\s*(\d+\.?\d*)/5", raw_output)
    lang_match = re.search(r"Language:\s*(\d+\.?\d*)/5", raw_output)
    
    # If we don't find the standard format, check for bullet points format
    if not (content_match and org_match and lang_match):
        # Try to extract from bullet point format
        bullet_points = re.findall(r"\*\s*(.*?)(?=\*|$)", raw_output, re.DOTALL)
        
        if len(bullet_points) >= 3:
            # Extract scores from descriptions if possible
            try:
                content_score = float(re.search(r"(\d+\.?\d*)", bullet_points[0]).group(1)) 
                org_score = float(re.search(r"(\d+\.?\d*)", bullet_points[1]).group(1))
                lang_score = float(re.search(r"(\d+\.?\d*)", bullet_points[2]).group(1))
            except:
                # If no scores found in bullet points, assign default scores based on content
                content_score = 3.0
                org_score = 3.0
                lang_score = 3.0
                
            # Create justifications from the bullet points
            content_just = bullet_points[0].strip().split('.')[0] + '.'
            org_just = bullet_points[1].strip().split('.')[0] + '.'
            lang_just = bullet_points[2].strip().split('.')[0] + '.'
            
            # Calculate overall score
            overall_score = content_score + org_score + lang_score
            
            # Format output
            formatted_result = f"""Content: {content_score}/5
Organization: {org_score}/5
Language: {lang_score}/5
Overall Score: {overall_score}/15

Justification for Content: {content_just}
Justification for Organization: {org_just}
Justification for Language: {lang_just}"""
            
            return formatted_result
    
    # If standard format was found, just return the original output
    return raw_output

# Main app
st.title("📝 Essay Evaluator")
st.markdown("""
    <style>
        .big-font { font-size:18px !important; }
    </style>
    <p class="big-font">Enter an essay to receive detailed feedback based on the DREsS rubric.</p>
""", unsafe_allow_html=True)

# Load model automatically on page load
if not st.session_state.model_loaded:
    with st.spinner("Loading evaluation model... This may take a moment."):
        model, tokenizer = load_model()
        if model is not None and tokenizer is not None:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            st.success("Evaluation model loaded successfully!")
            time.sleep(1)
            st.experimental_rerun()

# Input area and evaluation
if st.session_state.model_loaded:
    essay_prompt = st.text_area("Essay Prompt:", height=100, 
                              placeholder="Enter the essay prompt or question here...")
    essay_text = st.text_area(
        "Paste the student's essay here:",
        height=300,
        placeholder="Enter the essay text to evaluate..."
    )
    
    if st.button("Evaluate Essay", type="primary"):
        if not essay_text.strip():
            st.error("Please enter an essay to evaluate.")
        else:
            with st.spinner("Analyzing essay and generating feedback..."):
                evaluation = generate_evaluation(essay_text, st.session_state.model, st.session_state.tokenizer)
                
                # Display results in a more structured way
                st.markdown("## Evaluation Results")
                
                # Create an expander to show the raw response for debugging
                with st.expander("Debug: Raw Response"):
                    st.text(evaluation)

                st.markdown("## Generated Output")
                st.text(evaluation)
                