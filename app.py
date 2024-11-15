from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.cuda.amp import autocast
import re
import os

# Set memory configuration for PyTorch to help with GPU memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.cuda.empty_cache()

# Load the fine-tuned model and tokenizer
model_path = "C:/Users/Kiran/Downloads/GPT2_medium_trained_model-20241101T220221Z-001/GPT2_medium_trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Try moving the model to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model.to(device)
except torch.cuda.OutOfMemoryError:
    print("CUDA memory is insufficient. Switching to CPU.")
    device = "cpu"  # Fallback to CPU
    model.to(device)
model.eval()

# Initialize Flask app
app = Flask(__name__)

# Route for the main page (H.html)
@app.route('/')
def main_page():
    return render_template('index.html')

# Route for the input page (input.html)
@app.route('/input')
def input_page():
    return render_template('input.html')

# Endpoint to generate story based on user input
@app.route('/generate', methods=['POST'])
def generate_story():
    # Parse JSON data from the request
    data = request.get_json()
    character_name = data.get('character_name')
    setting = data.get('setting')
    traits = data.get('traits')
    goal = data.get('goal')
    obstacle = data.get('obstacle')
    genre = data.get('genre')
    twist = data.get('twist', '')  # Optional twist

    # Construct the story prompt
    prompt = (
        f"In this {genre} story, {character_name} lives in {setting}. "
        f"They are known for being {traits}, which drives their mission to {goal}. "
        f"However, their journey is not easy, as they face the greatest obstacle: {obstacle}. "
    )
    
    if twist:
        prompt += f"But things take a surprising turn: {twist}.\n"
    
    print("Constructed Prompt:", prompt)  # Debug: Print prompt to console

    # Tokenize input and move to the device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    try:
        # Generate story using mixed precision to save memory
        with torch.no_grad(), autocast():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=512,  # Temporarily reduced max_length for testing
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.7,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the generated story and clean up the text
        generated_story = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the start if it appears
        if generated_story.startswith(prompt):
            generated_story = generated_story[len(prompt):].strip()

        # Clean up any remaining meta-information or extra lines
        clean_story = re.sub(r'\b(Voice|Tense|Genre|Prompt|Title|Text):.*?(\n|\|)', '', generated_story)  # Remove meta info
        clean_story = re.sub(r'[_]{2,}', '', clean_story)  # Remove lines with only underscores
        clean_story = re.sub(r'^\s*[\|\-–]+\s*$', '', clean_story, flags=re.MULTILINE)  # Remove separators (like "|")
        clean_story = clean_story.strip()  # Final cleanup of extra whitespace
        print("Generated Story:", clean_story)  # Debug: Print generated story to console

        return jsonify({"story": clean_story})
    
    except RuntimeError as e:
        print("Error during generation:", e)
        if "CUDA out of memory" in str(e):
            print("Switching to CPU due to insufficient GPU memory.")
            model.to("cpu")
            inputs = inputs.to("cpu")
            
            # Retry generation on CPU
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=512,  # Reduced max_length to avoid memory issues
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.5,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.7,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_story = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from the start if it appears
            if generated_story.startswith(prompt):
                generated_story = generated_story[len(prompt):].strip()

            # Clean up any remaining meta-information or extra lines
            clean_story = re.sub(r'\b(Voice|Tense|Genre|Prompt|Title|Text):.*?(\n|\|)', '', generated_story)  # Remove meta info
            clean_story = re.sub(r'[_]{2,}', '', clean_story)  # Remove lines with only underscores
            clean_story = re.sub(r'^\s*[\|\-–]+\s*$', '', clean_story, flags=re.MULTILINE)  # Remove separators
            clean_story = clean_story.strip()  # Final cleanup of extra whitespace
            print("Generated Story (CPU Fallback):", clean_story)
            return jsonify({"story": clean_story})

        else:
            return jsonify({"story": "An error occurred during generation."}), 500

if __name__ == '__main__':
    app.run(debug=True)
