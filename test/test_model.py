import os
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Any
import torch
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import TeradataDatabase
from nl2sql import NL2SQL

# Global constants
MODELS_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
DEFAULT_PROMPT = "Write a Python function to calculate the factorial of a number."
MODEL_NAME = "code-llama-7b-instruct"  # Name of the model directory
DEFAULT_PAD_TOKEN = "[PAD]"  # Default padding token for testing


def check_model_files(model_path: str) -> List[str]:
    """
    Check if all required files for the model exist in the specified directory.

    Args:
        model_path: Path to the locally saved model directory.
    
    Returns:
        List of files found in the model directory.
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory '{model_path}' does not exist.")

    existing_files = os.listdir(model_path)
    missing_files = [file for file in required_files if file not in existing_files]

    # Check for at least one .bin file
    bin_files = [file for file in existing_files if file.endswith(".bin")]
    if not bin_files:
        missing_files.append("pytorch_model.bin (or equivalent .bin files)")

    if missing_files:
        raise FileNotFoundError(
            f"The following required files are missing in '{model_path}':\n" +
            "\n".join(missing_files) + 
            "\nEnsure the model is downloaded correctly using the 'download_model.py' script."
        )

    return existing_files


def load_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer using the same parameters as in nl2sql.py.

    Args:
        model_path: Path to the locally saved model directory.
        
    Returns:
        Loaded model and tokenizer.
    """
    # Load environment variables for configuration
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env')
    load_dotenv(env_path)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Ensure the tokenizer has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model using the same parameters as nl2sql.py
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            max_memory={0: "6GiB", "cpu": "16GiB"},
            offload_folder="offload",
            offload_state_dict=True,
            local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            offload_folder="offload",
            offload_state_dict=True,
            local_files_only=True
        )
    
    return model, tokenizer


def generate_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    """
    Generate a response from the model based on the given prompt.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        prompt: The input prompt for the model.
        
    Returns:
        The generated response as a string.
    """
    # Get environment variables for configuration
    max_length = int(os.getenv('MAX_LENGTH', 4096))
    input_context_length = int(os.getenv('INPUT_CONTEXT_LENGTH', 2048))
    token_generation_limit = int(os.getenv('TOKEN_GENERATION_LIMIT', 2048))
    
    # Tokenize the input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=input_context_length
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # Generate the response
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=token_generation_limit,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        temperature=1.0,
        num_beams=1
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response


def read_prompt_from_file(file_path: str) -> str:
    """
    Read prompt content from a file.
    
    Args:
        file_path: Path to the file containing the prompt
        
    Returns:
        Content of the file as a string
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading prompt file: {e}")


def test_nl2sql() -> Dict[str, Any]:
    """
    Test NL2SQL conversion with proper imports.

    Returns:
        Dictionary with test results
    """
    from db import TeradataDatabase
    from nl2sql import NL2SQL
    
    db = TeradataDatabase()
    nl2sql = NL2SQL(db)
    
    query = "Show me all transactions in Ohio"
    sql_query, raw_sql = nl2sql.nl2sql(query)  # Now unpacking the tuple
    
    return {
        "query": query,
        "sql": sql_query,
        "raw_sql": raw_sql
    }


def main() -> None:
    """
    Main function to test the downloaded model with a given prompt.
    """
    parser = argparse.ArgumentParser(description="Test the downloaded model with a prompt.")
    parser.add_argument("prompt", type=str, nargs="?", default=None, 
                        help="The input prompt for the model (optional).")
    args = parser.parse_args()

    # Path to the locally saved model
    model_path = os.path.join(MODELS_DIRECTORY, MODEL_NAME)

    # SECTION 1: Check model files
    print("\n" + "="*80)
    print("MODEL FILES CHECK")
    print("="*80)
    try:
        files = check_model_files(model_path)
        print(f"All required files are present in '{model_path}':")
        for file in files:
            print(f"- {file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # SECTION 2: Determine prompt source - simplified
    print("\n" + "="*80)
    print("PROMPT SETUP")
    print("="*80)
    
    # Use provided prompt or default
    prompt = args.prompt if args.prompt else DEFAULT_PROMPT
    
    # If using default, notify user
    if not args.prompt:
        print(f"No prompt provided. Using the default prompt:\n\n'{DEFAULT_PROMPT}'")
    
    print(f"Prompt: {prompt}")

    # SECTION 3: Load the model and tokenizer
    print("\n" + "="*80)
    print("MODEL LOADING")
    print("="*80)
    try:
        print(f"Loading model and tokenizer from: {model_path}")
        model, tokenizer = load_model_and_tokenizer(model_path)
        device = next(model.parameters()).device
        print(f"Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # SECTION 4: Generate response
    print("\n" + "="*80)
    print("MODEL INFERENCE")
    print("="*80)
    try:
        print("Generating response...")
        start_time = os.times()
        response = generate_response(model, tokenizer, prompt)
        end_time = os.times()
        elapsed = end_time.user - start_time.user + end_time.system - start_time.system
        print(f"Response generated in {elapsed:.2f} seconds")
        print("\nModel Response:")
        print("-"*50)
        print(response)
        print("-"*50)
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()