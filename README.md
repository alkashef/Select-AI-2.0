# Select AI
Natural Language to SQL Query Generator for Teradata.

## Overview

Select AI is a natural language to SQL assistant that enables users to query databases using plain English. The application leverages the Code Llama 7B Instruct model to translate natural language questions into SQL queries and run it on Teradata.

Key features:

- Natural language to SQL translation
- Interactive web interface with real-time feedback
- Query execution against Teradata databases
- Batch processing mode for multiple queries

## Technology Stack

- **Frontend**: Streamlit - Python library for interactive web applications
- **Backend**: Python 3.9+ - Core application logic
- **AI Model**: Code Llama 7B Instruct - For natural language to SQL translation
- **Database**: Teradata - Enterprise data warehouse solution
- **Model Optimization**: PyTorch, transformers, bitsandbytes - For model loading and inference
- **Testing**: Python unittest framework - For automated testing
- **Configuration**: dotenv - For environment variable management

## Important Files:

- Core Application: `app.py` - Main entry point with Streamlit interface
- Database Operations: `db.py` - Teradata connection handling
- NL-to-SQL Engine: `nl2sql.py` - Model interaction and query generation
- UI Components: `modules/*.py` - Modular interface components
- Model Template: `prompt.txt` - Prompt engineering template for the LLM
- Utilities: `scripts/*.py` - Helper scripts for model management

## Setup for Development

#### Conda Environment

1. Create a new Conda environment:

    ```bash
	conda create --name select-ai 
    ```

2. Install pip:

    ```bash
	conda install pip
    ```

3. Install project dependencies:

    ```bash
	pip install -r requirements.txt
    ```

#### GPU Configuration

1. Download and install the Nvidia driver appropriate for your GPU
2. Install the CUDA toolkit:
   - Download from: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
   - Follow the installation instructions

3. Install CUDA deep learning package (cuDNN):
   - Download from: https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
   - Extract and follow installation instructions

4. Set up PyTorch with CUDA support:
   ```bash
   # In your Conda environment
   pip uninstall torch torchvision torchaudio -y
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

5. Verify CUDA installation:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
   ```

### Mac GPUs (Apple Silicon or Metal-compatible Intel)
1. Ensure PyTorch 2.0+ is installed:
   ```bash
   pip install --upgrade torch
   ```

#### Download Model

The application uses the Code Llama 7B Instruct model for NL-to-SQL conversion.

1. Use the download script:
   ```bash
   python scripts/download_model.py --repo_id codellama/CodeLlama-7b-Instruct-hf --save_path ./models/code-llama-7b-instruct
   ```

2. The model will be downloaded to the `models/code-llama-7b-instruct/` directory.

3. For offline usage, set the environment variable:

   ```
   export HF_HUB_OFFLINE=1  # Linux/Mac
   set HF_HUB_OFFLINE=1     # Windows
   ```

#### Database Setup

1. Create a Teradata account on the Clearscape Analytics platform: https://clearscape.teradata.com/ 
2. Use the `scripts/td_init.sql` SQL script to create the database, the tables, and insert sample data.
3. Configure database credentials in `config/.env`:
   ```
   TD_HOST=your-teradata-host.com
   TD_NAME=your-database-name
   TD_USER=your-username
   TD_PASSWORD=your-password
   TD_PORT=1025
   ```

## How to Test

Run tests to verify the application components are working correctly:

#### Test Database Connection

```bash
python test/test_db.py
```

This test connects to the Teradata database, retrieves the schema, and executes a sample query to validate connectivity.

#### Test Model Integration

```bash
python test/test_model.py
```

This test verifies that:
- The model files are present
- The model loads correctly
- The model can generate responses to prompts

#### Test SQL Extraction

```bash
python test/test_extract_sql.py
```

This test validates the SQL extraction logic from model outputs.

## How to Run

The application can be run in two modes:

#### Web UI Mode (Default)

```bash
python app.py
```

This starts the web server on `http://localhost:5000`, where you can:
1. Connect to the database
2. Load the AI model
3. Enter natural language queries
4. Get translated SQL queries
5. Execute queries and view results

#### Batch Mode

```bash
python app.py --batch
```

This processes a batch of questions from the file specified in `QUESTIONS_PATH` in the `.env` file and outputs results to an Excel file.
