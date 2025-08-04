import os
import time
import streamlit as st
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv
from db import TeradataDatabase
from nl2sql import NL2SQL
import torch
from logger import logger
# Import page modules
from modules import assistant_page, database_page, dataset_page, model_page

def initialize_database() -> Optional[TeradataDatabase]:
    """
    Initialize database connection once during app startup.
    
    Returns:
        TeradataDatabase: Connected database instance or None if connection fails
    """
    try:
        logger.info("Initializing database connection at app startup")
        db = TeradataDatabase()    
        db.connect()
        # Cache the schema during initialization
        db.get_schema()
        logger.info("Database connection established and schema cached at startup")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize database at startup: {str(e)}")
        return None


def get_log_contents() -> str:
    """Read and return the contents of the current log file.
    
    Returns:
        str: Contents of the log file or error message if file cannot be read
    """
    try:
        log_file_path = logger.log_file_path
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                return f.read()
        return "No log file found"
    except Exception as e:
        return f"Error reading log file: {str(e)}"


def perform_model_warmup(nl2sql_converter: NL2SQL) -> None:
    """
    Performs a warmup inference to initialize model weights and improve initial response time.
    
    Args:
        nl2sql_converter: The NL2SQL instance to warm up
    """
    warmup_query = "how many branches are in texas?"
    logger.info(f"Performing model warmup with query: '{warmup_query}'")
    
    with st.spinner("Warming up model..."):
        try:
            start_time = time.time()
            warmup_sql, _ = nl2sql_converter.nl2sql(warmup_query)
            end_time = time.time()
            warmup_time = end_time - start_time
            logger.info(f"Model warmup completed in {warmup_time:.2f} seconds. Generated SQL: {warmup_sql}")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}. Continuing with initialization.")


def initialize_session_state() -> None:
    """Initialize the Streamlit session state variables if they don't exist."""
    if 'current_sql_query' not in st.session_state:
        st.session_state.current_sql_query = None
    if 'show_sql' not in st.session_state:
        st.session_state.show_sql = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "assistant"  # Default to assistant page


def setup_page_config() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(page_title="Select AI", page_icon="🤖")
    
    env_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
    load_dotenv(env_path)


def render_sidebar() -> None:
    """
    Render the sidebar content with logo, application title, and a single navigation selectbox.
    """
    with st.sidebar:
        logo_path = os.path.join(os.path.dirname(__file__), "img", "td_new_trans.png")
        if os.path.exists(logo_path):
            st.image(logo_path)

        st.markdown(
            """<p></p><p style='color: grey; margin-bottom: 20px;'><b>Select AI</b> is a natural language to SQL assistant.</p>
            <p style='color: grey; margin-bottom: 20px;'>Enter your query in natural language, the AI will generate the corresponding SQL query and execute it on Teradata.</p>""",
            unsafe_allow_html=True
        )

        st.markdown("---")
        
        # Navigation using selectbox
        pages = {
            "Assistant": "assistant",
            "Dataset": "dataset",
            "Database Connection": "database",
            "Model Parameters": "model"
        }
        
        st.markdown("## Pages")
        selected_page = st.selectbox(
            "Pages",
            options=list(pages.keys()),
            key="page_selector",
            index=list(pages.values()).index(st.session_state.current_page) if "current_page" in st.session_state else 0,
            label_visibility="collapsed"
        )
        
        # Check if page has changed and use the new rerun method
        if st.session_state.current_page != pages[selected_page]:
            st.session_state.current_page = pages[selected_page]
            st.rerun()

        st.markdown("---")
        # Github repo link
        st.markdown(
            "<a href='http://github.com/alkashef/Select-AI' target='_blank' style='color: #2c7be5; text-decoration: none;'>"
            "GitHub Repository</a>",
            unsafe_allow_html=True
        )
        # Copyright notice
        st.markdown(
            "<p style='color: grey;'>© 2025 Teradata.</p>",
            unsafe_allow_html=True
        )


def check_database_connection() -> bool:
    """
    Check if the database connection is established.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    if 'db' not in st.session_state:
        st.session_state.db = initialize_database()
    
    db = st.session_state.db
    db_status = db is not None
    
    if not db_status:
        st.error("Database connection failed. Please check your configuration and restart the application.")
    
    return db_status


def initialize_model() -> Optional[NL2SQL]:
    """
    Initialize the NL2SQL model if not already initialized.
    
    Returns:
        Optional[NL2SQL]: Initialized model or None if initialization fails
    """
    if 'nl2sql_converter' in st.session_state:
        return st.session_state.nl2sql_converter
        
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    try:          
        torch.cuda.empty_cache()
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        logger.log_model_loading("code-llama-7b-instruct", device_type, "started")
        
        nl2sql_converter = NL2SQL(st.session_state.db)
        
        # Call the separate warmup method
        perform_model_warmup(nl2sql_converter)
        
        st.session_state.nl2sql_converter = nl2sql_converter
        
        logger.log_model_loading("code-llama-7b-instruct", device_type, "completed")
        return nl2sql_converter
    except Exception as e:
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        logger.log_model_loading("code-llama-7b-instruct", device_type, "failed", 
                                {"error": str(e)})
        st.error(f"Failed to load model: {str(e)}")
        return None


def main() -> None:
    """Main application entry point."""
    print(f"Application started. All messages are being logged to: {logger.log_file_path}")
    
    setup_page_config()
    initialize_session_state()
    render_sidebar()
    
    if not check_database_connection():
        return

    nl2sql_converter = initialize_model()
    
    # Route to the appropriate page based on selection
    current_page = st.session_state.current_page
    
    if current_page == "assistant":
        assistant_page.render(nl2sql_converter)
    elif current_page == "dataset":
        dataset_page.render(st.session_state.db)
    elif current_page == "database":
        database_page.render(st.session_state.db)
    elif current_page == "model":
        model_page.render(nl2sql_converter)


if __name__ == "__main__":
    main()