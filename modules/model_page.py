"""
Model page module for the BorAI application.
Manages model configuration settings.
"""
import os
import streamlit as st
import torch
from typing import Optional
from dotenv import dotenv_values, set_key
from nl2sql import NL2SQL


def render(nl2sql_converter: Optional[NL2SQL]) -> None:
    """
    Render the model page interface.

    Args:
        nl2sql_converter: The NL2SQL instance
    """
    st.markdown("<h1 style='font-size: 3em;'>Select AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: grey;'>Model Parameters</h2>", unsafe_allow_html=True)
    
    # Enhanced model status indicator with GPU/CPU information
    model_loaded = nl2sql_converter and hasattr(nl2sql_converter, "model")
    model_status = "Loaded" if model_loaded else "Not Loaded"
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    status_color = "rgba(0, 255, 0, 0.5)" if model_loaded else "rgba(255, 0, 0, 0.5)"
    
    status_text = f"{model_status} on {device_type}" if model_loaded else model_status
    
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; background-color: {status_color}; color: white;">
        <strong>Status:</strong> {status_text}
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Load current values from .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env')
    env_vars = dotenv_values(env_path)

    # Create form for model settings
    with st.form("model_settings"):
        model_path = st.text_input("Model Path", value=env_vars.get("MODEL_PATH", "./models/code-llama-7b-instruct"))
        max_length = st.text_input("Maximum Length", value=env_vars.get("MAX_LENGTH", "4096"))
        token_gen_limit = st.text_input("Token Generation Limit", value=env_vars.get("TOKEN_GENERATION_LIMIT", "128"))
        context_length = st.text_input("Input Context Length", value=env_vars.get("INPUT_CONTEXT_LENGTH", "512"))
        result_limit = st.text_input("Result Limit", value=env_vars.get("RESULT_LIMIT", "1"))

        submitted = st.form_submit_button("Save Model Settings")

        if submitted:
            # Update .env file with new values
            set_key(env_path, "MODEL_PATH", model_path)
            set_key(env_path, "MAX_LENGTH", max_length)
            set_key(env_path, "TOKEN_GENERATION_LIMIT", token_gen_limit)
            set_key(env_path, "INPUT_CONTEXT_LENGTH", context_length)
            set_key(env_path, "RESULT_LIMIT", result_limit)
            st.success("Model settings updated! Restart the application to apply changes.")