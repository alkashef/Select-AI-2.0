"""
Connection page module for the BorAI application.
Manages database connection settings.
"""
import os
import streamlit as st
from typing import Optional
from dotenv import dotenv_values, set_key
from db import TeradataDatabase


def render(db: Optional[TeradataDatabase]) -> None:
    """
    Render the connection page interface.
    
    Args:
        db: The TeradataDatabase instance
    """
    st.markdown("<h1 style='font-size: 3em;'>Select AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: grey;'>Database Connection</h2>", unsafe_allow_html=True)

    # Connection status indicator
    connection_status = "Connected" if db and db.connection else "Disconnected"
    status_color = "rgba(0, 255, 0, 0.5)" if connection_status == "Connected" else "rgba(255, 0, 0, 0.5)"       
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; background-color: {status_color}; color: white;">
        <strong>Status:</strong> {connection_status}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    #st.markdown("#### Connection Parameters")
    
    # Load current values from .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env')
    env_vars = dotenv_values(env_path)
    
    # Create form for connection settings
    with st.form("connection_settings"):
        host = st.text_input("Host", value=env_vars.get("TD_HOST", ""))
        database = st.text_input("Database Name", value=env_vars.get("TD_NAME", ""))
        user = st.text_input("Username", value=env_vars.get("TD_USER", ""))
        password = st.text_input("Password", value=env_vars.get("TD_PASSWORD", ""), type="password")
        port = st.text_input("Port", value=env_vars.get("TD_PORT", "1025"))
        
        submitted = st.form_submit_button("Save Connection Settings")
        
        if submitted:
            # Update .env file with new values
            set_key(env_path, "TD_HOST", host)
            set_key(env_path, "TD_NAME", database)
            set_key(env_path, "TD_USER", user)
            set_key(env_path, "TD_PASSWORD", password)
            set_key(env_path, "TD_PORT", port)
            st.success("Connection settings updated! Restart the application to apply changes.")