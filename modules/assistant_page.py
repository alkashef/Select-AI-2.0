"""
Assistant page module for the BorAI application.
Provides the main natural language to SQL interface.
"""
import time
import streamlit as st
from typing import Optional
from nl2sql import NL2SQL
from logger import logger


def format_time(elapsed_seconds: float) -> str:
    """
    Format seconds into hours:minutes:seconds format.
    
    Args:
        elapsed_seconds: Time in seconds
        
    Returns:
        str: Formatted time string in HH:MM:SS format
    """
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def translate_query(nl2sql_converter: NL2SQL, natural_language_query: str) -> None:
    """
    Translate a natural language query to SQL.
    
    Args:
        nl2sql_converter: The model to use for translation
        natural_language_query: The natural language query to translate
    """
    if not natural_language_query.strip():
        return
        
    try:
        start_time = time.time()
        sql_query, _ = nl2sql_converter.nl2sql(natural_language_query)
        end_time = time.time()
        
        elapsed_seconds = end_time - start_time
        time_format = format_time(elapsed_seconds)
        
        # Store SQL query in session state for execution and display
        st.session_state.current_sql_query = sql_query
        st.session_state.show_sql = True
        
        # Log the translation time
        logger.info(f"NL2SQL translation completed in {time_format} - Query: {natural_language_query[:50]}...")
            
    except Exception as e:
        st.error(f"Error translating query: {str(e)}")
        logger.error(f"Translation error: {str(e)} - Query: {natural_language_query[:50]}...")


def display_sql_query() -> None:
    """Display the generated SQL query if available in session state."""
    if st.session_state.get('show_sql') and st.session_state.get('current_sql_query'):
        st.markdown("##### Generated SQL Query", unsafe_allow_html=True)
        sql_editor = st.text_area("", value=st.session_state.current_sql_query, height=200, label_visibility="collapsed")
        st.session_state.current_sql_query = sql_editor


def execute_sql_query() -> None:
    """Execute the current SQL query stored in the session state."""
    if not st.session_state.get('current_sql_query'):
        return
        
    try:
        sql_query = st.session_state.current_sql_query
        
        start_time = time.time()
        results = st.session_state.db.execute_query(sql_query)
        end_time = time.time()
        
        elapsed_seconds = end_time - start_time
        time_format = format_time(elapsed_seconds)
        
        logger.info(f"SQL execution completed in {time_format} - Query: {sql_query[:50]}...")
        
        if results is not None:
            st.markdown("##### Query Results")
            st.dataframe(results)
        else:
            st.info("Query executed successfully but returned no results.")
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        logger.error(f"SQL execution error: {str(e)} - Query: {sql_query[:50]}...")


def render(nl2sql_converter: Optional[NL2SQL]) -> None:
    """Render the assistant page interface."""
    st.markdown("<h1 style='font-size: 3em;'>Select AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: grey;'>Assistant</h2>", unsafe_allow_html=True)

    if not nl2sql_converter:
        st.error("Model is not loaded. Please check the model configuration.")
        return
    
    st.markdown("##### Natural Language Query")
    natural_language_query = st.text_area("", height=200, label_visibility="collapsed")

    if st.button("Translate to SQL"):
        translate_query(nl2sql_converter, natural_language_query)
    
    display_sql_query()
    
    if st.session_state.get('current_sql_query'):
        if st.button("Execute Query"):
            execute_sql_query()