"""
Dataset page module for the BorAI application.
Displays database schema information and sample data.
"""
import streamlit as st
import pandas as pd
import re
from typing import Optional, Dict, List, Tuple
from db import TeradataDatabase


def parse_schema_to_dataframe(schema: str) -> Dict[str, pd.DataFrame]:
    """
    Parse the database schema string into a dictionary of DataFrames, one per table.
    
    Args:
        schema: Raw schema text from database
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping table names to their column DataFrames
    """
    schema_dict = {}
    schema_sections = schema.split("\n\nTable: ")
    
    for i, section in enumerate(schema_sections):
        if i == 0 and not section.startswith("Table: "):
            continue
            
        if i > 0:
            section = f"Table: {section}"
            
        # Extract table name
        table_name = section.split("\n")[0].replace("Table: ", "")
        
        # Parse columns using regex
        column_pattern = r"- ([^\(]+) \(([^)]+)\)(?: - (.+))?"
        matches = re.findall(column_pattern, section)
        
        # Create DataFrame from parsed columns
        if matches:
            data = [{"Column Name": col.strip(), 
                     "Data Type": dtype.strip(), 
                     "Description": desc.strip() if desc else ""} 
                    for col, dtype, desc in matches]
            schema_dict[table_name] = pd.DataFrame(data)
    
    return schema_dict


def get_sample_data(db: TeradataDatabase, table_name: str, sample_size: int = 5) -> pd.DataFrame:
    """
    Fetch sample data from a database table.
    
    Args:
        db: Database connection
        table_name: Name of the table to sample
        sample_size: Number of rows to sample
        
    Returns:
        pd.DataFrame: Sample data from the table
    """
    try:
        # Create a safe SQL query to fetch sample data
        query = f"SELECT * FROM {table_name} SAMPLE {sample_size}"
        result = db.execute_query(query)
        
        # Convert result to DataFrame if it's not already
        if not isinstance(result, pd.DataFrame):
            # If result is a list of lists, convert it to DataFrame
            if result and isinstance(result, list):
                # Try to get column names from the first row if it exists
                if len(result) > 0 and isinstance(result[0], list) and len(result[0]) > 0:
                    return pd.DataFrame(result[1:], columns=result[0])
                else:
                    return pd.DataFrame(result)
            else:
                return pd.DataFrame()
        return result
    except Exception as e:
        st.error(f"Error fetching sample data from {table_name}: {str(e)}")
        return pd.DataFrame()


def render(db: Optional[TeradataDatabase]) -> None:
    """
    Render the dataset page interface with schema and sample data.
    
    Args:
        db: The TeradataDatabase instance
    """
    st.markdown("<h1 style='font-size: 3em;'>Select AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: grey;'>Dataset</h2>", unsafe_allow_html=True)

    if not db or not db.connection:
        st.error("Database is not connected. Please check the connection configuration.")
        return
    
    try:
        with st.spinner("Loading database schema..."):
            schema = db.get_schema()
            schema_dict = parse_schema_to_dataframe(schema)
        
        # Display schema and sample data for each table
        for table_name, columns_df in schema_dict.items():
            with st.expander(f"📊 {table_name}"):
                st.markdown("#### Table Schema")
                st.dataframe(columns_df, use_container_width=True)
                
                # Sample data section
                st.markdown("#### Sample Data")
                
                # Add a button to load sample data on demand
                if st.button(f"Load sample data from {table_name}", key=f"sample_{table_name}"):
                    with st.spinner(f"Loading sample data from {table_name}..."):
                        sample_df = get_sample_data(db, table_name)
                        
                    # Check if DataFrame is empty using pandas-specific methods
                    if isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
                        st.dataframe(sample_df, use_container_width=True)
                    else:
                        st.info(f"No data available in {table_name} or access restricted.")
                
    except Exception as e:
        st.error(f"Error processing database schema: {str(e)}")