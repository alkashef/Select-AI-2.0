#!/usr/bin/env python
"""
Test script for the extract_sql function.

This script tests the functionality of extracting SQL queries from model output text.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import os
import sys
from nl2sql import NL2SQL
from db import TeradataDatabase

# Add parent directory to path to import nl2sql module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#def extract_sql(text: str) -> str:
#    """
#    Extract SQL query from model output text.
#    
#    Args:
#        text: Model output text containing SQL query
#        
#    Returns:
#        Extracted SQL query
#        
#    Raises:
#        ValueError: If SQL query cannot be extracted
#    """
#    # Find SQLQuery pattern followed by the SQL statement
#    pattern = r"SQLQuery:\s*(.*?);+"
#    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
#    
#    if not match:
#        raise ValueError("No SQL query found in the text.")
#    
#    # Extract the SQL query, handling multiple matches by taking the last one
#    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
#    sql_query = f"{matches[-1]};"
#    
#    return sql_query


def test_extract_sql() -> None:
    """Test various inputs for the extract_sql function."""
    # Test cases for extract_sql
    test_cases = [
        # Clean case with proper format
        {
            "name": "Clean SQL Query",
            "input": "<SQL Query to run> Question: How many users are there?\n\nSQLQuery: SELECT COUNT(*) FROM users;",
            "expected": "SELECT COUNT(*) FROM raven.users;"
        },
        # Case with multiple SQLQuery mentions
        {
            "name": "Multiple SQLQuery mentions",
            "input": "... SQLQuery: SELECT * FROM table1; ... <SQL Query to run> ... SQLQuery: SELECT COUNT(*) FROM users;",
            "expected": "SELECT COUNT(*) FROM raven.users;"
        },
        # Case with additional text
        {
            "name": "SQL with surrounding text",
            "input": "<SQL Query to run> Here's a query to count users: SQLQuery: SELECT COUNT(*) FROM users; more text...",
            "expected": "SELECT COUNT(*) FROM raven.users;"
        },
        # Complex query with joins
        {
            "name": "Complex query with LIMIT keyword",
            "input": "<SQL Query to run> ... SQLQuery: SELECT u.\"name\", o.\"amount\" FROM \"users\" u JOIN \"orders\" o ON u.\"id\" = o.\"user_id\" WHERE o.\"amount\" > 100 ORDER BY o.\"amount\" DESC LIMIT 5;",
            "expected": "SELECT u.\"name\", o.\"amount\" FROM \"users\" u JOIN \"orders\" o ON u.\"id\" = o.\"user_id\" WHERE o.\"amount\" > 100 ORDER BY o.\"amount\" DESC;"
        },
        # Error case - no SQLQuery
        {
            "name": "No SQLQuery",
            "input": "This doesn't contain any SQL",
            "expected": "ERROR"
        },
        # No semicolon
        {
            "name": "No semicolon",
            "input": "SQLQuery: SELECT COUNT(*) FROM users",
            "expected": "SELECT COUNT(*) FROM raven.users;"
        },
        # Tri brackets example
        {
            "name": "Tri brackets",
            "input": "Question: What is the total amount of deposits in Ohio?\n\nSQLQuery: SELECT include < and > SUM(transaction_amount) FROM transactions WHERE transaction_type = 'deposit' AND branch_id IN (SELECT branch_id FROM branches WHERE governorate = 'Ohio');",
            "expected": '''SELECT include < and > SUM(transaction_amount) FROM raven.transactions WHERE transaction_type = 'deposit' AND branch_id IN (SELECT branch_id FROM raven.branches WHERE governorate = 'Ohio');'''
        },
        # Real example
        {
            "name": "Real example",
            "input": "prompt SQLQuery: prompt prompt prompt prompt SQLQuery:<SQL Query to run> prompt  prompt prompt prompt  prompt prompt prompt SQLQuery: SELECT SUM(transaction_amount) FROM transactions WHERE transaction_type = 'deposit' AND branch_id IN (SELECT branch_id FROM branches WHERE governorate = 'Ohio');",
            "expected": '''SELECT SUM(transaction_amount) FROM raven.transactions WHERE transaction_type = 'deposit' AND branch_id IN (SELECT branch_id FROM raven.branches WHERE governorate = 'Ohio');'''
        }
    ]
    
    # Run tests and print results
    print("=" * 80)
    print("TESTING extract_sql FUNCTION")
    print("=" * 80)
    
    db = TeradataDatabase()
    nl2sql = NL2SQL(db)

    for i, test_case in enumerate(test_cases, 1):
        name = test_case["name"]
        input_text = test_case["input"]
        expected = test_case["expected"]
        
        print(f"\nTest {i}: {name}")
        print(f"Input:\n{input_text}")
        
        try:
            result = nl2sql._extract_sql(input_text)
            print(f"Result: {result}")
            if result == expected:
                print("✅ PASSED")
            else:
                print(f"❌ FAILED - Expected: {expected}")
        except ValueError as e:
            if expected == "ERROR":
                print(f"Expected error: {str(e)}")
                print("✅ PASSED")
            else:
                print(f"❌ FAILED - Unexpected error: {str(e)}")


if __name__ == "__main__":
    test_extract_sql()