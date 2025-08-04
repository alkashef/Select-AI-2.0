import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from db import TeradataDatabase


if __name__ == "__main__":
    db = TeradataDatabase()
    db.connect()
    print(20*"-", "Schema", 20*"-")
    print(db.get_schema(), "\n")

    print(20*"-", "Execute Query", 20*"-")
    query = """SELECT * FROM raven.transactions WHERE branch_id = 1;"""
    row_no=1
    for row in db.execute_query(query):
        print(f"row{row_no}:")
        for key, value in row.items():
            print(f"\t{key}={value}")
        row_no += 1
