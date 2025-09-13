"""Manual DB script.

Run:
  python tests/test_db.py

Behavior:
  - Creates a TeradataDatabase instance (uses env vars TD_HOST, TD_NAME, TD_USER, TD_PASSWORD).
  - Prints a schema snapshot.
  - Executes a test query (TD_TEST_QUERY env or fallback) and prints row data.
"""

import os
import sys
from pathlib import Path

# Make the tests folder importable when run as a script
TESTS_DIR = Path(__file__).parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_db_helper import TestDBHelper  # local slim helper


def main() -> int:
    missing = [v for v in ["TD_HOST", "TD_NAME", "TD_USER", "TD_PASSWORD"] if not os.getenv(v)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)} (populate config/.env)")
        return 2

    db = TestDBHelper()
    try:
        db.connect()
        print("-" * 20, "Schema", "-" * 20)
        try:
            schema_txt = db.get_schema()
            print(schema_txt[:1500] + ("..." if len(schema_txt) > 1500 else ""), "\n")
        except Exception as e:  # schema may be large or restricted
            print(f"Schema retrieval failed: {e}\n")

        print("-" * 20, "Execute Query", "-" * 20)
        query = os.getenv("TD_TEST_QUERY", "SELECT CURRENT_DATE;")
        try:
            rows = db.execute_query(query)
            if not rows:
                print("(no rows)")
            for idx, row in enumerate(rows, start=1):
                print(f"row{idx}:")
                for k, v in row.items():
                    print(f"  {k}={v}")
        except Exception as e:
            print(f"Query failed: {e}")
            return 1
        return 0
    finally:
        try:
            db.disconnect()
        except Exception:
            pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
