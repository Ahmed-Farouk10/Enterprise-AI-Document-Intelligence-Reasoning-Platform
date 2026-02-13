import sys
import os

print("Checking syntax of cognee_setup.py...")
try:
    with open("backend/app/cognee_setup.py", "r") as f:
        content = f.read()
    compile(content, "backend/app/cognee_setup.py", "exec")
    print("✅ Syntax OK - Verification passed")
except Exception as e:
    print(f"❌ Syntax Error: {e}")
    sys.exit(1)
