import sys
import os

print("Checking syntax of cognee_engine.py...")
try:
    with open("backend/app/services/cognee_engine.py", "r", encoding="utf-8") as f:
        content = f.read()
    compile(content, "backend/app/services/cognee_engine.py", "exec")
    print("✅ Syntax OK")
except Exception as e:
    print(f"❌ Syntax Error: {e}")
    sys.exit(1)
