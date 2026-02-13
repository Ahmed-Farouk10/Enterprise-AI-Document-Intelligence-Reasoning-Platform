import sys
import os
import logging

# Add the backend directory to sys.path so we can import app modules
sys.path.append(os.path.join(os.getcwd(), 'backend'))

# Mimic the main.py import sequence
print("1. Importing cognee_setup...")
import app.cognee_setup

print("2. Importing cognee internals...")
from cognee.infrastructure.databases.relational import get_relational_engine

print("3. Calling get_relational_engine()...")
try:
    engine = get_relational_engine()
    print(f"   Engine object: {engine}")
    
    # Try to inspect the URL or path
    if hasattr(engine, "url"):
        print(f"   Engine URL: {engine.url}")
    elif hasattr(engine, "engine"):
         print(f"   Inner Engine URL: {engine.engine.url}")
    else:
        print("   Could not find .url attribute, checking __dict__...")
        print(f"   {engine.__dict__}")

    # Check os.makedirs patch
    print("4. Testing os.makedirs patch...")
    test_path = "c:\\python\\lib\\site-packages\\cognee\\.cognee_system\\test_dir"
    print(f"   Attempting to create: {test_path}")
    try:
        os.makedirs(test_path, exist_ok=True)
        print("   os.makedirs called without error.")
    except Exception as e:
        print(f"   os.makedirs threw exception (unexpected if patched): {e}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
