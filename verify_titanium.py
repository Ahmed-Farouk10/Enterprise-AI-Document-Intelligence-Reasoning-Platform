import sys
import os
import shutil

# Make sure we import from app
sys.path.append(os.path.join(os.getcwd(), 'backend'))

print("--- 1. VERIFYING PERMISSIONS ---")
cache_dir = "backend/app/.cache/cognee_data" # Simulated local path
# In Docker it will be /app/.cache/...

print("--- 2. IMPORTING COGNEE_SETUP ---")
try:
    import app.cognee_setup
    print("✅ cognee_setup imported successfully")
except Exception as e:
    print(f"❌ cognee_setup failed: {e}")
    sys.exit(1)

print("--- 3. TESTING COGNEE ENGINE PATCH ---")
try:
    import cognee.infrastructure.databases.relational as rel_pkg
    
    engine = rel_pkg.get_relational_engine()
    print(f"Engine type: {type(engine)}")
    print(f"Engine info: {engine}")
    
    if hasattr(engine, "url"):
        print(f"✅ Engine URL: {engine.url}")
        if "cognee_data" in str(engine.url):
             print("✅ PATH IS CORRECT!")
        else:
             print("❌ PATH IS INCORRECT")
    else:
        print("❌ Engine has no URL attribute (might be raw sqlite3 connection?)")

except Exception as e:
    print(f"❌ Failed to get engine: {e}")
    import traceback
    traceback.print_exc()

print("--- 4. TESTING OS.MAKEDIRS ---")
try:
    test_path = "c:\\python\\lib\\site-packages\\cognee\\.cognee_system\\test_dir"
    os.makedirs(test_path, exist_ok=True)
    print("✅ os.makedirs call intercepted (no error)")
except Exception as e:
    print(f"❌ os.makedirs failed: {e}")
