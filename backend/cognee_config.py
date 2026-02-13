import os

# Force paths for Hugging Face Spaces
if os.path.exists("/data"):
    COGNEE_ROOT_DIR = "/data/cognee_data"
else:
    COGNEE_ROOT_DIR = os.path.join(os.getcwd(), ".cognee_system")

# Ensure writable
ANONYMOUS_ID_PATH = os.path.join(COGNEE_ROOT_DIR, ".anon_id")
