import sys
import os
import asyncio
import logging

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

# Mock configured environment variables
os.environ["COGNEE_ROOT"] = "./tmp/cognee_test"
os.environ["LLM_API_KEY"] = "test"
os.environ["CO_API_KEY"] = "test"

# MOCK LLM SERVICE to avoid numpy/transformers import errors
import sys
from unittest.mock import MagicMock
sys.modules["app.services.llm_service"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["numpy"] = MagicMock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifier")

async def verify_imports():
    logger.info("🧪 Verifying imports...")
    
    try:
        logger.info("1. Importing Models...")
        from app.models.cognee_models import Resume, DataPoint
        assert issubclass(Resume, DataPoint)
        logger.info("✅ Models OK")

        logger.info("2. Importing Pipelines (ECL)...")
        from app.services.cognee_pipelines import default_ecl, ECLProcessor, PipelineConfig
        assert isinstance(default_ecl, ECLProcessor)
        logger.info("✅ Pipelines OK")

        logger.info("3. Importing Retrievers (Hybrid Search)...")
        from app.services.cognee_retrievers import ResumeRetriever
        assert hasattr(ResumeRetriever, "search_candidates")
        logger.info("✅ Retrievers OK")

        logger.info("4. Importing Background Service (Memify)...")
        from app.services.cognee_background import memify_service
        assert memify_service is not None
        logger.info("✅ Background Service OK")

        logger.info("🎉 All modules imported successfully!")

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(verify_imports())
