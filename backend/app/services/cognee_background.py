import asyncio
import logging
import uuid
from typing import List, Set
from app.core.cognee_config import settings as cognee_settings

try:
    from cognee.modules.users.models import User
except ImportError:
    class User:
        def __init__(self, id):
            self.id = id

logger = logging.getLogger(__name__)

class MemifyService:
    """
    Background service for Self-Improvement and Graph Maintenance.
    Implements the 'Memify' concept from the architecture report.
    """
    
    def __init__(self):
        self.active_datasets: Set[str] = set()
        self.is_running = False
        self._task = None

    def register_dataset(self, dataset_name: str):
        """Track datasets for maintenance"""
        self.active_datasets.add(dataset_name)

    async def start(self):
        """Start the background maintenance service"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("🧠 Memify Service started (Background Graph Optimization)")
        self._task = asyncio.create_task(self._maintenance_loop())

    async def stop(self):
        """Stop the background service"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("🧠 Memify Service stopped")

    async def _maintenance_loop(self):
        """Periodic maintenance loop"""
        while self.is_running:
            try:
                # Maintenance interval (e.g., every hour in production, 5 mins for demo)
                await asyncio.sleep(600)  # 10 minutes
                
                if self.active_datasets:
                    await self.run_maintenance()
                else:
                    logger.debug("🧠 Memify: No active datasets to optimize")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Memify loop error: {e}")
                await asyncio.sleep(60) # Backoff

    async def run_maintenance(self):
        """Execute graph condensation and cleanup"""
        logger.info("🧠 Memify: Starting graph maintenance cycle...")
        
        user = User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
        
        # 1. Prune/Condense (If supported by installed Cognee version, otherwise simulated)
        # Note: In Cognee 0.5.x, cognify *is* the condensation step. 
        # Rerunning it might consolidate new nodes.
        
        for dataset in list(self.active_datasets):
            try:
                logger.info(f"🧠 Memify: Optimizing dataset '{dataset}'")
                # We perform a lightweight cognify or specialized task here
                # For now, we re-run cognify to ensure graph integrity
                await cognee.cognify(datasets=[dataset], user=user)
                
            except Exception as e:
                logger.warning(f"⚠️ Memify optimization failed for {dataset}: {e}")

        logger.info("✅ Memify: Maintenance cycle complete")

# Global singleton
memify_service = MemifyService()
