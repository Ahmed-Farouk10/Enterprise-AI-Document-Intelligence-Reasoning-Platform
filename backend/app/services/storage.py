import os
import logging
from typing import BinaryIO, Optional
from pathlib import Path
from supabase import create_client, Client
from app.config import settings

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.use_supabase = settings.database.USE_SUPABASE_STORAGE
        self.local_upload_dir = Path(settings.database.UPLOAD_DIR)
        self.local_upload_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_supabase:
            url = (settings.database.SUPABASE_URL or "").strip().rstrip('/')
            key = (settings.database.SUPABASE_SERVICE_ROLE_KEY or "").strip()
            if not url or not key or "your-project" in url:
                logger.warning("SUPABASE_URL or KEY missing or invalid. Falling back to local storage.")

                self.use_supabase = False
            else:
                try:
                    self.supabase: Client = create_client(url, key)
                    self.bucket_name = "documents"
                    logger.info(f"Initialized Supabase Storage (Bucket: {self.bucket_name})")
                except Exception as e:
                    logger.error(f"Failed to initialize Supabase Storage: {e}. Falling back to local storage.")
                    self.use_supabase = False

    async def upload_file(self, file: BinaryIO, filename: str, content_type: str) -> str:
        """
        Uploads a file and returns its identifier (filename or remote path).
        """
        if self.use_supabase:
            try:
                # Read file content
                file_content = file.read()
                # Upload to Supabase Storage
                response = self.supabase.storage.from_(self.bucket_name).upload(
                    path=filename,
                    file=file_content,
                    file_options={"content-type": content_type}
                )
                logger.info(f"✅ Uploaded to Supabase: {filename}")
                return filename
            except Exception as e:
                logger.error(f"❌ Supabase upload failed: {e}")
                # Fallback to local
                return self._upload_local(file, filename)
        else:
            return self._upload_local(file, filename)

    def _upload_local(self, file: BinaryIO, filename: str) -> str:
        file_path = self.local_upload_dir / filename
        file.seek(0)
        with open(file_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(file, buffer)
        logger.info(f"✅ Uploaded to local storage: {filename}")
        return filename

    def get_file_path(self, filename: str) -> Optional[Path]:
        """Returns local path if file exists locally."""
        file_path = self.local_upload_dir / filename
        if file_path.exists():
            return file_path
        return None

    async def download_file(self, filename: str) -> Optional[bytes]:
        """Downloads file content from either storage provider."""
        if self.use_supabase:
            try:
                return self.supabase.storage.from_(self.bucket_name).download(filename)
            except Exception as e:
                logger.error(f"❌ Supabase download failed: {e}")
                return self._read_local(filename)
        else:
            return self._read_local(filename)

    def _read_local(self, filename: str) -> Optional[bytes]:
        file_path = self.local_upload_dir / filename
        if file_path.exists():
            return file_path.read_bytes()
        return None

    async def delete_file(self, filename: str) -> bool:
        if self.use_supabase:
            try:
                self.supabase.storage.from_(self.bucket_name).remove([filename])
                return True
            except Exception as e:
                logger.error(f"❌ Supabase delete failed: {e}")
        
        file_path = self.local_upload_dir / filename
        if file_path.exists():
            os.remove(file_path)
            return True
        return False

storage_service = StorageService()
