from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from zep_cloud.client import AsyncZep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
ZEP_API_KEY = os.getenv("ZEP_API_KEY")
NOVEL_USER_ID = "novel_reader_v2" 
CHECKPOINT_FILE = "zep_ingestion_checkpoint.json"

# UPDATED: Reduced to fit 10k char limit
MAX_WORDS_PER_CHUNK = 1500  # ~9000 characters
MIN_WORDS_PER_CHUNK = 300

class ZepFreeRateLimiter:
    def __init__(self, checkpoint_file: str, rpm: int = 4):
        self.checkpoint_file = checkpoint_file
        self.rpm = rpm
        self.state = self._load_checkpoint()
        self.request_times = []
        
    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {'chunks_processed': 0, 'user_created': False}
    
    def _save_checkpoint(self, chunks_processed: int, user_created: bool = None):
        state = {
            'chunks_processed': chunks_processed,
            'last_updated': datetime.now().isoformat(),
            'user_created': user_created if user_created is not None else self.state.get('user_created', False)
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
        self.state = state
        logger.info(f"💾 Checkpoint: {chunks_processed} chunks")
    
    async def acquire(self):
        now = time.time()
        # Filter out requests older than 60 seconds
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.rpm:
            wait_time = 60 - (now - self.request_times[0]) + 2
            logger.warning(f"⏳ Rate limit: {len(self.request_times)}/{self.rpm} RPM. Waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            # Re-filter after waiting
            self.request_times = [t for t in self.request_times if time.time() - t < 60]
            
        self.request_times.append(time.time())

def chunk_novel(text: str) -> list[str]:
    """Splits novel into large text chunks for Graph ingestion"""
    # Simple paragraph-based chunker
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for para in paragraphs:
        words = len(para.split())
        # Add to chunk if it fits
        if current_word_count + words > MAX_WORDS_PER_CHUNK:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_word_count = words
        else:
            current_chunk.append(para)
            current_word_count += words
            
    # Add the final chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
        
    return chunks

async def ingest_novel():
    logger.info("🔧 Initializing Zep Cloud (Graph Mode)...")
    if not ZEP_API_KEY:
        logger.error("❌ ZEP_API_KEY is missing from environment variables")
        return

    client = AsyncZep(api_key=ZEP_API_KEY)
    rate_limiter = ZepFreeRateLimiter(CHECKPOINT_FILE)
    
    # Load Novel
    local_path = os.path.expanduser("~/Books/The Count of Monte Cristo.txt")
    if not os.path.exists(local_path):
        logger.error(f"❌ File not found: {local_path}")
        return

    with open(local_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = chunk_novel(text)
    logger.info(f"📚 Split into {len(chunks)} large chunks for Knowledge Graph")
    logger.info(f"⏱️  Estimated time: ~{len(chunks)/4:.0f} minutes")

    # Create User
    if not rate_limiter.state.get('user_created', False):
        try:
            await rate_limiter.acquire()
            await client.user.add(
                user_id=NOVEL_USER_ID,
                metadata={"title": "The Count of Monte Cristo"}
            )
            rate_limiter._save_checkpoint(0, user_created=True)
            logger.info("👤 User created.")
        except Exception as e:
            if "exists" in str(e).lower():
                logger.info("👤 User already exists.")
                rate_limiter._save_checkpoint(0, user_created=True)
            else:
                logger.error(f"❌ User creation failed: {e}")
                return

    # Ingest Chunks
    start_idx = rate_limiter.state.get('chunks_processed', 0)
    logger.info(f"🚀 Ingesting chunks {start_idx}/{len(chunks)}...")
    
    for i in range(start_idx, len(chunks)):
        chunk_content = chunks[i]
        try:
            await rate_limiter.acquire()
            
            # --- UPDATED API CALL ---
            await client.graph.add(
                user_id=NOVEL_USER_ID,
                data=chunk_content,
                type="text"  # <--- Fixes the 'missing type' error
            )
            # ------------------------
            
            logger.info(f"✅ Chunk {i+1}/{len(chunks)} ingested")
            rate_limiter._save_checkpoint(i + 1)
            
        except Exception as e:
            logger.error(f"❌ Failed chunk {i}: {e}")
            if "403" in str(e):
                logger.critical("⛔ QUOTA EXHAUSTED. Cannot continue.")
                break
            elif "429" in str(e):
                logger.warning("🚫 Rate limited. Waiting 60s...")
                await asyncio.sleep(60)
                # Don't increment index, retry this loop iteration?
                # For simplicity in this script, we break. Rerun to resume.
                break 
            
            # Checkpoint safe to save current progress if we failed? 
            # Ideally we don't save checkpoint if we failed.
            # Script will resume from last successful save.
            break

    logger.info("🎉 Script finished.")

if __name__ == "__main__":
    asyncio.run(ingest_novel())