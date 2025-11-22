import logging
from typing import Any

logger = logging.getLogger(__name__)

class UnifiedMemoryLayer:
    def get_memory_stats(self):
        logger.info("Getting unified memory stats.")
        return {"stats": "ok (placeholder)"}
    
    def store_episode(self, content: Any, context: Any, importance: Any):
        logger.info(f"Storing episode: {content}")

def get_unified_memory():
    logger.info("Getting unified memory instance.")
    return UnifiedMemoryLayer()
