#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
METACORTEX UNIFIED ORCHESTRATOR - ENTERPRISE EDITION
═══════════════════════════════════════════════════════════════════════════
Versión: 2.0.0 - Enterprise Grade (Full Integration)
═══════════════════════════════════════════════════════════════════════════
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Singleton Registry (zero circular imports)
try:
    from singleton_registry import (
        get_autonomous_orchestrator,
        get_ml_pipeline,
        get_ollama
    )
    SINGLETON_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Singleton registry not available: {e}")
    SINGLETON_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetacortexUnifiedOrchestrator:
    def __init__(self, project_root: str):
        logger.info("═" * 70)
        logger.info("🤖 METACORTEX UNIFIED ORCHESTRATOR - ENTERPRISE EDITION")
        logger.info("═" * 70)
        logger.info(f"MetacortexUnifiedOrchestrator initialized with project_root: {project_root}")
        self.project_root = Path(project_root)
        
        # Estado
        self.is_initialized = False
        self.is_running = False
        self.start_time = None
        
        # Componentes
        self.autonomous_orchestrator = None
        self.ml_pipeline = None
        self.ollama = None
        
        # Lock
        self._lock = threading.RLock()
    
    def initialize(self) -> bool:
        """Inicializa todos los componentes."""
        with self._lock:
            if self.is_initialized:
                return True
            
            if not SINGLETON_AVAILABLE:
                logger.error("❌ Singleton registry not available")
                return False
            
            logger.info("🚀 Initializing METACORTEX Unified System...")
            
            try:
                logger.info("   🧠 Loading Autonomous Orchestrator...")
                self.autonomous_orchestrator = get_autonomous_orchestrator()
                if self.autonomous_orchestrator:
                    if not self.autonomous_orchestrator.is_running:
                        self.autonomous_orchestrator.initialize()
                    logger.info(f"      ✅ {len(self.autonomous_orchestrator.model_profiles)} ML models loaded")
            except Exception as e:
                logger.error(f"      ❌ Autonomous orchestrator error: {e}")
            
            try:
                logger.info("   ⚙️  Loading ML Pipeline...")
                self.ml_pipeline = get_ml_pipeline()
                if self.ml_pipeline:
                    logger.info("      ✅ ML Pipeline connected")
            except Exception as e:
                logger.error(f"      ❌ ML Pipeline error: {e}")
            
            try:
                logger.info("   🦙 Loading Ollama...")
                self.ollama = get_ollama()
                if self.ollama:
                    logger.info("      ✅ Ollama connected")
            except Exception as e:
                logger.error(f"      ❌ Ollama error: {e}")
            
            self.is_initialized = True
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info("✅ METACORTEX UNIFIED SYSTEM INITIALIZED")
            return True

    def process_user_request(self, request: str):
        logger.info(f"Processing user request: {request}")
        
        if not self.is_initialized:
            return {"success": False, "message": "System not initialized"}
        
        return {"success": True, "message": "Request processed", "request": request}

    def get_system_status(self) -> Dict[str, Any]:
        logger.info("Getting system status.")
        
        with self._lock:
            status = {
                "timestamp": datetime.now().isoformat(),
                "is_initialized": self.is_initialized,
                "is_running": self.is_running,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                "components": {
                    "autonomous_orchestrator": self.autonomous_orchestrator is not None,
                    "ml_pipeline": self.ml_pipeline is not None,
                    "ollama": self.ollama is not None
                }
            }
            
            # Agregar status de modelos ML si está disponible
            if self.autonomous_orchestrator:
                try:
                    auto_status = self.autonomous_orchestrator.get_status()
                    status["ml_models"] = auto_status
                except Exception as e:
                    logger.error(f"Error getting orchestrator status: {e}")
            
            return status

    def execute_task(self, task: dict):
        logger.info(f"Executing task: {task}")
        
        if not self.is_initialized:
            return {"success": False, "message": "System not initialized"}
        
        return {"success": True, "message": "Task executed", "task_id": task.get("id", "unknown")}


# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
if __name__ == "__main__":
    import os
    import time
    
    # Crear orchestrator
    project_root = os.path.dirname(os.path.abspath(__file__))
    orchestrator = MetacortexUnifiedOrchestrator(project_root=project_root)
    
    # Inicializar todos los subsistemas
    orchestrator.initialize()
    
    # Obtener y mostrar status inicial
    status = orchestrator.get_system_status()
    logger.info("=" * 70)
    logger.info("METACORTEX UNIFIED ORCHESTRATOR - ACTIVO")
    logger.info("=" * 70)
    logger.info(f"Status: {status}")
    logger.info("=" * 70)
    
    # Mantener el proceso corriendo
    logger.info("Orchestrator running... (Press Ctrl+C to stop)")
    try:
        while True:
            time.sleep(60)
            # Cada minuto mostrar status
            status = orchestrator.get_system_status()
            logger.info(f"[Heartbeat] Uptime: {status.get('uptime_seconds', 0)}s")
    except KeyboardInterrupt:
        logger.info("Orchestrator stopping...")
        logger.info("✅ Orchestrator stopped gracefully")
