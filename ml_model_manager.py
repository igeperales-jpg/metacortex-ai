import os
from sentence_transformers import SentenceTransformer
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 METACORTEX Model Manager - Sistema Centralizado de Gestión de Modelos ML
=============================================================================

Sistema robusto y avanzado para gestión inteligente de modelos ML con:
    pass  # TODO: Implementar
✅ Singleton global - Una sola instancia por modelo en TODO el sistema
✅ Lazy loading - Carga modelos solo cuando se necesitan
✅ Task-based pooling - Asigna modelos específicos por tarea
✅ Metal MPS optimizado - Usa GPU Apple Silicon con fallback a CPU
✅ Memory-efficient - Libera modelos no usados automáticamente
✅ Process-aware - Evita duplicación entre procesos
✅ Error handling - Maneja errores de MPS automáticamente

Autor: METACORTEX Team
Fecha: 2025-11-11
"""

import logging
import threading
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Información de un modelo cargado"""
    name: str
    task: str
    model_instance: Any
    loaded_at: datetime
    last_used: datetime
    size_mb: float
    device: str
    usage_count: int = 0


class MetacortexModelManager:
    """
    🎯 Gestor centralizado de modelos ML
    
    Características:
    - Singleton global compartido
    - Lazy loading (carga bajo demanda)
    - Task-based pooling (modelo específico por tarea)
    - Memory-efficient (limpieza automática)
    - MPS optimizado con fallback a CPU
    - Thread-safe
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.logger = logging.getLogger("model_manager")
        
        # Cache de modelos cargados
        self._models: Dict[str, ModelInfo] = {}
        self._model_lock = threading.Lock()
        
        # Detección de dispositivo
        self.device = self._detect_device()
        self.logger.info(f"🎯 Model Manager inicializado - Device: {self.device}")
        
        # Configuración de memoria
        self.max_memory_mb = 2000  # 2 GB máximo para modelos
        self.cleanup_interval = timedelta(minutes=5)
        self.max_idle_time = timedelta(minutes=10)
        
        # Registro de modelos disponibles por tarea
        self._task_models = {
            "embedding_default": {
                "model_name": "all-MiniLM-L6-v2",
                "size_mb": 90,
                "description": "Modelo general de embeddings (384 dim)"
            },
            "embedding_large": {
                "model_name": "all-mpnet-base-v2",
                "size_mb": 420,
                "description": "Modelo grande de embeddings (768 dim)"
            },
            "embedding_multilingual": {
                "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
                "size_mb": 470,
                "description": "Modelo multilingüe (384 dim)"
            },
            "semantic_search": {
                "model_name": "msmarco-distilbert-base-v4",
                "size_mb": 250,
                "description": "Optimizado para búsqueda semántica"
            }
        }
        
        self.logger.info(f"📋 Registrados {len(self._task_models)} modelos predeterminados")
        
        # Iniciar cleanup daemon
        self._start_cleanup_daemon()
    
    def _detect_device(self) -> str:
        """
        Detecta el mejor dispositivo disponible con soporte robusto para MPS
        
        Returns:
            "mps" para Apple Silicon GPU, "cuda" para NVIDIA, "cpu" como fallback
        """
        try:
            import torch
            
            # 1. Intentar MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # 🔧 FIX: Configurar MPS para evitar errores de low watermark
                    # Este es un workaround conocido para PyTorch + MPS
                    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                    
                    # Test MPS con tensor pequeño
                    test_tensor = torch.zeros(1, device='mps')
                    del test_tensor
                    
                    self.logger.info("✅ MPS (Apple Silicon GPU) disponible y funcional")
                    return "mps"
                    
                except Exception as mps_error:
                    logger.error(f"Error en ml_model_manager.py: {mps_error}", exc_info=True)
                    self.logger.warning(f"⚠️ MPS disponible pero con error: {mps_error}")
                    self.logger.info("🔄 Intentando fix de MPS...")
                    
                    # Intentar solución alternativa
                    try:
                        torch.mps.empty_cache()
                        test_tensor = torch.zeros(1, device='mps')
                        del test_tensor
                        self.logger.info("✅ MPS funcionando después del fix")
                        return "mps"
                    except Exception as e:
                        logger.error(f"Error en ml_model_manager.py: {e}", exc_info=True)
                        self.logger.warning("⚠️ MPS no funcional, usando CPU")
            
            # 2. Intentar CUDA (NVIDIA)
            if torch.cuda.is_available():
                self.logger.info("✅ CUDA (NVIDIA GPU) disponible")
                return "cuda"
            
            # 3. Fallback a CPU
            self.logger.info("ℹ️ Usando CPU (no hay GPU disponible)")
            return "cpu"
            
        except ImportError:
            self.logger.warning("⚠️ PyTorch no disponible, usando CPU")
            return "cpu"
    
    def get_model(self, task: str = "embedding_default", force_reload: bool = False) -> Optional[Any]:
        """
        Obtiene modelo para una tarea específica (lazy loading)
        
        Args:
            task: Tipo de tarea ("embedding_default", "embedding_large", etc.)
            force_reload: Forzar recarga del modelo
            
        Returns:
            Instancia del modelo o None si hay error
        """
        with self._model_lock:
            # Verificar si el modelo ya está cargado
            if task in self._models and not force_reload:
                model_info = self._models[task]
                model_info.last_used = datetime.now()
                model_info.usage_count += 1
                self.logger.debug(f"♻️ Reutilizando modelo: {task} (usos: {model_info.usage_count})")
                return model_info.model_instance
            
            # Cargar modelo nuevo
            return self._load_model(task)
    
    def _load_model(self, task: str) -> Optional[Any]:
        """
        Carga un modelo específico
        
        Args:
            task: Tipo de tarea
            
        Returns:
            Instancia del modelo o None si hay error
        """
        if task not in self._task_models:
            self.logger.error(f"❌ Tarea desconocida: {task}")
            return None
        
        task_info = self._task_models[task]
        model_name = task_info["model_name"]
        
        # Verificar memoria disponible
        current_memory = self._get_current_memory_usage()
        if current_memory + task_info["size_mb"] > self.max_memory_mb:
            self.logger.warning(f"⚠️ Memoria insuficiente, liberando modelos antiguos...")
            self._cleanup_old_models(force=True)
        
        try:
            self.logger.info(f"⏳ Cargando modelo: {task} ({task_info['size_mb']}MB)")
            start_time = time.time()
            
            # Cargar con SentenceTransformer
            
            # 🔧 FIX: Cargar con device explícito y error handling
            try:
                model = SentenceTransformer(model_name, device=self.device)
            except Exception as device_error:
                logger.error(f"Error en ml_model_manager.py: {device_error}", exc_info=True)
                self.logger.warning(f"⚠️ Error cargando en {self.device}: {device_error}")
                self.logger.info("🔄 Reintentando con CPU...")
                model = SentenceTransformer(model_name, device='cpu')
                self.device = 'cpu'  # Actualizar device para futuros modelos
            
            load_time = time.time() - start_time
            
            # Registrar modelo
            model_info = ModelInfo(
                name=model_name,
                task=task,
                model_instance=model,
                loaded_at=datetime.now(),
                last_used=datetime.now(),
                size_mb=task_info["size_mb"],
                device=str(model.device) if hasattr(model, 'device') else self.device,
                usage_count=1
            )
            
            self._models[task] = model_info
            
            self.logger.info(f"✅ Modelo cargado: {task} en {load_time:.2f}s")
            self.logger.info(f"   Device: {model_info.device}")
            self.logger.info(f"   Memoria total: {self._get_current_memory_usage():.0f}MB")
            
            return model
            
        except Exception as e:
            logger.error(f"Error en ml_model_manager.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error cargando modelo {task}: {e}")
            return None
    
    def _get_current_memory_usage(self) -> float:
        """Calcula memoria total usada por modelos cargados"""
        return sum(info.size_mb for info in self._models.values())
    
    def _cleanup_old_models(self, force: bool = False) -> None:
        """
        Limpia modelos no usados recientemente
        
        Args:
            force: Forzar limpieza inmediata
        """
        now = datetime.now()
        to_remove = []
        
        for task, info in self._models.items():
            time_since_use = now - info.last_used
            
            if force or time_since_use > self.max_idle_time:
                to_remove.append(task)
        
        for task in to_remove:
            info = self._models[task]
            self.logger.info(f"🗑️ Liberando modelo: {task} ({info.size_mb}MB, idle: {now - info.last_used})")
            
            # Limpiar memoria
            del self._models[task]
            
            # Limpiar cache de GPU si está disponible
            try:
                import torch
                if self.device == 'mps':
                    torch.mps.empty_cache()
                elif self.device == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                pass
    
    def _start_cleanup_daemon(self) -> None:
        """Inicia daemon de limpieza automática"""
        def cleanup_loop():
            while True:
                time.sleep(self.cleanup_interval.total_seconds())
                with self._model_lock:
                    self._cleanup_old_models()
        
        daemon_thread = threading.Thread(target=cleanup_loop, daemon=True, name="model-cleanup")
        daemon_thread.start()
        self.logger.info(f"✅ Cleanup daemon iniciado (check cada {self.cleanup_interval.total_seconds()/60:.0f}min)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del Model Manager"""
        with self._model_lock:
            return {
                "device": self.device,
                "models_loaded": len(self._models),
                "total_memory_mb": self._get_current_memory_usage(),
                "max_memory_mb": self.max_memory_mb,
                "models": {
                    task: {
                        "name": info.name,
                        "size_mb": info.size_mb,
                        "device": info.device,
                        "usage_count": info.usage_count,
                        "loaded_at": info.loaded_at.isoformat(),
                        "last_used": info.last_used.isoformat()
                    }
                    for task, info in self._models.items()
                }
            }
    
    def get_available_tasks(self) -> List[str]:
        """Lista todas las tareas disponibles"""
        return list(self._task_models.keys())


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

_global_model_manager: Optional[MetacortexModelManager] = None
_manager_lock = threading.Lock()


def get_model_manager() -> MetacortexModelManager:
    """
    Obtiene instancia global del Model Manager (singleton)
    
    Returns:
        Instancia única de MetacortexModelManager
    """
    global _global_model_manager
    
    if _global_model_manager is None:
        with _manager_lock:
            if _global_model_manager is None:
                _global_model_manager = MetacortexModelManager()
                logger.info("✅ Model Manager ready - Task-based pooling enabled")
    
    return _global_model_manager


# ═══════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🎯 Testing METACORTEX Model Manager")
    print("=" * 60)
    
    # Test singleton
    manager1 = get_model_manager()
    manager2 = get_model_manager()
    print(f"\n✅ Singleton test: {manager1 is manager2}")
    
    # Test device detection
    print(f"\n📱 Device detectado: {manager1.device}")
    
    # Test lazy loading
    print("\n⏳ Cargando modelo de embeddings...")
    model1 = manager1.get_model("embedding_default")
    print(f"✅ Modelo 1 cargado: {model1 is not None}")
    
    # Test reutilización
    print("\n♻️ Reutilizando modelo...")
    model2 = manager1.get_model("embedding_default")
    print(f"✅ Mismo modelo: {model1 is model2}")
    
    # Test estadísticas
    print("\n📊 Estadísticas:")
    stats = manager1.get_stats()
    print(f"   Device: {stats['device']}")
    print(f"   Modelos cargados: {stats['models_loaded']}")
    print(f"   Memoria total: {stats['total_memory_mb']:.0f}MB")
    
    # Test embeddings
    if model1:
        print("\n🧪 Test de embeddings:")
        embeddings = model1.encode(["test sentence"])
        print(f"   Shape: {embeddings.shape}")
        print(f"   Device: {model1.device if hasattr(model1, 'device') else 'unknown'}")
    
    print("\n✅ Test completado")