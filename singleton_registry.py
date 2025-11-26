#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎖️ SINGLETON REGISTRY - MILITARY GRADE IMPORT MANAGER
═══════════════════════════════════════════════════════════════════════════════

MISIÓN: Eliminar imports circulares y gestionar singletons de forma centralizada.

ARQUITECTURA:
┌─────────────────────────────────────────────────────────────┐
│  SINGLETON REGISTRY (Este archivo)                          │
│  • Un solo punto de verdad para todos los singletons        │
│  • Lazy loading inteligente                                 │
│  • Thread-safe con locks                                    │
│  • Zero circular dependencies                               │
└─────────────────────────────────────────────────────────────┘

VENTAJAS:
✅ NO más segmentation faults por imports circulares
✅ NO más inicializaciones duplicadas
✅ Thread-safe por diseño
✅ Lazy loading = startup ultrarrápido
✅ Testeable y mockeable

AUTOR: METACORTEX AUTONOMOUS SYSTEM
FECHA: 2025-11-26
"""

import logging
import threading
from typing import Dict, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class SingletonRegistry:
    """
    Registro global de singletons thread-safe.
    
    En lugar de que cada módulo importe directamente a otros y cause
    circular dependencies, todos registran sus singletons aquí.
    
    Beneficios:
    1. Un solo punto de verdad
    2. Lazy loading automático
    3. Thread-safe con locks
    4. Fácil de testear (mock registry)
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            # Storage para singletons
            self._singletons: Dict[str, Any] = {}
            
            # Storage para factories (lazy loading)
            self._factories: Dict[str, Callable] = {}
            
            # Locks individuales por singleton
            self._singleton_locks: Dict[str, threading.Lock] = {}
            
            self._initialized = True
            logger.info("🎖️ SingletonRegistry initialized")
    
    def register_factory(self, name: str, factory: Callable):
        """
        Registra una factory function para crear un singleton bajo demanda.
        
        Args:
            name: Nombre del singleton
            factory: Function que retorna la instancia
        """
        with self._lock:
            if name in self._factories:
                logger.warning(f"Factory '{name}' ya registrada, sobrescribiendo")
            
            self._factories[name] = factory
            self._singleton_locks[name] = threading.Lock()
            logger.debug(f"✅ Factory registrada: {name}")
    
    def get(self, name: str, *args, **kwargs) -> Optional[Any]:
        """
        Obtiene un singleton. Si no existe, intenta crearlo usando su factory.
        
        Args:
            name: Nombre del singleton
            *args, **kwargs: Argumentos para la factory (solo en primera creación)
        
        Returns:
            El singleton o None si no se puede crear
        """
        # Fast path: ya existe
        if name in self._singletons:
            return self._singletons[name]
        
        # Slow path: crear usando factory
        if name not in self._factories:
            logger.error(f"❌ Singleton '{name}' no tiene factory registrada")
            return None
        
        # Lock específico para este singleton
        with self._singleton_locks[name]:
            # Double-check: puede que otro thread lo haya creado
            if name in self._singletons:
                return self._singletons[name]
            
            try:
                logger.info(f"🏭 Creando singleton: {name}")
                factory = self._factories[name]
                instance = factory(*args, **kwargs)
                self._singletons[name] = instance
                logger.info(f"✅ Singleton creado: {name}")
                return instance
                
            except Exception as e:
                logger.error(f"❌ Error creando singleton '{name}': {e}", exc_info=True)
                return None
    
    def set(self, name: str, instance: Any):
        """
        Registra manualmente un singleton ya creado.
        
        Args:
            name: Nombre del singleton
            instance: Instancia a registrar
        """
        with self._lock:
            if name in self._singletons:
                logger.warning(f"Singleton '{name}' ya existe, sobrescribiendo")
            
            self._singletons[name] = instance
            logger.debug(f"✅ Singleton registrado manualmente: {name}")
    
    def exists(self, name: str) -> bool:
        """Verifica si un singleton existe."""
        return name in self._singletons
    
    def reset(self, name: str):
        """Elimina un singleton del registry (útil para tests)."""
        with self._lock:
            if name in self._singletons:
                del self._singletons[name]
                logger.debug(f"🗑️  Singleton eliminado: {name}")
    
    def reset_all(self):
        """Elimina todos los singletons (útil para tests)."""
        with self._lock:
            count = len(self._singletons)
            self._singletons.clear()
            logger.info(f"🗑️  {count} singletons eliminados")
    
    def list_all(self) -> Dict[str, str]:
        """Lista todos los singletons registrados."""
        with self._lock:
            return {
                name: type(instance).__name__
                for name, instance in self._singletons.items()
            }


# ═══════════════════════════════════════════════════════════════════════════
# 🌐 GLOBAL REGISTRY INSTANCE
# ═══════════════════════════════════════════════════════════════════════════

# Esta es la instancia global que todos importarán
registry = SingletonRegistry()


# ═══════════════════════════════════════════════════════════════════════════
# 🏭 FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _create_ml_pipeline():
    """Factory para ML Pipeline."""
    try:
        from ml_pipeline import get_ml_pipeline
        return get_ml_pipeline()
    except ImportError as e:
        logger.warning(f"ml_pipeline no disponible: {e}")
        return None


def _create_ollama():
    """Factory para Ollama Integration."""
    try:
        from ollama_integration import get_ollama_integration
        return get_ollama_integration()
    except ImportError as e:
        logger.warning(f"ollama_integration no disponible: {e}")
        return None


def _create_internet_search():
    """Factory para Internet Search."""
    try:
        from internet_search import get_internet_search
        return get_internet_search()
    except ImportError as e:
        logger.warning(f"internet_search no disponible: {e}")
        return None


def _create_world_model():
    """Factory para World Model."""
    try:
        from metacortex_sinaptico.world_model import WorldModel
        return WorldModel()
    except ImportError as e:
        logger.warning(f"world_model no disponible: {e}")
        return None


def _create_cognitive_agent():
    """Factory para Cognitive Agent."""
    try:
        from metacortex_sinaptico.core import CognitiveAgent
        return CognitiveAgent()
    except ImportError as e:
        logger.warning(f"cognitive_agent no disponible: {e}")
        return None


def _create_memory_system():
    """Factory para Memory System."""
    try:
        from memory_system import get_memory_system
        return get_memory_system()
    except ImportError as e:
        logger.warning(f"memory_system no disponible: {e}")
        return None


def _create_telegram_bot():
    """Factory para Telegram Bot."""
    try:
        # Import lazy para evitar circular deps
        import os
        from telegram import Bot
        
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            logger.warning("TELEGRAM_BOT_TOKEN no configurado")
            return None
        
        return Bot(token=token)
    except ImportError as e:
        logger.warning(f"telegram no disponible: {e}")
        return None


def _create_autonomous_orchestrator(models_dir: Path = None, max_parallel_tasks: int = 50, auto_start: bool = False):
    """
    Factory para Autonomous Model Orchestrator.
    
    Args:
        models_dir: Directorio de modelos ML
        max_parallel_tasks: Número máximo de tareas paralelas
        auto_start: Si True, inicia los loops automáticamente (solo desde Dashboard Enterprise)
    """
    try:
        from autonomous_model_orchestrator import AutonomousModelOrchestrator
        
        if models_dir is None:
            models_dir = Path(__file__).parent / "ml_models"
        
        orchestrator = AutonomousModelOrchestrator(
            models_dir=models_dir,
            max_parallel_tasks=max_parallel_tasks,
            enable_auto_task_generation=True  # ✅ MODO AUTÓNOMO ACTIVADO - Toma decisiones y ejecuta
        )
        
        logger.info("🏭 Autonomous Orchestrator creado (modo autónomo configurado)")
        logger.info(f"   ℹ️  auto_start={auto_start}")
        
        # SOLO inicializar si auto_start=True (llamado desde Dashboard Enterprise)
        if auto_start:
            orchestrator._discover_models()
            orchestrator._start_execution_threads()  # ✅ Inicia el executor + generator loops automáticamente
            
            logger.info("🚀 Autonomous Orchestrator INICIADO en MODO TOTALMENTE AUTÓNOMO")
            logger.info("   ✅ enable_auto_task_generation: TRUE")
            logger.info("   ✅ Task Executor Loop: ACTIVO")
            logger.info("   ✅ Task Generator Loop: ACTIVO")
        else:
            logger.info("   ⏸️  Orchestrator creado pero NO iniciado (esperando llamada explícita)")
        
        return orchestrator
        
    except ImportError as e:
        logger.warning(f"autonomous_model_orchestrator no disponible: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 📝 REGISTRO DE FACTORIES
# ═══════════════════════════════════════════════════════════════════════════

def register_all_factories():
    """Registra todas las factories en el registry."""
    registry.register_factory("ml_pipeline", _create_ml_pipeline)
    registry.register_factory("ollama", _create_ollama)
    registry.register_factory("internet_search", _create_internet_search)
    registry.register_factory("world_model", _create_world_model)
    registry.register_factory("cognitive_agent", _create_cognitive_agent)
    registry.register_factory("memory_system", _create_memory_system)
    registry.register_factory("telegram_bot", _create_telegram_bot)
    registry.register_factory("autonomous_orchestrator", _create_autonomous_orchestrator)
    
    logger.info("✅ Todas las factories registradas")


# Auto-register al importar
register_all_factories()


# ═══════════════════════════════════════════════════════════════════════════
# 🎯 CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_ml_pipeline():
    """Obtiene el singleton de ML Pipeline."""
    return registry.get("ml_pipeline")


def get_ollama():
    """Obtiene el singleton de Ollama."""
    return registry.get("ollama")


def get_internet_search():
    """Obtiene el singleton de Internet Search."""
    return registry.get("internet_search")


def get_world_model():
    """Obtiene el singleton de World Model."""
    return registry.get("world_model")


def get_cognitive_agent():
    """Obtiene el singleton de Cognitive Agent."""
    return registry.get("cognitive_agent")


def get_memory_system():
    """Obtiene el singleton de Memory System."""
    return registry.get("memory_system")


def get_telegram_bot():
    """Obtiene el singleton de Telegram Bot."""
    return registry.get("telegram_bot")


def get_autonomous_orchestrator(**kwargs):
    """Obtiene el singleton de Autonomous Orchestrator."""
    return registry.get("autonomous_orchestrator", **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# 🧪 TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 79)
    print("🎖️ SINGLETON REGISTRY - TEST")
    print("=" * 79 + "\n")
    
    # Test 1: Registry es singleton
    print("1️⃣  Testing singleton property...")
    r1 = SingletonRegistry()
    r2 = SingletonRegistry()
    assert r1 is r2, "Registry NO es singleton!"
    print("   ✅ Registry es singleton\n")
    
    # Test 2: Registrar y obtener
    print("2️⃣  Testing register/get...")
    registry.set("test_instance", {"data": "test"})
    retrieved = registry.get("test_instance")
    assert retrieved == {"data": "test"}
    print(f"   ✅ Retrieved: {retrieved}\n")
    
    # Test 3: Factory
    print("3️⃣  Testing factory...")
    def test_factory():
        return {"created": "by_factory"}
    
    registry.register_factory("test_factory", test_factory)
    instance = registry.get("test_factory")
    assert instance == {"created": "by_factory"}
    print(f"   ✅ Factory created: {instance}\n")
    
    # Test 4: Listar
    print("4️⃣  Listing all singletons...")
    all_singletons = registry.list_all()
    for name, type_name in all_singletons.items():
        print(f"   • {name}: {type_name}")
    
    print("\n" + "=" * 79)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 79 + "\n")
