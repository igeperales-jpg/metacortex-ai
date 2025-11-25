#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cognitive Agent Pool (Military-Grade)
=====================================

Sistema de gestión de alto rendimiento para un pool de agentes cognitivos,
permitiendo la reutilización, precarga y creación bajo demanda.
"""

import threading
import logging
import time
from typing import Any, Dict, Optional, List, cast, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

# --- Configuración de Ruta para Importaciones ---
import sys
from pathlib import Path
try:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except IndexError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
# --- Fin Configuración de Ruta ---

# --- Unified Robust Imports ---
try:
    from unified_logging import get_logger
    from agent_modules.telemetry_system import get_telemetry_system, TelemetrySystem
    from programming_agent import get_programming_agent, MetacortexUniversalProgrammingAgent as ProgrammingAgent
    imports_were_successful = True
except ImportError:
    imports_were_successful = False
    print("ADVERTENCIA: Módulos críticos no encontrados. CognitiveAgentPool funcionará en modo degradado.")

    # --- Fallbacks ---
    def get_logger(name: str = "DefaultLogger") -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    class _DummyMetric:
        def __init__(self, *args: Any, **kwargs: Any): pass
        def labels(self, *args: Any, **kwargs: Any) -> '_DummyMetric': return self
        def set(self, *args: Any, **kwargs: Any) -> None: pass
        def inc(self, *args: Any, **kwargs: Any) -> None: pass
    
    class _DummyTelemetry:
        def __init__(self, *args: Any, **kwargs: Any):
            self.agent_pool_size = _DummyMetric()
            self.agent_pool_active = _DummyMetric()
            self.agent_pool_hits = _DummyMetric()
            self.agent_pool_misses = _DummyMetric()
            self.agent_pool_creation_failures = _DummyMetric()

    TelemetrySystem = _DummyTelemetry # type: ignore
    def get_telemetry_system(*args: Any, **kwargs: Any) -> Any:
        return _DummyTelemetry()

    class _DummyAgent:
        """Fallback dummy agent class."""
        def __init__(self, *args: Any, **kwargs: Any):
            pass
    
    ProgrammingAgent = _DummyAgent # type: ignore
    def get_programming_agent(*args: Any, **kwargs: Any) -> Any:
        """Fallback dummy agent factory."""
        return _DummyAgent(*args, **kwargs)


class CognitiveAgentPool:
    """
    🏊‍♂️ Pool de Agentes Cognitivos (Grado Militar)

    Gestiona un conjunto de agentes reutilizables para minimizar la latencia
    de creación y optimizar el uso de recursos.
    """
    
    if TYPE_CHECKING:
        telemetry: TelemetrySystem
        _pool: Dict[str, List[ProgrammingAgent]]
        _active_agents: Dict[str, List[ProgrammingAgent]]

    def __init__(self, min_agents: int = 2, max_agents: int = 10, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.min_agents = min_agents
        self.max_agents = max_agents
        
        self.telemetry: TelemetrySystem = get_telemetry_system()
        
        self._pool: Dict[str, List[ProgrammingAgent]] = {"default": []}
        self._active_agents: Dict[str, List[ProgrammingAgent]] = {"default": []}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_agents, thread_name_prefix="agent_pool_preloader")

        self.logger.info(f"🏊‍♂️ CognitiveAgentPool (Grado Militar) inicializado. Min: {min_agents}, Max: {max_agents}")

    def preload_async(self, agent_type: str = "default", num_to_preload: Optional[int] = None):
        """
        Precarga agentes en segundo plano para llenar el pool hasta el mínimo.
        """
        num_to_preload = num_to_preload or self.min_agents
        self.logger.info(f"Iniciando precarga asíncrona de {num_to_preload} agentes tipo '{agent_type}'...")
        for _ in range(num_to_preload):
            self._executor.submit(self._create_and_add_agent, agent_type)

    def _create_agent(self, agent_type: str) -> Optional["ProgrammingAgent"]:
        """Crea una nueva instancia de un agente basado en su tipo."""
        try:
            if agent_type == "default" or agent_type == "programming":
                agent = get_programming_agent(force_new=True) 
                return cast(ProgrammingAgent, agent)
            else:
                self.logger.error(f"Tipo de agente desconocido: {agent_type}")
                return None
        except Exception as e:
            self.logger.error(f"Error creando agente tipo '{agent_type}': {e}", exc_info=True)
            self.telemetry.agent_pool_creation_failures.labels(type=agent_type).inc()
            return None

    def _create_and_add_agent(self, agent_type: str):
        """Crea un agente y lo añade al pool si hay espacio."""
        with self._lock:
            current_pool_size = len(self._pool.get(agent_type, []))
            if current_pool_size >= self.max_agents:
                self.logger.warning(f"Pool para '{agent_type}' ha alcanzado su capacidad máxima ({self.max_agents}). No se creará nuevo agente.")
                return

        agent = self._create_agent(agent_type)
        if agent:
            with self._lock:
                self._pool.setdefault(agent_type, []).append(agent)
                new_size = len(self._pool[agent_type])
                self.logger.info(f"Agente tipo '{agent_type}' añadido al pool. Tamaño actual: {new_size}")
                self.telemetry.agent_pool_size.labels(type=agent_type).set(new_size)

    def acquire(self, agent_type: str = "default") -> Optional["ProgrammingAgent"]:
        """
        Adquiere un agente del pool. Si el pool está vacío, crea uno nuevo.
        """
        agent: Optional["ProgrammingAgent"] = None
        with self._lock:
            pool_list = self._pool.get(agent_type)
            if pool_list:
                agent = pool_list.pop(0)
                self._active_agents.setdefault(agent_type, []).append(agent)
                self.logger.info(f"Agente '{agent_type}' adquirido del pool.")
                self.telemetry.agent_pool_hits.labels(type=agent_type).inc()
            else:
                self.logger.info(f"Pool para '{agent_type}' vacío. Creando nuevo agente bajo demanda.")
                self.telemetry.agent_pool_misses.labels(type=agent_type).inc()
                created_agent = self._create_agent(agent_type)
                if created_agent:
                    agent = created_agent
                    self._active_agents.setdefault(agent_type, []).append(agent)

            pool_size = len(self._pool.get(agent_type, []))
            active_size = len(self._active_agents.get(agent_type, []))
            self.telemetry.agent_pool_size.labels(type=agent_type).set(pool_size)
            self.telemetry.agent_pool_active.labels(type=agent_type).set(active_size)
            return agent

    def release(self, agent: "ProgrammingAgent", agent_type: str = "default"):
        """
        Devuelve un agente al pool para su reutilización.
        """
        with self._lock:
            active_list = self._active_agents.get(agent_type, [])
            try:
                if agent in active_list:
                    active_list.remove(agent)
                else:
                    self.logger.warning("Intento de liberar un agente que no estaba activo o ya fue liberado.")
                    return
            except ValueError:
                self.logger.warning("Intento de liberar un agente que no estaba en la lista de activos (ValueError).")
                return

            pool_list = self._pool.setdefault(agent_type, [])
            if len(pool_list) < self.max_agents:
                pool_list.append(agent)
                self.logger.info(f"Agente '{agent_type}' devuelto al pool.")
            else:
                self.logger.info(f"Pool para '{agent_type}' lleno. El agente será descartado.")
                pass
            
            self.telemetry.agent_pool_size.labels(type=agent_type).set(len(pool_list))
            self.telemetry.agent_pool_active.labels(type=agent_type).set(len(active_list))

    def shutdown(self):
        """Limpia el pool y detiene los hilos de trabajo."""
        self.logger.info("Cerrando el pool de agentes...")
        self._executor.shutdown(wait=True)
        with self._lock:
            self._pool.clear()
            self._active_agents.clear()
        self.logger.info("Pool de agentes cerrado.")


# --- Singleton Factory ---
_cognitive_agent_pool_instance: Optional[CognitiveAgentPool] = None
_lock_singleton = threading.Lock()

def get_cognitive_agent_pool(force_new: bool = False, **kwargs: Any) -> CognitiveAgentPool:
    """
    Factory para obtener una instancia de CognitiveAgentPool.
    """
    global _cognitive_agent_pool_instance
    if _cognitive_agent_pool_instance is None or force_new:
        with _lock_singleton:
            if _cognitive_agent_pool_instance is None or force_new:
                _cognitive_agent_pool_instance = CognitiveAgentPool(**kwargs)
    if _cognitive_agent_pool_instance is None:
        raise RuntimeError("Failed to create CognitiveAgentPool instance.")
    return _cognitive_agent_pool_instance


if __name__ == '__main__':
    print("Ejecutando CognitiveAgentPool en modo de prueba...")
    
    pool = get_cognitive_agent_pool(force_new=True, min_agents=2, max_agents=5)
    
    print("\n--- Precargando 2 agentes en background ---")
    pool.preload_async(num_to_preload=2)
    print("Esperando a que la precarga se complete...")
    time.sleep(2) 

    print("\n--- Adquiriendo 3 agentes ---")
    agent1 = pool.acquire()
    agent2 = pool.acquire()
    agent3 = pool.acquire()

    print(f"Agente 1 adquirido: {agent1 is not None}")
    print(f"Agente 2 adquirido: {agent2 is not None}")
    print(f"Agente 3 adquirido: {agent3 is not None}")

    print("\n--- Liberando 2 agentes ---")
    if agent1:
        pool.release(agent1)
    if agent2:
        pool.release(agent2)

    print("\n--- Adquiriendo 1 agente más (debería ser del pool) ---")
    agent4 = pool.acquire()
    print(f"Agente 4 adquirido: {agent4 is not None}")

    print("\n--- Liberando agentes restantes ---")
    if agent3:
        pool.release(agent3)
    if agent4:
        pool.release(agent4)
    
    print("\n--- Intentando liberar un agente ya liberado ---")
    if agent1:
        pool.release(agent1)

    pool.shutdown()
    print("\nPrueba finalizada.")
