#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resource Manager Module (Military-Grade)
========================================

Módulo centralizado para la gestión, asignación y limitación de recursos
del sistema (CPU, memoria, hilos) para los diferentes agentes y tareas.
"""

import threading
import logging
import time
from typing import Optional, Dict, Any, Type, List, cast
from types import TracebackType

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
    import psutil
    from unified_logging import get_logger
    from agent_modules.telemetry_system import get_telemetry_system, TelemetrySystem
    imports_were_successful = True
except ImportError:
    imports_were_successful = False
    print("ADVERTENCIA: Módulos críticos no encontrados. ResourceManager funcionará en modo degradado.")

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
        def labels(self, *args: Any, **kwargs: Any) -> '_DummyMetric': return self
        def set(self, *args: Any, **kwargs: Any) -> None: pass
        def inc(self, *args: Any, **kwargs: Any) -> None: pass

    class _DummyTelemetry:
        resource_allocations = _DummyMetric()
        resource_releases = _DummyMetric()
        resource_allocation_failures = _DummyMetric()
        active_resource_locks = _DummyMetric()

    TelemetrySystem = _DummyTelemetry  # type: ignore

    def get_telemetry_system(force_new: bool = False, **kwargs: Any) -> Any:
        return _DummyTelemetry()
    
    class _DummyPsutilProcess:
        def cpu_affinity(self, cpus: Optional[List[int]] = None) -> Optional[List[int]]: return None
        def nice(self, value: Optional[int] = None) -> Optional[int]: return None

    class _DummyPsutil:
        def Process(self) -> _DummyPsutilProcess: return _DummyPsutilProcess()
        def cpu_count(self) -> int: return 1

    psutil = _DummyPsutil()  # type: ignore
# --- End Unified Robust Imports ---


class ResourceLock:
    """
    Un context manager que representa un bloqueo de recursos para una tarea.
    Gestiona la adquisición y liberación de recursos de forma segura.
    """
    _manager: 'ResourceManager'
    _task_id: str
    _requirements: Dict[str, Any]
    _acquired: bool

    def __init__(self, manager: 'ResourceManager', task_id: str, requirements: Dict[str, Any]):
        self._manager = manager
        self._task_id = task_id
        self._requirements = requirements
        self._acquired = False

    def __enter__(self) -> 'ResourceLock':
        self._acquired = self._manager.acquire(self._task_id, self._requirements)
        if not self._acquired:
            raise RuntimeError(f"No se pudieron adquirir los recursos para la tarea '{self._task_id}'")
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        if self._acquired:
            self._manager.release(self._task_id)


class ResourceManager:
    """
    🏗️ Gestor de Recursos (Grado Militar)

    Orquesta la asignación de recursos computacionales para garantizar la
    estabilidad y el rendimiento del sistema.
    """
    logger: logging.Logger
    telemetry: "TelemetrySystem"
    total_cpus: int
    available_cpus: List[int]
    allocations: Dict[str, Dict[str, Any]]
    _lock: threading.Lock

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.telemetry = get_telemetry_system()
        
        self.total_cpus = psutil.cpu_count() or 1
        self.available_cpus = list(range(self.total_cpus))
        
        self.allocations = {}
        self._lock = threading.Lock()
        
        self.logger.info("🏗️ ResourceManager (Grado Militar) inicializado.")
        self.telemetry.active_resource_locks.labels(type='global').set(0)

    def request(self, task_id: str, requirements: Dict[str, Any]) -> ResourceLock:
        """
        Solicita recursos y devuelve un context manager para su uso seguro.
        
        Ejemplo de uso:
        with resource_manager.request("training_job_1", {"cpu_cores": 2, "priority": "high"}) as lock:
            # ... ejecutar tarea ...
        """
        return ResourceLock(self, task_id, requirements)

    def acquire(self, task_id: str, requirements: Dict[str, Any]) -> bool:
        """
        Intenta adquirir los recursos solicitados para una tarea.
        """
        with self._lock:
            self.logger.info(f"Solicitud de adquisición de recursos para '{task_id}': {requirements}")
            
            if task_id in self.allocations:
                self.logger.warning(f"La tarea '{task_id}' ya tiene recursos asignados.")
                return True # Idempotente

            # --- Lógica de asignación de CPU ---
            cpu_cores_needed = requirements.get('cpu_cores', 1)
            if cpu_cores_needed > len(self.available_cpus):
                self.logger.error(f"Fallo al asignar recursos para '{task_id}': No hay suficientes núcleos de CPU disponibles.")
                self.telemetry.resource_allocation_failures.labels(resource='cpu').inc()
                return False
            
            assigned_cpus: List[int] = self.available_cpus[:cpu_cores_needed]
            self.available_cpus = self.available_cpus[cpu_cores_needed:]
            
            # --- Lógica de Prioridad (nice) ---
            priority = requirements.get('priority', 'normal')
            nice_value = self._get_nice_value(priority)

            try:
                p = psutil.Process()
                p.cpu_affinity(assigned_cpus)
                p.nice(nice_value)
                self.logger.info(f"Recursos para '{task_id}' asignados: CPU cores {assigned_cpus}, Prioridad '{priority}' (nice {nice_value}).")
            except Exception as e:
                self.logger.error(f"Error al aplicar configuración de recursos para '{task_id}': {e}", exc_info=True)
                # Revertir asignación
                self.available_cpus.extend(assigned_cpus)
                self.available_cpus.sort()
                self.telemetry.resource_allocation_failures.labels(resource='system_config').inc()
                return False

            self.allocations[task_id] = {
                "cpus": assigned_cpus,
                "priority": priority,
                "nice": nice_value,
                "timestamp": time.time()
            }
            
            self.telemetry.resource_allocations.labels(priority=priority).inc()
            self.telemetry.active_resource_locks.labels(type='global').set(len(self.allocations))
            return True

    def release(self, task_id: str) -> bool:
        """
        Libera los recursos previamente asignados a una tarea.
        """
        with self._lock:
            if task_id not in self.allocations:
                self.logger.warning(f"Intento de liberar recursos para una tarea no registrada: '{task_id}'")
                return False
            
            allocation = self.allocations.pop(task_id)
            assigned_cpus = allocation.get('cpus', [])
            
            self.available_cpus.extend(assigned_cpus)
            self.available_cpus.sort()
            
            self.logger.info(f"Recursos para '{task_id}' liberados: CPU cores {assigned_cpus}.")
            self.telemetry.resource_releases.inc()
            self.telemetry.active_resource_locks.labels(type='global').set(len(self.allocations))
            return True

    def _get_nice_value(self, priority: str) -> int:
        """Traduce un nivel de prioridad a un valor 'nice' del sistema operativo."""
        # Valores más bajos = mayor prioridad
        priority_map = {
            "realtime": -10,
            "high": 0,
            "normal": 10,
            "low": 19
        }
        return priority_map.get(priority.lower(), 10)

# --- Singleton Factory ---
_resource_manager_instance: Optional[ResourceManager] = None

def get_resource_manager(force_new: bool = False, **kwargs: Any) -> ResourceManager:
    """
    Factory para obtener una instancia de ResourceManager.
    """
    global _resource_manager_instance
    if _resource_manager_instance is None or force_new:
        _resource_manager_instance = ResourceManager(**kwargs)
    return _resource_manager_instance

if __name__ == '__main__':
    print("Ejecutando ResourceManager en modo de prueba...")
    
    manager = get_resource_manager(force_new=True)

    if not imports_were_successful:
        print("\nADVERTENCIA: Ejecutando en modo degradado. La asignación de recursos será simulada.")

    print("\n--- Prueba 1: Adquisición y liberación exitosa ---")
    try:
        with manager.request("test_task_1", {"cpu_cores": 1, "priority": "high"}) as lock:
            print("Recursos para 'test_task_1' adquiridos. Trabajando...")
            time.sleep(1)
            print("Trabajo de 'test_task_1' completado.")
        print("Recursos para 'test_task_1' liberados automáticamente.")
    except Exception as e:
        print(f"Error en Prueba 1: {e}")

    print(f"Estado de asignaciones: {manager.allocations}")
    print(f"CPUs disponibles: {manager.available_cpus}")

    print("\n--- Prueba 2: Intento de adquirir más recursos de los disponibles ---")
    try:
        needed = manager.total_cpus + 1
        print(f"Intentando adquirir {needed} núcleos...")
        with manager.request("test_task_2", {"cpu_cores": needed}):
            print("Esto no debería imprimirse.")
    except RuntimeError as e:
        print(f"Capturado error esperado: {e}")

    print(f"Estado de asignaciones: {manager.allocations}")
    print(f"CPUs disponibles: {manager.available_cpus}")
    
    print("\nPrueba finalizada.")
