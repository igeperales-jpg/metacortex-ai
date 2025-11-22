#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performance Monitor Module (Military-Grade)
===========================================

Módulo dedicado a la monitorización activa y en tiempo real del rendimiento
del sistema, incluyendo CPU, memoria, I/O y latencia de red.
"""

import time
import logging
import threading
from typing import Optional, Any, List, Type

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

try:
    import psutil
    from unified_logging import get_logger
    from agent_modules.telemetry_system import get_telemetry_system, TelemetrySystem
    imports_were_successful = True
except ImportError:
    imports_were_successful = False
    print("ADVERTENCIA: Módulos críticos ('psutil', 'unified_logging', 'telemetry_system') no encontrados. PerformanceMonitor funcionará en modo degradado.")

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

    class _DummyLabel:
        def set(self, *args: Any, **kwargs: Any) -> None: pass

    class _DummyMetric:
        def __init__(self, *args: Any, **kwargs: Any): self._label = _DummyLabel()
        def set(self, *args: Any, **kwargs: Any) -> None: pass
        def labels(self, *args: Any, **kwargs: Any) -> _DummyLabel: return self._label

    class _DummyTelemetry:
        cpu_usage_percent = _DummyMetric()
        cpu_load_average_1m = _DummyMetric()
        memory_usage_percent = _DummyMetric()
        swap_usage_percent = _DummyMetric()
        disk_usage_percent = _DummyMetric()
        network_bytes_sent = _DummyMetric()
        network_bytes_recv = _DummyMetric()

    TelemetrySystem: Type[_DummyTelemetry] = _DummyTelemetry
    
    def get_telemetry_system(**kwargs: Any) -> _DummyTelemetry:
        return _DummyTelemetry()

    class _DummyPsutil:
        def cpu_percent(self, *args: Any, **kwargs: Any) -> float: return 0.0
        def getloadavg(self, *args: Any, **kwargs: Any) -> List[float]: return [0.0, 0.0, 0.0]
        def cpu_count(self, *args: Any, **kwargs: Any) -> int: return 1
        def virtual_memory(self, *args: Any, **kwargs: Any) -> Any:
            class Mem: percent = 0.0
            return Mem()
        def swap_memory(self, *args: Any, **kwargs: Any) -> Any:
            class Swap: percent = 0.0
            return Swap()
        def disk_partitions(self, *args: Any, **kwargs: Any) -> List: return []
        def disk_usage(self, *args: Any, **kwargs: Any) -> Any:
            class Usage: percent = 0.0
            return Usage()
        def net_io_counters(self, *args: Any, **kwargs: Any) -> Any:
            class NetIO: bytes_sent = 0; bytes_recv = 0
            return NetIO()

    psutil = _DummyPsutil()


class PerformanceMonitor:
    """
    ⏱️ Monitor de Rendimiento (Grado Militar)

    Realiza un seguimiento continuo de las métricas vitales del sistema y las
    reporta al sistema de telemetría.
    """

    def __init__(self, interval_seconds: int = 30, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.interval = interval_seconds
        self.is_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self.telemetry: TelemetrySystem = get_telemetry_system()
        
        self.logger.info(f"⏱️ PerformanceMonitor (Grado Militar) inicializado. Intervalo de sondeo: {self.interval}s.")

    def start(self):
        """Inicia el hilo de monitorización de rendimiento."""
        if self.is_running:
            self.logger.warning("El monitor de rendimiento ya está en ejecución.")
            return
        self.is_running = True
        self._monitor_thread = threading.Thread(target=self._run_monitor, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Monitor de rendimiento iniciado.")

    def stop(self):
        """Detiene el hilo de monitorización."""
        self.is_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        self.logger.info("Monitor de rendimiento detenido.")

    def _run_monitor(self):
        """El bucle principal que recolecta y reporta métricas periódicamente."""
        while self.is_running:
            try:
                self.collect_all_metrics()
            except Exception as e:
                self.logger.error(f"Error durante la recolección de métricas de rendimiento: {e}", exc_info=True)
            
            # Esperar para el próximo ciclo, respetando la señal de parada
            for _ in range(self.interval):
                if not self.is_running: break
                time.sleep(1)

    def collect_all_metrics(self):
        """Orquesta la recolección de todas las métricas de rendimiento."""
        self.logger.debug("Recolectando métricas de rendimiento del sistema...")
        self.collect_cpu_metrics()
        self.collect_memory_metrics()
        self.collect_disk_metrics()
        self.collect_network_metrics()

    def collect_cpu_metrics(self):
        """Recolecta y reporta métricas de uso de CPU."""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_load_avg = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
        self.telemetry.cpu_usage_percent.set(cpu_percent)
        self.telemetry.cpu_load_average_1m.set(cpu_load_avg[0])
        self.logger.debug(f"CPU Usage: {cpu_percent}%, Load Avg (1m): {cpu_load_avg[0]:.2f}%")

    def collect_memory_metrics(self):
        """Recolecta y reporta métricas de uso de memoria."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        self.telemetry.memory_usage_percent.set(mem.percent)
        self.telemetry.swap_usage_percent.set(swap.percent)
        self.logger.debug(f"Memory Usage: {mem.percent}%, Swap Usage: {swap.percent}%")

    def collect_disk_metrics(self):
        """Recolecta y reporta métricas de uso de disco y I/O."""
        partitions = self._get_relevant_partitions()
        for part in partitions:
            try:
                usage = psutil.disk_usage(part.mountpoint)
                self.telemetry.disk_usage_percent.labels(partition=part.mountpoint).set(usage.percent)
                self.logger.debug(f"Disk Usage ({part.mountpoint}): {usage.percent}%")
            except Exception as e:
                self.logger.warning(f"No se pudo obtener el uso de disco para {part.mountpoint}: {e}")

    def collect_network_metrics(self):
        """Recolecta y reporta métricas de red."""
        net_io = psutil.net_io_counters()
        self.telemetry.network_bytes_sent.set(net_io.bytes_sent)
        self.telemetry.network_bytes_recv.set(net_io.bytes_recv)
        self.logger.debug(f"Network I/O: Sent={net_io.bytes_sent}, Recv={net_io.bytes_recv}")

    def _get_relevant_partitions(self) -> List[Any]:
        """Obtiene una lista de particiones de disco relevantes para monitorizar."""
        relevant_partitions = []
        try:
            all_partitions = psutil.disk_partitions()
            for p in all_partitions:
                # Filtrar sistemas de archivos "virtuales" o temporales
                if 'rw' in p.opts and p.fstype and not p.mountpoint.startswith('/boot'):
                    relevant_partitions.append(p)
        except Exception as e:
            self.logger.error(f"No se pudieron listar las particiones del disco: {e}")
        return relevant_partitions

# --- Singleton Factory ---
_performance_monitor_instance: Optional[PerformanceMonitor] = None

def get_performance_monitor(force_new: bool = False, **kwargs: Any) -> PerformanceMonitor:
    """
    Factory para obtener una instancia de PerformanceMonitor.
    """
    global _performance_monitor_instance
    if _performance_monitor_instance is None or force_new:
        _performance_monitor_instance = PerformanceMonitor(**kwargs)
        if imports_were_successful:
            _performance_monitor_instance.start()
    return _performance_monitor_instance

if __name__ == '__main__':
    print("Ejecutando PerformanceMonitor en modo de prueba...")
    monitor = get_performance_monitor(force_new=True, interval_seconds=5)

    if not imports_were_successful:
        print("\nADVERTENCIA: Ejecutando en modo degradado. Las métricas serán simuladas.")

    print("El monitor de rendimiento está activo en segundo plano.")
    print("Recolectará métricas cada 5 segundos.")
    print("Presiona Ctrl+C para salir.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDeteniendo el monitor de rendimiento...")
        monitor.stop()
        print("Monitor detenido. Saliendo.")
