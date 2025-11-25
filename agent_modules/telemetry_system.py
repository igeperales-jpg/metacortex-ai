#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Telemetry System Module (Military-Grade)
========================================

Sistema centralizado para la recolección y exposición de métricas de
rendimiento y salud del sistema, utilizando el estándar de Prometheus.
"""

import time
import logging
import threading
from functools import wraps
from typing import Optional, Callable, Any, Type

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
    from prometheus_client import Gauge, Counter, Histogram, start_http_server
    from unified_logging import get_logger
    prometheus_available = True
except ImportError:
    prometheus_available = False
    print("ADVERTENCIA: Módulos críticos ('prometheus-client', 'unified_logging') no encontrados. El sistema de telemetría funcionará en modo degradado.")

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
        def __init__(self, *args: Any, **kwargs: Any):
            pass
        def inc(self, *args: Any, **kwargs: Any) -> None:
            pass
        def set(self, *args: Any, **kwargs: Any) -> None:
            pass
        def observe(self, *args: Any, **kwargs: Any) -> None:
            pass
        def labels(self, *args: Any, **kwargs: Any) -> '_DummyMetric':
            return self

    Gauge: Type[_DummyMetric] = _DummyMetric
    Counter: Type[_DummyMetric] = _DummyMetric
    Histogram: Type[_DummyMetric] = _DummyMetric

    def start_http_server(*args: Any, **kwargs: Any) -> None:
        pass


class TelemetrySystem:
    """
    📡 Sistema de Telemetría (Grado Militar)

    Proporciona una interfaz unificada para registrar y exponer métricas
    clave del sistema para monitoreo y observabilidad.
    """

    def __init__(self, port: int = 8000, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.port = port
        self._server_thread: Optional[threading.Thread] = None

        if not prometheus_available:
            self.logger.error("Prometheus client no está disponible. No se pueden registrar métricas reales.")
        
        # --- Definición de Métricas Clave ---
        # Estas definiciones funcionarán incluso si Prometheus no está disponible,
        # gracias a las clases _DummyMetric.
        self.requests_total = Counter(
            'metacortex_requests_total',
            'Total de peticiones procesadas por el sistema',
            ['module', 'operation']
        )
        self.requests_failed_total = Counter(
            'metacortex_requests_failed_total',
            'Total de peticiones fallidas',
            ['module', 'operation', 'reason']
        )
        self.request_latency = Histogram(
            'metacortex_request_latency_seconds',
            'Latencia de las peticiones',
            ['module', 'operation']
        )
        self.active_agents = Gauge(
            'metacortex_active_agents',
            'Número de agentes activos actualmente',
            ['agent_type']
        )
        self.disk_usage_percent = Gauge(
            'metacortex_disk_usage_percent',
            'Uso de disco en porcentaje',
            ['partition']
        )
        
        # Métricas para el sistema de auto-reparación
        self.repairs_attempted = Counter(
            'metacortex_repairs_attempted_total',
            'Total de intentos de auto-reparación',
            ['type']
        )
        self.repairs_successful = Counter(
            'metacortex_repairs_successful_total',
            'Total de reparaciones exitosas',
            ['type']
        )
        self.diagnosis_errors = Counter(
            'metacortex_diagnosis_errors_total',
            'Total de errores durante la fase de diagnóstico'
        )

        self.logger.info(f"📡 TelemetrySystem (Grado Militar) inicializado. Métrica en puerto {port}.")

    def start_server(self) -> None:
        """Inicia el servidor de métricas de Prometheus en un hilo separado."""
        if not prometheus_available:
            self.logger.warning("No se puede iniciar el servidor de métricas porque Prometheus no está disponible.")
            return

        if self._server_thread and self._server_thread.is_alive():
            self.logger.warning("El servidor de telemetría ya está en ejecución.")
            return

        try:
            # La función start_http_server real se usará aquí si está disponible
            self._server_thread = threading.Thread(
                target=lambda: start_http_server(self.port),
                daemon=True
            )
            self._server_thread.start()
            self.logger.info(f"Servidor de métricas Prometheus iniciado en http://localhost:{self.port}")
        except Exception as e:
            self.logger.critical(f"No se pudo iniciar el servidor de métricas de Prometheus: {e}", exc_info=True)

    def trace_operation(self, module: str, operation: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorador para instrumentar una función o método, registrando
        peticiones, fallos y latencia.
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    self.requests_total.labels(module=module, operation=operation).inc()
                    result = func(*args, **kwargs)
                    latency = time.time() - start_time
                    self.request_latency.labels(module=module, operation=operation).observe(latency)
                    return result
                except Exception as e:
                    reason = type(e).__name__
                    self.requests_failed_total.labels(module=module, operation=operation, reason=reason).inc()
                    latency = time.time() - start_time
                    self.request_latency.labels(module=module, operation=operation).observe(latency)
                    self.logger.error(f"Operación '{operation}' en módulo '{module}' falló: {e}", exc_info=False)
                    raise
            return wrapper
        return decorator

# --- Singleton Factory ---
_telemetry_system_instance: Optional[TelemetrySystem] = None

def get_telemetry_system(force_new: bool = False, **kwargs: Any) -> TelemetrySystem:
    """
    Factory para obtener una instancia de TelemetrySystem.
    """
    global _telemetry_system_instance
    
    if _telemetry_system_instance is None or force_new:
        _telemetry_system_instance = TelemetrySystem(**kwargs)
        if prometheus_available:
            _telemetry_system_instance.start_server()
            
    return _telemetry_system_instance

if __name__ == '__main__':
    print("Ejecutando TelemetrySystem en modo de prueba...")
    
    # No es necesario pasar el logger aquí si get_logger funciona globalmente
    telemetry = get_telemetry_system(force_new=True, port=8001)

    if prometheus_available:
        @telemetry.trace_operation(module="test_module", operation="do_work")
        def do_work(should_fail: bool):
            """Función de prueba para instrumentar."""
            print("Haciendo trabajo...")
            time.sleep(0.1)
            if should_fail:
                raise ValueError("Fallo intencional")
            print("Trabajo completado.")
            return "OK"

        # Simular algunas operaciones
        do_work(should_fail=False)
        do_work(should_fail=False)
        try:
            do_work(should_fail=True)
        except ValueError:
            print("Capturado fallo intencional.")

        telemetry.active_agents.labels(agent_type="programming").set(5)
        telemetry.disk_usage_percent.labels(partition="/").set(42.5)

        print("\nPrueba finalizada. Métricas expuestas en http://localhost:8001")
        print("Presiona Ctrl+C para salir.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nSaliendo.")
    else:
        print("\nPrueba omitida porque 'prometheus-client' no está instalado.")
