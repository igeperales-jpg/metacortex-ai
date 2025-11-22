"""
Sistema de Telemetría para METACORTEX
Proporciona funciones de monitoreo y trazabilidad para todos los agentes
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


class Telemetry:
    """Sistema de telemetría para monitorear operaciones de agentes"""

    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "operations": [],
            "errors": [],
            "performance": {},
            "start_time": datetime.now(),
        }
        self.enabled = True

    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool,
        metadata: Optional[Dict] = None,
    ):
        """Registra una operación"""
        if not self.enabled:
            return

        record = {
            "operation": operation,
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.metrics["operations"].append(record)

        # Mantener solo las últimas 1000 operaciones
        if len(self.metrics["operations"]) > 1000:
            self.metrics["operations"] = self.metrics["operations"][-1000:]

    def record_error(self, error: str, context: Optional[Dict] = None):
        """Registra un error"""
        if not self.enabled:
            return

        error_record = {
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
        }
        self.metrics["errors"].append(error_record)

        # Mantener solo los últimos 100 errores
        if len(self.metrics["errors"]) > 100:
            self.metrics["errors"] = self.metrics["errors"][-100:]

    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene todas las métricas"""
        return {
            **self.metrics,
            "uptime": (datetime.now() - self.metrics["start_time"]).total_seconds(),
            "total_operations": len(self.metrics["operations"]),
            "total_errors": len(self.metrics["errors"]),
        }

    def clear_metrics(self):
        """Limpia todas las métricas"""
        self.metrics = {
            "operations": [],
            "errors": [],
            "performance": {},
            "start_time": datetime.now(),
        }


# Instancia global de telemetría
_global_telemetry = Telemetry()


def get_telemetry() -> Telemetry:
    """
    Obtiene la instancia global de telemetría

    Returns:
        Telemetry: Instancia global de telemetría
    """
    return _global_telemetry


def trace_operation(operation_name: Optional[str] = None):
    """
    Decorador para trazar operaciones automáticamente

    Args:
        operation_name: Nombre de la operación (usa el nombre de la función si no se proporciona)

    Returns:
        Callable: Función decorada
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            start_time = time.time()
            success = False
            error_msg = None

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                error_msg = str(e)
                telemetry.record_error(
                    error=f"Error in {op_name}: {error_msg}",
                    context={"function": func.__name__, "args": str(args)[:100]},
                )
                raise
            finally:
                duration = time.time() - start_time
                telemetry.record_operation(
                    operation=op_name,
                    duration=duration,
                    success=success,
                    metadata={"error": error_msg} if error_msg else {},
                )

        return wrapper

    return decorator


# Funciones de utilidad adicionales
def log_performance(operation: str, duration: float):
    """Registra métricas de rendimiento"""
    telemetry = get_telemetry()
    if operation not in telemetry.metrics["performance"]:
        telemetry.metrics["performance"][operation] = []

    telemetry.metrics["performance"][operation].append(
        {"duration": duration, "timestamp": datetime.now().isoformat()}
    )

    # Mantener solo las últimas 100 mediciones por operación
    if len(telemetry.metrics["performance"][operation]) > 100:
        telemetry.metrics["performance"][operation] = telemetry.metrics["performance"][
            operation
        ][-100:]


def get_operation_stats(operation: str) -> Optional[Dict[str, float]]:
    """Obtiene estadísticas de una operación específica"""
    telemetry = get_telemetry()

    if operation not in telemetry.metrics["performance"]:
        return None

    durations = [m["duration"] for m in telemetry.metrics["performance"][operation]]

    if not durations:
        return None

    return {
        "count": len(durations),
        "avg": sum(durations) / len(durations),
        "min": min(durations),
        "max": max(durations),
        "total": sum(durations),
    }
