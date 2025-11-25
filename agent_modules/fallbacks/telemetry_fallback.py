"""
Fallback para el TelemetrySystem.
"""
from typing import Any

class _DummyMetric:
    def __init__(self, *args: Any, **kwargs: Any): pass
    def labels(self, *args: Any, **kwargs: Any) -> '_DummyMetric': return self
    def inc(self, *args: Any, **kwargs: Any) -> None: pass
    def set(self, *args: Any, **kwargs: Any) -> None: pass
    def observe(self, *args: Any, **kwargs: Any) -> None: pass

class _DummyTelemetry:
    def __init__(self, *args: Any, **kwargs: Any):
        self.repairs_attempted = _DummyMetric()
        self.repairs_successful = _DummyMetric()
        self.diagnosis_errors = _DummyMetric()

def get_telemetry_system(force_new: bool = False, **kwargs: Any) -> Any:
    """
    Firma idéntica a la real, devuelve una instancia dummy.
    """
    # El logger no es esencial en el fallback, pero mantenemos la firma.
    kwargs.pop('logger', None)
    return _DummyTelemetry(**kwargs)
