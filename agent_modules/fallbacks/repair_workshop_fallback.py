"""
Fallback para el SelfRepairWorkshop.
"""
from typing import Any, Dict, Optional

class _DummyRepairWorkshop:
    def __init__(self, *args: Any, **kwargs: Any): pass
    def repair_code(self, code: str, file_path: str, test_report: Any = None, max_attempts: int = 3) -> Dict[str, Any]:
        return {
            "file_path": file_path,
            "original_issues": 0,
            "repairs_attempted": 0,
            "repairs_successful": 0,
            "final_score": 0.0,
            "actions": [],
            "requires_manual_review": True,
            "execution_time_ms": 0.0
        }

def get_repair_workshop(
    testing_lab: Optional[Any] = None, 
    code_generator: Optional[Any] = None, 
    project_analyzer: Optional[Any] = None, 
    telemetry: Optional[Any] = None
) -> Any:
    """
    Firma idéntica a la real, devuelve una instancia dummy.
    """
    return _DummyRepairWorkshop()
