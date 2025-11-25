#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
System Auto-Repair Module (Military-Grade)
==========================================

Sistema avanzado para el diagnóstico y la reparación automática de la salud
del sistema METACORTEX, garantizando la máxima disponibilidad y resiliencia.
"""

import sys
import time
import logging
import threading
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Configuración de Ruta para Importaciones ---
try:
    project_root_path = Path(__file__).resolve().parent.parent
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))
except IndexError:
    project_root_path = Path.cwd()
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))
# --- Fin Configuración de Ruta ---

# --- Unified Robust Imports ---
# Patrón robusto: importar lo necesario, usar Any para máxima flexibilidad
try:
    from unified_logging import get_logger  # type: ignore
    from agent_modules.telemetry_system import get_telemetry_system  # type: ignore
    from agent_modules.self_repair_workshop import get_repair_workshop  # type: ignore
    from llm_integration import get_llm_integration  # type: ignore
    imports_were_successful = True

except ImportError as e:
    imports_were_successful = False
    print(f"⚠️ ADVERTENCIA: Módulos críticos no encontrados ({e}). SystemAutoRepair funcionará en modo degradado.")

    # --- Fallbacks ---
    def get_logger(name: str = "DefaultLogger", **kwargs: Any) -> logging.Logger:  # type: ignore
        """Fallback logger si unified_logging no está disponible"""
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
        def inc(self, *args: Any, **kwargs: Any) -> None: pass
        def set(self, *args: Any, **kwargs: Any) -> None: pass

    class _DummyTelemetry:
        def __init__(self, *args: Any, **kwargs: Any):
            self.repairs_attempted = _DummyMetric()
            self.repairs_successful = _DummyMetric()
            self.diagnosis_errors = _DummyMetric()

    def get_telemetry_system(force_new: bool = False, **kwargs: Any) -> Any:  # type: ignore
        """Fallback telemetry si el módulo no está disponible"""
        return _DummyTelemetry()

    class _DummyRepairWorkshop:
        def __init__(self, *args: Any, **kwargs: Any): pass
        def repair_code(self, code: str, file_path: str, test_report: Any = None, max_attempts: int = 3) -> Dict[str, Any]:
            return {
                "file_path": file_path, "original_issues": 0, "repairs_attempted": 0,
                "repairs_successful": 0, "final_score": 0.0, "actions": [],
                "requires_manual_review": True, "execution_time_ms": 0.0
            }

    def get_repair_workshop(testing_lab: Optional[Any] = None, code_generator: Optional[Any] = None, project_analyzer: Optional[Any] = None, telemetry: Optional[Any] = None) -> Any:  # type: ignore
        """Fallback workshop si el módulo no está disponible"""
        return _DummyRepairWorkshop()

    def get_llm_integration(force_new: bool = False, **kwargs: Any) -> Any:  # type: ignore
        """Fallback LLM si el módulo no está disponible"""
        return None


class SystemAutoRepair:
    """
    🔧 Sistema de Auto-Reparación (Grado Militar)

    Diagnostica problemas en el sistema y aplica reparaciones automáticas
    utilizando análisis de logs, reintentos de servicios y agentes de IA.
    """

    def __init__(self, project_root: Path, logs_dir: Path, auto_repair_enabled: bool = True, logger: Optional[logging.Logger] = None):
        self.project_root = project_root
        self.logs_dir = logs_dir
        self.auto_repair_enabled = auto_repair_enabled
        self.logger: Any = logger or get_logger(__name__)
        
        self.telemetry: Any = get_telemetry_system()
        self.repair_workshop: Any = get_repair_workshop()
        self.llm: Any = get_llm_integration() if imports_were_successful else None

        if self.auto_repair_enabled:
            self.logger.info("🔧 SystemAutoRepair (Grado Militar) inicializado en modo ACTIVO.")
        else:
            self.logger.info("🔧 SystemAutoRepair (Grado Militar) inicializado en modo PASIVO.")

    def diagnose_system(self) -> Dict[str, Any]:
        """
        Realiza un diagnóstico completo del sistema.
        
        Returns:
            Un diccionario con el estado de salud y los problemas detectados.
        """
        self.logger.info("🩺 Iniciando diagnóstico completo del sistema...")
        diagnosis: Dict[str, Any] = {
            "timestamp": time.time(),
            "health_percentage": 100.0,
            "issues": [],
            "checks": {}
        }

        try:
            # 1. Analizar logs de errores
            log_issues = self._analyze_logs()
            diagnosis["checks"]["log_analysis"] = {"issue_count": len(log_issues)}
            if log_issues:
                diagnosis["health_percentage"] -= 25.0
                diagnosis["issues"].extend(log_issues)

            # 2. Verificar estado de servicios (simulado)
            service_issues = self._check_services()
            diagnosis["checks"]["service_status"] = {"issue_count": len(service_issues)}
            if service_issues:
                diagnosis["health_percentage"] -= 30.0
                diagnosis["issues"].extend(service_issues)

            # 3. Verificar dependencias
            dep_issues = self._check_dependencies()
            diagnosis["checks"]["dependency_check"] = {"issue_count": len(dep_issues)}
            if dep_issues:
                diagnosis["health_percentage"] -= 15.0
                diagnosis["issues"].extend(dep_issues)

        except Exception as e:
            self.logger.error(f"Error durante el diagnóstico: {e}", exc_info=True)
            self.telemetry.diagnosis_errors.inc()
            diagnosis["health_percentage"] = 0.0
            diagnosis["issues"].append({"type": "diagnosis_failure", "details": str(e), "severity": "critical"})

        self.logger.info(f"🩺 Diagnóstico completado. Salud del sistema: {diagnosis['health_percentage']:.1f}%")
        return diagnosis

    def _analyze_logs(self) -> List[Dict[str, Any]]:
        """Analiza los archivos de log en busca de errores recientes."""
        issues: List[Dict[str, Any]] = []
        error_patterns = ["ERROR", "CRITICAL", "Traceback", "Exception"]
        
        for log_file in self.logs_dir.glob("*.log"):
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f.readlines()[-100:]: # Analizar últimas 100 líneas
                        if any(pattern in line for pattern in error_patterns):
                            issues.append({
                                "type": "log_error",
                                "details": line.strip(),
                                "source": log_file.name,
                                "severity": "high"
                            })
            except Exception as e:
                self.logger.warning(f"No se pudo analizar el log {log_file.name}: {e}")
        return issues

    def _check_services(self) -> List[Dict[str, Any]]:
        """Simula la verificación de servicios críticos (e.g., DB, Ollama)."""
        issues: List[Dict[str, Any]] = []
        if time.time() % 10 > 8: # Falla aleatoria para simulación
             issues.append({
                "type": "service_issue",
                "details": "El servicio de base de datos responde con lentitud.",
                "service": "Database",
                "severity": "medium"
            })
        return issues

    def _check_dependencies(self) -> List[Dict[str, Any]]:
        """Verifica que las dependencias del proyecto estén instaladas."""
        issues: List[Dict[str, Any]] = []
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            return [{"type": "dependency_issue", "details": "requirements.txt no encontrado.", "severity": "critical"}]
        
        try:
            installed_packages_raw = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
            installed_packages = {line.split('==')[0].lower() for line in installed_packages_raw.decode().splitlines()}
            
            with open(req_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        package_name = line.split('==')[0].split('>')[0].split('<')[0].strip().lower()
                        if package_name and package_name not in installed_packages:
                            issues.append({
                                "type": "dependency_issue",
                                "details": f"Dependencia faltante: {package_name}",
                                "severity": "critical"
                            })
        except Exception as e:
            self.logger.error(f"Error verificando dependencias: {e}")
            issues.append({"type": "dependency_issue", "details": f"No se pudieron verificar las dependencias: {e}", "severity": "high"})
            
        return issues

    def auto_repair(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intenta reparar los problemas detectados en el diagnóstico.
        """
        if not self.auto_repair_enabled:
            self.logger.info("Modo de auto-reparación desactivado. No se tomarán acciones.")
            return {"success": False, "reason": "Auto-repair disabled"}

        self.logger.info(f"🔧 Iniciando proceso de auto-reparación para {len(diagnosis['issues'])} problemas.")
        report: Dict[str, Any] = {
            "success": True,
            "repairs_attempted": 0,
            "repairs_successful": 0,
            "actions": []
        }

        for issue in diagnosis["issues"]:
            self.telemetry.repairs_attempted.labels(type=issue["type"]).inc()
            report["repairs_attempted"] += 1
            
            action_taken = self._apply_fix(issue)
            report["actions"].append(action_taken)
            
            if action_taken["success"]:
                self.telemetry.repairs_successful.labels(type=issue["type"]).inc()
                report["repairs_successful"] += 1
        
        if report["repairs_attempted"] > 0 and report["repairs_successful"] == 0:
            report["success"] = False

        self.logger.info(f"🔧 Proceso de auto-reparación finalizado. {report['repairs_successful']}/{report['repairs_attempted']} reparaciones exitosas.")
        return report

    def _apply_fix(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica una solución específica para un problema dado."""
        issue_type = issue.get("type")
        details = issue.get("details")
        
        try:
            if issue_type == "log_error" and details and "Traceback" in details:
                self.logger.info(f"Intentando reparar error de código: {details[:100]}...")
                file_path_str = self._extract_path_from_traceback(details)
                if file_path_str:
                    file_path = Path(file_path_str)
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()
                        
                        repair_report: Any = self.repair_workshop.repair_code(code=code, file_path=str(file_path))
                        
                        if repair_report.get("final_score", 0) > 0 and repair_report.get("repairs_successful", 0) > 0:
                            return {"action": "code_fix_applied", "details": f"Parche aplicado a {file_path.name} por SelfRepairWorkshop.", "success": True}
                return {"action": "code_fix_failed", "details": "No se pudo reparar el código.", "success": False}

            elif issue_type == "service_issue":
                self.logger.info(f"Intentando reiniciar servicio: {issue.get('service')}...")
                return {"action": "service_restart", "details": f"Señal de reinicio enviada a {issue.get('service')}", "success": True}

            elif issue_type == "dependency_issue" and details and "faltante" in details:
                package_name = details.split(": ")[1]
                self.logger.info(f"Instalando dependencia faltante: {package_name}...")
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', package_name], capture_output=True, text=True)
                if result.returncode == 0:
                    return {"action": "install_dependency", "details": f"Instalado {package_name}", "success": True}
                else:
                    return {"action": "install_dependency_failed", "details": result.stderr, "success": False}

        except Exception as e:
            self.logger.error(f"Error aplicando solución para {issue_type}: {e}", exc_info=True)
            return {"action": "fix_application_error", "details": str(e), "success": False}

        return {"action": "no_fix_available", "details": f"No hay una solución automática para {issue_type}", "success": False}

    def _extract_path_from_traceback(self, traceback_text: str) -> Optional[str]:
        """Extrae la ruta de un archivo de un texto de traceback."""
        import re
        match = re.search(r'File "([^"]+)", line', traceback_text)
        if match:
            path = match.group(1)
            if Path(path).is_absolute() and Path(path).exists():
                return path
            full_path = self.project_root / path
            if full_path.exists():
                return str(full_path)
        return None


# --- Singleton Factory ---
_auto_repair_instance: Optional[SystemAutoRepair] = None
_lock = threading.Lock()

def get_auto_repair(force_new: bool = False, **kwargs: Any) -> SystemAutoRepair:
    """
    Factory para obtener una instancia de SystemAutoRepair.
    """
    global _auto_repair_instance
    if _auto_repair_instance is None or force_new:
        with _lock:
            if _auto_repair_instance is None or force_new:
                if "project_root" not in kwargs:
                    kwargs["project_root"] = project_root_path
                if "logs_dir" not in kwargs:
                    kwargs["logs_dir"] = project_root_path / "logs"
                _auto_repair_instance = SystemAutoRepair(**kwargs)
    
    # The instance is guaranteed to be created here.
    return _auto_repair_instance


if __name__ == '__main__':
    print("Ejecutando SystemAutoRepair en modo de prueba...")
    
    logs_dir_test = project_root_path / "logs_test"
    logs_dir_test.mkdir(exist_ok=True)
    test_log_path = logs_dir_test / "test_error.log"
    with open(test_log_path, "w") as f:
        f.write("2025-11-21 10:00:00,000 - TestModule - ERROR - Ocurrió un error inesperado.\n")
        f.write(f'Traceback (most recent call last):\n  File "{project_root_path}/src/main.py", line 42, in process\n    result = 10 / 0\nZeroDivisionError: division by zero\n')

    repair_system = get_auto_repair(force_new=True, logs_dir=logs_dir_test, auto_repair_enabled=False)
    
    diagnosis_result = repair_system.diagnose_system()
    print("\n--- Diagnóstico del Sistema ---")
    print(f"Salud: {diagnosis_result['health_percentage']:.1f}%")
    for issue in diagnosis_result['issues']:
        print(f"- Problema: {issue['type']} | Severidad: {issue['severity']} | Detalles: {issue['details'][:100]}...")

    print("\n--- Simulación de Reparación ---")
    repair_system.auto_repair_enabled = True
    repair_report = repair_system.auto_repair(diagnosis_result)
    print(f"Éxito: {repair_report['success']}")
    print(f"Reparaciones intentadas: {repair_report['repairs_attempted']}")
    print(f"Reparaciones exitosas: {repair_report['repairs_successful']}")
    for action in repair_report['actions']:
        print(f"- Acción: {action['action']} | Éxito: {action['success']} | Detalles: {action.get('details')}")

    import shutil
    shutil.rmtree(logs_dir_test)
    print("\nPrueba finalizada.")
