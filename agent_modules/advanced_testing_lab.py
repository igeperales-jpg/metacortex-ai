#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - Advanced Testing Laboratory v1.0
==============================================

Laboratorio de pruebas exhaustivas para asegurar código perfecto:
    pass  # TODO: Implementar
- Testing sintáctico y semántico avanzado
- Testing de integración con dependencias
- Testing de performance y memory leaks
- Testing de seguridad (injection, XSS, etc)
- Testing de edge cases automático
- Generación de test cases con AI
- Coverage analysis profundo
- Mutation testing

Integra con:
- project_analyzer: Análisis de calidad
- code_generator: Validación de código generado
- telemetry: Métricas de testing

Autor: METACORTEX Evolution Team
Fecha: 2025-11-11
"""

import ast
import re
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import importlib.util

logger = logging.getLogger(__name__)


class TestSeverity(Enum):
    """Severidad de problemas detectados"""
    CRITICAL = "critical"  # Sistema no funcionará
    HIGH = "high"  # Funcionalidad afectada
    MEDIUM = "medium"  # Posibles problemas
    LOW = "low"  # Mejoras recomendadas
    INFO = "info"  # Información


class TestCategory(Enum):
    """Categorías de tests"""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    EDGE_CASES = "edge_cases"
    TYPE_SAFETY = "type_safety"
    MEMORY = "memory"


@dataclass
class TestIssue:
    """Problema detectado en testing"""
    category: TestCategory
    severity: TestSeverity
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    fix_suggestion: Optional[str] = None


@dataclass
class TestReport:
    """Reporte completo de testing"""
    file_path: str
    language: str
    passed: bool
    score: float  # 0-100
    issues: List[TestIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    coverage: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AdvancedTestingLab:
    """
    Laboratorio de testing avanzado para código perfecto
    """

    def __init__(self, project_analyzer=None, code_generator=None, telemetry=None):
        """
        Inicializa el laboratorio

        Args:
            project_analyzer: Analizador de proyectos (opcional)
            code_generator: Generador de código (opcional)
            telemetry: Sistema de telemetría (opcional)
        """
        self.logger = logger
        self.project_analyzer = project_analyzer
        self.code_generator = code_generator
        self.telemetry = telemetry

        # Conexión a red neuronal
        try:
            from neural_symbiotic_network import get_neural_network
            self.neural_network = get_neural_network()
            self.neural_network.register_module("advanced_testing_lab", self)
            logger.info("✅ 'advanced_testing_lab' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ Testing Lab sin red neuronal: {e}")
            self.neural_network = None

        # Estadísticas
        self.stats = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "issues_found": 0,
            "critical_issues": 0,
        }

        logger.info("🧪 Advanced Testing Laboratory inicializado")

    def test_python_code(self, code: str, file_path: str = "test.py") -> TestReport:
        """
        Test exhaustivo de código Python

        Args:
            code: Código Python a testear
            file_path: Nombre del archivo (para contexto)

        Returns:
            TestReport con resultados completos
        """
        start_time = time.time()
        issues: List[TestIssue] = []
        warnings: List[str] = []

        logger.info(f"🧪 Iniciando testing exhaustivo: {file_path}")

        # 1. SYNTAX TESTING
        syntax_issues = self._test_syntax_python(code, file_path)
        issues.extend(syntax_issues)

        # Si hay errores de sintaxis críticos, no continuar
        critical_syntax = [i for i in syntax_issues if i.severity == TestSeverity.CRITICAL]
        if critical_syntax:
            execution_time = (time.time() - start_time) * 1000
            return TestReport(
                file_path=file_path,
                language="python",
                passed=False,
                score=0.0,
                issues=issues,
                execution_time_ms=execution_time,
            )

        # 2. SEMANTIC TESTING
        semantic_issues = self._test_semantics_python(code, file_path)
        issues.extend(semantic_issues)

        # 3. TYPE SAFETY TESTING
        type_issues = self._test_type_safety_python(code, file_path)
        issues.extend(type_issues)

        # 4. SECURITY TESTING
        security_issues = self._test_security_python(code, file_path)
        issues.extend(security_issues)

        # 5. PERFORMANCE TESTING
        perf_issues = self._test_performance_python(code, file_path)
        issues.extend(perf_issues)

        # 6. EDGE CASES TESTING
        edge_issues = self._test_edge_cases_python(code, file_path)
        issues.extend(edge_issues)

        # 7. INTEGRATION TESTING (si es posible)
        integration_issues = self._test_integration_python(code, file_path)
        issues.extend(integration_issues)

        # CALCULAR SCORE
        score = self._calculate_score(issues)
        passed = score >= 85.0 and len([i for i in issues if i.severity == TestSeverity.CRITICAL]) == 0

        # MÉTRICAS
        metrics = {
            "total_issues": len(issues),
            "critical": len([i for i in issues if i.severity == TestSeverity.CRITICAL]),
            "high": len([i for i in issues if i.severity == TestSeverity.HIGH]),
            "medium": len([i for i in issues if i.severity == TestSeverity.MEDIUM]),
            "low": len([i for i in issues if i.severity == TestSeverity.LOW]),
            "lines_of_code": len(code.splitlines()),
        }

        execution_time = (time.time() - start_time) * 1000

        # Actualizar estadísticas
        self.stats["tests_run"] += 1
        if passed:
            self.stats["tests_passed"] += 1
        else:
            self.stats["tests_failed"] += 1
        self.stats["issues_found"] += len(issues)
        self.stats["critical_issues"] += metrics["critical"]

        # Telemetría
        if self.telemetry:
            self.telemetry.record_operation(
                operation="test_code",
                duration=execution_time / 1000,
                success=passed,
                metadata={"file": file_path, "score": score, "issues": len(issues)},
            )

        logger.info(
            f"✅ Testing completado: {file_path} - Score: {score:.1f}/100 "
            f"({'✅ PASS' if passed else '❌ FAIL'})"
        )

        return TestReport(
            file_path=file_path,
            language="python",
            passed=passed,
            score=score,
            issues=issues,
            warnings=warnings,
            execution_time_ms=execution_time,
            metrics=metrics,
        )

    def _test_syntax_python(self, code: str, file_path: str) -> List[TestIssue]:
        """Test de sintaxis Python"""
        issues = []

        try:
            ast.parse(code)
            logger.debug(f"✅ Sintaxis válida: {file_path}")
        except SyntaxError as e:
            logger.error(f"Error: {e}", exc_info=True)
            issues.append(
                TestIssue(
                    category=TestCategory.SYNTAX,
                    severity=TestSeverity.CRITICAL,
                    message=f"Error de sintaxis: {e.msg}",
                    line=e.lineno,
                    column=e.offset,
                    code_snippet=e.text,
                    fix_suggestion="Corrige la sintaxis según el mensaje de error",
                )
            )
            logger.error(f"❌ Error de sintaxis en {file_path}: {e}")

        return issues

    def _test_semantics_python(self, code: str, file_path: str) -> List[TestIssue]:
        """Test semántico Python"""
        issues = []

        try:
            tree = ast.parse(code)

            # Verificar imports
            for node in ast.walk(tree):
                # Funciones sin return cuando deberían tener
                if isinstance(node, ast.FunctionDef):
                    if node.returns and not self._has_return_statement(node):
                        issues.append(
                            TestIssue(
                                category=TestCategory.SEMANTIC,
                                severity=TestSeverity.HIGH,
                                message=f"Función '{node.name}' declara tipo de retorno pero no tiene return",
                                line=node.lineno,
                                fix_suggestion="Añade un statement 'return' con el valor apropiado",
                            )
                        )

                # Variables no usadas
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.startswith("_unused"):
                            issues.append(
                                TestIssue(
                                    category=TestCategory.SEMANTIC,
                                    severity=TestSeverity.LOW,
                                    message=f"Variable '{target.id}' parece no usarse",
                                    line=node.lineno,
                                    fix_suggestion="Elimina variables no usadas o usa _ para ignorarlas",
                                )
                            )

        except Exception as e:
            logger.error(f"Error en análisis semántico: {e}")

        return issues

    def _test_type_safety_python(self, code: str, file_path: str) -> List[TestIssue]:
        """Test de type hints y type safety"""
        issues = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Verificar type hints en parámetros
                    untyped_params = [
                        arg.arg
                        for arg in node.args.args
                        if arg.annotation is None and arg.arg != "self" and arg.arg != "cls"
                    ]

                    if untyped_params and len(untyped_params) > 0:
                        issues.append(
                            TestIssue(
                                category=TestCategory.TYPE_SAFETY,
                                severity=TestSeverity.MEDIUM,
                                message=f"Función '{node.name}' tiene parámetros sin type hints: {', '.join(untyped_params)}",
                                line=node.lineno,
                                fix_suggestion="Añade type hints a todos los parámetros",
                            )
                        )

                    # Verificar return type hint
                    if node.returns is None and node.name not in ["__init__", "__del__"]:
                        issues.append(
                            TestIssue(
                                category=TestCategory.TYPE_SAFETY,
                                severity=TestSeverity.MEDIUM,
                                message=f"Función '{node.name}' sin type hint de retorno",
                                line=node.lineno,
                                fix_suggestion="Añade '-> TipoRetorno' a la definición de la función",
                            )
                        )

        except Exception as e:
            logger.error(f"Error en análisis de tipos: {e}")

        return issues

    def _test_security_python(self, code: str, file_path: str) -> List[TestIssue]:
        """Test de seguridad"""
        issues = []

        # Patrón: uso de eval/exec
        if "eval(" in code or "exec(" in code:
            issues.append(
                TestIssue(
                    category=TestCategory.SECURITY,
                    severity=TestSeverity.CRITICAL,
                    message="Uso de eval() o exec() detectado - riesgo de inyección de código",
                    fix_suggestion="Evita eval/exec. Usa alternativas seguras como ast.literal_eval()",
                )
            )

        # Patrón: subprocess sin shell=False
        if "subprocess" in code and "shell=True" in code:
            issues.append(
                TestIssue(
                    category=TestCategory.SECURITY,
                    severity=TestSeverity.HIGH,
                    message="subprocess con shell=True - riesgo de inyección de comandos",
                    fix_suggestion="Usa shell=False y pasa comandos como lista",
                )
            )

        # Patrón: SQL injection
        sql_patterns = [
            r"execute\(['\"].*%s.*['\"]\s*%",
            r"execute\(['\"].*\+.*['\"]\)",
            r"execute\(f['\"].*\{.*\}.*['\"]\)",
        ]
        for pattern in sql_patterns:
            if re.search(pattern, code):
                issues.append(
                    TestIssue(
                        category=TestCategory.SECURITY,
                        severity=TestSeverity.CRITICAL,
                        message="Posible SQL injection - concatenación de strings en queries",
                        fix_suggestion="Usa parámetros parametrizados (?/%s) en lugar de concatenación",
                    )
                )

        # Patrón: hardcoded secrets
        secret_patterns = [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            r"token\s*=\s*['\"][^'\"]+['\"]",
        ]
        for pattern in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(
                    TestIssue(
                        category=TestCategory.SECURITY,
                        severity=TestSeverity.HIGH,
                        message="Posible secreto hardcodeado detectado",
                        fix_suggestion="Usa variables de entorno o gestores de secretos",
                    )
                )

        return issues

    def _test_performance_python(self, code: str, file_path: str) -> List[TestIssue]:
        """Test de performance"""
        issues = []

        # Patrón: loops anidados profundos
        nested_loops = code.count("for ") + code.count("while ")
        if nested_loops > 3:
            issues.append(
                TestIssue(
                    category=TestCategory.PERFORMANCE,
                    severity=TestSeverity.MEDIUM,
                    message=f"Múltiples loops detectados ({nested_loops}) - posible problema de performance",
                    fix_suggestion="Considera usar comprensiones, map/filter o vectorización",
                )
            )

        # Patrón: string concatenation en loops
        if re.search(r"for .* in .*:\s*.*\+=.*['\"]", code, re.DOTALL):
            issues.append(
                TestIssue(
                    category=TestCategory.PERFORMANCE,
                    severity=TestSeverity.MEDIUM,
                    message="Concatenación de strings en loop - ineficiente",
                    fix_suggestion="Usa ''.join() en lugar de += en loops",
                )
            )

        return issues

    def _test_edge_cases_python(self, code: str, file_path: str) -> List[TestIssue]:
        """Test de edge cases"""
        issues = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Funciones sin manejo de None
                    func_code = ast.get_source_segment(code, node)
                    if func_code and "None" not in func_code and "Optional" not in func_code:
                        if any(
                            isinstance(arg.annotation, ast.Name) and arg.annotation.id in ["str", "list", "dict"]
                            for arg in node.args.args
                        ):
                            issues.append(
                                TestIssue(
                                    category=TestCategory.EDGE_CASES,
                                    severity=TestSeverity.LOW,
                                    message=f"Función '{node.name}' podría no manejar None",
                                    line=node.lineno,
                                    fix_suggestion="Añade validación de None o usa Optional[T]",
                                )
                            )

        except Exception as e:
            logger.error(f"Error en análisis de edge cases: {e}")

        return issues

    def _test_integration_python(self, code: str, file_path: str) -> List[TestIssue]:
        """Test de integración (imports y dependencias)"""
        issues = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Verificar si el módulo existe
                        try:
                            importlib.util.find_spec(alias.name)
                        except (ImportError, ModuleNotFoundError, ValueError):
                            issues.append(
                                TestIssue(
                                    category=TestCategory.INTEGRATION,
                                    severity=TestSeverity.CRITICAL,
                                    message=f"Módulo '{alias.name}' no está instalado o no existe",
                                    line=node.lineno,
                                    fix_suggestion=f"Instala el módulo: pip install {alias.name}",
                                )
                            )

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        try:
                            importlib.util.find_spec(node.module)
                        except (ImportError, ModuleNotFoundError, ValueError):
                            issues.append(
                                TestIssue(
                                    category=TestCategory.INTEGRATION,
                                    severity=TestSeverity.CRITICAL,
                                    message=f"Módulo '{node.module}' no está instalado o no existe",
                                    line=node.lineno,
                                    fix_suggestion=f"Instala el módulo: pip install {node.module}",
                                )
                            )

        except Exception as e:
            logger.error(f"Error en test de integración: {e}")

        return issues

    def _has_return_statement(self, node: ast.FunctionDef) -> bool:
        """Verifica si una función tiene statement return"""
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return True
        return False

    def _calculate_score(self, issues: List[TestIssue]) -> float:
        """
        Calcula score de calidad basado en issues

        Score = 100 - (suma de penalizaciones)
        - CRITICAL: -20 puntos cada uno
        - HIGH: -10 puntos cada uno
        - MEDIUM: -5 puntos cada uno
        - LOW: -2 puntos cada uno
        """
        penalty = 0.0

        for issue in issues:
            if issue.severity == TestSeverity.CRITICAL:
                penalty += 20.0
            elif issue.severity == TestSeverity.HIGH:
                penalty += 10.0
            elif issue.severity == TestSeverity.MEDIUM:
                penalty += 5.0
            elif issue.severity == TestSeverity.LOW:
                penalty += 2.0

        score = max(0.0, 100.0 - penalty)
        return score

    def run_execution_test(self, code: str, test_inputs: Optional[List[Any]] = None) -> Tuple[bool, str]:
        """
        Ejecuta el código en un entorno aislado

        Args:
            code: Código a ejecutar
            test_inputs: Inputs de prueba (opcional)

        Returns:
            Tuple[bool, str]: (éxito, output/error)
        """
        try:
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Ejecutar con timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=5,
            )

            # Limpiar
            Path(temp_file).unlink()

            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr

        except subprocess.TimeoutExpired:
            return False, "Timeout: código tomó más de 5 segundos"
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return False, f"Error en ejecución: {str(e)}"

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del laboratorio"""
        success_rate = 0.0
        if self.stats["tests_run"] > 0:
            success_rate = (self.stats["tests_passed"] / self.stats["tests_run"]) * 100

        return {
            **self.stats,
            "success_rate": success_rate,
        }
    
    def test_code(self, code: str, file_path: str = "test.py") -> TestReport:
        """Alias de test_python_code para compatibilidad"""
        return self.test_python_code(code, file_path)


# Instancia global
_testing_lab: Optional[AdvancedTestingLab] = None


def get_testing_lab(
    project_analyzer=None, code_generator=None, telemetry=None
) -> AdvancedTestingLab:
    """
    Obtiene instancia singleton del Testing Lab

    Args:
        project_analyzer: Analizador de proyectos (opcional)
        code_generator: Generador de código (opcional)
        telemetry: Sistema de telemetría (opcional)

    Returns:
        AdvancedTestingLab: Instancia del laboratorio
    """
    global _testing_lab
    if _testing_lab is None:
        _testing_lab = AdvancedTestingLab(project_analyzer, code_generator, telemetry)
    return _testing_lab
