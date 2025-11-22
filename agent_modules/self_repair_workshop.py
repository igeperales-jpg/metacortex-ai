#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - Self-Repair Workshop v1.0
=======================================

Taller de auto-reparación inteligente que:
    pass  # TODO: Implementar
- Detecta problemas en código generado
- Aplica fixes automáticos basados en patterns
- Aprende de fixes exitosos
- Reintenta generación con contexto mejorado
- Valida fixes con Testing Lab
- Evoluciona estrategias de reparación

Integra con:
- advanced_testing_lab: Detección de problemas
- code_generator: Regeneración de código
- project_analyzer: Análisis de calidad
- neural_symbiotic_network: Aprendizaje de patterns

Autor: METACORTEX Evolution Team
Fecha: 2025-11-11
"""

import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

if TYPE_CHECKING:
    from agent_modules.advanced_testing_lab import TestReport, AdvancedTestingLab
    from agent_modules.code_generator import CodeGenerator
    from agent_modules.project_analyzer import ProjectAnalyzer
    from agent_modules.telemetry_system import TelemetrySystem
    from neural_symbiotic_network import MetacortexNeuralSymbioticNetworkV2

logger = logging.getLogger(__name__)


class RepairStrategy(Enum):
    """Estrategias de reparación"""
    SYNTAX_FIX = "syntax_fix"  # Corrección de sintaxis
    IMPORT_FIX = "import_fix"  # Corrección de imports
    TYPE_HINT_ADD = "type_hint_add"  # Añadir type hints
    SECURITY_FIX = "security_fix"  # Corrección de seguridad
    PERFORMANCE_FIX = "performance_fix"  # Optimización
    REGENERATE = "regenerate"  # Regenerar completamente
    MANUAL_REVIEW = "manual_review"  # Requiere revisión manual


@dataclass
class RepairAction:
    """Acción de reparación"""
    strategy: RepairStrategy
    description: str
    code_before: str
    code_after: str
    confidence: float  # 0-1
    applied: bool = False
    success: Optional[bool] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RepairReport:
    """Reporte de reparación"""
    file_path: str
    original_issues: int
    repairs_attempted: int
    repairs_successful: int
    final_score: float
    actions: List["RepairAction"] = field(default_factory=list)
    requires_manual_review: bool = False
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class RepairPattern:
    """Define la estructura de un patrón de reparación."""
    pattern: str
    fix: Callable[[re.Match[str]], str]
    confidence: float
    strategy: RepairStrategy
    requires_import: Optional[str] = None


@dataclass
class ImportPattern:
    """Define la estructura de un patrón de reparación de importación."""
    trigger: str
    fix: str
    confidence: float


class SelfRepairWorkshop:
    """
    Taller de auto-reparación inteligente
    """

    def __init__(
        self, 
        testing_lab: Optional["AdvancedTestingLab"] = None, 
        code_generator: Optional["CodeGenerator"] = None, 
        project_analyzer: Optional["ProjectAnalyzer"] = None, 
        telemetry: Optional["TelemetrySystem"] = None
    ):
        """
        Inicializa el taller

        Args:
            testing_lab: Laboratorio de testing
            code_generator: Generador de código
            project_analyzer: Analizador de proyectos
            telemetry: Sistema de telemetría
        """
        self.logger = logger
        self.testing_lab = testing_lab
        self.code_generator = code_generator
        self.project_analyzer = project_analyzer
        self.telemetry = telemetry
        self.neural_network: Optional["MetacortexNeuralSymbioticNetworkV2"] = None

        # Conexión a red neuronal
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("self_repair_workshop", self)
            logger.info("✅ 'self_repair_workshop' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ Self-Repair Workshop sin red neuronal: {e}")
            self.neural_network = None

        # Base de conocimiento de patterns de reparación
        self.repair_patterns = self._initialize_repair_patterns()

        # Estadísticas
        self.stats = {
            "repairs_attempted": 0,
            "repairs_successful": 0,
            "patterns_learned": len(self.repair_patterns),
        }

        logger.info("🔧 Self-Repair Workshop inicializado")

    @staticmethod
    def _fix_missing_colon(m: re.Match[str]) -> str:
        return m.group(0) + ":"

    @staticmethod
    def _fix_missing_parenthesis(m: re.Match[str]) -> str:
        return f"print({m.group(1)})"

    @staticmethod
    def _fix_eval_to_literal_eval(m: re.Match[str]) -> str:
        return f"ast.literal_eval({m.group(1)})"

    @staticmethod
    def _fix_shell_true_to_false(m: re.Match[str]) -> str:
        return f"subprocess.{m.group(1)}({m.group(2)}, shell=False"

    @staticmethod
    def _fix_string_concat_to_join(m: re.Match[str]) -> str:
        return f"# Usa {m.group(1)} = ''.join([...]) en lugar de +="

    def _initialize_repair_patterns(self) -> Dict[str, RepairPattern]:
        """Inicializa patterns de reparación conocidos"""
        return {
            # SYNTAX FIXES
            "missing_colon": RepairPattern(
                pattern=r"(if|elif|else|for|while|def|class)\s+[^:]+$",
                fix=self._fix_missing_colon,
                confidence=0.95,
                strategy=RepairStrategy.SYNTAX_FIX
            ),
            "missing_parenthesis": RepairPattern(
                pattern=r"print\s+(['\"].*['\"])",
                fix=self._fix_missing_parenthesis,
                confidence=0.90,
                strategy=RepairStrategy.SYNTAX_FIX
            ),
            # SECURITY FIXES
            "eval_to_literal_eval": RepairPattern(
                pattern=r"eval\(([^)]+)\)",
                fix=self._fix_eval_to_literal_eval,
                confidence=0.85,
                strategy=RepairStrategy.SECURITY_FIX,
                requires_import="import ast\n"
            ),
            "shell_true_to_false": RepairPattern(
                pattern=r"subprocess\.(run|call|Popen)\(([^)]+),\s*shell=True",
                fix=self._fix_shell_true_to_false,
                confidence=0.80,
                strategy=RepairStrategy.SECURITY_FIX
            ),
            # PERFORMANCE FIXES
            "string_concat_to_join": RepairPattern(
                pattern=r"for .+ in .+:\s+(\w+)\s*\+=\s*['\"]",
                fix=self._fix_string_concat_to_join,
                confidence=0.70,
                strategy=RepairStrategy.PERFORMANCE_FIX
            ),
        }

    def repair_code(
        self,
        code: str,
        file_path: str = "repaired.py",
        test_report: Optional["TestReport"] = None,
        max_attempts: int = 3,
    ) -> RepairReport:
        """
        Repara código automáticamente

        Args:
            code: Código a reparar
            file_path: Nombre del archivo
            test_report: Reporte de testing (opcional)
            max_attempts: Máximo intentos de reparación

        Returns:
            RepairReport con resultados
        """
        start_time = time.time()
        actions: List[RepairAction] = []
        current_code = code
        original_issues = 0

        logger.info(f"🔧 Iniciando auto-reparación: {file_path}")

        # Si no hay test report, ejecutar testing primero
        if test_report is None and self.testing_lab:
            test_report = self.testing_lab.test_python_code(code, file_path)

        if test_report:
            original_issues = len(test_report.issues)
            logger.info(f"   Issues detectados: {original_issues}")

        # INTENTAR REPARACIONES
        for attempt in range(max_attempts):
            logger.info(f"   Intento {attempt + 1}/{max_attempts}")

            # Ejecutar estrategias de reparación
            repaired = False

            # 1. SYNTAX FIXES
            if test_report and any(
                i.category.value == "syntax" for i in test_report.issues
            ):
                current_code, syntax_actions = self._apply_pattern_fixes(
                    current_code, [self.repair_patterns["missing_colon"], self.repair_patterns["missing_parenthesis"]]
                )
                actions.extend(syntax_actions)
                repaired = repaired or len(syntax_actions) > 0

            # 2. IMPORT FIXES (Mantiene su lógica separada por ahora)
            if test_report and any(
                i.category.value == "integration" for i in test_report.issues
            ):
                current_code, import_actions = self._apply_import_fixes(
                    current_code, test_report
                )
                actions.extend(import_actions)
                repaired = repaired or len(import_actions) > 0

            # 3. SECURITY FIXES
            if test_report and any(
                i.category.value == "security" for i in test_report.issues
            ):
                current_code, security_actions = self._apply_pattern_fixes(
                    current_code, [self.repair_patterns["eval_to_literal_eval"], self.repair_patterns["shell_true_to_false"]]
                )
                actions.extend(security_actions)
                repaired = repaired or len(security_actions) > 0

            # 4. TYPE HINT FIXES
            if test_report and any(
                i.category.value == "type_safety" for i in test_report.issues
            ):
                current_code, type_actions = self._apply_type_hint_fixes(
                    current_code, test_report
                )
                actions.extend(type_actions)
                repaired = repaired or len(type_actions) > 0

            # Re-testear si se hicieron cambios
            if repaired and self.testing_lab:
                new_test_report = self.testing_lab.test_python_code(current_code, file_path)
                if new_test_report:
                    test_report = new_test_report
                    logger.info(
                        f"   Re-test: Score {test_report.score:.1f}/100, "
                        f"Issues: {len(test_report.issues)}"
                    )

                    # Si el score es aceptable, terminar
                    if test_report.score >= 85.0:
                        logger.info(f"   ✅ Reparación exitosa - Score: {test_report.score:.1f}/100")
                        break
            else:
                logger.info("   No se aplicaron reparaciones en este intento")
                break

        # CALCULAR RESULTADOS
        execution_time = (time.time() - start_time) * 1000
        final_score = test_report.score if test_report else 0.0
        repairs_successful = len([a for a in actions if a.success])
        requires_manual = final_score < 70.0 or any(
            a.strategy == RepairStrategy.MANUAL_REVIEW for a in actions
        )

        # Actualizar estadísticas
        self.stats["repairs_attempted"] += len(actions)
        self.stats["repairs_successful"] += repairs_successful

        # Telemetría
        if self.telemetry:
            self.telemetry.requests_total.labels(module="self_repair_workshop", operation="repair_code").inc()
            if final_score < 85.0:
                self.telemetry.requests_failed_total.labels(
                    module="self_repair_workshop", 
                    operation="repair_code", 
                    reason="low_score"
                ).inc()
            self.telemetry.request_latency.labels(
                module="self_repair_workshop", 
                operation="repair_code"
            ).observe(execution_time / 1000)


        logger.info(
            f"✅ Auto-reparación completada: {file_path} - "
            f"Score: {final_score:.1f}/100 - "
            f"Repairs: {repairs_successful}/{len(actions)}"
        )

        return RepairReport(
            file_path=file_path,
            original_issues=original_issues,
            repairs_attempted=len(actions),
            repairs_successful=repairs_successful,
            final_score=final_score,
            actions=actions,
            requires_manual_review=requires_manual,
            execution_time_ms=execution_time,
        )

    def _apply_pattern_fixes(self, code: str, patterns: List[RepairPattern]) -> Tuple[str, List[RepairAction]]:
        """Aplica una lista de patrones de reparación basados en regex."""
        actions: List[RepairAction] = []
        fixed_code = code
        
        for pattern in patterns:
            matches = list(re.finditer(pattern.pattern, fixed_code, re.MULTILINE))
            for match in matches:
                code_after = pattern.fix(match)
                action = RepairAction(
                    strategy=pattern.strategy,
                    description=f"Aplicado patrón: {pattern.strategy.value}",
                    code_before=match.group(0),
                    code_after=code_after,
                    confidence=pattern.confidence,
                )
                fixed_code = fixed_code.replace(match.group(0), code_after)
                action.applied = True
                action.success = True
                actions.append(action)

                # Añadir import si es necesario
                if pattern.requires_import and pattern.requires_import not in fixed_code:
                    fixed_code = pattern.requires_import + fixed_code
        
        return fixed_code, actions


    def _apply_import_fixes(self, code: str, test_report: "TestReport") -> Tuple[str, List[RepairAction]]:
        """Aplica fixes de imports"""
        actions: List[RepairAction] = []
        fixed_code = code

        # Detectar imports faltantes por uso de tipos
        missing_imports_patterns: List[ImportPattern] = [
            ImportPattern(trigger="Optional[", fix="from typing import Optional\n", confidence=1.0),
            ImportPattern(trigger="List[", fix="from typing import List\n", confidence=1.0),
            ImportPattern(trigger="Dict[", fix="from typing import Dict\n", confidence=1.0),
            ImportPattern(trigger="Tuple[", fix="from typing import Tuple\n", confidence=1.0),
            ImportPattern(trigger="Any", fix="from typing import Any\n", confidence=1.0),
            ImportPattern(trigger="Path(", fix="from pathlib import Path\n", confidence=1.0),
            ImportPattern(trigger="@dataclass", fix="from dataclasses import dataclass\n", confidence=1.0),
        ]

        imports_to_add: List[ImportPattern] = []
        for pattern_info in missing_imports_patterns:
            if pattern_info.trigger in fixed_code:
                # Verificar que no esté ya importado
                if pattern_info.fix.strip() not in fixed_code:
                    imports_to_add.append(pattern_info)

        # Añadir imports al principio
        if imports_to_add:
            import_section = ""
            for pattern_info in imports_to_add:
                import_section += pattern_info.fix
                action = RepairAction(
                    strategy=RepairStrategy.IMPORT_FIX,
                    description=f"Añadir import faltante: {pattern_info.fix.strip()}",
                    code_before="",
                    code_after=pattern_info.fix,
                    confidence=pattern_info.confidence,
                )
                action.applied = True
                action.success = True
                actions.append(action)

            # Insertar imports después del docstring si existe
            if '"""' in fixed_code:
                parts = fixed_code.split('"""', 2)
                if len(parts) >= 3:
                    fixed_code = parts[0] + '"""' + parts[1] + '"""' + "\n" + import_section + parts[2]
                else:
                    fixed_code = import_section + "\n" + fixed_code
            else:
                fixed_code = import_section + "\n" + fixed_code

        return fixed_code, actions

    def _apply_type_hint_fixes(self, code: str, test_report: "TestReport") -> Tuple[str, List[RepairAction]]:
        """Aplica fixes de type hints"""
        actions: List[RepairAction] = []
        fixed_code = code

        try:
            tree = ast.parse(fixed_code)

            # Encontrar funciones sin type hints
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Por ahora solo reportamos que se necesita manual review
                    if node.returns is None and node.name not in ["__init__", "__del__"]:
                        action = RepairAction(
                            strategy=RepairStrategy.MANUAL_REVIEW,
                            description=f"Función '{node.name}' necesita type hints - requiere revisión manual",
                            code_before="",
                            code_after="",
                            confidence=0.5,
                        )
                        actions.append(action)

        except Exception as e:
            logger.error(f"Error aplicando type hint fixes: {e}")

        return fixed_code, actions

    def learn_from_repair(self, repair_report: RepairReport):
        """
        Aprende de una reparación exitosa

        Args:
            repair_report: Reporte de reparación
        """
        if repair_report.final_score >= 85.0:
            # Analizar qué estrategias funcionaron
            successful_strategies = [
                a.strategy for a in repair_report.actions if a.success
            ]

            logger.info(
                f"📚 Aprendiendo de reparación exitosa: "
                f"Estrategias: {', '.join([s.value for s in successful_strategies])}"
            )

            # Aquí se podría implementar ML para aprender patterns
            # Por ahora solo loggear

            # Si tenemos neural network, compartir conocimiento
            if self.neural_network:
                self.neural_network.share_knowledge(
                    "self_repair_workshop",
                    {
                        "type": "successful_repair",
                        "strategies": [s.value for s in successful_strategies],
                        "score": repair_report.final_score,
                    },
                )

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del taller"""
        success_rate = 0.0
        if self.stats["repairs_attempted"] > 0:
            success_rate = (
                self.stats["repairs_successful"] / self.stats["repairs_attempted"]
            ) * 100

        return {
            **self.stats,
            "success_rate": success_rate,
        }


# Instancia global
_repair_workshop: Optional[SelfRepairWorkshop] = None


def get_repair_workshop(
    testing_lab: Optional["AdvancedTestingLab"] = None, 
    code_generator: Optional["CodeGenerator"] = None, 
    project_analyzer: Optional["ProjectAnalyzer"] = None, 
    telemetry: Optional["TelemetrySystem"] = None
) -> SelfRepairWorkshop:
    """
    Obtiene instancia singleton del Self-Repair Workshop

    Args:
        testing_lab: Laboratorio de testing
        code_generator: Generador de código
        project_analyzer: Analizador de proyectos
        telemetry: Sistema de telemetría

    Returns:
        SelfRepairWorkshop: Instancia del taller
    """
    global _repair_workshop
    if _repair_workshop is None:
        _repair_workshop = SelfRepairWorkshop(
            testing_lab, code_generator, project_analyzer, telemetry
        )
    return _repair_workshop
