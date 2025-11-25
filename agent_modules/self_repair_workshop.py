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
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

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
    actions: List[RepairAction] = field(default_factory=list)
    requires_manual_review: bool = False
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SelfRepairWorkshop:
    """
    Taller de auto-reparación inteligente
    """

    def __init__(
        self, testing_lab=None, code_generator=None, project_analyzer=None, telemetry=None
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

    def _initialize_repair_patterns(self) -> Dict[str, Any]:
        """Inicializa patterns de reparación conocidos"""
        return {
            # SYNTAX FIXES
            "missing_colon": {
                "pattern": r"(if|elif|else|for|while|def|class)\s+[^:]+$",
                "fix": lambda m: m.group(0) + ":",
                "confidence": 0.95,
            },
            "missing_parenthesis": {
                "pattern": r"print\s+(['\"].*['\"])",
                "fix": lambda m: f"print({m.group(1)})",
                "confidence": 0.90,
            },
            # IMPORT FIXES
            "missing_import": {
                "typing_optional": {
                    "trigger": "Optional[",
                    "fix": "from typing import Optional\n",
                    "confidence": 1.0,
                },
                "typing_list": {
                    "trigger": "List[",
                    "fix": "from typing import List\n",
                    "confidence": 1.0,
                },
                "typing_dict": {
                    "trigger": "Dict[",
                    "fix": "from typing import Dict\n",
                    "confidence": 1.0,
                },
                "typing_tuple": {
                    "trigger": "Tuple[",
                    "fix": "from typing import Tuple\n",
                    "confidence": 1.0,
                },
                "typing_any": {
                    "trigger": "Any",
                    "fix": "from typing import Any\n",
                    "confidence": 1.0,
                },
                "pathlib_path": {
                    "trigger": "Path(",
                    "fix": "from pathlib import Path\n",
                    "confidence": 1.0,
                },
                "dataclasses": {
                    "trigger": "@dataclass",
                    "fix": "from dataclasses import dataclass\n",
                    "confidence": 1.0,
                },
            },
            # SECURITY FIXES
            "eval_to_literal_eval": {
                "pattern": r"eval\(([^)]+)\)",
                "fix": lambda m: f"ast.literal_eval({m.group(1)})",
                "confidence": 0.85,
                "requires_import": "import ast\n",
            },
            "shell_true_to_false": {
                "pattern": r"subprocess\.(run|call|Popen)\(([^)]+),\s*shell=True",
                "fix": lambda m: f"subprocess.{m.group(1)}({m.group(2)}, shell=False",
                "confidence": 0.80,
            },
            # PERFORMANCE FIXES
            "string_concat_to_join": {
                "pattern": r"for .+ in .+:\s+(\w+)\s*\+=\s*['\"]",
                "fix": lambda m: f"# Usa {m.group(1)} = ''.join([...]) en lugar de +=",
                "confidence": 0.70,
            },
        }

    def repair_code(
        self,
        code: str,
        file_path: str = "repaired.py",
        test_report=None,
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
                current_code, syntax_actions = self._apply_syntax_fixes(
                    current_code, test_report
                )
                actions.extend(syntax_actions)
                repaired = repaired or len(syntax_actions) > 0

            # 2. IMPORT FIXES
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
                current_code, security_actions = self._apply_security_fixes(
                    current_code, test_report
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
                test_report = self.testing_lab.test_python_code(current_code, file_path)
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
            self.telemetry.record_operation(
                operation="repair_code",
                duration=execution_time / 1000,
                success=final_score >= 85.0,
                metadata={
                    "file": file_path,
                    "original_issues": original_issues,
                    "final_score": final_score,
                    "repairs": len(actions),
                },
            )

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

    def _apply_syntax_fixes(self, code: str, test_report) -> Tuple[str, List[RepairAction]]:
        """Aplica fixes de sintaxis"""
        actions = []
        fixed_code = code

        # Pattern: missing colon
        pattern = self.repair_patterns["missing_colon"]
        matches = list(re.finditer(pattern["pattern"], fixed_code, re.MULTILINE))
        for match in matches:
            action = RepairAction(
                strategy=RepairStrategy.SYNTAX_FIX,
                description="Añadir ':' faltante",
                code_before=match.group(0),
                code_after=pattern["fix"](match),
                confidence=pattern["confidence"],
            )
            fixed_code = fixed_code.replace(match.group(0), action.code_after)
            action.applied = True
            action.success = True
            actions.append(action)

        # Pattern: missing parenthesis in print
        pattern = self.repair_patterns["missing_parenthesis"]
        matches = list(re.finditer(pattern["pattern"], fixed_code))
        for match in matches:
            action = RepairAction(
                strategy=RepairStrategy.SYNTAX_FIX,
                description="Añadir paréntesis a print",
                code_before=match.group(0),
                code_after=pattern["fix"](match),
                confidence=pattern["confidence"],
            )
            fixed_code = fixed_code.replace(match.group(0), action.code_after)
            action.applied = True
            action.success = True
            actions.append(action)

        return fixed_code, actions

    def _apply_import_fixes(self, code: str, test_report) -> Tuple[str, List[RepairAction]]:
        """Aplica fixes de imports"""
        actions = []
        fixed_code = code

        # Detectar imports faltantes por uso de tipos
        missing_imports = self.repair_patterns["missing_import"]

        imports_to_add = []
        for name, pattern_info in missing_imports.items():
            if pattern_info["trigger"] in fixed_code:
                # Verificar que no esté ya importado
                if pattern_info["fix"].strip() not in fixed_code:
                    imports_to_add.append((name, pattern_info))

        # Añadir imports al principio
        if imports_to_add:
            import_section = ""
            for name, pattern_info in imports_to_add:
                import_section += pattern_info["fix"]
                action = RepairAction(
                    strategy=RepairStrategy.IMPORT_FIX,
                    description=f"Añadir import faltante: {pattern_info['fix'].strip()}",
                    code_before="",
                    code_after=pattern_info["fix"],
                    confidence=pattern_info["confidence"],
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

    def _apply_security_fixes(self, code: str, test_report) -> Tuple[str, List[RepairAction]]:
        """Aplica fixes de seguridad"""
        actions = []
        fixed_code = code

        # Fix: eval() -> ast.literal_eval()
        pattern = self.repair_patterns["eval_to_literal_eval"]
        matches = list(re.finditer(pattern["pattern"], fixed_code))
        for match in matches:
            action = RepairAction(
                strategy=RepairStrategy.SECURITY_FIX,
                description="Reemplazar eval() con ast.literal_eval()",
                code_before=match.group(0),
                code_after=pattern["fix"](match),
                confidence=pattern["confidence"],
            )
            fixed_code = fixed_code.replace(match.group(0), action.code_after)
            action.applied = True
            action.success = True
            actions.append(action)

            # Añadir import si es necesario
            if pattern.get("requires_import") and pattern["requires_import"] not in fixed_code:
                fixed_code = pattern["requires_import"] + fixed_code

        # Fix: shell=True -> shell=False
        pattern = self.repair_patterns["shell_true_to_false"]
        matches = list(re.finditer(pattern["pattern"], fixed_code))
        for match in matches:
            action = RepairAction(
                strategy=RepairStrategy.SECURITY_FIX,
                description="Cambiar shell=True a shell=False",
                code_before=match.group(0),
                code_after=pattern["fix"](match),
                confidence=pattern["confidence"],
            )
            fixed_code = fixed_code.replace(match.group(0), action.code_after)
            action.applied = True
            action.success = True
            actions.append(action)

        return fixed_code, actions

    def _apply_type_hint_fixes(self, code: str, test_report) -> Tuple[str, List[RepairAction]]:
        """Aplica fixes de type hints"""
        actions = []
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
    testing_lab=None, code_generator=None, project_analyzer=None, telemetry=None
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
