#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - ProjectAnalyzer v1.0
==================================

Analizador de calidad de código y estructura de proyectos:
    pass  # TODO: Implementar
- Métricas de calidad
- Detección de code smells
- Análisis de complejidad
- Recomendaciones de mejora

Autor: METACORTEX Evolution Team
Fecha: 2025-01-16
"""

import re
from pathlib import Path
from typing import Dict, List, Any
import logging


class ProjectAnalyzer:
    """
    Analizador avanzado de proyectos y calidad de código
    """

    def __init__(self, workspace_scanner, logger: logging.Logger):
        """
        Inicializa el analizador

        Args:
            workspace_scanner: Escáner de workspace
            logger: Logger para mensajes
        """
        self.workspace_scanner = workspace_scanner
        self.logger = logger

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("project_analyzer", self)
            logger.info("✅ 'project_analyzer' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

    def analyze_code_quality(self, file_path: Path) -> Dict[str, Any]:
        """
        Analiza calidad de un archivo de código

        Args:
            file_path: Ruta al archivo

        Returns:
            Dict con métricas de calidad
        """
        try:
            content = file_path.read_text(encoding="utf-8")

            result = {
                "file": str(file_path),
                "lines": len(content.split("\n")),
                "complexity": self._calculate_complexity(content),
                "comments_ratio": self._calculate_comments_ratio(content),
                "code_smells": self._detect_code_smells(content),
                "maintainability_index": self._calculate_maintainability(content),
                "recommendations": [],
            }

            # Generar recomendaciones
            if result["complexity"] > 20:
                result["recommendations"].append(
                    "⚠️ Alta complejidad ciclomática - considera refactorizar"
                )

            if result["comments_ratio"] < 0.1:
                result["recommendations"].append(
                    "📝 Bajo ratio de comentarios - añade documentación"
                )

            if len(result["code_smells"]) > 5:
                result["recommendations"].append(
                    "🐛 Múltiples code smells detectados - revisar calidad"
                )

            return result

        except Exception as e:
            logger.error(f"Error en project_analyzer.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error analizando {file_path}: {e}")
            return {"error": str(e)}

    def _calculate_complexity(self, content: str) -> int:
        """
        Calcula complejidad ciclomática (aproximación)

        Args:
            content: Contenido del archivo

        Returns:
            Complejidad estimada
        """
        # Contar estructuras de control
        complexity = 1  # Base

        patterns = [
            r"\bif\b",
            r"\belif\b",
            r"\belse\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\band\b",
            r"\bor\b",
            r"\bexcept\b",
            r"\btry\b",
            r"\bcase\b",
            r"\bswitch\b",
        ]

        for pattern in patterns:
            complexity += len(re.findall(pattern, content))

        return complexity

    def _calculate_comments_ratio(self, content: str) -> float:
        """
        Calcula ratio de comentarios vs código

        Args:
            content: Contenido del archivo

        Returns:
            Ratio (0.0 a 1.0)
        """
        lines = content.split("\n")
        total_lines = len(lines)

        if total_lines == 0:
            return 0.0

        # Contar líneas con comentarios
        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith("//")
                or stripped.startswith("/*")
            ):
                comment_lines += 1

        return comment_lines / total_lines

    def _detect_code_smells(self, content: str) -> List[str]:
        """
        Detecta code smells comunes

        Args:
            content: Contenido del archivo

        Returns:
            Lista de code smells encontrados
        """
        smells = []

        # Funciones muy largas (> 50 líneas)
        if content.count("\n") > 300:
            smells.append("Long file (>300 lines)")

        # Muchos parámetros en funciones
        if re.search(r"def \w+\([^)]{80,}\)", content):
            smells.append("Long parameter list")

        # Código duplicado (patterns repetidos)
        lines = content.split("\n")
        seen = set()
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 20:
                if stripped in seen:
                    smells.append("Duplicated code")
                    break
                seen.add(stripped)

        # Magic numbers
        if re.search(r"\b(?!0|1|2|10|100)\d{3,}\b", content):
            smells.append("Magic numbers")

        # Too many imports
        imports = re.findall(r"^import |^from .* import", content, re.MULTILINE)
        if len(imports) > 20:
            smells.append("Too many imports")

        # Nested loops
        if re.search(r"for.*:\s+for", content, re.DOTALL):
            smells.append("Nested loops")

        # Empty exception handlers
        if re.search(r"except.*:\s+pass", content):
            smells.append("Empty exception handler")

        return list(set(smells))  # Eliminar duplicados

    def _calculate_maintainability(self, content: str) -> float:
        """
        Calcula índice de mantenibilidad (0-100)

        Args:
            content: Contenido del archivo

        Returns:
            Índice de mantenibilidad
        """
        # Fórmula simplificada basada en métricas
        lines = len(content.split("\n"))
        complexity = self._calculate_complexity(content)
        comments_ratio = self._calculate_comments_ratio(content)

        # Cálculo (escala 0-100)
        base_score = 100

        # Penalizar complejidad
        base_score -= min(complexity * 2, 40)

        # Penalizar archivos muy largos
        if lines > 500:
            base_score -= 20
        elif lines > 300:
            base_score -= 10

        # Bonificar buenos comentarios
        base_score += comments_ratio * 20

        # Límites
        return max(0, min(100, base_score))

    def analyze_project_structure(self, project_root: Path) -> Dict[str, Any]:
        """
        Analiza la estructura completa del proyecto

        Args:
            project_root: Directorio raíz

        Returns:
            Análisis de estructura
        """
        self.logger.info("🔍 Analizando estructura del proyecto...")

        workspace_info = self.workspace_scanner.scan_workspace()

        result = {
            "project_info": workspace_info,
            "architecture_quality": self._assess_architecture(project_root),
            "testing_coverage": self._estimate_test_coverage(project_root),
            "documentation_quality": self._assess_documentation(project_root),
            "dependencies_health": self._check_dependencies(project_root),
            "overall_score": 0,
        }

        # Calcular score general
        scores = [
            result["architecture_quality"]["score"],
            result["testing_coverage"]["score"],
            result["documentation_quality"]["score"],
            result["dependencies_health"]["score"],
        ]
        result["overall_score"] = sum(scores) / len(scores)

        self.logger.info(
            f"✅ Análisis completado - Score: {result['overall_score']:.1f}/100"
        )
        return result

    def _assess_architecture(self, project_root: Path) -> Dict[str, Any]:
        """Evalúa calidad arquitectónica"""
        score = 50  # Base
        issues = []

        # Verificar estructura de directorios
        expected_dirs = ["tests", "docs", "src"]
        has_structure = sum(1 for d in expected_dirs if (project_root / d).exists())
        score += has_structure * 10

        if has_structure < 2:
            issues.append("Missing standard directories (tests, docs, src)")

        # Verificar archivos de configuración
        config_files = [
            "setup.py",
            "pyproject.toml",
            "requirements.txt",
            "package.json",
        ]
        has_config = sum(1 for f in config_files if (project_root / f).exists())
        score += has_config * 5

        # Verificar README
        if (project_root / "README.md").exists():
            score += 10
        else:
            issues.append("Missing README.md")

        return {
            "score": min(100, score),
            "issues": issues,
            "recommendations": [
                "Organizar código en módulos claros",
                "Separar lógica de negocio de infraestructura",
            ],
        }

    def _estimate_test_coverage(self, project_root: Path) -> Dict[str, Any]:
        """Estima cobertura de tests"""
        score = 0
        issues = []

        # Buscar archivos de test
        test_files = list(project_root.rglob("test_*.py")) + list(
            project_root.rglob("*_test.py")
        )
        code_files = list(project_root.rglob("*.py"))

        # Filtrar archivos en directorios ignorados
        code_files = [
            f
            for f in code_files
            if "test" not in str(f) and "__pycache__" not in str(f)
        ]

        if code_files:
            coverage_ratio = len(test_files) / len(code_files)
            score = min(100, coverage_ratio * 200)  # 50% coverage = 100 points

        if score < 30:
            issues.append("Low test coverage - add more tests")

        return {
            "score": score,
            "test_files": len(test_files),
            "code_files": len(code_files),
            "issues": issues,
        }

    def _assess_documentation(self, project_root: Path) -> Dict[str, Any]:
        """Evalúa calidad de documentación"""
        score = 0
        issues = []

        # Verificar README
        if (project_root / "README.md").exists():
            readme = (project_root / "README.md").read_text()
            score += 30
            if len(readme) > 500:
                score += 20
        else:
            issues.append("Missing README.md")

        # Verificar docs/
        if (project_root / "docs").exists():
            score += 30

        # Verificar CONTRIBUTING
        if (project_root / "CONTRIBUTING.md").exists():
            score += 10

        # Verificar LICENSE
        if (project_root / "LICENSE").exists():
            score += 10

        return {"score": score, "issues": issues}

    def _check_dependencies(self, project_root: Path) -> Dict[str, Any]:
        """Verifica salud de dependencias"""
        score = 70  # Base (asume OK si no hay info)
        issues = []

        # Verificar requirements.txt
        req_file = project_root / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text()
            lines = [
                l.strip()
                for l in content.split("\n")
                if l.strip() and not l.startswith("#")
            ]

            # Verificar pinning de versiones
            pinned = sum(1 for l in lines if "==" in l or ">=" in l)
            if pinned < len(lines) * 0.5:
                issues.append("Many dependencies without version pinning")
                score -= 20

            # Advertir si hay muchas dependencias
            if len(lines) > 50:
                issues.append(f"High number of dependencies ({len(lines)})")
                score -= 10

        return {"score": score, "issues": issues}

    def get_refactoring_suggestions(self, file_path: Path) -> List[str]:
        """
        Genera sugerencias de refactorización

        Args:
            file_path: Archivo a analizar

        Returns:
            Lista de sugerencias
        """
        suggestions = []
        quality = self.analyze_code_quality(file_path)

        if quality.get("complexity", 0) > 15:
            suggestions.append("🔧 Reducir complejidad ciclomática: extraer métodos")

        if quality.get("lines", 0) > 300:
            suggestions.append("📂 Dividir archivo en módulos más pequeños")

        if "Duplicated code" in quality.get("code_smells", []):
            suggestions.append(
                "♻️ Eliminar código duplicado mediante funciones auxiliares"
            )

        if quality.get("comments_ratio", 0) < 0.1:
            suggestions.append("📝 Añadir docstrings y comentarios explicativos")

        if "Magic numbers" in quality.get("code_smells", []):
            suggestions.append("🔢 Reemplazar magic numbers con constantes nombradas")

        return suggestions


def get_project_analyzer(workspace_scanner, logger: logging.Logger) -> ProjectAnalyzer:
    """Factory function"""
    return ProjectAnalyzer(workspace_scanner, logger)
