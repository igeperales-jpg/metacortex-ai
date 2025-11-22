#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - WorkspaceScanner v1.0
===================================

Escáner inteligente de workspace que detecta:
    pass  # TODO: Implementar
- Lenguajes de programación
- Frameworks y librerías
- Estructura del proyecto
- Dependencias
- Patrones arquitectónicos

Autor: METACORTEX Evolution Team
Fecha: 2025-10-16
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import logging


class WorkspaceScanner:
    """
    Escáner inteligente de workspace con análisis profundo
    """

    # Directorios a ignorar
    IGNORE_DIRS = {
        "node_modules",
        ".git",
        "__pycache__",
        "venv",
        "env",
        ".venv",
        "build",
        "dist",
        "target",
        ".idea",
        ".vscode",
        ".pytest_cache",
        "coverage",
        ".mypy_cache",
        ".tox",
        "eggs",
        "*.egg-info",
    }

    # Extensiones de código por lenguaje
    LANGUAGE_EXTENSIONS = {
        "python": {".py", ".pyw", ".pyx"},
        "javascript": {".js", ".jsx", ".mjs"},
        "typescript": {".ts", ".tsx"},
        "java": {".java"},
        "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".h"},
        "c": {".c", ".h"},
        "rust": {".rs"},
        "go": {".go"},
        "ruby": {".rb"},
        "php": {".php"},
        "csharp": {".cs"},
        "swift": {".swift"},
        "kotlin": {".kt"},
        "dart": {".dart"},
    }

    def __init__(self, project_root: Path, logger: logging.Logger):
        """
        Inicializa el escáner

        Args:
            project_root: Directorio raíz del proyecto
            logger: Logger para mensajes
        """
        self.project_root = project_root
        self.logger = logger

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("workspace_scanner", self)
            logger.info("✅ 'workspace_scanner' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

    def scan_workspace(self) -> Dict[str, Any]:
        """
        Escanea el workspace completo

        Returns:
            Dict con información completa del workspace
        """
        self.logger.info("🔍 Iniciando escaneo de workspace...")

        result = {
            "project_type": self._detect_project_type(),
            "languages_used": self._detect_languages(),
            "frameworks_detected": self._detect_frameworks(),
            "build_systems": self._detect_build_systems(),
            "testing_frameworks": self._detect_testing_frameworks(),
            "documentation": self._detect_documentation(),
            "dependencies": self._analyze_dependencies(),
            "architecture_patterns": self._detect_architecture_patterns(),
            "file_count": self._count_files(),
            "line_count": self._count_lines(),
            "project_size": self._calculate_size(),
        }

        self.logger.info(f"✅ Escaneo completado: {result['project_type']}")
        return result

    def _detect_project_type(self) -> str:
        """Detecta el tipo de proyecto"""
        indicators = {
            "web_app": ["package.json", "requirements.txt", "templates/", "static/"],
            "api_service": ["main.py", "api.py", "routes/", "endpoints/", "Dockerfile"],
            "ml_project": [
                "model.py",
                "train.py",
                "data/",
                "notebooks/",
                "requirements.txt",
            ],
            "desktop_app": ["setup.py", "gui/", "windows/"],
            "library": ["__init__.py", "lib/", "src/", "setup.py"],
            "microservice": ["docker-compose.yml", "Dockerfile", "service/"],
            "cli_tool": ["cli.py", "commands/", "bin/", "scripts/"],
        }

        scores = {}
        for project_type, files in indicators.items():
            score = sum(1 for f in files if list(self.project_root.glob(f"**/{f}")))
            if score > 0:
                scores[project_type] = score

        if scores:
            return max(scores, key=scores.get)
        return "unknown"

    def _detect_languages(self) -> List[str]:
        """Detecta lenguajes de programación usados"""
        languages = set()

        for lang, extensions in self.LANGUAGE_EXTENSIONS.items():
            for ext in extensions:
                if self._has_files_with_extension(ext):
                    languages.add(lang)

        return sorted(list(languages))

    def _detect_frameworks(self) -> List[str]:
        """Detecta frameworks usados"""
        frameworks = []

        # Python frameworks
        if (self.project_root / "requirements.txt").exists():
            content = (self.project_root / "requirements.txt").read_text()
            if "flask" in content.lower():
                frameworks.append("Flask")
            if "django" in content.lower():
                frameworks.append("Django")
            if "fastapi" in content.lower():
                frameworks.append("FastAPI")
            if "tensorflow" in content.lower():
                frameworks.append("TensorFlow")
            if "pytorch" in content.lower():
                frameworks.append("PyTorch")

        # JavaScript frameworks
        if (self.project_root / "package.json").exists():
            try:
                pkg = json.loads((self.project_root / "package.json").read_text())
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

                if "react" in deps:
                    frameworks.append("React")
                if "vue" in deps:
                    frameworks.append("Vue")
                if "angular" in deps or "@angular/core" in deps:
                    frameworks.append("Angular")
                if "express" in deps:
                    frameworks.append("Express")
                if "next" in deps:
                    frameworks.append("Next.js")
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                pass

        return frameworks

    def _detect_build_systems(self) -> List[str]:
        """Detecta sistemas de build"""
        build_systems = []

        if (self.project_root / "setup.py").exists():
            build_systems.append("setuptools")
        if (self.project_root / "pyproject.toml").exists():
            build_systems.append("poetry/pip")
        if (self.project_root / "Makefile").exists():
            build_systems.append("make")
        if (self.project_root / "CMakeLists.txt").exists():
            build_systems.append("cmake")
        if (self.project_root / "build.gradle").exists():
            build_systems.append("gradle")
        if (self.project_root / "pom.xml").exists():
            build_systems.append("maven")
        if (self.project_root / "Cargo.toml").exists():
            build_systems.append("cargo")
        if (self.project_root / "package.json").exists():
            build_systems.append("npm/yarn")

        return build_systems

    def _detect_testing_frameworks(self) -> List[str]:
        """Detecta frameworks de testing"""
        testing = []

        # Python testing
        if self._has_files_matching("**/test_*.py") or self._has_files_matching(
            "**/*_test.py"
        ):
            testing.append("pytest/unittest")

        # JavaScript testing
        if (self.project_root / "package.json").exists():
            try:
                pkg = json.loads((self.project_root / "package.json").read_text())
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

                if "jest" in deps:
                    testing.append("Jest")
                if "mocha" in deps:
                    testing.append("Mocha")
                if "chai" in deps:
                    testing.append("Chai")
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                pass

        return testing

    def _detect_documentation(self) -> List[str]:
        """Detecta documentación"""
        docs = []

        if (self.project_root / "README.md").exists():
            docs.append("README.md")
        if (self.project_root / "docs").exists():
            docs.append("docs/")
        if (self.project_root / "CONTRIBUTING.md").exists():
            docs.append("CONTRIBUTING.md")
        if (self.project_root / "LICENSE").exists():
            docs.append("LICENSE")

        return docs

    def _analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analiza dependencias del proyecto"""
        dependencies = {}

        # Python dependencies
        if (self.project_root / "requirements.txt").exists():
            content = (self.project_root / "requirements.txt").read_text()
            deps = [
                line.split("==")[0].strip()
                for line in content.split("\n")
                if line.strip() and not line.startswith("#")
            ]
            dependencies["python"] = deps[:20]  # Primeras 20

        # JavaScript dependencies
        if (self.project_root / "package.json").exists():
            try:
                pkg = json.loads((self.project_root / "package.json").read_text())
                deps = list(pkg.get("dependencies", {}).keys())
                dependencies["javascript"] = deps[:20]
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                pass

        return dependencies

    def _detect_architecture_patterns(self) -> List[str]:
        """Detecta patrones arquitectónicos"""
        patterns = []

        # MVC
        if self._has_directories(["models", "views", "controllers"]):
            patterns.append("MVC")

        # Microservices
        if (self.project_root / "docker-compose.yml").exists():
            patterns.append("Microservices")

        # Layered
        if self._has_directories(["domain", "application", "infrastructure"]):
            patterns.append("Layered/Clean Architecture")

        # API REST
        if self._has_files_matching("**/api.py") or self._has_files_matching(
            "**/routes.py"
        ):
            patterns.append("REST API")

        return patterns

    def _count_files(self) -> int:
        """Cuenta archivos de código"""
        count = 0
        for ext_set in self.LANGUAGE_EXTENSIONS.values():
            for ext in ext_set:
                count += len(list(self.project_root.rglob(f"*{ext}")))
        return count

    def _count_lines(self) -> int:
        """Cuenta líneas de código"""
        total_lines = 0
        for ext_set in self.LANGUAGE_EXTENSIONS.values():
            for ext in ext_set:
                for file_path in self.project_root.rglob(f"*{ext}"):
                    if self._should_ignore(file_path):
                        continue
                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            total_lines += len(f.readlines())
                    except Exception as e:
                        logger.error(f"Error: {e}", exc_info=True)
                        pass
        return total_lines

    def _calculate_size(self) -> str:
        """Calcula tamaño del proyecto"""
        total_bytes = 0
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not self._should_ignore(file_path):
                try:
                    total_bytes += file_path.stat().st_size
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    pass

        # Convertir a MB
        mb = total_bytes / (1024 * 1024)
        return f"{mb:.2f} MB"

    def _has_files_with_extension(self, ext: str) -> bool:
        """Verifica si existen archivos con extensión específica"""
        try:
            return any(self.project_root.rglob(f"*{ext}"))
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return False

    def _has_files_matching(self, pattern: str) -> bool:
        """Verifica si existen archivos que coincidan con patrón"""
        try:
            return any(self.project_root.glob(pattern))
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return False

    def _has_directories(self, dirs: List[str]) -> bool:
        """Verifica si existen directorios"""
        return all((self.project_root / d).exists() for d in dirs)

    def _should_ignore(self, path: Path) -> bool:
        """Determina si un path debe ser ignorado"""
        parts = set(path.parts)
        return bool(parts & self.IGNORE_DIRS)


def get_workspace_scanner(
    project_root: Path, logger: logging.Logger
) -> WorkspaceScanner:
    """Factory function"""
    return WorkspaceScanner(project_root, logger)
