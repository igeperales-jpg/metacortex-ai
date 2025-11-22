#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Exponential Capability Engine
=================================

Motor de descubrimiento de capacidades exponenciales para Metacortex.
Analiza el código fuente en tiempo real para descubrir, catalogar y aprender
nuevas capacidades y agentes, permitiendo al sistema evolucionar de forma autónoma.

Características Avanzadas:
- Análisis AST: Utiliza el Árbol de Sintaxis Abstracta (AST) para analizar
  código Python de forma segura, sin ejecutarlo.
- Descubrimiento Recursivo: Escanea directorios para encontrar nuevos módulos
  y clases de agentes.
- Aprendizaje de Keywords: Extrae keywords de docstrings para entender la
  semántica de las capacidades.
- Cache Inteligente: Cachea los resultados del análisis para un rendimiento
  exponencialmente rápido en escaneos sucesivos.
- Singleton Robusto: Garantiza una única instancia del motor para todo el sistema.
"""
from __future__ import annotations

import ast
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import re
from dataclasses import dataclass, field

# Configuración del logging
logger = logging.getLogger(__name__)

@dataclass
class DiscoveredCapability:
    """Representa una capacidad (método) descubierta en un agente."""
    name: str
    docstring: str
    args: List[str]
    keywords: Set[str] = field(default_factory=set)

@dataclass
class DiscoveredAgent:
    """Representa un agente (clase) descubierto en el código fuente."""
    name: str
    module_path: str
    class_name: str
    docstring: str
    capabilities: List[DiscoveredCapability] = field(default_factory=list)

class ExponentialCapabilityEngine:
    """
    Motor que escanea, analiza y cataloga las capacidades del ecosistema Metacortex.
    """
    _instance: Optional[ExponentialCapabilityEngine] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> ExponentialCapabilityEngine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, project_root: Path):
        if self._initialized:
            return
            
        self.project_root: Path = project_root
        self.discovered_agents: Dict[str, DiscoveredAgent] = {}
        self.learned_keywords: Set[str] = set()
        self.file_cache: Dict[Path, float] = {}  # Cache de archivos y sus mtime
        self.stats: Dict[str, Any] = {
            "scans": 0,
            "modules_analyzed": 0,
            "agents_discovered": 0,
            "capabilities_found": 0,
            "keywords_learned": 0,
            "cache_hits": 0,
        }
        self._initialized = True
        logger.info("🚀 Exponential Capability Engine inicializado.")

    def discover_agents_in_directory(self, directory: Path, recursive: bool = True) -> List[DiscoveredAgent]:
        """
        Escanea un directorio en busca de agentes y sus capacidades.

        Args:
            directory: El directorio a escanear.
            recursive: Si es True, escanea subdirectorios.

        Returns:
            Una lista de los agentes descubiertos en este escaneo.
        """
        self.stats["scans"] += 1
        newly_discovered: List[DiscoveredAgent] = []
        
        glob_pattern = "**/*.py" if recursive else "*.py"
        
        for py_file in directory.glob(glob_pattern):
            if "test" in str(py_file) or "venv" in str(py_file) or ".venv" in str(py_file):
                continue

            # Usar cache para evitar re-analizar archivos sin cambios
            try:
                last_modified = py_file.stat().st_mtime
                if py_file in self.file_cache and self.file_cache[py_file] == last_modified:
                    self.stats["cache_hits"] += 1
                    continue
                
                self.file_cache[py_file] = last_modified
                self.stats["modules_analyzed"] += 1

                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Heurística para identificar un "Agente"
                            if "Agent" in node.name or "Service" in node.name or "Manager" in node.name or "Engine" in node.name:
                                if node.name not in self.discovered_agents:
                                    agent = self._analyze_class_node(node, str(py_file))
                                    self.discovered_agents[agent.name] = agent
                                    newly_discovered.append(agent)
                                    self.stats["agents_discovered"] += 1
                                    self.stats["capabilities_found"] += len(agent.capabilities)
            except FileNotFoundError:
                logger.warning(f"⚠️ Archivo no encontrado durante el escaneo: {py_file}")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo analizar el archivo {py_file}: {e}")

        self.stats["keywords_learned"] = len(self.learned_keywords)
        return newly_discovered

    def _analyze_class_node(self, node: ast.ClassDef, file_path: str) -> DiscoveredAgent:
        """Analiza un nodo de clase AST para extraer información del agente."""
        docstring = ast.get_docstring(node) or ""
        
        agent = DiscoveredAgent(
            name=node.name,
            module_path=file_path,
            class_name=node.name,
            docstring=docstring
        )

        for body_item in node.body:
            if isinstance(body_item, ast.FunctionDef) and not body_item.name.startswith("_"):
                capability = self._analyze_function_node(body_item)
                agent.capabilities.append(capability)
                self.learned_keywords.update(capability.keywords)
        
        return agent

    def _analyze_function_node(self, node: ast.FunctionDef) -> DiscoveredCapability:
        """Analiza un nodo de función AST para extraer información de la capacidad."""
        docstring = ast.get_docstring(node) or ""
        args = [arg.arg for arg in node.args.args if arg.arg != "self"]
        
        # Extraer keywords del docstring (simple, se puede mejorar con NLP)
        keywords = set(re.findall(r'\b[a-zA-Z_]{4,}\b', docstring.lower()))

        return DiscoveredCapability(
            name=node.name,
            docstring=docstring,
            args=args,
            keywords=keywords
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna las estadísticas actuales del motor."""
        return self.stats

    def get_agent(self, name: str) -> Optional[DiscoveredAgent]:
        """Obtiene un agente descubierto por su nombre."""
        return self.discovered_agents.get(name)

# Singleton global para el motor
_engine_instance: Optional[ExponentialCapabilityEngine] = None

def get_exponential_engine(project_root: Optional[Path] = None) -> ExponentialCapabilityEngine:
    """
    Fábrica para obtener la instancia singleton del ExponentialCapabilityEngine.
    """
    global _engine_instance
    if _engine_instance is None:
        if project_root is None:
            # Fallback robusto si no se provee el root
            project_root = Path.cwd()
        _engine_instance = ExponentialCapabilityEngine(project_root)
    return _engine_instance

if __name__ == '__main__':
    print("🧪 Probando Exponential Capability Engine...")
    
    # Usar el directorio padre como raíz del proyecto para el test
    engine = get_exponential_engine(Path(__file__).parent.parent)
    
    start_time = time.time()
    discovered = engine.discover_agents_in_directory(engine.project_root)
    end_time = time.time()
    
    print(f"✅ Escaneo completado en {end_time - start_time:.4f} segundos.")
    print("\n--- ESTADÍSTICAS ---")
    print(engine.get_statistics())
    
    print(f"\n--- {len(discovered)} AGENTES NUEVOS DESCUBIERTOS ---")
    for agent in discovered:
        print(f"\n🤖 Agente: {agent.name} (en {Path(agent.module_path).name})")
        print(f"   Capabilities: {len(agent.capabilities)}")
        for cap in agent.capabilities[:3]: # Mostrar hasta 3 capabilities
            print(f"     - {cap.name}({', '.join(cap.args)})")
    
    print("\n\n--- RE-ESCANEO (debería usar cache) ---")
    start_time = time.time()
    engine.discover_agents_in_directory(engine.project_root)
    end_time = time.time()
    print(f"✅ Re-escaneo completado en {end_time - start_time:.4f} segundos.")
    print(engine.get_statistics())
