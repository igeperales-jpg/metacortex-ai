#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestor de Capacidades de Metacortex
===================================

Este módulo se encarga de descubrir, inicializar y gestionar todas las
capacidades (herramientas) que el Agente Cognitivo puede utilizar.

Funciona como un registro centralizado que permite al agente acceder
dinámicamente a funcionalidades como programación, búsqueda web, etc.
"""
import logging
from typing import Dict, Any, Protocol, List
from pathlib import Path
import importlib.util
import inspect
import sys

logger = logging.getLogger(__name__)

# Definir el protocolo Capability
class Capability(Protocol):
    """Protocolo que todas las capacidades deben seguir."""
    def get_status(self) -> Dict[str, Any]:
        ...

class CapabilityManager:
    """
    Descubre, inicializa y gestiona las capacidades del agente de forma dinámica.
    """
    def __init__(self, capability_paths: List[Path]):
        """
        Inicializa el gestor de capacidades.

        Args:
            capability_paths (List[Path]): Lista de rutas donde buscar capacidades.
        """
        self.capabilities: Dict[str, Capability] = {}
        self.capability_paths = capability_paths
        self.logger = logger.getChild("CapabilityManager")

    def discover_and_initialize_capabilities(self) -> Dict[str, Capability]:
        """
        Descubre e inicializa dinámicamente todas las capacidades disponibles en las rutas especificadas.
        """
        self.logger.info(f"Iniciando descubrimiento de capacidades en: {self.capability_paths}")
        
        for path in self.capability_paths:
            if not path.is_dir():
                self.logger.warning(f"La ruta de capacidad '{path}' no es un directorio o no existe.")
                continue

            for file_path in path.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue

                module_name = file_path.stem
                if module_name in sys.modules:
                    self.logger.debug(f"El módulo '{module_name}' ya está importado, saltando.")
                    # Podríamos recargarlo si es necesario con: importlib.reload(sys.modules[module_name])
                    # Pero por ahora lo saltamos para evitar efectos secundarios.
                    continue

                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        self.logger.debug(f"Módulo '{module_name}' cargado desde '{file_path}'.")

                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            # La clase no debe ser el propio protocolo 'Capability'
                            if obj is not Capability and hasattr(obj, "__mro__") and Capability in obj.__mro__:
                                self.logger.info(f"Capacidad potencial encontrada: '{name}' en {module_name}.py")
                                try:
                                    instance = obj()
                                    # Usar un nombre de capacidad normalizado (ej. 'programming' de 'ProgrammingCapability')
                                    capability_name = name.replace("Capability", "").replace("Agent", "").lower()
                                    self.capabilities[capability_name] = instance
                                    self.logger.info(f"✅ Capacidad '{capability_name}' ({name}) registrada e instanciada.")
                                except Exception as e:
                                    self.logger.error(f"❌ Error al instanciar la capacidad '{name}': {e}", exc_info=True)
                    else:
                        self.logger.warning(f"No se pudo crear la especificación para el módulo en '{file_path}'")

                except Exception as e:
                    self.logger.error(f"❌ Error al cargar o inspeccionar el módulo '{file_path.name}': {e}", exc_info=True)
        
        self.logger.info(f"Descubrimiento finalizado. {len(self.capabilities)} capacidades cargadas.")
        return self.capabilities

    def get_capability(self, name: str) -> Any:
        """
        Obtiene una capacidad por su nombre.
        """
        return self.capabilities.get(name)

    def get_all_statuses(self) -> Dict[str, Any]:
        """
        Obtiene el estado de todas las capacidades cargadas.
        """
        statuses: Dict[str, Any] = {}
        for name, cap in self.capabilities.items():
            try:
                statuses[name] = cap.get_status()
            except Exception as e:
                statuses[name] = {"status": "error", "details": str(e)}
        return statuses
