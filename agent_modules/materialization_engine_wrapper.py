from agent_modules.language_handlers import LanguageHandlerRegistry
from agent_modules.project_analyzer import ProjectAnalyzer
from pathlib import Path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper para MaterializationEngine con inicialización automática
Generado automáticamente por MetacortexUltraVerifier
"""

import logging
from typing import Optional
from agent_modules.code_generator import CodeGenerator
from agent_modules.workspace_scanner import WorkspaceScanner

logger = logging.getLogger(__name__)

_materialization_engine_instance = None

def get_materialization_engine_auto():
    """
    Obtiene MaterializationEngine con parámetros por defecto
    
    Returns:
        MaterializationEngine inicializado
    """
    global _materialization_engine_instance
    
    if _materialization_engine_instance is not None:
        return _materialization_engine_instance
    
    try:
        # Importar dependencias con manejo de errores individual
        try:
            from agent_modules.materialization_engine import get_materialization_engine
        except ImportError as e:
            logger.error(f"❌ No se puede importar materialization_engine: {e}")
            return None
        
        # USAR SISTEMAS EXISTENTES CORRECTOS (no duplicados)
        # Inicializar dependencias necesarias para CodeGenerator y WorkspaceScanner
        try:
            from agent_modules.template_system import TemplateSystem
            
            # Inicializar componentes base
            project_root = Path.cwd()
            template_system = TemplateSystem(logger=logger)  # ✅ AGREGAR logger parameter
            language_handlers = LanguageHandlerRegistry(
                template_system=template_system,
                logger=logger
            )  # ✅ AGREGAR template_system y logger
            project_analyzer = ProjectAnalyzer(project_root, logger)
            
            # CodeGenerator REAL del sistema modular
            code_gen = CodeGenerator(
                template_system=template_system,
                language_handlers=language_handlers,
                project_analyzer=project_analyzer,
                logger=logger
            )
            
            # WorkspaceScanner REAL del sistema modular
            workspace = WorkspaceScanner(
                project_root=project_root,
                logger=logger
            )
            
            logger.info("✅ Sistemas reales inicializados correctamente")
            
        except ImportError as e:
            logger.warning(f"⚠️ No se pueden cargar sistemas modulares: {e}")
            # Fallback: usar MaterializationEngine sin parámetros opcionales
            code_gen = None
            workspace = None
        except Exception as e:
            logger.warning(f"⚠️ Error inicializando sistemas: {e}")
            code_gen = None
            workspace = None
        
        # Intentar obtener cognitive bridge (opcional)
        try:
            from ml_cognitive_bridge import MLCognitiveBridge
            cognitive_agent = MLCognitiveBridge()
        except ImportError:
            cognitive_agent = None
            logger.debug("MLCognitiveBridge no disponible, usando None")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            cognitive_agent = None
            logger.debug(f"Error cargando MLCognitiveBridge: {e}")
        
        # Crear instancia con sistemas REALES (o None si no disponibles)
        _materialization_engine_instance = get_materialization_engine(
            code_generator=code_gen,
            workspace_scanner=workspace,
            cognitive_agent=cognitive_agent,
            logger=logger
        )
        
        logger.info("✅ MaterializationEngine inicializado con wrapper")
        return _materialization_engine_instance
        
    except Exception as e:
        logger.error(f"❌ Error inicializando MaterializationEngine: {e}")
        return None

def enable_autonomous_materialization():
    """Habilita materialización autónoma"""
    engine = get_materialization_engine_auto()
    if engine and hasattr(engine, 'enable_autonomous_development_mode'):
        engine.enable_autonomous_development_mode()
        logger.info("✅ Modo autónomo de materialización activado")
        return True
    return False
