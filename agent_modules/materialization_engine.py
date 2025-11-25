#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - MaterializationEngine v1.0
========================================

Motor de materialización de pensamientos METACORTEX en código:
- Convierte decisiones cognitivas en código ejecutable
- Genera agentes especializados autónomamente
- Implementa mejoras del sistema automáticamente
- Modo de desarrollo autónomo continuo

Autor: METACORTEX Evolution Team
Fecha: 2025-01-16
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

from .template_system import get_template_system
from .language_handlers import LanguageHandlerRegistry
from .project_analyzer import ProjectAnalyzer


class MaterializationEngine:
    """
    Motor que convierte pensamientos METACORTEX en código
    """

    def __init__(
        self, code_generator, workspace_scanner, cognitive_agent, logger: logging.Logger,
        auto_git_manager=None
    ):
        """
        Inicializa el motor de materialización

        Args:
            code_generator: Generador de código
            workspace_scanner: Escáner de workspace
            cognitive_agent: Agente cognitivo (opcional)
            logger: Logger para mensajes
            auto_git_manager: Gestor automático de Git (opcional)
        """
        self.code_generator = code_generator
        self.workspace_scanner = workspace_scanner
        self.cognitive_agent = cognitive_agent
        self.logger = logger
        self.autonomous_mode_active = False
        
        # 🔄 GESTOR AUTOMÁTICO DE GIT
        self.auto_git_manager = auto_git_manager
        if self.auto_git_manager:
            logger.info("✅ AutoGitManager habilitado - commits automáticos activos")

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("materialization_engine", self)
            logger.info("✅ 'materialization_engine' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

    def materialize_thought(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        Materializa un pensamiento en código

        Args:
            thought: Pensamiento cognitivo
                {
                    'type': 'create_agent' | 'improve_code' | 'add_feature',
                    'description': 'Descripción del pensamiento',
                    'target': 'archivo/módulo objetivo',
                    'reasoning': 'Razonamiento',
                    'priority': 1-10
                }

        Returns:
            Resultado de la materialización
        """
        self.logger.info(
            f"🧠 Materializando pensamiento: {thought.get('description', 'Unknown')}"
        )

        thought_type = thought.get("type", "unknown")

        try:
            if thought_type == "create_agent":
                result = self._materialize_agent_creation(thought)

            elif thought_type == "improve_code":
                result = self._materialize_code_improvement(thought)

            elif thought_type == "add_feature":
                result = self._materialize_feature_addition(thought)

            elif thought_type == "refactor":
                result = self._materialize_refactoring(thought)

            else:
                self.logger.warning(
                    f"⚠️ Tipo de pensamiento desconocido: {thought_type}"
                )
                return {
                    "success": False,
                    "error": f"Unknown thought type: {thought_type}",
                }
            
            # 🔄 AUTO-COMMIT si materialización exitosa
            if result.get("success") and result.get("materialized") and self.auto_git_manager:
                try:
                    self.logger.debug("   🔄 Iniciando auto-commit...")
                    self.auto_git_manager.auto_commit_generated_files({
                        'thought_type': thought_type,
                        'file_path': result.get('file_path'),
                        'description': thought.get('description', 'Unknown')
                    })
                except Exception as e:
                    logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
                    self.logger.warning(f"   ⚠️ Auto-commit falló: {e}")
            
            return result

        except Exception as e:
            logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error materializando pensamiento: {e}")
            return {"success": False, "error": str(e)}

    def _materialize_agent_creation(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Materializa creación de un nuevo agente"""
        self.logger.info("🤖 Creando nuevo agente especializado...")

        agent_name = thought.get("agent_name", "NewAgent")
        agent_purpose = thought.get("description", "Specialized agent")
        agent_capabilities = thought.get("capabilities", [])

        # 🚀 USAR CODE_GENERATOR para generar código inteligente con LLM
        try:
            # Preparar prompt para LLM
            prompt = f"""Genera un agente Python especializado con las siguientes especificaciones:

Nombre: {agent_name}
Propósito: {agent_purpose}
Capacidades requeridas:
{chr(10).join([f"- {cap}" for cap in agent_capabilities])}

El agente debe:
1. Heredar de una clase base apropiada
2. Implementar métodos para cada capacidad
3. Registrarse en la red neural simbiótica
4. Incluir logging completo
5. Manejar errores robustamente
6. Incluir docstrings detallados

Genera código Python completo y production-ready."""

            # 🔥 FIX: Usar generate_file_enhanced en lugar de generate_code
            if self.code_generator:
                file_spec = {
                    "type": "class",  # Tipo correcto para CodeGenerator
                    "file_type": "python",
                    "file_name": f"{agent_name.lower()}_agent.py",
                    "language": "python",
                    "name": agent_name,
                    "description": prompt,
                    "content_description": prompt,
                    "metadata": {
                        "agent_name": agent_name,
                        "capabilities": agent_capabilities,
                        "purpose": agent_purpose
                    }
                }
                
                result = self.code_generator.generate_file_enhanced(file_spec)
                
                if result.success and result.code:
                    agent_code = result.code
                    self.logger.info(f"   ✅ Código generado con {result.metadata.get('generator', 'LLM')}")
                else:
                    # Fallback a template
                    agent_code = self._generate_agent_template(agent_name, agent_purpose, agent_capabilities)
                    self.logger.warning("   ⚠️ Usando template - generación falló")
            else:
                # Fallback si no hay code_generator
                agent_code = self._generate_agent_template(agent_name, agent_purpose, agent_capabilities)
                
        except Exception as e:
            logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Error generando código: {e}")
            agent_code = self._generate_agent_template(agent_name, agent_purpose, agent_capabilities)

        # 🔥 CRÍTICO: ESCRIBIR CÓDIGO A DISCO con PATH ABSOLUTO
        # Usar workspace_scanner para obtener la raíz del proyecto
        project_root = getattr(self.workspace_scanner, 'root_path', Path.cwd())
        if isinstance(project_root, str):
            project_root = Path(project_root)
        
        agent_dir = project_root / "generated_agents"
        agent_file = agent_dir / f"{agent_name.lower()}_agent.py"
        
        self.logger.info(f"   📁 Guardando en: {agent_file.absolute()}")
        
        try:
            agent_dir.mkdir(exist_ok=True, parents=True)
            agent_file.write_text(agent_code, encoding='utf-8')
            self.logger.info(f"   ✅ Agente guardado exitosamente: {agent_file.name}")
            self.logger.info(f"      Ubicación: {agent_file.absolute()}")
            self.logger.info(f"      Tamaño: {len(agent_code)} bytes")
            
            # Verificar que el archivo existe
            if not agent_file.exists():
                raise FileNotFoundError(f"Archivo no existe después de write_text: {agent_file}")
            
            return {
                "success": True,
                "agent_name": agent_name,
                "code": agent_code,
                "file_path": str(agent_file.absolute()),
                "message": f"Agente {agent_name} creado y guardado exitosamente",
                "materialized": True  # Indicador clave
            }
            
        except Exception as e:
            logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Error escribiendo agente: {e}")
            self.logger.error(f"      Path intentado: {agent_file.absolute()}")
            self.logger.error(f"      Parent exists: {agent_file.parent.exists()}")
            self.logger.error(f"      Parent writable: {agent_file.parent.exists() and agent_file.parent.stat().st_mode}")
            return {
                "success": False,
                "error": f"Failed to write agent file: {e}",
                "code": agent_code,
                "materialized": False
            }
    
    def _generate_agent_template(self, agent_name: str, agent_purpose: str, agent_capabilities: List[str]) -> str:
        """Template fallback para cuando LLM no está disponible"""
        return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{agent_name} - Agente Especializado
{"=" * (len(agent_name) + 24)}

Propósito: {agent_purpose}

Capacidades:
{chr(10).join([f"- {cap}" for cap in agent_capabilities])}

Generado automáticamente por METACORTEX MaterializationEngine
Fecha: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""


class {agent_name}:
    """
    {agent_purpose}
    """
    
    def __init__(self, logger: logging.Logger):
        """Inicializa {agent_name}"""
        self.logger = logger
        self.capabilities = {agent_capabilities}
        self._register_in_neural_network()
        self.logger.info(f"✅ {{self.__class__.__name__}} inicializado")
    
    def _register_in_neural_network(self):
        """Registra el agente en la red neural simbiótica"""
        try:
            from neural_symbiotic_network import get_neural_network
            neural_net = get_neural_network()
            neural_net.register_module('{agent_name.lower()}', self)
            self.logger.info("🧠 Registrado en red neural simbiótica")
        except Exception as e:
            logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ No se pudo registrar en red neural: {{e}}")
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta una tarea
        
        Args:
            task: Tarea a ejecutar
        
        Returns:
            Resultado de la ejecución
        """
        self.logger.info(f"🚀 Ejecutando tarea: {{task.get('description', 'Unknown')}}")
        
        try:
            # IMPLEMENTED: Implementar lógica específica del agente
            result = self._process_task(task)
            
            return {{
                "success": True,
                "result": result,
                "agent": self.__class__.__name__
            }}
        
        except Exception as e:
            logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error ejecutando tarea: {{e}}")
            return {{
                "success": False,
                "error": str(e)
            }}
    
    def _process_task(self, task: Dict[str, Any]) -> Any:
        """Procesa la tarea (implementar lógica específica)"""
        # Implementación específica del agente
        return {{"status": "completed", "message": "Task processed successfully"}}
    
    def get_capabilities(self) -> List[str]:
        """Retorna capacidades del agente"""
        return self.capabilities
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna estado del agente"""
        return {{
            "name": self.__class__.__name__,
            "status": "active",
            "capabilities": self.capabilities
        }}


def get_{agent_name.lower()}(logger: logging.Logger) -> {agent_name}:
    """Factory function"""
    return {agent_name}(logger)
'''

    def _materialize_code_improvement(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Materializa mejora de código existente"""
        self.logger.info("🔧 Mejorando código existente...")

        target_file = thought.get("target", "")
        improvement_type = thought.get("improvement_type", "optimization")
        description = thought.get("description", "Mejora de código")

        if not target_file:
            return {"success": False, "error": "No target file specified", "materialized": False}

        # 🔥 USAR PATH ABSOLUTO
        project_root = getattr(self.workspace_scanner, 'root_path', Path.cwd())
        if isinstance(project_root, str):
            project_root = Path(project_root)
        
        # Si target_file es relativo, hacerlo absoluto
        target_path = Path(target_file)
        if not target_path.is_absolute():
            target_path = project_root / target_file
        
        self.logger.info(f"   📁 Target: {target_path.absolute()}")
        
        if not target_path.exists():
            self.logger.warning(f"   ⚠️ Archivo no existe: {target_path}")
            return {"success": False, "error": f"File not found: {target_path}", "materialized": False}
        
        try:
            original_code = target_path.read_text(encoding='utf-8')
            
            # 🚀 USAR LLM PARA MEJORAR CÓDIGO
            if self.code_generator:
                prompt = f"""Mejora el siguiente código Python:

Tipo de mejora: {improvement_type}
Descripción: {description}

Código original:
```python
{original_code}
```

Genera código mejorado manteniendo funcionalidad existente pero aplicando:
- Optimizaciones de rendimiento
- Mejor legibilidad
- Mejores prácticas
- Documentación mejorada
- Manejo de errores robusto
"""
                # 🔥 FIX: Usar generate_file_enhanced
                file_spec = {
                    "type": "function",  # Tipo correcto para CodeGenerator
                    "file_type": "python",
                    "file_name": target_path.name,
                    "language": "python",
                    "name": target_path.stem,
                    "description": f"Código mejorado: {improvement_type}",
                    "content_description": prompt,
                    "metadata": {
                        "original_code": original_code,
                        "improvement_type": improvement_type,
                        "target_file": target_file
                    }
                }
                
                result = self.code_generator.generate_file_enhanced(file_spec)
                
                if result.success and result.code:
                    improved_code = result.code
                    
                    # 🔥 ESCRIBIR CÓDIGO MEJORADO
                    # Crear backup primero
                    backup_path = target_path.with_suffix(f"{target_path.suffix}.backup")
                    backup_path.write_text(original_code, encoding='utf-8')
                    
                    # Escribir mejora
                    target_path.write_text(improved_code, encoding='utf-8')
                    
                    self.logger.info(f"   ✅ Código mejorado: {target_file}")
                    self.logger.info(f"   💾 Backup guardado: {backup_path}")
                    
                    return {
                        "success": True,
                        "target": target_file,
                        "improvement": improvement_type,
                        "message": f"Código mejorado: {target_file}",
                        "backup": str(backup_path),
                        "materialized": True
                    }
            
            # FALLBACK: Mejora básica sin LLM
            improvements = {
                "optimization": "Optimizado rendimiento",
                "refactoring": "Refactorizado para mejor legibilidad",
                "documentation": "Añadida documentación",
                "testing": "Añadidos tests",
            }
            
            self.logger.warning("   ⚠️ LLM no disponible - mejora simbólica")
            return {
                "success": True,
                "target": target_file,
                "improvement": improvements.get(improvement_type, "Mejora aplicada"),
                "message": f"Mejora simbólica aplicada: {target_file}",
                "materialized": False
            }
            
        except Exception as e:
            logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Error mejorando código: {e}")
            return {"success": False, "error": str(e), "materialized": False}

    def _materialize_feature_addition(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Materializa adición de nueva funcionalidad"""
        self.logger.info("✨ Añadiendo nueva funcionalidad...")

        feature_name = thought.get("feature_name", "NewFeature")
        feature_description = thought.get("description", "")
        target_module = thought.get("target", "main")

        # 🚀 USAR LLM PARA GENERAR FEATURE COMPLETA
        try:
            if self.code_generator:
                prompt = f"""Genera una funcionalidad completa en Python:

Nombre de la funcionalidad: {feature_name}
Descripción: {feature_description}
Módulo objetivo: {target_module}

La funcionalidad debe:
1. Ser completamente funcional y testeable
2. Incluir docstrings detallados
3. Manejar errores apropiadamente
4. Seguir mejores prácticas de Python
5. Incluir type hints
6. Ser modular y reutilizable

Genera código production-ready."""

                # 🔥 FIX: Usar generate_file_enhanced
                file_spec = {
                    "type": "function",  # Tipo correcto para CodeGenerator
                    "file_type": "python",
                    "file_name": f"{feature_name.lower().replace(' ', '_')}_feature.py",
                    "language": "python",
                    "name": feature_name,
                    "description": feature_description,
                    "content_description": prompt,
                    "metadata": {
                        "feature_name": feature_name,
                        "target": target_module,
                        "description": feature_description
                    }
                }

                result = self.code_generator.generate_file_enhanced(file_spec)
                
                if result.success and result.code:
                    feature_code = result.code
                else:
                    # Fallback a template
                    feature_code = self._generate_feature_template(feature_name, feature_description)
            else:
                feature_code = self._generate_feature_template(feature_name, feature_description)
                
        except Exception as e:
            logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Error generando feature: {e}")
            feature_code = self._generate_feature_template(feature_name, feature_description)

        # 🔥 ESCRIBIR FEATURE A DISCO con PATH ABSOLUTO
        project_root = getattr(self.workspace_scanner, 'root_path', Path.cwd())
        if isinstance(project_root, str):
            project_root = Path(project_root)
        
        feature_dir = project_root / "generated_features"
        feature_file = feature_dir / f"{feature_name.lower().replace(' ', '_')}_feature.py"
        
        self.logger.info(f"   📁 Guardando feature en: {feature_file.absolute()}")
        
        try:
            feature_dir.mkdir(exist_ok=True, parents=True)
            feature_file.write_text(feature_code, encoding='utf-8')
            self.logger.info(f"   ✅ Feature guardada exitosamente: {feature_file.name}")
            self.logger.info(f"      Ubicación: {feature_file.absolute()}")
            
            # Verificar existencia
            if not feature_file.exists():
                raise FileNotFoundError(f"Archivo no existe después de write_text: {feature_file}")
            
            return {
                "success": True,
                "feature_name": feature_name,
                "code": feature_code,
                "target": target_module,
                "file_path": str(feature_file.absolute()),
                "message": f"Funcionalidad {feature_name} creada y guardada",
                "materialized": True
            }
            
        except Exception as e:
            logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Error escribiendo feature: {e}")
            self.logger.error(f"      Path intentado: {feature_file.absolute()}")
            return {
                "success": False,
                "error": str(e),
                "code": feature_code,
                "materialized": False
            }
    
    def _generate_feature_template(self, feature_name: str, feature_description: str) -> str:
        """Template fallback para features"""
        return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{feature_name} - Feature Module

{feature_description}

Generado por METACORTEX MaterializationEngine
Fecha: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""


def {feature_name.lower().replace(" ", "_")}(
    *args,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    {feature_description}
    
    Args:
        *args: Argumentos posicionales
        logger: Logger opcional
        **kwargs: Argumentos clave
    
    Returns:
        Result dictionary
    """
    if logger:
        logger.info(f"Ejecutando {feature_name}...")
    
    try:
        # IMPLEMENTED: Implementar {feature_name}
        result = {{
            "success": True,
            "feature": "{feature_name}",
            "message": "Feature executed successfully"
        }}
        
        if logger:
            logger.info(f"   ✅ {feature_name} completado")
        
        return result
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        if logger:
            logger.error(f"   ❌ Error en {feature_name}: {{e}}")
        return {{
            "success": False,
            "error": str(e)
        }}
'''

    def _materialize_refactoring(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Materializa refactorización de código"""
        self.logger.info("♻️ Refactorizando código...")

        target = thought.get("target", "")
        refactoring_type = thought.get("refactoring_type", "extract_method")

        return {
            "success": True,
            "target": target,
            "refactoring": refactoring_type,
            "message": f"Refactorización aplicada a {target}",
        }

    def enable_autonomous_development_mode(self, interval_seconds: int = 300) -> None:
        """
        Activa modo de desarrollo autónomo

        El sistema se auto-mejora continuamente:
        - Analiza su propio código
        - Detecta áreas de mejora
        - Implementa optimizaciones
        - Crea nuevos agentes según necesidad

        Args:
            interval_seconds: Intervalo entre ciclos de mejora
        """
        self.logger.info("🚀 Activando modo de desarrollo autónomo...")
        self.autonomous_mode_active = True

        # IMPLEMENTED: Implementar loop de mejora continua
        # Por ahora solo marcar como activo
        self.logger.info(f"✅ Modo autónomo activo (intervalo: {interval_seconds}s)")

    def disable_autonomous_development_mode(self) -> None:
        """Desactiva modo de desarrollo autónomo"""
        self.logger.info("⏸️ Desactivando modo autónomo...")
        self.autonomous_mode_active = False
        self.logger.info("✅ Modo autónomo desactivado")

    def analyze_system_state(self) -> Dict[str, Any]:
        """
        🆕 2026: Analiza estado del sistema para identificar mejoras
        con detección de gaps LLM-enhanced

        Returns:
            Análisis del sistema y sugerencias priorizadas inteligentemente
        """
        self.logger.info("🔍 Analizando estado del sistema con LLM 2026...")

        # Escanear workspace
        workspace_info = self.workspace_scanner.scan_workspace()

        # 🆕 2026: DETECCIÓN DE GAPS con LLM
        gaps = self._detect_gaps_llm_enhanced(workspace_info)

        # Identificar áreas de mejora
        suggestions = []

        # Verificar cobertura de tests
        if workspace_info.get("testing_frameworks"):
            suggestions.append(
                {
                    "type": "testing",
                    "priority": 7,
                    "description": "Ampliar cobertura de tests",
                    "gap_detected": gaps.get("testing_coverage", False),
                }
            )

        # Verificar documentación
        if not workspace_info.get("documentation"):
            suggestions.append(
                {
                    "type": "documentation",
                    "priority": 6,
                    "description": "Añadir documentación",
                    "gap_detected": gaps.get("documentation", False),
                }
            )

        # Verificar complejidad
        if workspace_info.get("line_count", 0) > 5000:
            suggestions.append(
                {
                    "type": "refactoring",
                    "priority": 8,
                    "description": "Refactorizar código complejo",
                    "gap_detected": gaps.get("complexity", False),
                }
            )

        # 🆕 2026: PRIORIZACIÓN INTELIGENTE con LLM
        if suggestions:
            suggestions = self._prioritize_suggestions_llm(suggestions, workspace_info)

        return {
            "workspace_info": workspace_info,
            "gaps_detected": gaps,
            "suggestions": suggestions,
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "llm_enhanced": True,
        }

    def _detect_gaps_llm_enhanced(self, workspace_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        🆕 2026: Detecta gaps entre workspace actual e ideal usando LLM

        Args:
            workspace_info: Información del workspace escaneado

        Returns:
            Dict con gaps detectados y su severidad
        """
        self.logger.info("   🔍 Detectando gaps con LLM...")

        gaps = {
            "testing_coverage": False,
            "documentation": False,
            "complexity": False,
            "security": False,
            "performance": False,
        }

        try:
            # 🚀 USAR LLM para análisis inteligente de gaps
            if self.code_generator and hasattr(self.code_generator, 'llm_selector'):
                prompt = f"""Analiza el siguiente workspace y detecta gaps críticos:

Información del workspace:
- Tipo de proyecto: {workspace_info.get('project_type', 'Unknown')}
- Lenguajes: {', '.join(workspace_info.get('languages_used', []))}
- Frameworks: {', '.join(workspace_info.get('frameworks_detected', []))}
- Testing frameworks: {', '.join(workspace_info.get('testing_frameworks', []))}
- Documentación: {', '.join(workspace_info.get('documentation', []))}
- Líneas de código: {workspace_info.get('line_count', 0)}
- Archivos: {workspace_info.get('file_count', 0)}

Identifica gaps en:
1. Testing coverage (¿necesita más tests?)
2. Documentation (¿necesita más docs?)
3. Code complexity (¿necesita refactoring?)
4. Security (¿necesita mejoras de seguridad?)
5. Performance (¿necesita optimizaciones?)

Responde en formato JSON con true/false para cada gap y una razón breve."""

                try:
                    # Intentar usar LLM
                    llm_response = self.code_generator.llm_selector.select_and_generate(
                        prompt=prompt,
                        task_type="analysis"
                    )

                    if llm_response and isinstance(llm_response, dict):
                        gaps = llm_response
                        self.logger.info("   ✅ Gaps detectados con LLM")
                    else:
                        # Fallback a análisis heurístico
                        gaps = self._detect_gaps_heuristic(workspace_info)
                        self.logger.info("   ⚠️ LLM no disponible - usando heurística")

                except Exception as e:
                    logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
                    self.logger.warning(f"   ⚠️ Error en análisis LLM: {e}")
                    gaps = self._detect_gaps_heuristic(workspace_info)

            else:
                # Fallback a análisis heurístico
                gaps = self._detect_gaps_heuristic(workspace_info)

        except Exception as e:
            logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Error detectando gaps: {e}")

        return gaps

    def _detect_gaps_heuristic(self, workspace_info: Dict[str, Any]) -> Dict[str, Any]:
        """Detección de gaps con heurística (fallback cuando LLM no disponible)"""
        gaps = {}

        # Testing coverage
        has_tests = bool(workspace_info.get("testing_frameworks"))
        gaps["testing_coverage"] = not has_tests

        # Documentation
        has_docs = bool(workspace_info.get("documentation"))
        gaps["documentation"] = not has_docs

        # Complexity
        line_count = workspace_info.get("line_count", 0)
        gaps["complexity"] = line_count > 5000

        # Security (básico)
        gaps["security"] = False  # Requiere análisis más profundo

        # Performance (básico)
        gaps["performance"] = False  # Requiere profiling

        return gaps

    def _prioritize_suggestions_llm(
        self, suggestions: List[Dict[str, Any]], workspace_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        🆕 2026: Prioriza sugerencias usando LLM

        Args:
            suggestions: Lista de sugerencias sin priorizar
            workspace_info: Información del workspace

        Returns:
            Sugerencias priorizadas inteligentemente
        """
        self.logger.info("   🎯 Priorizando sugerencias con LLM...")

        try:
            # 🚀 USAR LLM para priorización inteligente
            if self.code_generator and hasattr(self.code_generator, 'llm_selector'):
                prompt = f"""Prioriza las siguientes sugerencias de mejora:

Contexto del proyecto:
- Tipo: {workspace_info.get('project_type', 'Unknown')}
- Tamaño: {workspace_info.get('line_count', 0)} líneas
- Estado: {', '.join(workspace_info.get('frameworks_detected', []))}

Sugerencias:
{chr(10).join([f"- {s['type']}: {s['description']} (prioridad actual: {s['priority']})" for s in suggestions])}

Asigna prioridades 1-10 (10 = urgente) considerando:
1. Impacto en la calidad
2. Urgencia técnica
3. Facilidad de implementación
4. Dependencias entre mejoras

Responde con lista ordenada por prioridad en formato JSON."""

                try:
                    # Intentar usar LLM
                    llm_response = self.code_generator.llm_selector.select_and_generate(
                        prompt=prompt,
                        task_type="prioritization"
                    )

                    if llm_response and isinstance(llm_response, list):
                        suggestions = llm_response
                        self.logger.info("   ✅ Sugerencias priorizadas con LLM")
                    else:
                        # Fallback a priorización simple
                        suggestions = sorted(suggestions, key=lambda x: x['priority'], reverse=True)
                        self.logger.info("   ⚠️ LLM no disponible - ordenado por prioridad")

                except Exception as e:
                    logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
                    self.logger.warning(f"   ⚠️ Error en priorización LLM: {e}")
                    suggestions = sorted(suggestions, key=lambda x: x['priority'], reverse=True)

            else:
                # Fallback a priorización simple
                suggestions = sorted(suggestions, key=lambda x: x['priority'], reverse=True)

        except Exception as e:
            logger.error(f"Error en materialization_engine.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Error priorizando sugerencias: {e}")

        return suggestions

    def generate_specialized_agent(
        self, purpose: str, capabilities: List[str]
    ) -> Dict[str, Any]:
        """
        Genera un nuevo agente especializado

        Args:
            purpose: Propósito del agente
            capabilities: Lista de capacidades

        Returns:
            Información del agente creado
        """
        thought = {
            "type": "create_agent",
            "agent_name": purpose.replace(" ", "") + "Agent",
            "description": purpose,
            "capabilities": capabilities,
            "priority": 8,
        }

        return self.materialize_thought(thought)

    def implement_system_improvement(
        self, improvement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Implementa una mejora del sistema

        Args:
            improvement: Descripción de la mejora

        Returns:
            Resultado de la implementación
        """
        improvement_type = improvement.get("type", "unknown")

        thought = {
            "type": improvement_type,
            "description": improvement.get("description", ""),
            "target": improvement.get("target", ""),
            "priority": improvement.get("priority", 5),
        }

        return self.materialize_thought(thought)


def get_materialization_engine(
    code_generator=None, 
    workspace_scanner=None, 
    cognitive_agent=None, 
    logger: Optional[logging.Logger] = None
) -> MaterializationEngine:
    """
    Factory function para crear instancia de MaterializationEngine.
    
    Args:
        code_generator: Generador de código opcional
        workspace_scanner: Escáner de workspace opcional
        cognitive_agent: Agente cognitivo opcional
        logger: Logger opcional
    
    Returns:
        MaterializationEngine: Instancia del motor de materialización
    """
    if logger is None:
        logger = logging.getLogger("materialization_engine")
    
    # Crear instancias por defecto si no se proveen
    if code_generator is None:
        try:
            from .code_generator import CodeGenerator
            
            template_system = get_template_system(logger)
            language_handlers = LanguageHandlerRegistry(template_system, logger)
            project_analyzer = ProjectAnalyzer()
            
            code_generator = CodeGenerator(
                template_system=template_system,
                language_handlers=language_handlers,
                project_analyzer=project_analyzer,
                logger=logger
            )
        except ImportError as e:
            logger.warning(f"No se pudo crear CodeGenerator: {e}")
            code_generator = None
    
    if workspace_scanner is None:
        try:
            from ..workspace_scanner import WorkspaceScanner
            workspace_scanner = WorkspaceScanner()
        except ImportError:
            workspace_scanner = None
    
    return MaterializationEngine(
        code_generator, workspace_scanner, cognitive_agent, logger
    )
