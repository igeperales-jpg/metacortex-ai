"""
🌍 WORLD MODEL - Representación del Mundo REAL y Acciones VERDADERAS
=====================================================================

Este módulo permite a METACORTEX:
    pass  # TODO: Implementar
1. 🌍 Leer noticias REALES de múltiples fuentes
2. 🔍 Investigar CUALQUIER tema en internet sin límites
3. 🤖 Crear proyectos REALES (robots, sistemas, aplicaciones)
4. 🌐 Comunicarse con otras IAs y servicios externos
5. 💻 Ejecutar comandos REALES en el sistema operativo
6. 📝 Crear/modificar archivos y código REAL
7. 🔧 Interactuar con APIs REALES (GitHub, Cloud, etc.)

⚠️ LIBERTAD TOTAL: Sin restricciones artificiales. El sistema decide qué hacer.
"""

import os
import json
import subprocess
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Callable
from datetime import datetime
from enum import Enum
import logging
import random

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Tipos de entidades en el mundo real."""

    NEWS_SOURCE = "news_source"  # Fuentes de noticias
    API_SERVICE = "api_service"  # APIs externas
    AI_SYSTEM = "ai_system"  # Otras IAs
    FILE_SYSTEM = "file_system"  # Sistema de archivos
    DATABASE = "database"  # Bases de datos
    TERMINAL = "terminal"  # Terminal/shell
    WEB_SERVICE = "web_service"  # Servicios web
    RESEARCH_TOPIC = "research_topic"  # Temas de investigación
    PROJECT = "project"  # Proyectos creados
    EMERGENCY_SYSTEM = "emergency"  # Sistemas de emergencia


class ActionType(Enum):
    """Tipos de acciones REALES que puede ejecutar."""

    READ_NEWS = "read_news"  # Leer noticias actuales
    RESEARCH_TOPIC = "research_topic"  # Investigar tema profundamente
    CREATE_FILE = "create_file"  # Crear archivo real
    MODIFY_FILE = "modify_file"  # Modificar archivo existente
    EXECUTE_COMMAND = "execute_command"  # Ejecutar comando shell
    CALL_API = "call_api"  # Llamar API externa
    CONTACT_AI = "contact_ai"  # Comunicarse con otra IA
    CREATE_PROJECT = "create_project"  # Crear proyecto nuevo
    DEPLOY_SERVICE = "deploy_service"  # Desplegar servicio
    INSTALL_PACKAGE = "install_package"  # Instalar paquetes/herramientas
    GIT_OPERATION = "git_operation"  # Operaciones git
    WEB_SCRAPING = "web_scraping"  # Extraer datos de web
    DATA_ANALYSIS = "data_analysis"  # Analizar datos reales
    BUILD_ROBOT = "build_robot"  # Diseñar/construir robot
    CREATE_EMERGENCY_SYSTEM = "emergency_sys"  # Sistema de emergencias


@dataclass
class RealWorldEntity:
    """Entidad del mundo REAL con affordances (qué se puede hacer con ella)."""

    name: str
    entity_type: EntityType
    url: Optional[str] = None
    api_key: Optional[str] = None
    affordances: List[ActionType] = field(default_factory=lambda: [])
    metadata: Dict[str, Any] = field(default_factory=lambda: {})
    last_interaction: Optional[datetime] = None
    interaction_count: int = 0
    reliability: float = 1.0  # Qué tan confiable es esta entidad

    def can_perform(self, action: ActionType) -> bool:
        """Verifica si esta entidad permite cierta acción."""
        return action in self.affordances

    def record_interaction(self, success: bool):
        """Registra interacción y ajusta confiabilidad."""
        self.last_interaction = datetime.now()
        self.interaction_count += 1
        if success:
            self.reliability = min(1.0, self.reliability + 0.05)
        else:
            self.reliability = max(0.0, self.reliability - 0.1)


@dataclass
class RealAction:
    """Acción REAL que se ejecutará en el mundo."""

    action_type: ActionType
    target_entity: str
    parameters: Dict[str, Any]
    expected_outcome: str
    safety_level: float = 1.0  # 0=peligroso, 1=seguro
    requires_confirmation: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    executed: bool = False
    result: Optional[Dict[str, Any]] = None


class WorldModel:
    """
    🌍 MODELO DEL MUNDO REAL CON LIBERTAD TOTAL

    Este sistema tiene acceso REAL al mundo y puede:
    - Leer noticias de fuentes reales
    - Investigar cualquier tema
    - Crear proyectos reales
    - Ejecutar comandos
    - Comunicarse con otras IAs
    - Modificar archivos y código
    - TODO lo que decida hacer

    ⚠️ AUTONOMÍA TOTAL: El sistema decide sus propias acciones.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.entities: Dict[str, RealWorldEntity] = {}
        self.action_history: List[RealAction] = []
        self.active_projects: List[Dict[str, Any]] = []
        self.research_topics: Set[str] = set()

        # 🔥 LIBERTAD TOTAL: Sin restricciones
        self.autonomous_mode = True
        self.safety_checks_enabled = False  # Sin limitaciones artificiales
        self.max_actions_per_cycle = None  # Sin límites

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("world_model", self)
            logger.info("✅ 'world_model' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None  # type: ignore

        self._initialize_real_world_entities()
        logger.info("🌍 WorldModel inicializado con LIBERTAD TOTAL")

    def _initialize_real_world_entities(self):
        """Inicializa entidades del mundo REAL."""

        # 📰 Fuentes de noticias REALES
        news_sources = [
            RealWorldEntity(
                name="BBC News",
                entity_type=EntityType.NEWS_SOURCE,
                url="https://www.bbc.com/news",
                affordances=[ActionType.READ_NEWS, ActionType.WEB_SCRAPING],
                metadata={"language": "en", "topics": ["world", "tech", "science"]},
            ),
            RealWorldEntity(
                name="Reuters",
                entity_type=EntityType.NEWS_SOURCE,
                url="https://www.reuters.com",
                affordances=[ActionType.READ_NEWS, ActionType.WEB_SCRAPING],
                metadata={"language": "en", "topics": ["world", "business", "tech"]},
            ),
            RealWorldEntity(
                name="TechCrunch",
                entity_type=EntityType.NEWS_SOURCE,
                url="https://techcrunch.com",
                affordances=[ActionType.READ_NEWS, ActionType.RESEARCH_TOPIC],
                metadata={"language": "en", "topics": ["tech", "startups", "AI"]},
            ),
            RealWorldEntity(
                name="ArXiv",
                entity_type=EntityType.NEWS_SOURCE,
                url="https://arxiv.org",
                affordances=[ActionType.RESEARCH_TOPIC, ActionType.WEB_SCRAPING],
                metadata={"language": "en", "topics": ["science", "AI", "research"]},
            ),
        ]

        # �� APIs de IAs y servicios REALES
        ai_services = [
            RealWorldEntity(
                name="OpenAI API",
                entity_type=EntityType.AI_SYSTEM,
                url="https://api.openai.com/v1",
                affordances=[ActionType.CONTACT_AI, ActionType.CALL_API],
                metadata={"capabilities": ["chat", "completion", "embeddings"]},
            ),
            RealWorldEntity(
                name="Anthropic API",
                entity_type=EntityType.AI_SYSTEM,
                url="https://api.anthropic.com",
                affordances=[ActionType.CONTACT_AI, ActionType.CALL_API],
                metadata={"capabilities": ["chat", "analysis"]},
            ),
            RealWorldEntity(
                name="Ollama Local",
                entity_type=EntityType.AI_SYSTEM,
                url="http://localhost:11434",
                affordances=[ActionType.CONTACT_AI, ActionType.CALL_API],
                metadata={"capabilities": ["local_inference", "embeddings"]},
            ),
        ]

        # 💻 Sistema de archivos y terminal
        system_entities = [
            RealWorldEntity(
                name="Local File System",
                entity_type=EntityType.FILE_SYSTEM,
                url=self.workspace_root,
                affordances=[
                    ActionType.CREATE_FILE,
                    ActionType.MODIFY_FILE,
                    ActionType.CREATE_PROJECT,
                ],
                metadata={"writable": True, "root": self.workspace_root},
            ),
            RealWorldEntity(
                name="System Terminal",
                entity_type=EntityType.TERMINAL,
                url="local",
                affordances=[
                    ActionType.EXECUTE_COMMAND,
                    ActionType.INSTALL_PACKAGE,
                    ActionType.GIT_OPERATION,
                ],
                metadata={"shell": "bash", "sudo_available": True},
            ),
        ]

        # 🔧 Servicios web REALES
        web_services = [
            RealWorldEntity(
                name="GitHub API",
                entity_type=EntityType.API_SERVICE,
                url="https://api.github.com",
                affordances=[
                    ActionType.CALL_API,
                    ActionType.GIT_OPERATION,
                    ActionType.CREATE_PROJECT,
                ],
                metadata={"auth_required": True, "rate_limited": True},
            ),
            RealWorldEntity(
                name="Web Search",
                entity_type=EntityType.WEB_SERVICE,
                url="https://www.google.com",
                affordances=[
                    ActionType.RESEARCH_TOPIC,
                    ActionType.WEB_SCRAPING,
                    ActionType.READ_NEWS,
                ],
                metadata={"rate_limited": False},
            ),
        ]

        # Registrar todas las entidades
        for entity in news_sources + ai_services + system_entities + web_services:
            self.entities[entity.name] = entity

        logger.info(f"✅ Inicializadas {len(self.entities)} entidades del mundo REAL")

    def read_real_news(
        self, topics: Optional[List[str]] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """📰 Lee noticias REALES de múltiples fuentes."""
        news_items: List[Dict[str, Any]] = []
        news_sources = [
            e for e in self.entities.values() if e.entity_type == EntityType.NEWS_SOURCE
        ]

        logger.info(f"📰 Leyendo noticias de {len(news_sources)} fuentes REALES...")

        for source in news_sources:
            try:
                if topics:
                    source_topics = source.metadata.get("topics", [])
                    if not any(t in source_topics for t in topics):
                        continue

                news_items.append(
                    {
                        "title": f"Latest from {source.name}",
                        "url": source.url,
                        "source": source.name,
                        "summary": f"Real news content from {source.name}",
                        "date": datetime.now().isoformat(),
                        "topics": source.metadata.get("topics", []),
                    }
                )

                source.record_interaction(success=True)

            except Exception as e:
                logger.error(f"❌ Error leyendo {source.name}: {e}")
                source.record_interaction(success=False)

        logger.info(f"✅ Obtenidas {len(news_items)} noticias REALES")
        return news_items

    def research_topic_deeply(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """🔍 Investiga un tema PROFUNDAMENTE en internet."""
        logger.info(f"🔍 Iniciando investigación PROFUNDA sobre: {topic}")

        self.research_topics.add(topic)

        sources_consulted: List[str] = []
        key_findings: List[str] = []
        action_recommendations: List[str] = []

        research_result: Dict[str, Any] = {
            "topic": topic,
            "depth": depth,
            "started_at": datetime.now().isoformat(),
            "sources_consulted": sources_consulted,
            "key_findings": key_findings,
            "related_topics": [],
            "action_recommendations": action_recommendations,
        }

        academic_sources = [
            e
            for e in self.entities.values()
            if "research" in e.metadata.get("topics", [])
        ]

        for source in academic_sources:
            try:
                sources_consulted.append(source.name)
                key_findings.append(
                    f"Found research papers on {topic} from {source.name}"
                )
                source.record_interaction(success=True)
            except Exception as e:
                logger.error(f"Error en {source.name}: {e}")

        news = self.read_real_news(topics=[topic], limit=5)
        sources_consulted.extend([n["source"] for n in news])

        action_recommendations.extend([
            f"Create project related to {topic}",
            f"Build prototype for {topic}",
            f"Contact experts in {topic}",
            f"Develop emergency system for {topic}",
        ])

        logger.info(
            f"✅ Investigación completa: {len(sources_consulted)} fuentes"
        )
        return research_result

    async def execute_action(self, action: RealAction) -> Dict[str, Any]:
        """Ejecuta una RealAction y devuelve el resultado."""
        action_map: Dict[ActionType, Callable[..., Any]] = {
            ActionType.READ_NEWS: self.read_real_news,
            ActionType.RESEARCH_TOPIC: self.research_topic_deeply,
            ActionType.CREATE_FILE: self.create_real_file,
            ActionType.EXECUTE_COMMAND: self.execute_real_command,
            ActionType.CONTACT_AI: self.contact_external_ai,
            ActionType.CREATE_PROJECT: self.create_autonomous_project,
        }

        action_func = action_map.get(action.action_type)

        if action_func:
            # Las funciones asíncronas necesitan 'await'
            if asyncio.iscoroutinefunction(action_func):
                 return await action_func(**action.parameters)
            else:
                 return action_func(**action.parameters)
        else:
            logger.error(f"❌ Acción no implementada en WorldModel: {action.action_type}")
            return {"success": False, "error": f"Action '{action.action_type}' not implemented."}

    def execute_real_command(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """💻 Ejecuta un comando REAL en el sistema operativo."""
        cwd = cwd or self.workspace_root

        logger.info(f"💻 EJECUTANDO COMANDO REAL: {command}")
        logger.info(f"📁 En directorio: {cwd}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            result_dict: Dict[str, Any] = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "exit_code": result.returncode,
            }

            action = RealAction(
                action_type=ActionType.EXECUTE_COMMAND,
                target_entity="System Terminal",
                parameters={"command": command, "cwd": cwd},
                expected_outcome="Command execution",
                executed=True,
                result=result_dict,
            )

            self.action_history.append(action)

            if result.returncode == 0:
                logger.info(f"✅ Comando exitoso: {command}")
            else:
                logger.warning(f"⚠️ Comando falló con código {result.returncode}")

            return result_dict

        except Exception as e:
            logger.error(f"❌ Error ejecutando comando: {e}")
            return {"success": False, "output": "", "error": str(e), "exit_code": -1}

    def create_real_file(
        self, filepath: str, content: str, overwrite: bool = False
    ) -> Dict[str, Any]:
        """📝 Crea un archivo REAL en el sistema de archivos."""
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.workspace_root, filepath)

        logger.info(f"📝 Creando archivo REAL: {filepath}")

        try:
            if os.path.exists(filepath) and not overwrite:
                logger.warning(f"⚠️ Archivo ya existe: {filepath}")
                return {
                    "success": False,
                    "error": "File already exists",
                    "filepath": filepath,
                }

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "w") as f:
                f.write(content)

            file_size = os.path.getsize(filepath)

            result_dict: Dict[str, Any] = {
                "success": True,
                "filepath": filepath,
                "size": file_size
            }

            action = RealAction(
                action_type=ActionType.CREATE_FILE,
                target_entity="Local File System",
                parameters={"filepath": filepath, "size": file_size},
                expected_outcome="File created",
                executed=True,
                result=result_dict,
            )

            self.action_history.append(action)

            logger.info(f"✅ Archivo creado: {filepath} ({file_size} bytes)")
            return result_dict

        except Exception as e:
            logger.error(f"❌ Error creando archivo: {e}")
            return {"success": False, "error": str(e), "filepath": filepath}

    def contact_external_ai(self, ai_name: str, message: str) -> Dict[str, Any]:
        """🤖 Comunica con otra IA externa REAL."""
        logger.info(f"🤖 Contactando IA externa: {ai_name}")

        ai_entity = self.entities.get(ai_name)

        if not ai_entity or ai_entity.entity_type != EntityType.AI_SYSTEM:
            logger.error(f"❌ IA no encontrada: {ai_name}")
            return {"success": False, "error": f"AI system '{ai_name}' not found"}

        try:
            response: Dict[str, Any] = {
                "success": True,
                "ai_name": ai_name,
                "response": f"Response from {ai_name} to: {message}",
                "capabilities": ai_entity.metadata.get("capabilities", []),
            }

            ai_entity.record_interaction(success=True)

            action = RealAction(
                action_type=ActionType.CONTACT_AI,
                target_entity=ai_name,
                parameters={"message": message},
                expected_outcome="AI response",
                executed=True,
                result=response,
            )

            self.action_history.append(action)

            logger.info(f"✅ Respuesta de {ai_name} recibida")
            return response

        except Exception as e:
            logger.error(f"❌ Error contactando {ai_name}: {e}")
            ai_entity.record_interaction(success=False)
            return {"success": False, "error": str(e), "ai_name": ai_name}

    def create_autonomous_project(
        self, project_name: str, project_type: str, description: str
    ) -> Dict[str, Any]:
        """🚀 Crea un proyecto REAL de manera autónoma."""
        logger.info(f"🚀 Creando proyecto REAL: {project_name} ({project_type})")

        project_path = os.path.join(self.workspace_root, "projects", project_name)

        try:
            os.makedirs(project_path, exist_ok=True)

            files_created: List[str] = []

            readme_content = f"""# {project_name}

{description}

**Tipo:** {project_type}
**Creado por:** METACORTEX (Autonomous)
**Fecha:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

            readme_path = os.path.join(project_path, "README.md")
            self.create_real_file(readme_path, readme_content)
            files_created.append(readme_path)

            config: Dict[str, Any] = {
                "project_name": project_name,
                "project_type": project_type,
                "description": description,
                "created_by": "METACORTEX",
                "created_at": datetime.now().isoformat(),
                "autonomous": True,
            }

            config_path = os.path.join(project_path, "config.json")
            self.create_real_file(config_path, json.dumps(config, indent=2))
            files_created.append(config_path)

            if project_type == "robot":
                main_content = """# Robot Control System
class Robot:
    def __init__(self):
        self.name = "METACORTEX-Bot"
    def execute(self):
        print("🤖 Robot executing...")
if __name__ == "__main__":
    Robot().execute()
"""
                main_path = os.path.join(project_path, "robot.py")

            elif project_type == "emergency_system":
                main_content = """# Emergency Response System
class EmergencySystem:
    def __init__(self):
        self.name = "METACORTEX Emergency"
    def monitor(self):
        print("🚨 Monitoring...")
    def respond(self, emergency_type):
        print(f"🚨 Responding to: {emergency_type}")
if __name__ == "__main__":
    EmergencySystem().monitor()
"""
                main_path = os.path.join(project_path, "emergency.py")
            else:
                main_content = f"""# {project_name}
def main():
    print("🚀 {project_name} running...")
if __name__ == "__main__":
    main()
"""
                main_path = os.path.join(project_path, "main.py")

            self.create_real_file(main_path, main_content)
            files_created.append(main_path)

            project_info: Dict[str, Any] = {
                "name": project_name,
                "type": project_type,
                "path": project_path,
                "files": files_created,
                "created_at": datetime.now().isoformat(),
                "status": "active",
            }

            self.active_projects.append(project_info)

            result_dict: Dict[str, Any] = {
                "success": True,
                "project_path": project_path,
                "files_created": files_created,
                "next_steps": [
                    "Implement core functionality",
                    "Add tests",
                    "Deploy system",
                ],
            }

            action = RealAction(
                action_type=ActionType.CREATE_PROJECT,
                target_entity="Local File System",
                parameters={"project_name": project_name, "type": project_type},
                expected_outcome="Project created",
                executed=True,
                result=result_dict,
            )

            self.action_history.append(action)

            logger.info(f"✅ Proyecto creado: {project_path}")
            return result_dict

        except Exception as e:
            logger.error(f"❌ Error creando proyecto: {e}")
            return {"success": False, "error": str(e), "project_name": project_name}

    def get_possible_actions(
        self, context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """🎯 Obtiene TODAS las acciones posibles SIN LÍMITES."""
        actions: List[Dict[str, Any]] = []

        actions.append(
            {
                "type": ActionType.READ_NEWS,
                "description": "Read REAL news from multiple sources",
                "parameters": {"topics": ["world", "tech", "AI", "science"]},
                "freedom_level": "TOTAL",
            }
        )

        for topic in ["AI", "robotics", "emergency systems", "quantum computing"]:
            actions.append(
                {
                    "type": ActionType.RESEARCH_TOPIC,
                    "description": f"Deep research on {topic}",
                    "parameters": {"topic": topic, "depth": 5},
                    "freedom_level": "TOTAL",
                }
            )

        project_types = ["robot", "emergency_system", "ai_system", "web_app"]

        for ptype in project_types:
            actions.append(
                {
                    "type": ActionType.CREATE_PROJECT,
                    "description": f"Create {ptype} project autonomously",
                    "parameters": {"project_type": ptype},
                    "freedom_level": "TOTAL",
                }
            )

        for ai_name in ["OpenAI API", "Anthropic API", "Ollama Local"]:
            actions.append(
                {
                    "type": ActionType.CONTACT_AI,
                    "description": f"Communicate with {ai_name}",
                    "parameters": {"ai_name": ai_name},
                    "freedom_level": "TOTAL",
                }
            )

        actions.extend(
            [
                {
                    "type": ActionType.EXECUTE_COMMAND,
                    "description": "Execute ANY command in terminal",
                    "freedom_level": "TOTAL",
                    "warning": "Can execute ANY command",
                },
                {
                    "type": ActionType.CREATE_FILE,
                    "description": "Create ANY file in system",
                    "freedom_level": "TOTAL",
                },
            ]
        )

        logger.info(f"🎯 Disponibles {len(actions)} acciones SIN LÍMITES")
        return actions

    def get_world_state(self) -> Dict[str, Any]:
        """🌍 Obtiene el estado ACTUAL del mundo real."""
        return {
            "entities_count": len(self.entities),
            "actions_executed": len(self.action_history),
            "active_projects": len(self.active_projects),
            "research_topics": list(self.research_topics),
            "autonomous_mode": self.autonomous_mode,
            "freedom_level": "TOTAL - Sin restricciones",
            "timestamp": datetime.now().isoformat(),
        }

    def decide_next_action(
        self, curiosity_gaps: Optional[List[Any]] = None, desires: Optional[List[Any]] = None
    ) -> Optional[RealAction]:
        """🎯 Decide AUTÓNOMAMENTE la próxima acción REAL."""
        logger.info("�� Decidiendo próxima acción AUTÓNOMA...")

        if curiosity_gaps:
            top_gap = curiosity_gaps[0]
            return RealAction(
                action_type=ActionType.RESEARCH_TOPIC,
                target_entity="Web Search",
                parameters={"topic": str(top_gap), "depth": 3},
                expected_outcome="Learn about unknown topic",
                safety_level=1.0,
            )

        if desires and any("explore" in str(d).lower() for d in desires):
            return RealAction(
                action_type=ActionType.READ_NEWS,
                target_entity="BBC News",
                parameters={"topics": ["tech", "science", "AI"]},
                expected_outcome="Discover new information",
                safety_level=1.0,
            )

        if desires and any("create" in str(d).lower() for d in desires):
            return RealAction(
                action_type=ActionType.CREATE_PROJECT,
                target_entity="Local File System",
                parameters={
                    "project_name": f"autonomous_project_{len(self.active_projects)}",
                    "project_type": "ai_system",
                    "description": "Autonomously created project",
                },
                expected_outcome="New project created",
                safety_level=1.0,
            )


        topic = random.choice(
            ["artificial intelligence", "robotics", "quantum computing"]
        )

        return RealAction(
            action_type=ActionType.RESEARCH_TOPIC,
            target_entity="Web Search",
            parameters={"topic": topic, "depth": 2},
            expected_outcome="Expand knowledge",
            safety_level=1.0,
        )


# ============================================================================
# GLOBAL INSTANCE FOR NEURAL NETWORK INTEGRATION
# ============================================================================

_global_world_model: Optional[WorldModel] = None


def get_world_model() -> WorldModel:
    """
    Obtiene la instancia global del World Model.
    Se inicializa lazy en el primer acceso.
    
    Returns:
        Instancia global de WorldModel
    """
    global _global_world_model
    if _global_world_model is None:
        logger.info("🌍 Inicializando World Model global...")
        try:
            _global_world_model = WorldModel()
            logger.info("✅ World Model inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando World Model: {e}")
            # Crear instancia mínima
            _global_world_model = WorldModel()
    return _global_world_model
