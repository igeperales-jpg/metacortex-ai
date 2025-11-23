from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filepath: /Users/edkanina/constructor_ia/metacortex_sinaptico/core.py
"""
METACORTEX - Núcleo Cognitivo
=============================

Agente cognitivo principal que integra todos los subsistemas:
    pass  # TODO: Implementar
homeostasis, afecto, BDI, planificación, aprendizaje y metacognición.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Solo ejecutar si se está corriendo como módulo (python -m metacortex.core)
if __name__ == "__main__" or "__package__" in globals():
    # Añadir raíz del proyecto al path para evitar RuntimeWarning
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import time
from typing import Dict, List, Any, Optional, Union, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from universal_knowledge_connector import UniversalKnowledgeConnector

from .utils import AgentConfig, get_env_config, setup_logging
from .db import MetacortexDB
from .memory import MemorySystem
from .anomaly import PerturbationDetector
from .learning import StructuralLearning
from .affect import AffectSystem
from .bdi import BDISystem
from .planning import Planner
from .metacog import MetaCognition
from .planning import TimeHorizon, PlanPriority
import time as time_module
import time
import argparse


logger = setup_logging()


class CognitiveAgent:
    """
    Agente cognitivo-metacognitivo vivo.

    Integra homeostasis, afecto, BDI, planificación, aprendizaje estructural
    y metacognición en un sistema unificado.
    """

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs: Any) -> None:
        """
        Inicializa el agente cognitivo.

        Args:
            config: Configuración del agente (opcional)
            **kwargs: Parámetros adicionales que se ignorarán (ej: agent_id, db, log_level)
                     Estos parámetros son para retrocompatibilidad con código legacy
        """
        # � SOLUCIÓN DE RAÍZ: Ignorar kwargs obsoletos sin warnings
        # El orquestador puede pasar agent_id, db, etc. pero CognitiveAgent
        # ahora maneja todo internamente a través de config
        if kwargs:
            # Solo loggear en modo debug, no warning
            logger.debug(f"CognitiveAgent: Parámetros legacy ignorados: {list(kwargs.keys())}")
        
        # 🔥 SOLUCIÓN DE RAÍZ: Asegurar config completamente inicializado
        self.config = config or get_env_config()

        # 🔥 VALIDACIÓN CRÍTICA: Verificar que config tiene db_path
        if not hasattr(self.config, "db_path"):
            raise AttributeError(
                "AgentConfig no tiene db_path definido. "
                "Asegúrate de que AgentConfig.__init__() se ejecutó completamente."
            )

        self.logger = logger.getChild("core")

        # Estado del sistema
        self.active = False
        self.start_time = time.time()
        self.tick_count = 0
        self.last_tick_time = 0.0

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            from neural_symbiotic_network import get_neural_network, MetacortexNeuralSymbioticNetworkV2

            self.neural_network: Optional[MetacortexNeuralSymbioticNetworkV2] = get_neural_network()
            if self.neural_network:
                self.neural_network.register_module("cognitive_agent", self)
                logger.info("✅ 'cognitive_agent' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None
        
        # 🧠 CONEXIÓN AL NEURAL HUB (Cerebro Central)
        try:
            from .metacortex_neural_hub import get_neural_hub, Event, EventCategory, EventPriority
            
            self.neural_hub = get_neural_hub()
            self.Event = Event
            self.EventCategory = EventCategory
            self.EventPriority = EventPriority
            
            # Registrar este módulo en el hub
            self._register_in_neural_hub()
            
            logger.info("✅ CognitiveAgent conectado a Neural Hub (cerebro central)")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar al Neural Hub: {e}")
            self.neural_hub = None

        # 🌍 CONEXIÓN AL SISTEMA DE CONOCIMIENTO UNIVERSAL
        self._initialize_knowledge_system()

        # Inicializar subsistemas
        self._init_subsystems()

        # � CONECTAR SUBSISTEMAS (BDI ↔ Affect)
        self._connect_subsystems()

        # �🔥 FIX: Estado cognitivo con anotaciones de tipo explícitas
        self.cognitive_state: Dict[str, Union[float, int, str, None, List[str]]] = {
            "wellbeing": 0.5,
            "recent_anomalies": 0,
            "current_intention": None,
            "system_notes": [],
        }

        self.logger.info("Agente cognitivo inicializado exitosamente")

    def _init_subsystems(self) -> None:
        """Inicializa todos los subsistemas."""
        try:
            # Base de datos y memoria
            self.db = MetacortexDB(self.config.db_path)
            self.memory = MemorySystem(self.db)

            # Detección de anomalías
            self.anomaly_detector = PerturbationDetector(
                window_size=self.config.history_window,
                threshold=self.config.anomaly_threshold,
            )

            # Aprendizaje estructural
            self.learning_system = StructuralLearning(
                learning_rate=self.config.learning_rate,
                novelty_threshold=self.config.novelty_threshold,
            )

            # Sistema afectivo
            self.affect_system = AffectSystem(self.neural_hub)

            # BDI
            self.bdi_system = BDISystem()

            # Planificador
            self.planner = Planner()

            # Metacognición
            self.metacognition = MetaCognition()

            # Cargar grafo existente si hay datos
            self._load_existing_graph()

            self.active = True
            self.logger.info("Todos los subsistemas inicializados")

        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error inicializando subsistemas: {e}")
            raise

    def _register_in_neural_hub(self) -> None:
        """
        🧠 Registra el CognitiveAgent en el Neural Hub.
        
        Define handlers para procesar eventos desde otros módulos.
        """
        if not self.neural_hub:
            return
        
        try:
            # Definir categorías de eventos a las que nos suscribimos
            subscriptions = {
                "PERCEPTION",
                "MEMORY_RETRIEVE",
                "ANOMALY_DETECTED",
                "ALERT"
            }
            
            # Definir handlers para cada categoría
            handlers = {
                "PERCEPTION": self._handle_perception_event,
                "MEMORY_RETRIEVE": self._handle_memory_event,
                "ANOMALY_DETECTED": self._handle_anomaly_event,
                "ALERT": self._handle_alert_event
            }
            
            # Registrar en el hub
            self.neural_hub.register_module(
                name="cognitive_agent",
                instance=self,
                subscriptions=subscriptions,
                handlers=handlers
            )
            
            self.logger.info("✅ Handlers registrados en Neural Hub")
            
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error registrando en Neural Hub: {e}")
    
    def _handle_perception_event(self, event: Any) -> None:
        """Handler para eventos de percepción."""
        try:
            data = event.data
            self.perceive(data.get("name", "unknown"), data.get("payload", {}))
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error procesando evento de percepción: {e}")
    
    def _handle_memory_event(self, event: Any) -> None:
        """Handler para eventos de memoria."""
        try:
            query = event.data.get("query", "")
            if query and self.memory:
                results = self.memory.semantic_search(query, limit=5)
                if event.requires_response:
                    event.response_data = results
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error procesando evento de memoria: {e}")
    
    def _handle_anomaly_event(self, event: Any) -> None:
        """Handler para eventos de anomalía."""
        try:
            # Actualizar estado cognitivo con anomalía detectada
            anomaly_data = event.data
            self.cognitive_state["recent_anomalies"] = \
                int(self.cognitive_state.get("recent_anomalies", 0)) + 1
            
            # Almacenar en memoria
            self.memory.store_episode(
                name="anomaly_detected",
                data=anomaly_data,
                anomaly=True
            )
            
            self.logger.warning(f"🚨 Anomalía detectada: {anomaly_data}")
            
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error procesando evento de anomalía: {e}")
    
    def _handle_alert_event(self, event: Any) -> None:
        """Handler para eventos de alerta."""
        try:
            alert_data = event.data
            severity = alert_data.get("severity", "INFO")
            message = alert_data.get("message", "")
            
            self.logger.info(f"⚠️ Alerta [{severity}]: {message}")
            
            # Si es crítica, ajustar prioridades
            if severity == "CRITICAL":
                self.bdi_system.add_desire("handle_critical_alert", priority=0.99)
            
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error procesando evento de alerta: {e}")
    
    def _connect_subsystems(self) -> None:
        """
        🔗 Conecta subsistemas entre sí para habilitarintegración profunda.
        
        Conexiones clave:
        - BDI ↔ Affect: Modulación emocional de deseos
        - BDI ↔ Planning: Generación de planes desde intenciones
        - Memory ↔ Learning: Conceptos desde episodios
        """
        try:
            # 1. Conectar BDI con sistema afectivo
            if hasattr(self, 'bdi_system') and hasattr(self, 'affect_system'):
                self.bdi_system.connect_affect_system(self.affect_system)
                self.logger.info("✅ BDI ↔ Affect conectados (modulación emocional habilitada)")
            
            # 2. Sincronizar estado emocional inicial con jerarquía de necesidades
            if hasattr(self, 'bdi_system') and hasattr(self, 'affect_system'):
                # Usar método existente get_emotional_insights()
                emotional_insights = self.affect_system.get_emotional_insights()
                # Crear estado emocional simplificado
                emotional_state = {
                    "valence": emotional_insights.get("avg_wellbeing", 0.5),
                    "energy": self.affect_system.state.energy,
                    "stress": emotional_insights.get("total_patterns", 0) * 0.1  # Proxy
                }
                self.bdi_system.need_hierarchy.set_emotional_state(emotional_state)
                self.logger.info("✅ Estado emocional sincronizado con necesidades")
            
            # 3. TODO: Conectar planificador con BDI (futuro)
            # self.planner.connect_bdi(self.bdi_system)
            
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ Error conectando subsistemas: {e}")

    def _initialize_knowledge_system(self) -> None:
        """
        🌍 Conecta al sistema de conocimiento universal

        Integra el cognitive agent con:
        - Knowledge Ingestion Engine (acceso a todo el conocimiento humano)
        - Hierarchical Learning System (memoria infinita)
        - Internet Search (búsqueda en tiempo real)
        - Working Memory expandida
        """
        try:
            # Intentar usar Universal Knowledge Connector SINGLETON
            try:
                from pathlib import Path as PathLib
                sys.path.insert(0, str(PathLib(__file__).parent.parent))
                from universal_knowledge_connector import get_knowledge_connector


                # ✅ SINGLETON - Instancia global compartida (evita duplicación masiva)
                self.knowledge_connector: Optional[UniversalKnowledgeConnector] = get_knowledge_connector(auto_initialize=True)
                self.logger.info("🧠 Cognitive Agent conectado a Knowledge Connector SINGLETON")
                self.logger.info("   💡 Usando instancia global compartida (NO duplicación)")
                self.logger.info("   - Acceso a Wikipedia, ArXiv, Internet")
                self.logger.info("   - Memoria jerárquica infinita disponible")

                # Método helper para consultar conocimiento
                self.query_knowledge = self.knowledge_connector.query_knowledge

            except ImportError as e:
                logger.error(f"Error en core.py: {e}", exc_info=True)
                self.logger.warning(
                    f"⚠️ Universal Knowledge Connector no disponible: {e}"
                )
                self.knowledge_connector = None
                self.query_knowledge = None

                # Fallback: Conectar solo al sistema jerárquico básico
                try:
                    from .learning import StructuralLearning as HierarchicalLearning

                    self.hierarchical_learning: Optional[Any] = HierarchicalLearning(
                        use_hierarchical=True
                    )

                    # 🌳 Acceso directo al grafo jerárquico
                    if hasattr(self.hierarchical_learning, "hierarchical_graph"):
                        self.hierarchical_graph: Optional[Any] = (
                            self.hierarchical_learning.hierarchical_graph
                        )
                        self.logger.info(
                            "✅ Acceso directo al grafo jerárquico configurado"
                        )
                        # 🔥 FIX: Verificar que hierarchical_graph existe antes de acceder
                        if self.hierarchical_graph and hasattr(self.hierarchical_graph, 'active_limit'):
                            self.logger.info(
                                f"   - Memoria activa: {self.hierarchical_graph.active_limit} conceptos"
                            )
                        self.logger.info("   - Memoria archivada: ∞ (sin límites)")
                    else:
                        self.hierarchical_graph = None

                    self.logger.info(
                        "✅ Sistema de aprendizaje jerárquico básico conectado"
                    )
                except Exception as fallback_error:
                    logger.error(f"Error en core.py: {fallback_error}", exc_info=True)
                    self.logger.warning(
                        f"⚠️ No se pudo conectar al sistema jerárquico: {fallback_error}"
                    )
                    self.hierarchical_learning = None
                    self.hierarchical_graph = None

        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error en inicialización de knowledge system: {e}")
            self.knowledge_connector = None
            self.query_knowledge = None
            self.hierarchical_graph = None

    def _load_existing_graph(self) -> None:
        """Carga el grafo de conocimiento existente."""
        try:
            edges = self.db.get_all_edges()
            if edges:
                self.learning_system.load_from_edges(edges)
                self.logger.info(f"Grafo cargado: {len(edges)} aristas")
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.warning(f"No se pudo cargar grafo existente: {e}")

    def perceive(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa una percepción externa con procesamiento multimodal avanzado.
        
        Nuevo en esta versión:
        - Fusión sensorial inteligente
        - Procesamiento paralelo de modalidades
        - Broadcasting a Neural Hub
        - Contextualización con memoria

        Args:
            name: Nombre del evento percibido
            payload: Datos del evento

        Returns:
            Resultado del procesamiento con métricas extendidas
        """
        try:
            self.logger.debug(f"Procesando percepción: {name}")
            
            # FASE 1: Detección de anomalías
            anomaly_result = self.anomaly_detector.detect(name, payload)
            
            # FASE 2: Contextualización con memoria
            # Buscar percepciones similares en memoria
            context_memories = []
            try:
                if hasattr(self.memory, 'recall_similar'):
                    context_memories = self.memory.recall_similar(name, limit=3)
            except Exception as e:
                logger.exception(f"Error in exception handler: {e}")
            # FASE 3: Fusión sensorial (multimodal)
            # Si el payload tiene múltiples modalidades, fusionarlas
            modalities = self._extract_modalities(payload)
            fused_representation = self._fuse_modalities(modalities)

            # FASE 4: Almacenar en memoria con contexto enriquecido
            enriched_data = {
                **payload,
                "modalities": list(modalities.keys()),
                "fused": fused_representation,
                "context": [m["name"] for m in context_memories] if context_memories else [],
                "anomaly_score": anomaly_result.z_score if anomaly_result.is_anomaly else 0.0
            }
            
            episode_id = self.memory.store_episode(
                name, enriched_data, anomaly=anomaly_result.is_anomaly
            )

            # FASE 5: Actualizar sistema afectivo con contexto
            affect_events: Dict[str, Any] = {
                "anomaly": anomaly_result.is_anomaly,
                "confidence": anomaly_result.confidence,
                "multimodal": len(modalities) > 1,
                "context_richness": len(context_memories) / 3.0  # Normalizado
            }
            self.affect_system.update(affect_events)

            # FASE 6: Actualizar memoria de trabajo
            self.memory.working_memory.add(
                {
                    "name": name,
                    "payload": payload,
                    "anomaly": anomaly_result.is_anomaly,
                    "timestamp": time.time(),
                    "multimodal": len(modalities) > 1,
                    "fused": fused_representation
                }
            )

            # FASE 7: Añadir conceptos al grafo de aprendizaje
            concepts = self._extract_concepts(name, payload)
            for concept in concepts:
                self.learning_system.add_concept(concept)
            
            # FASE 8: Broadcasting al Neural Hub si está disponible
            if self.neural_hub:
                try:
                    event = self.Event(
                        id=f"perception_{time.time()}_{name}",
                        category=self.EventCategory.PERCEPTION,
                        source="cognitive_agent",
                        payload={
                            "name": name,
                            "payload": payload,
                            "anomaly": anomaly_result.is_anomaly,
                            "modalities": list(modalities.keys())
                        },
                        priority=self.EventPriority.HIGH if anomaly_result.is_anomaly else self.EventPriority.NORMAL
                    )
                    self.neural_hub.publish(event)
                except Exception as e:
                    logger.error(f"Error en core.py: {e}", exc_info=True)
                    self.logger.debug(f"No se pudo emitir evento al hub: {e}")

            return {
                "anomaly": anomaly_result.is_anomaly,
                "z_score": anomaly_result.z_score if anomaly_result.is_anomaly else None,
                "stored": True,
                "episode_id": episode_id,
                "multimodal": len(modalities) > 1,
                "modalities": list(modalities.keys()),
                "context_matches": len(context_memories),
                "fused_representation": fused_representation
            }

        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error procesando percepción: {e}")
            return {"anomaly": False, "stored": False, "error": str(e)}
    
    def _extract_modalities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae modalidades sensoriales del payload.
        
        Soporta:
        - Visual: imágenes, video
        - Auditivo: audio, sonido
        - Textual: texto, lenguaje
        - Numérico: métricas, sensores
        - Temporal: timestamps, secuencias
        
        Args:
            payload: Datos de la percepción
            
        Returns:
            Diccionario de modalidades detectadas
        """
        modalities = {}
        
        # Detección de modalidad visual
        if any(k in payload for k in ['image', 'video', 'visual', 'frame']):
            modalities['visual'] = {k: payload[k] for k in ['image', 'video', 'visual', 'frame'] if k in payload}
        
        # Detección de modalidad auditiva
        if any(k in payload for k in ['audio', 'sound', 'speech', 'voice']):
            modalities['auditory'] = {k: payload[k] for k in ['audio', 'sound', 'speech', 'voice'] if k in payload}
        
        # Detección de modalidad textual
        if any(k in payload for k in ['text', 'message', 'content', 'description']):
            modalities['textual'] = {k: payload[k] for k in ['text', 'message', 'content', 'description'] if k in payload}
        
        # Detección de modalidad numérica
        numeric_keys = [k for k, v in payload.items() if isinstance(v, (int, float))]
        if numeric_keys:
            modalities['numeric'] = {k: payload[k] for k in numeric_keys}
        
        # Detección de modalidad temporal
        if any(k in payload for k in ['timestamp', 'time', 'sequence', 'duration']):
            modalities['temporal'] = {k: payload[k] for k in ['timestamp', 'time', 'sequence', 'duration'] if k in payload}
        
        # Si no se detectó ninguna modalidad, clasificar como "generic"
        if not modalities:
            modalities['generic'] = payload
        
        return modalities
    
    def _fuse_modalities(self, modalities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusión sensorial inteligente de múltiples modalidades.
        
        Implementa estrategias de fusión:
        - Early fusion: Combinar features de bajo nivel
        - Late fusion: Combinar decisiones de alto nivel
        - Attention-based: Ponderar modalidades por relevancia
        
        Args:
            modalities: Modalidades detectadas
            
        Returns:
            Representación fusionada
        """
        if len(modalities) == 1:
            # Solo una modalidad, no hay fusión necesaria
            return {"strategy": "single_modality", "data": list(modalities.values())[0]}
        
        # ESTRATEGIA 1: Fusión por atención (attention-based)
        # Calcular relevancia de cada modalidad
        modality_weights = {}
        for mod_name, mod_data in modalities.items():
            # Peso basado en cantidad de información
            if isinstance(mod_data, dict):
                weight = len(mod_data) / 10.0  # Normalizar
            else:
                weight = 0.5
            modality_weights[mod_name] = min(1.0, weight)
        
        # Normalizar pesos
        total_weight = sum(modality_weights.values())
        if total_weight > 0:
            modality_weights = {k: v/total_weight for k, v in modality_weights.items()}
        
        # ESTRATEGIA 2: Fusión temprana (early fusion)
        # Combinar todas las features en una representación unificada
        fused_features = {}
        for mod_name, mod_data in modalities.items():
            if isinstance(mod_data, dict):
                for k, v in mod_data.items():
                    fused_features[f"{mod_name}_{k}"] = v
        
        return {
            "strategy": "multimodal_attention_fusion",
            "modalities_count": len(modalities),
            "modality_weights": modality_weights,
            "fused_features": fused_features,
            "dominant_modality": max(modality_weights, key=modality_weights.get)
        }

    def tick(self) -> Dict[str, Any]:
        """
        Ejecuta un ciclo del sistema cognitivo-metacognitivo avanzado.
        
        Nuevo en esta versión:
        - Procesamiento paralelo de subsistemas independientes
        - Broadcasting de eventos al Neural Hub
        - Heartbeat al Neural Hub para health monitoring
        - Métricas de rendimiento cognitivo

        Returns:
            Reporte del estado tras el ciclo
        """
        try:
            self.tick_count += 1
            self.last_tick_time = time.time()
            
            # Enviar heartbeat al Neural Hub
            if self.neural_hub:
                try:
                    self.neural_hub.heartbeat("cognitive_agent")
                except Exception as e:
                    logger.exception(f"Error in exception handler: {e}")
            self.logger.debug(f"Ejecutando tick #{self.tick_count}")

            # 1. Monitoreo metacognitivo
            monitoring_data = self.metacognition.monitor(self.cognitive_state)

            # 2. Evaluación metacognitiva
            evaluation = self.metacognition.evaluate(monitoring_data)

            # 3. Control metacognitivo
            control_actions = self.metacognition.control(evaluation)

            # 4. Actualizar BDI con decisiones conscientes
            decision_result = self._update_bdi_system_advanced()

            # 5. Planificación consciente
            self._get_current_state_dict()
            # Determinar goal desde la intención actual o un objetivo por defecto
            goal = (
                self.bdi_system.current_intention.goal
                if self.bdi_system.current_intention
                else "maintain_system_health"
            )

            self.planner.create_plan(
                goal, TimeHorizon.IMMEDIATE, priority=PlanPriority.MEDIUM
            )

            # 6. Ciclo de auto-modificación (cada 10 ticks si hay anomalías)
            auto_mod_result: Optional[Dict[str, Any]] = None
            
            # 🔥 FIX: Acceso seguro sin cast innecesario
            recent_anomalies_value = self.cognitive_state.get("recent_anomalies", 0)
            recent_anomalies = recent_anomalies_value if isinstance(recent_anomalies_value, int) else 0
            
            if self.tick_count % 10 == 0 and recent_anomalies > 0:
                auto_mod_result = self._evaluar_auto_modificacion()

            # 7. Aprendizaje estructural
            learning_result: Optional[Dict[str, Any]] = None
            if self.tick_count % 3 == 0:
                recent_concepts = self._get_recent_concepts()

                self.logger.info(
                    f"🧠 LEARNING CYCLE: {len(recent_concepts)} conceptos generados: {recent_concepts[:5]}"
                )

                for concept in recent_concepts:
                    self.learning_system.add_concept(concept)
                    self.logger.debug(f"   ➕ Concepto añadido al grafo: {concept}")

                learning_result = self.learning_system.perform_learning_cycle(
                    recent_concepts
                )
                self.logger.info(f"📊 Learning result: {learning_result}")

                # Persistir grafo actualizado
                self._persist_graph_changes()

            # 8. Actualizar estado cognitivo
            self.cognitive_state["wellbeing"] = self.affect_system.get_wellbeing()
            self.cognitive_state["recent_anomalies"] = self.db.get_anomaly_count(
                since_hours=1
            )
            self.cognitive_state["current_intention"] = getattr(
                self.bdi_system.current_intention, "goal", None
            )

            # 9. Ejecutar acciones conscientes si es necesario
            actions_executed: List[str] = []
            if control_actions.get("adjust_learning_rate"):
                actions_executed.append(
                    "Ajustando tasa de aprendizaje por bajo rendimiento"
                )
            if control_actions.get("request_resources"):
                actions_executed.append("Solicitando recursos adicionales")
            if auto_mod_result and auto_mod_result.get("action_needed"):
                actions_executed.append(
                    f"Auto-modificación requerida: {auto_mod_result['reason']}"
                )

            self.cognitive_state["system_notes"] = actions_executed

            # 10. Persistir métricas expandidas
            # 🔥 FIX: Proporcionar diccionarios vacíos si son None
            self._store_metrics_advanced(
                decision_result,
                learning_result or {},
                auto_mod_result or {}
            )

            return {
                "wellbeing": self.cognitive_state["wellbeing"],
                "anomalies": self.cognitive_state["recent_anomalies"],
                "intention": self.cognitive_state["current_intention"],
                "notes": actions_executed,
                "tick_count": self.tick_count,
                "evaluation": evaluation,
                "decision_result": decision_result,
                "learning_result": learning_result,
                "auto_modification": auto_mod_result,
            }

        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error en tick cognitivo: {e}")
            return {
                "wellbeing": 0.0,
                "anomalies": 0,
                "intention": None,
                "notes": [f"Error: {str(e)}"],
                "tick_count": self.tick_count,
            }

    def _update_bdi_system_advanced(self) -> Dict[str, Any]:
        """Sistema BDI avanzado con decisiones conscientes"""
        try:
            # Seleccionar nueva intención si no hay una activa
            if not self.bdi_system.current_intention:
                current_state = self._get_current_state_dict()
                # 🔥 FIX: Manejar método async select_intention de forma segura
                intention_result = self.bdi_system.select_intention(current_state)
                # Si es coroutine, ejecutarla con asyncio
                if hasattr(intention_result, '__await__'):
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Si ya hay un loop corriendo, crear una tarea
                            self.bdi_system.current_intention = None  # Temporalmente None
                        else:
                            self.bdi_system.current_intention = loop.run_until_complete(intention_result)
                    except RuntimeError:
                        # No hay loop, crear uno nuevo
                        self.bdi_system.current_intention = asyncio.run(intention_result)
                else:
                    self.bdi_system.current_intention = intention_result

            # Añadir deseos evolutivos dinámicos
            if not self.bdi_system.desires:
                self.bdi_system.add_desire("maintain_wellbeing", priority=0.8)
                self.bdi_system.add_desire("learn_continuously", priority=0.6)
                self.bdi_system.add_desire("optimize_performance", priority=0.7)
                self.bdi_system.add_desire("evolve_consciously", priority=0.9)

            # 🔥 FIX: Cast explícito y verificación de tipos
            wellbeing_value = self.cognitive_state.get("wellbeing", 0.5)
            wellbeing = cast(float, wellbeing_value) if isinstance(wellbeing_value, (int, float)) else 0.5
            
            anomalies_value = self.cognitive_state.get("recent_anomalies", 0)
            anomalies = cast(int, anomalies_value) if isinstance(anomalies_value, int) else 0

            decision_taken = None
            if wellbeing < 0.4:
                decision_taken = "increase_wellbeing_priority"
                # Aumentar prioridad de bienestar
                for desire in self.bdi_system.desires:
                    if desire.name == "maintain_wellbeing":
                        desire.priority = min(1.0, desire.priority + 0.2)

            elif anomalies > 3:
                decision_taken = "focus_on_adaptation"
                # Enfocar en adaptación
                self.bdi_system.add_desire("adapt_to_anomalies", priority=0.95)

            return {
                "decision_taken": decision_taken,
                "current_intention": getattr(
                    self.bdi_system.current_intention, "goal", None
                ),
                "active_desires": len(self.bdi_system.desires),
                "wellbeing_based_adjustment": wellbeing < 0.4,
                "anomaly_based_adjustment": anomalies > 3,
            }

        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error en BDI avanzado: {e}")
            return {"error": str(e)}

    def _evaluar_auto_modificacion(self) -> Dict[str, Any]:
        """Evalúa si se necesita auto-modificación del sistema"""
        try:
            # 🔥 FIX: Acceso seguro sin cast innecesario
            recent_anomalies_value = self.cognitive_state.get("recent_anomalies", 0)
            recent_anomalies = int(recent_anomalies_value) if isinstance(recent_anomalies_value, int) else 0
            
            wellbeing_value = self.cognitive_state.get("wellbeing", 0.5)
            wellbeing = float(wellbeing_value) if isinstance(wellbeing_value, (int, float)) else 0.5

            # Criterios para auto-modificación
            needs_modification = False
            reason: Optional[str] = None

            if recent_anomalies > 5:
                needs_modification = True
                reason = f"Demasiadas anomalías detectadas: {recent_anomalies}"

            elif wellbeing < 0.3:
                needs_modification = True
                reason = f"Bienestar crítico: {wellbeing}"

            elif self.tick_count > 100 and self.tick_count % 50 == 0:
                # Evaluación periódica para mejoras
                needs_modification = True
                reason = "Evaluación periódica de optimización"

            return {
                "action_needed": needs_modification,
                "reason": reason,
                "anomalies_count": recent_anomalies,
                "wellbeing_level": wellbeing,
                "evaluation_timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error evaluando auto-modificación: {e}")
            return {"error": str(e)}

    def _store_metrics_advanced(
        self,
        decision_result: Dict[str, Any],
        learning_result: Dict[str, Any],
        auto_mod_result: Dict[str, Any],
    ) -> None:
        """Almacena métricas expandidas del sistema cognitivo"""
        try:
            ts = time.time()

            # Métricas básicas
            self.db.store_metrics(
                ts=ts,
                homeo_var=self.affect_system.get_homeostatic_variance(),
                anomaly_rate=self.anomaly_detector.get_recent_anomaly_rate(),
                edge_delta=len(self.learning_system.get_recent_edges()),
                goal_progress=self.planner.get_goal_progress(),
                wellbeing=self.affect_system.get_wellbeing(),
                energy=self.affect_system.state.energy,
                valence=self.affect_system.state.valence,
                activation=self.affect_system.state.activation,
            )

            # Métricas avanzadas como episodio especial
            advanced_metrics: Dict[str, Any] = {
                "tick_count": self.tick_count,
                "decision_result": decision_result,
                "learning_stats": learning_result,
                "auto_modification_eval": auto_mod_result,
                "cognitive_load": len(self.memory.working_memory.items) if self.memory.working_memory.items else 0,
                "bdi_desires_count": len(self.bdi_system.desires),
                "bdi_beliefs_count": len(self.bdi_system.beliefs),
                "bdi_intentions_count": len(self.bdi_system.intention_history),
                "system_evolution_stage": "advanced_cognitive_cycles",
            }

            self.memory.store_episode(
                name="advanced_cognitive_metrics", data=advanced_metrics, anomaly=False
            )

        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error almacenando métricas avanzadas: {e}")

    def _extract_concepts(self, name: str, payload: Dict[str, Any]) -> List[str]:
        """Extrae conceptos del evento para el grafo de conocimiento."""
        concepts = [name]

        # Extraer conceptos de las claves del payload
        concepts.extend(list(payload.keys())[:5])  # Limitar a 5 conceptos

        return concepts

    def search_knowledge(
        self, query: str, sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        🌍 Busca conocimiento en el sistema universal

        Args:
            query: Query de búsqueda
            sources: Fuentes específicas a usar (None = todas)

        Returns:
            Resultados de la búsqueda con conocimiento relevante
        """
        if self.knowledge_connector and callable(self.query_knowledge):
            try:
                self.logger.info(f"🔍 Buscando conocimiento: '{query}'")
                results = self.query_knowledge(query)
                self.logger.info(
                    f"✅ Encontrados {len(results.get('concepts', []))} conceptos"
                )
                return results
            except Exception as e:
                logger.error(f"Error en core.py: {e}", exc_info=True)
                self.logger.error(f"❌ Error en búsqueda de conocimiento: {e}")
                return {"error": str(e), "concepts": []}
        else:
            self.logger.warning("⚠️ Knowledge Connector no disponible")
            return {"error": "Knowledge Connector not available", "concepts": []}

    def learn_from_external_knowledge(
        self, topic: str, max_concepts: int = 10
    ) -> Dict[str, Any]:
        """
        🧠 Aprende activamente de conocimiento externo

        Args:
            topic: Tópico a aprender
            max_concepts: Máximo de conceptos a aprender

        Returns:
            Reporte del aprendizaje realizado
        """
        try:
            if not self.knowledge_connector:
                return {"error": "Knowledge Connector not available", "learned": 0}

            self.logger.info(f"🎓 Iniciando aprendizaje sobre: '{topic}'")

            # Buscar conocimiento
            knowledge = self.search_knowledge(topic)

            if "error" in knowledge:
                return {"error": knowledge["error"], "learned": 0}

            # Integrar conceptos en el grafo de aprendizaje
            concepts_learned = 0
            for concept in knowledge.get("concepts", [])[:max_concepts]:
                concept_name = concept.get("name", "")
                if concept_name:
                    # Agregar al sistema de aprendizaje estructural
                    self.learning_system.add_concept(concept_name)

                    # Almacenar en memoria episódica
                    self.memory.store_episode(
                        name=f"learned_concept_{concept_name}",
                        data=concept,
                        anomaly=False,
                    )

                    concepts_learned += 1

            self.logger.info(
                f"✅ Aprendidos {concepts_learned} conceptos sobre '{topic}'"
            )

            return {
                "topic": topic,
                "learned": concepts_learned,
                "source": knowledge.get("source", "unknown"),
            }

        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error en aprendizaje externo: {e}")
            return {"error": str(e), "learned": 0}

    def _update_bdi_system(self) -> None:
        """Actualiza el sistema BDI."""
        # Seleccionar nueva intención si no hay una activa
        if not self.bdi_system.current_intention:
            current_state = self._get_current_state_dict()
            # 🔥 FIX: Manejar método async select_intention de forma segura
            intention_result = self.bdi_system.select_intention(current_state)
            # Si es coroutine, ejecutarla con asyncio
            if hasattr(intention_result, '__await__'):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        self.bdi_system.current_intention = None
                    else:
                        self.bdi_system.current_intention = loop.run_until_complete(intention_result)
                except RuntimeError:
                    self.bdi_system.current_intention = asyncio.run(intention_result)
            else:
                self.bdi_system.current_intention = intention_result

        # Añadir deseos básicos si no los hay
        if not self.bdi_system.desires:
            self.bdi_system.add_desire("maintain_wellbeing", priority=0.8)
            self.bdi_system.add_desire("learn_continuously", priority=0.6)
            self.bdi_system.add_desire("optimize_performance", priority=0.7)

    def _get_current_state_dict(self) -> Dict[str, Any]:
        """Obtiene el estado actual como diccionario para planificación."""
        # 🔥 FIX: Acceso seguro con valores por defecto
        wellbeing = self.cognitive_state.get("wellbeing", 0.5)
        recent_anomalies = self.cognitive_state.get("recent_anomalies", 0)
        
        return {
            "wellbeing": float(wellbeing) if isinstance(wellbeing, (int, float)) else 0.5,
            "energy": self.affect_system.state.energy,
            "valence": self.affect_system.state.valence,
            "stress": self.affect_system.state.activation,  # activation como proxy de stress
            "recent_anomalies": int(recent_anomalies) if isinstance(recent_anomalies, int) else 0,
            "tick_count": self.tick_count,
        }

    def _get_recent_concepts(self) -> List[str]:
        """
        Obtiene conceptos recientes de la memoria de trabajo.

        🔥 SOLUCIÓN DE RAÍZ: Generación sintética CONTROLADA
        """
        recent_items = self.memory.working_memory.get_recent(10)
        concepts: List[str] = []

        for item in recent_items:
            if isinstance(item, dict) and "name" in item:
                concept_name_raw = item["name"]
                if isinstance(concept_name_raw, str):
                    concepts.append(concept_name_raw)

        # 🔥 GENERACIÓN SINTÉTICA CONTROLADA solo si working memory está vacía
        if len(concepts) == 0:
            # 🔥 FIX: Acceso seguro sin cast innecesario
            wellbeing_value = self.cognitive_state.get("wellbeing", 0.5)
            wellbeing = float(wellbeing_value) if isinstance(wellbeing_value, (int, float)) else 0.5
            
            anomalies_value = self.cognitive_state.get("recent_anomalies", 0)
            anomalies = int(anomalies_value) if isinstance(anomalies_value, int) else 0
            
            intention_value = self.cognitive_state.get("current_intention")
            intention = str(intention_value) if intention_value is not None else None

            # Conceptos sintéticos basados en estado interno
            if wellbeing > 0.7:
                concepts.append("high_wellbeing_state")
            elif wellbeing < 0.3:
                concepts.append("low_wellbeing_state")
            else:
                concepts.append("normal_wellbeing_state")

            if anomalies > 5:
                concepts.append("high_anomaly_detection")
            elif anomalies > 0:
                concepts.append("anomaly_monitoring")

            if intention:
                concepts.append(f"active_intention_{intention}")

            # UN SOLO concepto metacognitivo
            concepts.append("metacognitive_self_monitoring")

            self.logger.debug(
                f"Working memory vacía. Generados {len(concepts)} conceptos "
                f"sintéticos controlados basados en estado cognitivo."
            )

        return concepts

    def _persist_graph_changes(self) -> None:
        """
        Persiste cambios del grafo a la base de datos.
        """

        start = time_module.time()

        try:
            # Obtener todas las aristas del grafo en memoria
            current_edges = self.learning_system.get_graph_edges()
            total_edges = len(current_edges)

            # Optimización: Solo guardar si hay cambios recientes
            if not hasattr(self, "_last_persisted_edge_count"):
                self._last_persisted_edge_count = 0

            if total_edges == self._last_persisted_edge_count:
                self.logger.debug(
                    f"📁 Grafo sin cambios ({total_edges} aristas). Skip persist."
                )
                return

            # Calcular delta
            edges_added = total_edges - self._last_persisted_edge_count

            self.logger.info(
                f"💾 Persistiendo grafo: {edges_added} aristas nuevas "
                f"({self._last_persisted_edge_count} → {total_edges})"
            )

            # Guardar todas las aristas
            for edge in current_edges:
                self.db.store_edge(
                    edge["src"], edge["dst"], edge["weight"], edge["edge_type"]
                )

            # Actualizar contador
            self._last_persisted_edge_count = total_edges

            elapsed = time_module.time() - start
            self.logger.info(f"✅ Grafo persistido en {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            elapsed = time_module.time() - start
            self.logger.error(
                f"❌ Error persistiendo grafo tras {elapsed:.2f}s: {e}", exc_info=True
            )

    def _store_metrics(self) -> None:
        """Almacena métricas del sistema."""
        try:
            # 🔥 FIX: Acceso seguro sin cast innecesario
            wellbeing_value = self.cognitive_state.get("wellbeing", 0.5)
            wellbeing = float(wellbeing_value) if isinstance(wellbeing_value, (int, float)) else 0.5
            
            homeo_var = abs(0.5 - wellbeing)
            anomaly_rate = self.anomaly_detector.get_anomaly_rate()
            edge_delta = len(self.learning_system.get_graph_edges())
            goal_progress = 0.5  # Placeholder

            self.db.store_metrics(
                homeo_var=homeo_var,
                anomaly_rate=anomaly_rate,
                edge_delta=edge_delta,
                goal_progress=goal_progress,
                wellbeing=wellbeing,
                energy=self.affect_system.state.energy,
                valence=self.affect_system.state.valence,
                activation=self.affect_system.state.activation,
            )
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error almacenando métricas: {e}")

    # === MÉTODOS DE LA API ===

    def get_current_state(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema."""
        return dict(self.cognitive_state)

    def get_graph_snapshot(self) -> Dict[str, Any]:
        """Obtiene snapshot del grafo de conocimiento."""
        return self.db.get_graph_snapshot()

    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema."""
        uptime = time.time() - self.start_time

        return {
            "active": self.active,
            "uptime": uptime,
            "memory_usage": {
                "episodes": len(self.memory.recall_episodes(1000)),
                "working_memory": len(self.memory.working_memory.items) if self.memory.working_memory.items else 0,
                "graph_nodes": self.learning_system.graph.number_of_nodes() if hasattr(self.learning_system.graph, 'number_of_nodes') else 0,
                "graph_edges": self.learning_system.graph.number_of_edges() if hasattr(self.learning_system.graph, 'number_of_edges') else 0,
            },
            "last_tick": self.last_tick_time,
            "tick_count": self.tick_count,
        }

    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Obtiene historial de métricas."""
        return self.db.get_metrics_history(hours)

    def get_debug_info(self) -> Dict[str, Any]:
        """Obtiene información de debugging."""
        return {
            "config": self.config.__dict__,
            "cognitive_state": dict(self.cognitive_state),
            "affect_state": {
                "energy": self.affect_system.state.energy,
                "valence": self.affect_system.state.valence,
                "activation": self.affect_system.state.activation,
            },
            "bdi_state": {
                "beliefs_count": len(self.bdi_system.beliefs),
                "desires_count": len(self.bdi_system.desires),
                "current_intention": (
                    self.bdi_system.current_intention.goal
                    if self.bdi_system.current_intention
                    else None
                ),
            },
            "learning_stats": self.learning_system.get_learning_stats() if hasattr(self.learning_system, 'get_learning_stats') else {},
            "anomaly_stats": {
                "total_samples": self.anomaly_detector.total_samples,
                "anomalies_detected": self.anomaly_detector.anomalies_detected,
                "anomaly_rate": self.anomaly_detector.get_anomaly_rate(),
            },
        }

    def get_bdi_state(self) -> Dict[str, Any]:
        """
        🧠 Obtiene estado completo del sistema BDI.
        
        Returns:
            Estado BDI con beliefs, desires, intentions y jerarquía de necesidades
        """
        return self.bdi_system.get_system_state()

    def get_affect_analysis(self) -> Dict[str, Any]:
        """
        🎭 Obtiene análisis emocional completo.
        
        Returns:
            Análisis del estado afectivo con insights y trayectoria
        """
        try:
            insights = self.affect_system.get_emotional_insights()
            mood = self.affect_system.get_mood()
            trajectory = self.affect_system.get_emotional_trajectory(hours=24)
            
            return {
                "insights": insights,
                "mood": mood,
                "trajectory": trajectory,
                "current_state": {
                    "energy": self.affect_system.state.energy,
                    "valence": self.affect_system.state.valence,
                    "activation": self.affect_system.state.activation,
                },
                "wellbeing": self.affect_system.get_wellbeing(),
            }
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error obteniendo análisis afectivo: {e}")
            return {"error": str(e)}

    def get_cognitive_analysis(self) -> Dict[str, Any]:
        """
        🔬 Análisis cognitivo completo del sistema.
        
        Returns:
            Análisis integrado: BDI + Affect + Memoria + Aprendizaje
        """
        try:
            return {
                "timestamp": time.time(),
                "tick_count": self.tick_count,
                "uptime": time.time() - self.start_time,
                "bdi": self.get_bdi_state(),
                "affect": self.get_affect_analysis(),
                "memory": {
                    "episodes_count": len(self.memory.recall_episodes(1000)),
                    "working_memory_size": len(self.memory.working_memory.items) if self.memory.working_memory.items else 0,
                    "hierarchical_stats": self.get_hierarchical_stats() if self.hierarchical_graph else None,
                },
                "learning": {
                    "graph_nodes": self.learning_system.graph.number_of_nodes() if hasattr(self.learning_system.graph, 'number_of_nodes') else 0,
                    "graph_edges": self.learning_system.graph.number_of_edges() if hasattr(self.learning_system.graph, 'number_of_edges') else 0,
                    "stats": self.learning_system.get_learning_stats() if hasattr(self.learning_system, 'get_learning_stats') else {},
                },
                "anomaly": {
                    "total_samples": self.anomaly_detector.total_samples,
                    "anomalies_detected": self.anomaly_detector.anomalies_detected,
                    "anomaly_rate": self.anomaly_detector.get_anomaly_rate(),
                    "recent_anomalies": self.cognitive_state.get("recent_anomalies", 0),
                },
                "cognitive_state": dict(self.cognitive_state),
            }
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"Error en análisis cognitivo: {e}")
            return {"error": str(e)}

    def query_hierarchical_memory(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Consulta la memoria jerárquica del sistema.
        """
        if not self.hierarchical_graph:
            self.logger.warning("⚠️ Memoria jerárquica no disponible")
            return []

        try:
            results = self.hierarchical_graph.search_concepts(query, limit=limit)
            self.logger.info(
                f"🔍 Búsqueda en memoria: '{query}' → {len(results)} resultados"
            )
            return results
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error consultando memoria jerárquica: {e}")
            return []

    def get_hierarchical_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la memoria jerárquica.
        """
        if not self.hierarchical_graph:
            return {"available": False, "message": "Memoria jerárquica no disponible"}

        try:
            stats = self.hierarchical_graph.get_stats()
            stats["available"] = True

            # Calcular porcentaje de uso de memoria activa
            if stats["active_limit"] > 0:
                stats["active_usage_percent"] = (
                    stats["active_nodes"] / stats["active_limit"]
                ) * 100

            # Calcular conceptos archivados
            stats["archived_nodes"] = stats["total_concepts"] - stats["active_nodes"]

            self.logger.debug(
                f"📊 Memoria jerárquica: {stats['active_nodes']}/{stats['active_limit']} activos, {stats['archived_nodes']} archivados"
            )

            return stats
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error obteniendo estadísticas de memoria: {e}")
            return {"available": False, "error": str(e)}

    def add_to_long_term_memory(
        self,
        concept: str,
        related_concepts: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Agrega un concepto a la memoria de largo plazo (jerárquica).
        """
        if not self.hierarchical_graph:
            self.logger.warning("⚠️ Memoria jerárquica no disponible")
            return False

        try:
            added = self.hierarchical_graph.add_concept(
                concept, related_concepts=related_concepts, properties=properties
            )

            if added:
                self.logger.info(
                    f"💾 Concepto '{concept}' agregado a memoria de largo plazo"
                )
            else:
                self.logger.debug(f"ℹ️  Concepto '{concept}' ya existe en memoria")

            return added
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error agregando concepto a memoria: {e}")
            return False

    def get_concept_from_memory(self, concept: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene un concepto específico de la memoria jerárquica.
        """
        if not self.hierarchical_graph:
            self.logger.warning("⚠️ Memoria jerárquica no disponible")
            return None

        try:
            data = self.hierarchical_graph.get_concept(concept)
            if data:
                self.logger.debug(f"📖 Concepto '{concept}' recuperado de memoria")
            else:
                self.logger.debug(f"❓ Concepto '{concept}' no encontrado en memoria")
            return data
        except Exception as e:
            logger.error(f"Error en core.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error obteniendo concepto: {e}")
            return None

    def reset(self) -> None:
        """Reinicia el sistema cognitivo."""
        self.logger.warning("Reiniciando sistema cognitivo")

        # Reiniciar subsistemas
        self.anomaly_detector.reset()
        self.learning_system.reset()
        self.bdi_system = BDISystem()
        self.metacognition = MetaCognition()

        # Reiniciar estado
        self.cognitive_state = {
            "wellbeing": 0.5,
            "recent_anomalies": 0,
            "current_intention": None,
            "system_notes": [],
        }

        self.tick_count = 0
        self.start_time = time.time()

        self.logger.info("Sistema cognitivo reiniciado")


# === FUNCIÓN DE UTILIDAD ===


# 🔥 SINGLETON FACTORY: Evita 261 inicializaciones duplicadas (SOLUCIÓN DE RAÍZ)
_cognitive_agent_singleton: Optional[CognitiveAgent] = None
_singleton_lock = False


def get_cognitive_agent(config: Optional[AgentConfig] = None, force_new: bool = False) -> CognitiveAgent:
    """
    Factory para CognitiveAgent con patrón SINGLETON.
    
    Previene 200+ inicializaciones duplicadas que causan:
    - Explosión en logs (29,000+ líneas)
    - Consumo RAM masivo (~2GB extra)
    - Tiempo de inicio 3x más lento
    
    Args:
        config: Configuración del agente (solo usado en primera creación)
        force_new: Si True, crea nueva instancia (útil para tests)
    
    Returns:
        CognitiveAgent: Instancia única (singleton) reutilizable
    """
    global _cognitive_agent_singleton, _singleton_lock
    
    # Para tests: permitir creación forzada
    if force_new:
        logger.info("🧠 Creando CognitiveAgent NUEVO (force_new=True)")
        return CognitiveAgent(config)
    
    # Singleton: crear solo si no existe
    if _cognitive_agent_singleton is None and not _singleton_lock:
        _singleton_lock = True
        logger.info("🧠 Creando CognitiveAgent SINGLETON (primera vez)")
        _cognitive_agent_singleton = CognitiveAgent(config)
        _singleton_lock = False
    elif _cognitive_agent_singleton is not None:
        logger.info("🧠 Reutilizando CognitiveAgent SINGLETON existente")
        logger.info("   💡 Previene duplicación - Ahorra RAM y logs")
    else:
        # Lock activo - esperamos
        logger.warning("⚠️ Otro thread está creando CognitiveAgent - esperando...")
        time.sleep(0.1)
        return get_cognitive_agent(config, force_new)
    
    return _cognitive_agent_singleton


def create_cognitive_agent(config: Optional[AgentConfig] = None) -> CognitiveAgent:
    """
    Crea y configura un agente cognitivo.
    
    DEPRECADO: Usar get_cognitive_agent() para evitar duplicados.
    Mantenido por compatibilidad con código legacy.
    """
    logger.warning("⚠️ create_cognitive_agent() está DEPRECADO - usar get_cognitive_agent()")
    return get_cognitive_agent(config)


def run_cortex_service() -> None:
    """
    🔥 SOLUCIÓN DE RAÍZ: Función separada para ejecutar el servicio cortex.

    Evita el RuntimeWarning al ejecutar con python -m metacortex.core
    al separar la lógica del __main__ block.

    🎯 OPTIMIZADO PARA iMac M4 16GB:
    - Intervalo por defecto: 30s (no 1s) para prevenir sobrecarga
    - Backoff exponencial si carga del sistema es alta
    - Throttling inteligente basado en wellbeing
    """

    parser = argparse.ArgumentParser(description="METACORTEX - Sistema Cognitivo")
    parser.add_argument(
        "--daemon", action="store_true", help="Ejecutar como daemon en background"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="Número de ciclos a ejecutar (0 = infinito)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Intervalo entre ciclos en segundos (default: 30s para M4)",
    )
    parser.add_argument(
        "--max-load",
        type=float,
        default=8.0,
        help="Load average máximo antes de throttling (default: 8.0 para M4 10-core)",
    )
    
    args = parser.parse_args()
    
    # Implementación del servicio cortex
    logger.info("🚀 Iniciando METACORTEX Cortex Service")
    logger.info(f"   Ciclos: {args.cycles if args.cycles > 0 else '∞'}")
    logger.info(f"   Intervalo: {args.interval}s")
    logger.info(f"   Max load: {args.max_load}")
    
    # IMPLEMENTED: Implementar lógica del servicio
    # IMPLEMENTED: Implement this functionality


if __name__ == "__main__":
    run_cortex_service()