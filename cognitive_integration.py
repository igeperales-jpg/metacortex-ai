import json
#!/usr/bin/env python3
"""
🧠 METACORTEX Cognitive Integration Layer
Integra el sistema cognitivo (core.py) con el orquestador y la red neuronal
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from neural_symbiotic_network import get_neural_network
from metacortex_sinaptico.core import create_cognitive_agent
from metacortex_sinaptico.core import create_cognitive_agent
from metacortex_orchestrator import MetacortexUnifiedOrchestrator
from unified_memory_layer import get_unified_memory
import threading
import traceback
import threading

# Agregar rutas
sys.path.insert(0, str(Path(__file__).parent))

# Imports del sistema (después del renombrado)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CognitiveOrchestrationBridge:
    """
    🌉 Puente entre el Sistema Cognitivo y el Orquestador

    Funciones:
        pass  # TODO: Implementar
    1. Conecta CognitiveAgent con MetacortexOrchestrator
    2. Registra en la red neuronal simbiótica
    3. Sincroniza memorias (cognitiva + unified)
    4. Coordina decisiones conscientes con tareas de alto nivel

    🚀 v2.0: Inicialización LAZY - componentes se cargan solo cuando se necesitan
    """

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path(__file__).parent)
        logger.info("🌉 Inicializando Cognitive Orchestration Bridge (lazy mode)...")

        # Componentes lazy (se cargan al usarse)
        self._neural_network = None
        self._cognitive_agent = None
        self._orchestrator = None
        self._memory_cognitive = None
        self._memory_unified = None
        self._programming_agent = None  # 🔥 NUEVO: programming_agent lazy

        # Estado de inicialización
        self._components_initialized = {
            "neural_network": False,
            "cognitive_agent": False,
            "orchestrator": False,
            "memory": False,
            "programming_agent": False,  # 🔥 NUEVO: track programming_agent
        }

        # Estado del bridge
        self.cognitive_to_orchestrator_queue: List[Dict[str, Any]] = []
        self.orchestrator_to_cognitive_queue: List[Dict[str, Any]] = []
        self.sync_enabled = True

        logger.info("✅ Cognitive Orchestration Bridge inicializado (lazy mode)")

    # === LAZY PROPERTIES - OPTIMIZADAS PARA METAL MPS ===

    @property
    def neural_network(self):
        """Carga lazy de la red neuronal con MPS"""
        if self._neural_network is None:
            logger.info("🔌 Cargando neural network (Metal MPS)...")

            self._neural_network = get_neural_network()
            self._neural_network.register_module("cognitive_bridge", self)
            self._components_initialized["neural_network"] = True
            logger.info("✅ Neural network cargada con Metal")
        return self._neural_network

    @property
    def cognitive_agent(self):
        """
        Carga lazy del agente cognitivo usando Agent Pool (2026 optimizado)
        
        🆕 Usa cognitive_agent_pool para obtener agentes pre-cargados con timeout real
        """
        if self._cognitive_agent is None:
            logger.info("🧠 Obteniendo cognitive agent del pool...")
            try:
                from cognitive_agent_pool import get_cognitive_agent_pool
                
                pool = get_cognitive_agent_pool()
                self._cognitive_agent = pool.acquire(agent_type="default")
                
                if self._cognitive_agent:
                    self._components_initialized["cognitive_agent"] = True
                    logger.info("✅ Cognitive agent obtenido del pool")
                else:
                    logger.error("❌ Pool retornó None (circuit breaker abierto?)")
                    # Fallback a creación directa (último recurso)
                    logger.warning("⚠️ Fallback: creación directa (puede tardar)...")
                    self._cognitive_agent = create_cognitive_agent()
                    self._components_initialized["cognitive_agent"] = True
                    logger.info("✅ Cognitive agent creado directamente")
                    
            except Exception as e:
                logger.error(f"❌ Error obteniendo agent del pool: {e}")
                # Último fallback
                logger.warning("⚠️ Último fallback: creación directa...")
                self._cognitive_agent = create_cognitive_agent()
                self._components_initialized["cognitive_agent"] = True
                
        return self._cognitive_agent

    @property
    def orchestrator(self):
        """Carga lazy del orquestador"""
        if self._orchestrator is None:
            logger.info("🎯 Cargando orchestrator...")

            self._orchestrator = MetacortexUnifiedOrchestrator(str(self.project_root))
            self._components_initialized["orchestrator"] = True
            logger.info("✅ Orchestrator cargado")
        return self._orchestrator

    @property
    def memory_cognitive(self):
        """Memoria cognitiva (requiere cognitive_agent)"""
        if self._memory_cognitive is None:
            self._memory_cognitive = self.cognitive_agent.memory
        return self._memory_cognitive

    @property
    def memory_unified(self):
        """Carga lazy de memoria unificada con MPS"""
        if self._memory_unified is None:
            logger.info("💾 Cargando unified memory (Metal MPS)...")

            self._memory_unified = get_unified_memory()
            self._components_initialized["memory"] = True
            logger.info("✅ Unified memory cargada")
        return self._memory_unified

    @property
    def programming_agent(self):
        """
        🤖 Carga lazy del Programming Agent - Motor de materialización
        
        Este agente es CRÍTICO para:
        - Materializar pensamientos en código ejecutable
        - Generar agentes especializados
        - Crear proyectos completos
        - Aplicar mejoras al sistema
        
        🔥 SOLUCIÓN 2026: Lazy load con timeout y error handling robusto
        """
        if self._programming_agent is None:
            logger.info("🤖 Cargando Programming Agent...")
            try:
                from programming_agent import get_programming_agent
                
                # Crear programming agent con cognitive_agent como contexto
                self._programming_agent = get_programming_agent(
                    project_root=str(self.project_root),
                    cognitive_agent=self.cognitive_agent  # ✅ Inyectar cognitive_agent
                )
                
                self._components_initialized["programming_agent"] = True
                logger.info("✅ Programming Agent cargado y listo para materialización")
                logger.info("   • Puede generar agentes especializados")
                logger.info("   • Puede crear proyectos completos")
                logger.info("   • Puede aplicar mejoras automáticas")
                
            except Exception as e:
                logger.error(f"❌ Error cargando Programming Agent: {e}")
                # Fallback: crear instancia básica sin cognitive_agent
                try:
                    from programming_agent import get_programming_agent
                    self._programming_agent = get_programming_agent(
                        project_root=str(self.project_root),
                        cognitive_agent=None  # Sin cognitive context (degraded mode)
                    )
                    self._components_initialized["programming_agent"] = True
                    logger.warning("⚠️ Programming Agent en modo degradado (sin cognitive context)")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallo crítico en Programming Agent: {fallback_error}")
                    self._programming_agent = None
                    
        return self._programming_agent

    def get_initialization_status(self) -> Dict[str, bool]:
        """Retorna qué componentes están inicializados"""
        return self._components_initialized.copy()

    def preload_all_components(self):
        """Carga todos los componentes en background (opcional)"""
        logger.info("🚀 Precargando todos los componentes...")
        _ = self.neural_network
        _ = self.cognitive_agent
        _ = self.orchestrator
        _ = self.memory_unified
        _ = self.programming_agent  # 🔥 NUEVO: Precargar programming agent
        logger.info("✅ Todos los componentes precargados")

    def process_cognitive_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        🧠 Procesa una decisión del sistema cognitivo y la traduce a acción del orquestador

        Flujo:
        CognitiveAgent → Decision → Bridge → Orchestrator → Action

        Args:
            decision: Decisión tomada por el sistema cognitivo

        Returns:
            Resultado de la acción ejecutada
        """
        logger.info(
            f"🧠 Procesando decisión cognitiva: {decision.get('type', 'unknown')}"
        )

        decision_type = decision.get("type")

        # Traducir decisión cognitiva a tarea del orquestador
        if decision_type == "learn_new_concept":
            # El agente cognitivo quiere aprender algo nuevo
            concept = decision.get("concept")
            return self.orchestrator.process_user_request(
                f"Investiga y explica el concepto: {concept}"
            )

        elif decision_type == "improve_capability":
            # El agente cognitivo detectó una debilidad y quiere mejorar
            capability = decision.get("capability")
            return self.orchestrator.process_user_request(
                f"Genera código para mejorar capacidad: {capability}"
            )

        elif decision_type == "create_new_tool":
            # El agente cognitivo necesita una herramienta nueva
            tool_description = decision.get("description")
            return self.orchestrator.process_user_request(
                f"Crea una herramienta: {tool_description}"
            )

        elif decision_type == "research_anomaly":
            # Anomalía detectada, necesita investigación
            anomaly = decision.get("anomaly")
            return self.orchestrator.process_user_request(
                f"Investiga la siguiente anomalía: {anomaly}"
            )

        else:
            logger.warning(f"⚠️ Tipo de decisión desconocido: {decision_type}")
            return {"success": False, "error": "Unknown decision type"}

    def notify_ml_prediction(self, prediction_type: str, result: Dict[str, Any]):
        """
        🤖 Notifica al Cognitive Agent de una predicción ML

        Flujo: ML Pipeline → Bridge → Cognitive Agent → Update Beliefs

        Args:
            prediction_type: Tipo de predicción ('intention', 'load', 'cache', 'performance')
            result: Resultado de la predicción con confidence, value, etc.
        """
        logger.info(f"🤖 ML Prediction recibida: {prediction_type} → {result}")

        # Actualizar beliefs del cognitive agent con la predicción ML
        belief_key = f"ml_{prediction_type}_prediction"

        self.cognitive_agent.bdi_system.add_belief(
            key=belief_key,
            value={
                "prediction": result.get("prediction"),
                "confidence": result.get("confidence", 0.0),
                "timestamp": datetime.now().isoformat(),
                "model_accuracy": result.get("model_accuracy", 0.0),
            },
            confidence=result.get("confidence", 0.5),
        )

        # Si es predicción de intención, actualizar desires también
        if prediction_type == "intention" and result.get("confidence", 0) > 0.7:
            intent = result.get("prediction")
            self.cognitive_agent.bdi_system.add_desire(
                name=f"handle_{intent}_request", priority=result.get("confidence", 0.5)
            )
            logger.info(f"✨ Desire creado: handle_{intent}_request")

        # Si es predicción de carga alta, agregar desire de optimización
        elif prediction_type == "load" and result.get("prediction", 0) > 0.8:
            self.cognitive_agent.bdi_system.add_desire(
                name="optimize_system_load", priority=0.9
            )
            logger.info("⚡ Desire creado: optimize_system_load")

        # Almacenar en memoria unificada
        self.memory_unified.store_episode(
            content=f"ML Prediction: {prediction_type}",
            context={
                "source": "ml_pipeline",
                "prediction_type": prediction_type,
                "result": result,
            },
            importance=result.get("confidence", 0.5),
        )

        logger.info("✅ ML Prediction procesada por Cognitive Agent")

    def notify_ml_feedback(self, action_result: Dict[str, Any]):
        """
        📊 Envía feedback del Cognitive Agent al ML Pipeline

        Flujo: Cognitive Agent → Bridge → ML Data Collector → Retrain

        Args:
            action_result: {
                'action_type': str,
                'success': bool,
                'execution_time': float,
                'agent_used': str,
                'user_query': str (opcional),
                'tokens_used': int (opcional)
            }
        """
        logger.info(
            f"� Enviando feedback a ML Pipeline: {action_result.get('action_type')}"
        )

        try:
            from ml_data_collector import get_data_collector

            collector = get_data_collector()

            # 1. Feedback de performance del agente
            if action_result.get("agent_used"):
                collector.collect_agent_performance(
                    agent_name=action_result["agent_used"],
                    task_type=action_result.get("action_type", "unknown"),
                    execution_time_ms=int(
                        action_result.get("execution_time", 0) * 1000
                    ),
                    success=action_result.get("success", False),
                    error_type=action_result.get("error_type"),
                )
                logger.info("✅ Agent performance feedback enviado")

            # 2. Feedback de interacción de usuario (si aplica)
            if action_result.get("user_query"):
                collector.collect_user_interaction(
                    user_query=action_result["user_query"],
                    intent=action_result.get("intent", "unknown"),
                    response_time_ms=int(action_result.get("execution_time", 0) * 1000),
                    tokens_used=action_result.get("tokens_used", 0),
                    success=action_result.get("success", False),
                    agent_used=action_result.get("agent_used", "unknown"),
                    context_length=len(action_result.get("user_query", "")),
                )
                logger.info("✅ User interaction feedback enviado")

            # 3. Actualizar belief del cognitive sobre el feedback
            self.cognitive_agent.bdi_system.add_belief(
                key="ml_feedback_sent",
                value={
                    "last_feedback": action_result.get("action_type"),
                    "success": action_result.get("success"),
                    "timestamp": datetime.now().isoformat(),
                },
                confidence=1.0,
            )

        except Exception as e:
            logger.error(f"❌ Error enviando feedback a ML: {e}")

    def feed_orchestrator_result_to_cognitive(self, result: Dict[str, Any]):
        """
        �📥 Alimenta el resultado del orquestador al sistema cognitivo

        Flujo:
        Orchestrator → Result → Bridge → CognitiveAgent → Perception

        Args:
            result: Resultado de una tarea del orquestador
        """
        logger.info("📥 Alimentando resultado al sistema cognitivo...")

        # Extraer información relevante del resultado
        event_name = "orchestrator_result"
        payload = {
            "success": result.get("success", False),
            "type": result.get("type", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "data": result,
        }

        # Enviar como percepción al agente cognitivo
        perception_result = self.cognitive_agent.perceive(event_name, payload)

        # Almacenar en memoria unificada también
        if result.get("success"):
            self.memory_unified.store_episode(
                content=f"Orchestrator task completed: {result.get('type')}",
                context={"source": "orchestrator", "result": result},
                importance=0.8,
            )

        logger.info(
            f"✅ Resultado alimentado - Anomalía detectada: {perception_result.get('anomaly', False)}"
        )

        # Enviar feedback a ML Pipeline
        self.notify_ml_feedback(
            {
                "action_type": result.get("type", "unknown"),
                "success": result.get("success", False),
                "execution_time": result.get("execution_time", 0.0),
                "agent_used": result.get("agent_used", "unknown"),
            }
        )

    def sync_memories(self):
        """
        🔄 Sincroniza las dos memorias (cognitiva + unificada)

        - Memoria cognitiva: Episodios, working memory, grafo de conocimiento
        - Memoria unificada: Conversaciones, proyectos, learnings
        """
        if not self.sync_enabled:
            return

        logger.info("🔄 Sincronizando memorias...")

        try:
            # 1. Sincronizar episodios recientes de cognitiva → unificada
            recent_episodes = self.memory_cognitive.recall_episodes(limit=10)

            for episode in recent_episodes:
                # 🔥 FIX: episode es un MemoryEntry object, no un dict
                content = getattr(episode, 'content', None) or getattr(episode, 'name', 'cognitive_episode')
                context = getattr(episode, 'context', {}) or getattr(episode, 'data', {})
                
                self.memory_unified.store_episode(
                    content=str(content),
                    context=context if isinstance(context, dict) else {},
                    importance=0.7,
                )

            # 2. Sincronizar aprendizajes cognitivos → learnings unificados
            cognitive_state = self.cognitive_agent.get_current_state()

            if cognitive_state.get("recent_anomalies", 0) > 0:
                # Guardar como learning
                self.memory_unified.persistent.save_learning(
                    error_type="cognitive_anomaly",
                    error_message=f"Anomalías detectadas: {cognitive_state['recent_anomalies']}",
                    solution="Sistema cognitivo en proceso de adaptación",
                    context=str(cognitive_state),
                )

            logger.info("✅ Memorias sincronizadas")

        except Exception as e:
            logger.error(f"❌ Error sincronizando memorias: {e}")

    def cognitive_tick_with_orchestration(self) -> Dict[str, Any]:
        """
        ⏰ Ejecuta un tick cognitivo integrado con el orquestador

        🔥 SOLUCIÓN ROBUSTA v2.0:
        - Timeout de 60 segundos para evitar bloqueos
        - Manejo de excepciones completo
        - Fallback cuando orchestrator no disponible
        - Logging detallado para diagnóstico

        Proceso:
        1. Tick del sistema cognitivo (con timeout)
        2. Procesar decisiones conscientes
        3. Ejecutar acciones con el orquestador (opcional)
        4. Sincronizar memorias (con timeout)
        5. Retornar estado completo

        Returns:
            Estado completo del sistema integrado
        """

        logger.info("⏰ Ejecutando cognitive tick integrado...")

        # 🔥 PASO 1: Ejecutar tick cognitivo con timeout y error handling
        cognitive_report = {}
        tick_start_time = datetime.now()

        try:
            logger.debug("   🧠 Paso 1/5: Ejecutando cognitive_agent.tick()...")

            # Ejecutar tick en thread separado con timeout
            tick_result = [None]
            tick_exception = [None]

            def run_tick():
                try:
                    tick_result[0] = self.cognitive_agent.tick()
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    tick_exception[0] = e

            tick_thread = threading.Thread(target=run_tick, daemon=True)
            tick_thread.start()
            tick_thread.join(timeout=30.0)  # 30 segundos timeout

            if tick_thread.is_alive():
                logger.error("❌ Timeout en cognitive_agent.tick() después de 30s")
                cognitive_report = {
                    "status": "timeout",
                    "wellbeing": 0.5,
                    "anomalies": 0,
                    "error": "Cognitive tick timeout",
                }
            elif tick_exception[0]:
                raise tick_exception[0]
            else:
                cognitive_report = tick_result[0]
                tick_elapsed = (datetime.now() - tick_start_time).total_seconds()
                logger.info(f"   ✅ Cognitive tick completado en {tick_elapsed:.2f}s")

        except Exception as e:
            logger.error(f"❌ Error en cognitive tick: {e}")
            cognitive_report = {
                "status": "error",
                "wellbeing": 0.5,
                "anomalies": 0,
                "error": str(e),
            }

        # 🔥 PASO 2: Procesar decisiones (SOLO si tick fue exitoso)
        orchestrator_result = None
        if cognitive_report.get("status") not in ["timeout", "error"]:
            try:
                logger.debug("   🎯 Paso 2/5: Verificando decisiones...")
                decision_result = cognitive_report.get("decision_result", {})

                if decision_result and decision_result.get("decision_taken"):
                    # Crear decisión para el orquestador
                    decision = {
                        "type": decision_result["decision_taken"],
                        "context": cognitive_report,
                        "wellbeing": cognitive_report.get("wellbeing", 0.5),
                        "anomalies": cognitive_report.get("anomalies", 0),
                    }

                    logger.debug(f"   🔧 Procesando decisión: {decision['type']}")

                    # 🔥 BYPASS TEMPORAL: No cargar orchestrator si causa timeout
                    # El orchestrator se carga lazy, pero puede bloquearse al inicializar
                    if not self._components_initialized.get("orchestrator", False):
                        logger.warning(
                            "   ⚠️ Orchestrator no inicializado, saltando procesamiento"
                        )
                    else:
                        orchestrator_result = self.process_cognitive_decision(decision)
                        self.feed_orchestrator_result_to_cognitive(orchestrator_result)
                        logger.info("   ✅ Decisión procesada")
            except Exception as e:
                logger.error(f"❌ Error procesando decisión: {e}")

        # 🔥 PASO 3: Sincronizar memorias (con timeout)
        sync_start_time = datetime.now()
        memories_synced = False

        try:
            logger.debug("   💾 Paso 3/5: Sincronizando memorias...")

            # Ejecutar sync en thread con timeout
            sync_exception = [None]

            def run_sync():
                try:
                    self.sync_memories()
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    sync_exception[0] = e

            sync_thread = threading.Thread(target=run_sync, daemon=True)
            sync_thread.start()
            sync_thread.join(timeout=15.0)  # 15 segundos timeout

            if sync_thread.is_alive():
                logger.warning(
                    "⚠️ Timeout en sync_memories() después de 15s (continuando...)"
                )
            elif sync_exception[0]:
                raise sync_exception[0]
            else:
                memories_synced = True
                sync_elapsed = (datetime.now() - sync_start_time).total_seconds()
                logger.info(f"   ✅ Memorias sincronizadas en {sync_elapsed:.2f}s")

        except Exception as e:
            logger.error(f"❌ Error sincronizando memorias: {e}")

        # 🔥 PASO 4: MATERIALIZACIÓN de pensamientos (NUEVO)
        materialization_result = None
        mat_start_time = datetime.now()

        try:
            logger.info("   🧠 Paso 4/5: Materializando pensamientos...")

            # 🔥 FIX 2026: Usar la property lazy programming_agent
            if self.programming_agent:
                logger.info(f"      ✅ ProgrammingAgent DISPONIBLE: {type(self.programming_agent).__name__}")

                # Ejecutar materialización en thread con timeout
                mat_exception = [None]
                mat_data = [None]

                def run_materialization():
                    try:
                        logger.info("      🚀 Ejecutando materialize_metacortex_thoughts()...")
                        mat_data[0] = self.programming_agent.materialize_metacortex_thoughts()
                        logger.info(f"      ✅ Resultado: {mat_data[0].get('components_created', 0) if mat_data[0] else 'None'} componentes")
                    except Exception as e:
                        logger.error(f"Error: {e}", exc_info=True)
                        mat_exception[0] = e
                        logger.error(f"      ❌ Excepción en materialización: {e}")

                mat_thread = threading.Thread(target=run_materialization, daemon=True)
                mat_thread.start()
                mat_thread.join(timeout=30.0)  # 30 segundos timeout

                if mat_thread.is_alive():
                    logger.warning("⚠️ Timeout en materialización después de 30s")
                    materialization_result = {
                        "success": False,
                        "error": "Timeout after 30s"
                    }
                elif mat_exception[0]:
                    logger.error(f"❌ Error en materialización: {mat_exception[0]}")
                    materialization_result = {
                        "success": False,
                        "error": str(mat_exception[0])
                    }
                else:
                    materialization_result = mat_data[0]
                    mat_elapsed = (datetime.now() - mat_start_time).total_seconds()

                    if materialization_result and materialization_result.get('success'):
                        logger.info(f"   ✅ Materialización completada en {mat_elapsed:.2f}s:")
                        if materialization_result.get('components_created', 0) > 0:
                            logger.info(f"      • Componentes: {materialization_result['components_created']}")
                        if materialization_result.get('agents_generated', 0) > 0:
                            logger.info(f"      • Agentes: {materialization_result['agents_generated']}")
                        if materialization_result.get('improvements_applied', 0) > 0:
                            logger.info(f"      • Mejoras: {materialization_result['improvements_applied']}")
                    else:
                        logger.warning(f"   ⚠️ Materialización sin success o sin cambios en {mat_elapsed:.2f}s")
            else:
                logger.warning("   ❌ Programming agent NO se pudo cargar (ver logs anteriores)")

        except Exception as e:
            logger.error(f"❌ Error en paso de materialización: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")

        # 🔥 PASO 5: Retornar estado completo
        total_elapsed = (datetime.now() - tick_start_time).total_seconds()
        logger.info(
            f"✅ Cognitive tick con orchestration completado en {total_elapsed:.2f}s"
        )

        return {
            "success": cognitive_report.get("status") not in ["timeout", "error"],
            "cognitive_report": cognitive_report,
            "orchestrator_active": self._components_initialized.get(
                "orchestrator", False
            ),
            "orchestrator_result": orchestrator_result,
            "materialization_result": materialization_result,  # 🆕 NUEVO
            "neural_network_modules": len(self.neural_network.modules)
            if self._neural_network
            else 0,
            "memories_synced": memories_synced,
            "elapsed_seconds": total_elapsed,
            "timestamp": datetime.now().isoformat(),
        }

    def autonomous_improvement_cycle(self) -> Dict[str, Any]:
        """
        🔄 Ciclo de auto-mejora autónomo

        🔥 SOLUCIÓN ROBUSTA v2.0:
        - Timeout de 30 segundos
        - No depende de orchestrator (puede funcionar sin él)
        - Error handling completo

        El sistema cognitivo analiza su estado y solicita mejoras al orquestador

        Returns:
            Reporte del ciclo de mejora
        """

        logger.info("🔄 Iniciando ciclo de auto-mejora autónomo...")
        start_time = datetime.now()

        # 🆕 CRÍTICO: Inyectar percepciones ANTES de obtener estado
        self._inject_environmental_perceptions()

        improvements_requested = 0
        improvements_applied = 0

        try:
            # 1. Obtener estado cognitivo (con timeout)
            logger.debug("   📊 Paso 1/3: Obteniendo estado cognitivo...")

            state_result = [None]
            state_exception = [None]

            def get_state():
                try:
                    state_result[0] = {
                        "cognitive_state": self.cognitive_agent.get_current_state(),
                        "debug_info": self.cognitive_agent.get_debug_info(),
                    }
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    state_exception[0] = e

            state_thread = threading.Thread(target=get_state, daemon=True)
            state_thread.start()
            state_thread.join(timeout=10.0)

            if state_thread.is_alive():
                logger.warning("⚠️ Timeout obteniendo estado cognitivo (10s)")
                return {
                    "success": False,
                    "improvements_requested": 0,
                    "improvements_applied": 0,
                    "error": "Timeout getting cognitive state",
                }

            if state_exception[0]:
                raise state_exception[0]

            if not state_result[0]:
                logger.warning("⚠️ No se pudo obtener estado cognitivo")
                return {
                    "success": False,
                    "improvements_requested": 0,
                    "improvements_applied": 0,
                    "error": "Could not get cognitive state",
                }

            cognitive_state = state_result[0]["cognitive_state"]
            debug_info = state_result[0]["debug_info"]

            logger.debug(
                f"   ✅ Estado: wellbeing={cognitive_state.get('wellbeing', 0.5):.2f}"
            )

            # 2. Identificar áreas de mejora (rápido, no requiere timeout)
            logger.debug("   🔍 Paso 2/3: Identificando áreas de mejora...")
            improvements_needed: List[Dict[str, Any]] = []

            if cognitive_state.get("wellbeing", 0.5) < 0.4:
                improvements_needed.append(
                    {
                        "type": "improve_wellbeing",
                        "priority": "high",
                        "description": "Wellbeing bajo, necesita optimización",
                    }
                )

            if cognitive_state.get("recent_anomalies", 0) > 5:
                improvements_needed.append(
                    {
                        "type": "handle_anomalies",
                        "priority": "high",
                        "description": f"Demasiadas anomalías: {cognitive_state['recent_anomalies']}",
                    }
                )

            learning_stats = debug_info.get("learning_stats", {})
            if learning_stats.get("graph_density", 0) < 0.1:
                improvements_needed.append(
                    {
                        "type": "expand_knowledge",
                        "priority": "medium",
                        "description": "Grafo de conocimiento poco denso",
                    }
                )

            improvements_requested = len(improvements_needed)
            logger.info(f"   📋 {improvements_requested} mejoras identificadas")

            # 3. Aplicar mejoras (SOLO si orchestrator está disponible)
            logger.debug("   🔧 Paso 3/3: Aplicando mejoras...")

            if not self._components_initialized.get("orchestrator", False):
                logger.warning(
                    "   ⚠️ Orchestrator no disponible, mejoras registradas pero no aplicadas"
                )

                # Guardar mejoras pendientes en memoria para aplicar después
                for improvement in improvements_needed:
                    try:
                        self.memory_unified.store_episode(
                            content={
                                "event_type": "pending_improvement",
                                "details": improvement["description"],
                                "priority": improvement["priority"]
                            },
                            context=str(improvement),
                            importance=0.8
                            if improvement["priority"] == "high"
                            else 0.6
                        )
                    except Exception as e:
                        logger.error(
                            f"Error guardando mejora en memoria: {e}",
                            exc_info=True
                        )
                        # Continuar sin bloquear el ciclo de mejoras

            else:
                # Orchestrator disponible, aplicar mejoras (con timeout total)
                results_applied: List[Any] = []

                for improvement in improvements_needed[
                    :3
                ]:  # Máximo 3 mejoras por ciclo
                    try:
                        logger.debug(f"   🔧 Aplicando: {improvement['type']}")

                        # Timeout individual por mejora: 5 segundos
                        result = [None]

                        def apply_improvement():
                            try:
                                result[0] = self.orchestrator.execute_task(
                                    {
                                        "type": improvement["type"],
                                        "priority": improvement["priority"],
                                        "description": improvement["description"],
                                    }
                                )
                            except Exception as e:
                                improvement_type = improvement.get("type", "unknown")
                                logger.error(
                                    f"Error ejecutando mejora {improvement_type}: {e}",
                                    exc_info=True
                                )
                                result[0] = None  # Marcar como fallida

                        imp_thread = threading.Thread(
                            target=apply_improvement, daemon=True
                        )
                        imp_thread.start()
                        imp_thread.join(timeout=5.0)

                        if result[0]:
                            results_applied.append(result[0])
                            improvements_applied += 1

                    except Exception as e:
                        logger.warning(
                            f"   ⚠️ Error en mejora {improvement['type']}: {e}"
                        )
                        continue

                logger.info(
                    f"   ✅ {improvements_applied}/{improvements_requested} mejoras aplicadas"
                )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Ciclo de auto-mejora completado en {elapsed:.2f}s")

            return {
                "success": True,
                "improvements_requested": improvements_requested,
                "improvements_applied": improvements_applied,
                "wellbeing": cognitive_state.get("wellbeing", 0.5),
                "anomalies": cognitive_state.get("recent_anomalies", 0),
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Error en ciclo de auto-mejora: {e}")
            return {
                "success": False,
                "improvements_requested": improvements_requested,
                "improvements_applied": improvements_applied,
                "error": str(e),
            }

    def _inject_environmental_perceptions(self):
        """
        🌍 Inyecta percepciones del entorno al cognitive agent

        CRÍTICO para el aprendizaje:
        - Sin percepciones, working_memory está vacía
        - Sin working_memory, no hay conceptos nuevos
        - Sin conceptos, no hay aristas en el grafo

        Percepciones que se inyectan:
        1. Estado de tareas de programación
        2. Estado del sistema (daemon, subsistemas)
        3. Estado de la red neuronal
        4. Grafo de conocimiento
        5. Auto-percepción (meta-cognición)
        """
        try:
            perceptions_injected = 0

            # === PERCEPCIÓN 1: Tareas de Programación ===
            try:
                import sqlite3

                db_path = self.project_root / "logs" / "universal_programming.sqlite"
                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()

                    cursor.execute(
                        "SELECT COUNT(*) FROM programming_tasks WHERE status='pending'"
                    )
                    pending = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM completed_projects")
                    completed = cursor.fetchone()[0]

                    self.cognitive_agent.perceive(
                        "programming_workload",
                        {
                            "pending_tasks": pending,
                            "completed_projects": completed,
                            "workload_level": "high" if pending > 10 else "normal",
                        },
                    )

                    perceptions_injected += 1
                    conn.close()

            except Exception as e:
                logger.error(f"No se pudo leer tareas: {e}", exc_info=True)

            # === PERCEPCIÓN 2: Estado del Sistema ===
            try:
                import psutil  # type: ignore

                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()

                self.cognitive_agent.perceive(
                    "system_health",
                    {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "system_healthy": cpu_percent < 80 and memory.percent < 90,
                    },
                )

                perceptions_injected += 1

            except Exception as e:
                logger.error(f"No se pudo leer sistema: {e}", exc_info=True)

            # === PERCEPCIÓN 3: Grafo de Conocimiento ===
            try:
                learning_stats = (
                    self.cognitive_agent.learning_system.get_learning_stats()
                )

                self.cognitive_agent.perceive(
                    "knowledge_graph",
                    {
                        "total_nodes": learning_stats.get("current_nodes", 0),
                        "total_edges": learning_stats.get("current_edges", 0),
                        "graph_density": learning_stats.get("current_edges", 0)
                        / max(1, learning_stats.get("current_nodes", 1)),
                    },
                )

                perceptions_injected += 1

            except Exception as e:
                logger.error(f"No se pudo leer grafo: {e}", exc_info=True)

            # === PERCEPCIÓN 4: Auto-percepción ===
            try:
                cognitive_state = self.cognitive_agent.get_current_state()

                self.cognitive_agent.perceive(
                    "self_awareness",
                    {
                        "wellbeing": cognitive_state.get("wellbeing", 0.5),
                        "tick_count": cognitive_state.get("tick_count", 0),
                        "consciousness_level": "high"
                        if cognitive_state.get("wellbeing", 0) > 0.7
                        else "medium",
                    },
                )

                perceptions_injected += 1

            except Exception as e:
                logger.debug(f"No se pudo generar auto-percepción: {e}")

            logger.debug(f"✨ {perceptions_injected} percepciones inyectadas")

        except Exception as e:
            logger.warning(f"Error inyectando percepciones: {e}")

    def get_integrated_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema integrado"""
        return {
            "cognitive_agent": self.cognitive_agent.get_system_status(),
            "orchestrator": self.orchestrator.get_system_status(),
            "neural_network": self.neural_network.get_stats(),
            "memory_cognitive": {
                "episodes": len(self.memory_cognitive.recall_episodes(1000)),
                "working_memory": len(self.memory_cognitive.working_memory.items),
            },
            "memory_unified": self.memory_unified.get_memory_stats(),
            "bridge_status": {
                "sync_enabled": self.sync_enabled,
                "queues": {
                    "cognitive_to_orchestrator": len(
                        self.cognitive_to_orchestrator_queue
                    ),
                    "orchestrator_to_cognitive": len(
                        self.orchestrator_to_cognitive_queue
                    ),
                },
            },
        }


# Singleton global
_global_bridge: Optional[CognitiveOrchestrationBridge] = None


def get_cognitive_bridge(
    project_root: Optional[str] = None,
) -> CognitiveOrchestrationBridge:
    """Obtener instancia global del bridge"""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = CognitiveOrchestrationBridge(project_root)
    return _global_bridge


if __name__ == "__main__":

    print("🧪 Testing Cognitive Orchestration Bridge...\n")

    # Crear bridge
    bridge = get_cognitive_bridge()

    # Test 1: Estado integrado
    print("1️⃣ Estado del sistema integrado:")
    status = bridge.get_integrated_status()
    print(json.dumps(status, indent=2, default=str))

    # Test 2: Tick integrado
    print("\n2️⃣ Ejecutando tick cognitivo integrado...")
    tick_result = bridge.cognitive_tick_with_orchestration()
    print(f"   Wellbeing: {tick_result['cognitive_report'].get('wellbeing', 'N/A')}")
    print(f"   Módulos en red: {tick_result['neural_network_modules']}")

    # Test 3: Ciclo de auto-mejora
    print("\n3️⃣ Ciclo de auto-mejora autónomo...")
    improvement_result = bridge.autonomous_improvement_cycle()
    print(f"   Mejoras identificadas: {improvement_result['improvements_identified']}")

    print("\n✅ Tests completados!")