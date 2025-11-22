#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
METACORTEX - API FastAPI
========================

Endpoints REST para interacción con el sistema cognitivo-metacognitivo.

Endpoints:
    pass  # TODO: Implementar
- POST /perceive: Procesa percepción externa
- POST /tick: Avanza un ciclo del sistema
- GET /report: Estado metacognitivo
- GET /graph: Snapshot del grafo de conocimiento
- GET /status: Estado del sistema
"""
import logging
import time
from typing import Any, Optional, Dict, List

from fastapi import APIRouter, HTTPException, Depends

from .emotional_models import Emotion as EmotionType
from .utils import (
    PerceptionInput,
    PerceptionOutput,
    MetaReport,
    GraphSnapshot,
    SystemStatus,
    setup_logging,
)

logger = logging.getLogger(__name__)


# === CONFIGURACIÓN ===

logger = setup_logging()
router = APIRouter(prefix="/metacortex", tags=["Metacortex"])

# Variable global para el agente (será inyectado)
_cognitive_agent: Optional[Any] = None


# === DEPENDENCIAS ===


def get_cognitive_agent() -> Any:
    """Dependency injection para el agente cognitivo."""
    if _cognitive_agent is None:
        raise HTTPException(status_code=503, detail="Sistema cognitivo no inicializado")
    return _cognitive_agent


def set_cognitive_agent(agent: Any) -> None:
    """Configura el agente cognitivo global."""
    global _cognitive_agent
    _cognitive_agent = agent
    logger.info("Agente cognitivo configurado para API")


# === ENDPOINTS ===


@router.post(
    "/perceive",
    response_model=PerceptionOutput,
    summary="Procesar percepción",
    description="Envía un evento de percepción al sistema cognitivo",
)
async def perceive(
    perception: PerceptionInput, agent: Any = Depends(get_cognitive_agent)
) -> PerceptionOutput:
    """
    Procesa una percepción externa y detecta anomalías.

    Args:
        perception: Datos de la percepción (nombre y payload)

    Returns:
        Resultado del procesamiento con flag de anomalía
    """
    try:
        logger.info(f"Procesando percepción: {perception.name}")

        # Procesar a través del agente cognitivo
        result = agent.perceive(perception.name, perception.payload)

        return PerceptionOutput(
            anomaly=result.get("anomaly", False),
            z_score=result.get("z_score"),
            stored=result.get("stored", True),
        )

    except Exception as e:
        logger.error(f"Error procesando percepción: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error procesando percepción: {str(e)}"
        )


@router.post(
    "/tick",
    response_model=MetaReport,
    summary="Ciclo cognitivo",
    description="Ejecuta un ciclo del sistema cognitivo-metacognitivo",
)
async def tick(agent: Any = Depends(get_cognitive_agent)) -> MetaReport:
    """
    Ejecuta un tick/ciclo del sistema cognitivo.

    Returns:
        Reporte del estado metacognitivo tras el ciclo
    """
    try:
        logger.info("Ejecutando tick cognitivo")

        # Ejecutar ciclo a través del agente
        report = agent.tick()

        return MetaReport(
            wellbeing=report.get("wellbeing", 0.5),
            anomalies=report.get("anomalies", 0),
            intention=report.get("intention"),
            notes=report.get("notes", []),
            timestamp=time.time(),
        )

    except Exception as e:
        logger.error(f"Error ejecutando tick: {e}")
        raise HTTPException(status_code=500, detail=f"Error ejecutando tick: {str(e)}")


@router.get(
    "/report",
    response_model=MetaReport,
    summary="Estado metacognitivo",
    description="Obtiene el último reporte del estado metacognitivo",
)
async def get_report(agent: Any = Depends(get_cognitive_agent)) -> MetaReport:
    """
    Obtiene el estado actual del sistema metacognitivo.

    Returns:
        Reporte detallado del estado interno
    """
    try:
        logger.debug("Obteniendo reporte metacognitivo")

        # Obtener estado actual del agente
        state = agent.get_current_state()

        return MetaReport(
            wellbeing=state.get("wellbeing", 0.5),
            anomalies=state.get("recent_anomalies", 0),
            intention=state.get("current_intention"),
            notes=state.get("system_notes", []),
            timestamp=time.time(),
        )

    except Exception as e:
        logger.error(f"Error obteniendo reporte: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo reporte: {str(e)}"
        )


@router.get(
    "/graph",
    response_model=GraphSnapshot,
    summary="Grafo de conocimiento",
    description="Obtiene snapshot del grafo de conocimiento actual",
)
async def get_graph(agent: Any = Depends(get_cognitive_agent)) -> GraphSnapshot:
    """
    Obtiene un snapshot del grafo de conocimiento.

    Returns:
        Estructura del grafo con nodos, aristas y métricas
    """
    try:
        logger.debug("Obteniendo snapshot del grafo")

        # Obtener grafo del agente
        graph_data = agent.get_graph_snapshot()

        return GraphSnapshot(
            nodes=graph_data.get("nodes", []),
            edges=graph_data.get("edges", []),
            metrics=graph_data.get("metrics", {}),
        )

    except Exception as e:
        logger.error(f"Error obteniendo grafo: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo grafo: {str(e)}")


@router.get(
    "/status",
    response_model=SystemStatus,
    summary="Estado del sistema",
    description="Información del estado general del sistema",
)
async def get_status(agent: Any = Depends(get_cognitive_agent)) -> SystemStatus:
    """
    Obtiene el estado general del sistema cognitivo.

    Returns:
        Estado y métricas del sistema
    """
    try:
        logger.debug("Obteniendo estado del sistema")

        # Obtener estado del agente
        status = agent.get_system_status()

        return SystemStatus(
            active=status.get("active", True),
            uptime=status.get("uptime", 0.0),
            memory_usage=status.get("memory_usage", {}),
            last_tick=status.get("last_tick"),
        )

    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo estado: {str(e)}"
        )


# === ENDPOINTS DE ADMINISTRACIÓN ===


@router.post(
    "/reset",
    summary="Reiniciar sistema",
    description="Reinicia el estado del sistema cognitivo",
)
async def reset_system(
    confirm: bool = False, agent: Any = Depends(get_cognitive_agent)
) -> Dict[str, str]:
    """
    Reinicia el sistema cognitivo.

    Args:
        confirm: Confirmación requerida para reset

    Returns:
        Confirmación del reset
    """
    if not confirm:
        raise HTTPException(
            status_code=400, detail="Confirmación requerida (confirm=true)"
        )

    try:
        logger.warning("Reiniciando sistema cognitivo")

        # Reiniciar a través del agente
        agent.reset()

        return {"status": "success", "message": "Sistema reiniciado"}

    except Exception as e:
        logger.error(f"Error reiniciando sistema: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error reiniciando sistema: {str(e)}"
        )


@router.get(
    "/metrics",
    summary="Métricas históricas",
    description="Obtiene métricas históricas del sistema",
)
async def get_metrics(
    hours: int = 24, agent: Any = Depends(get_cognitive_agent)
) -> Dict[str, Any]:
    """
    Obtiene métricas históricas del sistema.

    Args:
        hours: Ventana temporal en horas

    Returns:
        Lista de métricas históricas
    """
    try:
        logger.debug(f"Obteniendo métricas de {hours} horas")

        # Obtener métricas del agente
        metrics = agent.get_metrics_history(hours)

        return {
            "metrics": metrics,
            "hours": hours,
            "count": len(metrics),
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo métricas: {str(e)}"
        )


@router.get(
    "/debug",
    summary="Información de debug",
    description="Información detallada para debugging",
)
async def get_debug_info(agent: Any = Depends(get_cognitive_agent)) -> Dict[str, Any]:
    """
    Obtiene información detallada para debugging.

    Returns:
        Información interna del sistema
    """
    try:
        logger.debug("Obteniendo información de debug")

        # Obtener info de debug del agente
        debug_info = agent.get_debug_info()

        return debug_info

    except Exception as e:
        logger.error(f"Error obteniendo debug info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo debug info: {str(e)}"
        )


# === MANEJO DE ERRORES ===

# @router.exception_handler(Exception)
# async def metacortex_exception_handler(request, exc):
#     """Manejo global de excepciones para Metacortex."""
#     logger.error(f"Error no manejado en Metacortex: {exc}")
#
#     return JSONResponse(
#         status_code=500,
#         content={
#             "detail": "Error interno del sistema cognitivo",
#             "error": str(exc),
#             "timestamp": time.time()
#         }
#     )


# === FUNCIONES DE UTILIDAD ===


def get_router() -> APIRouter:
    """Retorna el router configurado."""
    return router


def init_api(cognitive_agent: Any) -> APIRouter:
    """Inicializa la API con un agente cognitivo."""
    set_cognitive_agent(cognitive_agent)
    logger.info("API Metacortex inicializada")
    return router


# === ENDPOINTS ADICIONALES PARA VERIFICACIÓN ===


@router.get(
    "/health",
    summary="Estado de salud del sistema",
    description="Verifica que la DB y el agente estén operativos",
)
async def get_health(agent: Any = Depends(get_cognitive_agent)) -> Dict[str, Any]:
    """Estado de salud del sistema."""
    try:
        status = agent.get_status()
        db_ok = hasattr(agent, "db") and agent.db is not None
        return {
            "status": "healthy",
            "database": "ok" if db_ok else "error",
            "agent": "ready",
            "uptime_s": time.time() - status.get("start_time", time.time()),
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Sistema no saludable: {str(e)}")


@router.get(
    "/meta-state",
    summary="Estado metacognitivo JSON",
    description="Estado metacognitivo estructurado en JSON",
)
async def meta_state(agent: Any = Depends(get_cognitive_agent)) -> Dict[str, Any]:
    """Estado metacognitivo estructurado en JSON."""
    try:
        status = agent.get_status()
        return {
            "wellbeing": status.get("wellbeing", 0.5),
            "anomalies": status.get("anomalies_detected", 0),
            "graph_nodes": status.get("graph_nodes", 0),
            "intentions": status.get("current_intentions", []),
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/act/health",
    summary="Estado del motor de acciones",
    description="Verifica que el motor de acciones esté disponible",
)
async def act_health() -> Dict[str, Any]:
    """Estado del motor de acciones."""
    return {
        "status": "ready",
        "available_actions": ["write_note"],
        "timestamp": time.time(),
    }


@router.post(
    "/act/execute",
    summary="Ejecutar acción",
    description="Ejecuta acciones simples como write_note",
)
async def execute_action(
    payload: Dict[str, Any], agent: Any = Depends(get_cognitive_agent)
) -> Dict[str, List[Dict[str, Any]]]:
    """Ejecuta acciones del plan."""
    try:
        plan = payload.get("plan", [])
        results: List[Dict[str, Any]] = []
        for action in plan:
            if isinstance(action, dict) and action.get("name") == "write_note":
                # Simulación de write_note
                results.append(
                    {
                        "action": "write_note",
                        "status": "completed",
                        "params": action.get("params", {}),
                        "timestamp": time.time(),
                    }
                )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════
# 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
# ═══════════════════════════════════════════════════════════════════════


# === ENDPOINTS AVANZADOS: SISTEMA AFECTIVO ===


@router.get(
    "/affect/mood",
    summary="Estado de ánimo actual",
    description="Obtiene el estado emocional actual del sistema",
)
async def get_current_mood(agent: Any = Depends(get_cognitive_agent)) -> Dict[str, Any]:
    """
    Obtiene el estado de ánimo actual completo.
    
    Returns:
        Estado emocional con emociones activas, valencia, energía, etc.
    """
    try:
        if hasattr(agent, "affect_system"):
            mood = agent.affect_system.get_current_mood()
            return mood
        return {"error": "Sistema afectivo no disponible"}
    except Exception as e:
        logger.error(f"Error obteniendo mood: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/affect/insights",
    summary="Insights emocionales",
    description="Análisis profundo del estado emocional",
)
async def get_emotional_insights(agent: Any = Depends(get_cognitive_agent)) -> Dict[str, Any]:
    """
    Obtiene insights profundos sobre el estado emocional.
    
    Returns:
        Análisis con patrones, diversidad emocional, bienestar promedio, etc.
    """
    try:
        if hasattr(agent, "affect_system"):
            insights = agent.affect_system.get_emotional_insights()
            return insights
        return {"error": "Sistema afectivo no disponible"}
    except Exception as e:
        logger.error(f"Error obteniendo insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/affect/trajectory",
    summary="Trayectoria emocional",
    description="Obtiene la trayectoria emocional en el tiempo",
)
async def get_emotional_trajectory(
    hours: int = 24, agent: Any = Depends(get_cognitive_agent)
) -> Dict[str, Any]:
    """
    Obtiene la trayectoria emocional en las últimas N horas.
    
    Args:
        hours: Ventana temporal en horas
        
    Returns:
        Lista de puntos (timestamp, valencia) en el tiempo
    """
    try:
        if hasattr(agent, "affect_system") and hasattr(agent.affect_system, "emotional_memory"):
            trajectory = agent.affect_system.emotional_memory.get_emotional_trajectory(hours)
            return {
                "trajectory": [
                    {"timestamp": t.isoformat(), "valence": v}
                    for t, v in trajectory
                ],
                "hours": hours,
                "points": len(trajectory),
            }
        return {"error": "Memoria emocional no disponible"}
    except Exception as e:
        logger.error(f"Error obteniendo trayectoria: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/affect/trigger",
    summary="Disparar emoción",
    description="Dispara manualmente una emoción en el sistema",
)
async def trigger_emotion(
    emotion_data: Dict[str, Any], agent: Any = Depends(get_cognitive_agent)
) -> Dict[str, Any]:
    """
    Dispara una emoción específica en el sistema afectivo.
    
    Body esperado:
        {
            "emotion_type": "JOY",
            "intensity": 0.8,
            "trigger": "manual_test",
            "context": {...}
        }
        
    Returns:
        Confirmación y datos de la emoción creada
    """
    try:
        if not hasattr(agent, "affect_system"):
            raise HTTPException(status_code=503, detail="Sistema afectivo no disponible")
        
        
        emotion_type_str = emotion_data.get("emotion_type", "JOY")
        emotion_type = EmotionType(emotion_type_str.lower())
        intensity = float(emotion_data.get("intensity", 0.5))
        trigger = emotion_data.get("trigger", "api_manual")
        context = emotion_data.get("context", {})
        
        emotion = agent.affect_system.trigger_emotion(
            emotion_type=emotion_type,
            intensity=intensity,
            trigger=trigger,
            context=context
        )
        
        return {
            "success": True,
            "emotion": {
                "type": emotion.emotion_type.value,
                "intensity": emotion.intensity,
                "trigger": emotion.trigger,
                "valence": emotion.emotional_valence,
                "timestamp": emotion.timestamp.isoformat(),
            }
        }
    except ValueError as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Tipo de emoción inválido: {str(e)}")
    except Exception as e:
        logger.error(f"Error disparando emoción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === ENDPOINTS AVANZADOS: DETECCIÓN DE ANOMALÍAS ===


@router.get(
    "/anomaly/statistics",
    summary="Estadísticas de anomalías",
    description="Obtiene estadísticas completas del detector de anomalías",
)
async def get_anomaly_statistics(agent: Any = Depends(get_cognitive_agent)) -> Dict[str, Any]:
    """
    Obtiene estadísticas completas del detector de anomalías.
    
    Returns:
        Estadísticas por dimensión con medias, tendencias, volatilidad, etc.
    """
    try:
        if hasattr(agent, "anomaly_detector"):
            stats = agent.anomaly_detector.get_statistics()
            return {
                "statistics": stats,
                "dimensions_tracked": len(stats),
                "timestamp": time.time(),
            }
        return {"error": "Detector de anomalías no disponible"}
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/anomaly/performance",
    summary="Rendimiento del detector",
    description="Métricas de rendimiento del detector de anomalías",
)
async def get_anomaly_performance(agent: Any = Depends(get_cognitive_agent)) -> Dict[str, Any]:
    """
    Obtiene métricas de rendimiento del detector.
    
    Returns:
        Métricas: tiempo promedio, tasa de anomalías, patrones aprendidos, etc.
    """
    try:
        if hasattr(agent, "anomaly_detector"):
            metrics = agent.anomaly_detector.get_performance_metrics()
            return metrics
        return {"error": "Detector de anomalías no disponible"}
    except Exception as e:
        logger.error(f"Error obteniendo rendimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/anomaly/patterns",
    summary="Patrones aprendidos",
    description="Obtiene patrones de anomalías aprendidos por el sistema",
)
async def get_learned_patterns(agent: Any = Depends(get_cognitive_agent)) -> Dict[str, Any]:
    """
    Obtiene patrones de anomalías aprendidos.
    
    Returns:
        Diccionario de patrones con sus estadísticas
    """
    try:
        if hasattr(agent, "anomaly_detector"):
            patterns = agent.anomaly_detector.get_learned_patterns()
            return {
                "patterns": patterns,
                "count": len(patterns),
                "timestamp": time.time(),
            }
        return {"error": "Detector de anomalías no disponible"}
    except Exception as e:
        logger.error(f"Error obteniendo patrones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════
# 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA (MÓDULO WRAPPER)
# ═══════════════════════════════════════════════════════════════════════


class ApiModule:
    """
    Wrapper singleton para conectar módulo funcional 'api'
    a la red neuronal simbiótica
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("api", self)
            logger.info("✅ Módulo 'api' registrado en red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo registrar en red neuronal: {e}")
            self.neural_network = None

        self._initialized = True

    def __repr__(self):
        return "<ApiModule: 21 funciones>"


# Instanciar wrapper automáticamente al importar
_module_wrapper = ApiModule()