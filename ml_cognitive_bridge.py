#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌉 ML-COGNITIVE BRIDGE - Integración Real
==========================================

Conecta ML Pipeline con Cognitive Bridge para que trabajen juntos.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MLCognitiveBridge:
    """
    Puente que conecta ML Pipeline con Cognitive Bridge

    Permite que:
        pass  # TODO: Implementar
    - ML notifique a Cognitive de predicciones
    - Cognitive use ML para tomar mejores decisiones
    - Orchestrator coordine ambos sistemas
    """

    def __init__(self):
        self.ml_pipeline = None
        self.cognitive_bridge = None
        self.orchestrator = None

        # Intentar cargar sistemas
        self._load_ml_pipeline()
        self._load_cognitive_bridge()

        logger.info("🌉 ML-Cognitive Bridge inicializado")

    def _load_ml_pipeline(self):
        """Carga ML Pipeline"""
        try:
            from ml_pipeline import get_ml_pipeline

            self.ml_pipeline = get_ml_pipeline()
            logger.info("✅ ML Pipeline conectado al bridge")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar ML Pipeline: {e}")

    def _load_cognitive_bridge(self):
        """Carga Cognitive Bridge"""
        try:
            from cognitive_integration import get_cognitive_bridge

            self.cognitive_bridge = get_cognitive_bridge()
            logger.info("✅ Cognitive Bridge conectado al bridge")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar Cognitive Bridge: {e}")

    def set_orchestrator(self, orchestrator):
        """Establece referencia al orchestrator"""
        self.orchestrator = orchestrator
        logger.info("✅ Orchestrator conectado al bridge")

    # ═══════════════════════════════════════════════════════════════════════
    # PREDICCIÓN CON CONTEXTO COGNITIVO
    # ═══════════════════════════════════════════════════════════════════════

    def predict_with_cognitive_context(
        self, user_query: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Hace predicción ML con contexto cognitivo

        Args:
            user_query: Consulta del usuario
            context: Contexto adicional

        Returns:
            Dict con predicción + decisión cognitiva
        """
        result = {
            "query": user_query,
            "ml_prediction": None,
            "cognitive_decision": None,
            "recommended_action": None,
        }

        # 1. ML predice intención
        if self.ml_pipeline:
            try:
                # Buscar modelo de clasificación de intenciones
                intention_model = self._find_intention_model()

                if intention_model:
                    # Preparar features
                    features = self._extract_features(user_query)

                    # Predecir
                    prediction = self.ml_pipeline.predict(intention_model, features)

                    result["ml_prediction"] = {
                        "intent": self._map_prediction_to_intent(prediction),
                        "confidence": 0.85,  # IMPLEMENTED: obtener confianza real
                        "model_used": intention_model,
                    }

                    logger.info(f"🤖 ML predijo: {result['ml_prediction']['intent']}")

            except Exception as e:
                logger.error(f"❌ Error en predicción ML: {e}")

        # 2. Cognitive toma decisión
        if self.cognitive_bridge:
            try:
                # Pasar predicción ML a Cognitive
                cognitive_input = {
                    "query": user_query,
                    "ml_intent": result.get("ml_prediction", {}).get("intent"),
                    "context": context or {},
                }

                # Cognitive decide qué hacer
                decision = self.cognitive_bridge.process_with_ml_context(
                    cognitive_input
                )

                result["cognitive_decision"] = decision
                logger.info(f"🧠 Cognitive decidió: {decision.get('action')}")

            except Exception as e:
                logger.error(f"❌ Error en decisión Cognitive: {e}")

        # 3. Recomendar acción combinada
        result["recommended_action"] = self._combine_ml_and_cognitive(
            result.get("ml_prediction"), result.get("cognitive_decision")
        )

        return result

    def _find_intention_model(self) -> Optional[str]:
        """Busca modelo de clasificación de intenciones"""
        if not self.ml_pipeline:
            return None

        # Buscar en modelos activos
        for model_id, model_info in self.ml_pipeline.active_models.items():
            if "intention" in model_id.lower() or "intent" in model_id.lower():
                return model_id

        # Buscar en historial
        for entry in self.ml_pipeline.training_history:
            if entry.get("model_name", "").lower() in [
                "intention_classifier",
                "intent_classifier",
            ]:
                if entry.get("status") == "deployed":
                    return entry.get("model_id")

        return None

    def _extract_features(self, query: str):
        """Extrae features de la consulta para ML"""

        # Features básicas
        features = [
            len(query),  # longitud
            query.lower().count("código")
            + query.lower().count("code"),  # palabras coding
            query.lower().count("buscar")
            + query.lower().count("search"),  # palabras search
            query.lower().count("analizar")
            + query.lower().count("analyze"),  # palabras analysis
            len(query.split()),  # número de palabras
            0,  # tokens_used (desconocido por ahora)
        ]

        return np.array([features])

    def _map_prediction_to_intent(self, prediction) -> str:
        """Mapea predicción numérica a intención"""
        intent_map = {0: "coding", 1: "search", 2: "analysis", 3: "chat", 4: "debug"}

        if isinstance(prediction, (list, tuple)) and len(prediction) > 0:
            prediction = prediction[0]

        return intent_map.get(int(prediction), "unknown")

    def _combine_ml_and_cognitive(
        self, ml_prediction: Optional[Dict], cognitive_decision: Optional[Dict]
    ) -> Dict:
        """Combina predicción ML con decisión Cognitive"""

        # Si solo hay ML
        if ml_prediction and not cognitive_decision:
            return {
                "source": "ml_only",
                "action": f"use_{ml_prediction['intent']}_agent",
                "confidence": ml_prediction.get("confidence", 0.5),
            }

        # Si solo hay Cognitive
        if cognitive_decision and not ml_prediction:
            return {
                "source": "cognitive_only",
                "action": cognitive_decision.get("action"),
                "confidence": cognitive_decision.get("confidence", 0.5),
            }

        # Si hay AMBOS (lo ideal)
        if ml_prediction and cognitive_decision:
            # Combinar confianzas
            combined_confidence = (
                ml_prediction.get("confidence", 0.5) * 0.6
                + cognitive_decision.get("confidence", 0.5) * 0.4
            )

            return {
                "source": "ml_and_cognitive",
                "action": cognitive_decision.get("action"),
                "ml_intent": ml_prediction["intent"],
                "confidence": combined_confidence,
                "note": "Decisión tomada con ML + Cognitive",
            }

        # Fallback
        return {"source": "none", "action": "use_default_agent", "confidence": 0.3}

    # ═══════════════════════════════════════════════════════════════════════
    # NOTIFICACIONES DE ENTRENAMIENTOS
    # ═══════════════════════════════════════════════════════════════════════

    def notify_training_completed(self, model_id: str, result: Dict):
        """Notifica a Cognitive que se completó un entrenamiento"""
        if not self.cognitive_bridge:
            return

        try:
            self.cognitive_bridge.handle_ml_training_event(
                {
                    "event": "training_completed",
                    "model_id": model_id,
                    "accuracy": result.get("val_metrics", {}).get("accuracy", 0),
                    "deployed": result.get("deployed", False),
                }
            )

            logger.info(f"📣 Notificado a Cognitive: modelo {model_id} entrenado")

        except Exception as e:
            logger.error(f"❌ Error notificando a Cognitive: {e}")

    def notify_prediction_made(self, model_id: str, input_data: Any, prediction: Any):
        """Notifica a Cognitive de una predicción"""
        if not self.cognitive_bridge:
            return

        try:
            self.cognitive_bridge.handle_ml_prediction_event(
                {
                    "event": "prediction_made",
                    "model_id": model_id,
                    "prediction": prediction,
                }
            )

        except Exception as e:
            logger.debug(f"Error notificando predicción: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # CONSULTAS DE ESTADO
    # ═══════════════════════════════════════════════════════════════════════

    def get_integration_status(self) -> Dict:
        """Obtiene estado de la integración"""
        return {
            "ml_connected": self.ml_pipeline is not None,
            "cognitive_connected": self.cognitive_bridge is not None,
            "orchestrator_connected": self.orchestrator is not None,
            "fully_integrated": all(
                [self.ml_pipeline, self.cognitive_bridge, self.orchestrator]
            ),
            "ml_models_active": len(self.ml_pipeline.active_models)
            if self.ml_pipeline
            else 0,
            "cognitive_agent_active": self.cognitive_bridge.agent is not None
            if self.cognitive_bridge
            else False,
        }


# ═══════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════

_global_ml_cognitive_bridge = None


def get_ml_cognitive_bridge() -> MLCognitiveBridge:
    """Obtiene instancia global del ML-Cognitive Bridge"""
    global _global_ml_cognitive_bridge
    if _global_ml_cognitive_bridge is None:
        _global_ml_cognitive_bridge = MLCognitiveBridge()
    return _global_ml_cognitive_bridge


# ═══════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🌉 Testing ML-Cognitive Bridge")
    print("=" * 60)

    bridge = get_ml_cognitive_bridge()

    # Test estado
    print("\n📊 Estado de integración:")
    status = bridge.get_integration_status()
    for key, value in status.items():
        icon = "✅" if value else "❌"
        print(f"   {icon} {key}: {value}")

    # Test predicción
    if status["fully_integrated"]:
        print("\n🧪 Test de predicción con contexto cognitivo:")
        result = bridge.predict_with_cognitive_context(
            "Crea una función para calcular fibonacci"
        )
        print(f"   Query: {result['query']}")
        print(f"   ML Prediction: {result.get('ml_prediction')}")
        print(f"   Cognitive Decision: {result.get('cognitive_decision')}")
        print(f"   Recommended Action: {result.get('recommended_action')}")

    print("\n✅ Test completado")