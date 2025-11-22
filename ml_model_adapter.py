#!/usr/bin/env python3
"""
🔄 ML MODEL ADAPTER - Sistema de Adaptación y Re-entrenamiento
==============================================================

Sistema DUAL que:
    pass  # TODO: Implementar
1. Adapta features en tiempo real para modelos antiguos
2. Re-entrena modelos automáticamente con features nuevas
3. Gestiona transición gradual entre modelos viejos → nuevos

Autor: METACORTEX Team
Fecha: 2025-10-20
"""

import logging
import pickle
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class MLModelAdapter:
    """
    Sistema adaptativo para manejar modelos con diferentes dimensiones de features

    Estrategias:
    1. Feature Selection: Seleccionar subset de features que el modelo espera
    2. Feature Mapping: Mapear features nuevas a features antiguas conocidas
    3. Zero-padding: Rellenar con ceros si el modelo espera MÁS features
    4. Scheduled Retraining: Programar re-entrenamiento automático
    """

    def __init__(self, models_dir: str = "ml_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # Cache de información de modelos
        self.model_info_cache: Dict[str, Dict] = {}

        # Cola de re-entrenamiento
        self.retraining_queue: List[Dict] = []
        self.retraining_lock = threading.Lock()

        # Métricas
        self.adaptations_count = 0
        self.retrainings_scheduled = 0

        logger.info("✅ ML Model Adapter inicializado")

    def adapt_features(
        self, features: np.ndarray, model_id: str, expected_features: int
    ) -> np.ndarray:
        """
        Adapta features dinámicamente según lo que el modelo espera

        Args:
            features: Array de features actual (shape: [1, N])
            model_id: ID del modelo
            expected_features: Número de features que el modelo espera

        Returns:
            Features adaptadas (shape: [1, expected_features])
        """
        current_features = features.shape[1]

        if current_features == expected_features:
            return features  # No necesita adaptación

        self.adaptations_count += 1

        if current_features > expected_features:
            # CASO 1: Tenemos MÁS features de las que el modelo espera
            # Estrategia: Seleccionar las primeras N (más básicas/importantes)
            adapted = features[:, :expected_features]
            logger.debug(
                f"🔄 Adapter: {current_features} → {expected_features} features (selection)"
            )

            # Programar re-entrenamiento para aprovechar TODAS las features
            self._schedule_retraining(model_id, reason="more_features_available")

        else:
            # CASO 2: Tenemos MENOS features de las que el modelo espera
            # Estrategia: Zero-padding (rellenar con ceros)
            padding_needed = expected_features - current_features
            padding = np.zeros((features.shape[0], padding_needed))
            adapted = np.concatenate([features, padding], axis=1)
            logger.debug(
                f"🔄 Adapter: {current_features} → {expected_features} features (padding)"
            )

            # Programar re-entrenamiento con features correctas
            self._schedule_retraining(model_id, reason="insufficient_features")

        return adapted

    def _schedule_retraining(self, model_id: str, reason: str):
        """
        Programa re-entrenamiento automático del modelo

        Args:
            model_id: ID del modelo a re-entrenar
            reason: Razón del re-entrenamiento
        """
        with self.retraining_lock:
            # Verificar si ya está en cola
            if any(item["model_id"] == model_id for item in self.retraining_queue):
                return  # Ya programado

            self.retraining_queue.append(
                {
                    "model_id": model_id,
                    "reason": reason,
                    "scheduled_at": datetime.now().isoformat(),
                    "priority": "high"
                    if reason == "more_features_available"
                    else "medium",
                }
            )

            self.retrainings_scheduled += 1
            logger.info(f"📅 Re-entrenamiento programado: {model_id[:8]} ({reason})")

    def get_retraining_queue(self) -> List[Dict]:
        """Obtiene la cola de re-entrenamientos pendientes"""
        with self.retraining_lock:
            return self.retraining_queue.copy()

    def clear_retraining_queue(self):
        """Limpia la cola de re-entrenamientos"""
        with self.retraining_lock:
            cleared = len(self.retraining_queue)
            self.retraining_queue.clear()
            logger.info(f"🗑️ Cola de re-entrenamiento limpiada: {cleared} items")

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Obtiene información del modelo desde cache o disco

        Args:
            model_id: ID del modelo

        Returns:
            Dict con info del modelo o None si no existe
        """
        if model_id in self.model_info_cache:
            return self.model_info_cache[model_id]

        model_path = self.models_dir / f"{model_id}.pkl"
        if not model_path.exists():
            return None

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            info = {
                "model_id": model_id,
                "type": type(model).__name__,
                "n_features": getattr(model, "n_features_in_", None),
                "trained_at": datetime.fromtimestamp(
                    model_path.stat().st_mtime
                ).isoformat(),
            }

            self.model_info_cache[model_id] = info
            return info

        except Exception as e:
            logger.error(f"❌ Error obteniendo info del modelo {model_id[:8]}: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del adaptador"""
        return {
            "adaptations_count": self.adaptations_count,
            "retrainings_scheduled": self.retrainings_scheduled,
            "retraining_queue_size": len(self.retraining_queue),
            "models_cached": len(self.model_info_cache),
        }


# Singleton
_global_adapter: Optional[MLModelAdapter] = None


def get_model_adapter(**kwargs) -> MLModelAdapter:
    """Obtener instancia global del adaptador"""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = MLModelAdapter(**kwargs)
    return _global_adapter


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    adapter = get_model_adapter()

    # Simular adaptación
    features = np.random.rand(1, 28)
    adapted = adapter.adapt_features(features, "test_model_123", expected_features=6)

    print(f"\n✅ Features: {features.shape} → {adapted.shape}")
    print(f"📊 Stats: {adapter.get_stats()}")
    print(f"📅 Re-training queue: {adapter.get_retraining_queue()}")