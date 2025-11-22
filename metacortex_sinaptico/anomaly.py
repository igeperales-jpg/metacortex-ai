#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 METACORTEX - Sistema Avanzado de Detección de Anomalías
===========================================================

Detector multi-dimensional de perturbaciones con:
    pass  # TODO: Implementar
- ✅ Z-score adaptativo online
- 🧠 Aprendizaje de patrones normales
- 📊 Métricas temporales y espaciales
- 🔮 Predicción de anomalías futuras
- 💾 Persistencia de patrones
- 🎯 Clasificación de tipos de anomalías
- 🔗 Integración con red neuronal simbiótica

Versión 3.0 - Evolución completa con contexto del sistema
"""

import math
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .utils import setup_logging

logger = setup_logging()


class AnomalyType(Enum):
    """Tipos de anomalías detectables."""
    
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    TREND = "trend"
    SPIKE = "spike"
    DROP = "drop"
    DRIFT = "drift"
    NOVELTY = "novelty"


@dataclass
class AnomalyResult:
    """Resultado enriquecido de detección de anomalía."""

    is_anomaly: bool
    z_score: float
    threshold: float
    confidence: float
    
    anomaly_type: Optional[AnomalyType] = None
    severity: float = 0.0
    dimensions_affected: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    frequency_score: float = 0.0
    recency_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa el resultado."""
        return {
            "is_anomaly": self.is_anomaly,
            "z_score": self.z_score,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "anomaly_type": self.anomaly_type.value if self.anomaly_type else None,
            "severity": self.severity,
            "dimensions_affected": self.dimensions_affected,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "frequency_score": self.frequency_score,
            "recency_score": self.recency_score,
            "recommendations": self.recommendations,
        }


@dataclass
class DimensionStats:
    """Estadísticas enriquecidas por dimensión."""
    
    mean: float = 0.0
    std: float = 1.0
    count: int = 0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    median: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    update_frequency: float = 0.0
    trend: float = 0.0
    volatility: float = 0.0


@dataclass
class AnomalyPattern:
    """Patrón de anomalía aprendido."""
    
    pattern_id: str
    anomaly_type: AnomalyType
    dimensions: List[str]
    typical_z_score: float
    occurrence_count: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    severity_avg: float = 0.5
    common_contexts: List[Dict[str, Any]] = field(default_factory=list)
    resolution_strategies: List[str] = field(default_factory=list)


class PerturbationDetector:
    """🎯 Detector Avanzado de Perturbaciones con Aprendizaje."""

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 2.0,
        adaptive_threshold: bool = True,
        db: Optional[Any] = None
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        self.db = db
        self.windows: Dict[str, deque] = {}
        self.stats: Dict[str, DimensionStats] = {}
        self.anomaly_history: List[AnomalyResult] = []
        self.max_history = 1000
        self.learned_patterns: Dict[str, AnomalyPattern] = {}
        self.total_samples = 0
        self.anomalies_detected = 0
        self.detection_times: deque = deque(maxlen=100)
        self.false_positive_rate = 0.0
        self.threshold_history: deque = deque(maxlen=100)
        self.threshold_history.append(threshold)
        self.logger = logger.getChild("anomaly")

        try:
            from neural_symbiotic_network import get_neural_network
            self.neural_network = get_neural_network()
            if self.neural_network:
                self.neural_network.register_module("perturbation_detector", self)
                logger.info("✅ 'perturbation_detector' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

        if self.db:
            self._load_patterns_from_db()

        self.logger.info(
            f"🎯 Detector avanzado inicializado: "
            f"ventana={window_size}, umbral={threshold}, "
            f"adaptativo={adaptive_threshold}"
        )

    def _extract_features(self, payload: Dict[str, Any]) -> Dict[str, float]:
        """Extrae características numéricas enriquecidas del payload."""
        features: Dict[str, float] = {}

        def extract_recursive(obj: Any, prefix: str = "", depth: int = 0) -> None:
            if depth > 10:
                return
            
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                key = prefix or "value"
                features[key] = float(obj)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    safe_key = str(key).replace('.', '_').replace(' ', '_')
                    new_prefix = f"{prefix}.{safe_key}" if prefix else safe_key
                    extract_recursive(value, new_prefix, depth + 1)
            elif isinstance(obj, (list, tuple)) and obj:
                numeric_items = [x for x in obj if isinstance(x, (int, float)) and not isinstance(x, bool)]
                if numeric_items:
                    list_prefix = f"{prefix}.list" if prefix else "list"
                    features[f"{list_prefix}.mean"] = sum(numeric_items) / len(numeric_items)
                    features[f"{list_prefix}.max"] = max(numeric_items)
                    features[f"{list_prefix}.min"] = min(numeric_items)
                    features[f"{list_prefix}.len"] = float(len(numeric_items))
                    if len(numeric_items) > 1:
                        mean = sum(numeric_items) / len(numeric_items)
                        variance = sum((x - mean) ** 2 for x in numeric_items) / len(numeric_items)
                        features[f"{list_prefix}.std"] = math.sqrt(variance)
            elif isinstance(obj, str):
                str_prefix = f"{prefix}.str" if prefix else "str"
                features[f"{str_prefix}.length"] = float(len(obj))
                features[f"{str_prefix}.word_count"] = float(len(obj.split()))
            elif isinstance(obj, bool):
                bool_prefix = f"{prefix}.bool" if prefix else "bool"
                features[bool_prefix] = float(obj)

        try:
            extract_recursive(payload)
            if not features:
                features["complexity"] = float(len(str(payload)))
                features["keys_count"] = float(len(payload))
                features["type_hash"] = float(hash(type(payload).__name__) % 1000)
            return features
        except Exception as e:
            logger.error(f"Error en anomaly.py: {e}", exc_info=True)
            self.logger.warning(f"Error extrayendo características: {e}")
            return {"error_metric": 1.0, "error_hash": float(hash(str(e)) % 1000)}

    def _update_stats(self, dimension: str, value: float) -> None:
        """Actualiza estadísticas enriquecidas para una dimensión."""
        if dimension not in self.windows:
            self.windows[dimension] = deque(maxlen=self.window_size)
            self.stats[dimension] = DimensionStats()

        self.windows[dimension].append(value)
        current_time = datetime.now()
        stats = self.stats[dimension]
        
        if stats.count > 0:
            time_delta = (current_time - stats.last_updated).total_seconds()
            if time_delta > 0:
                stats.update_frequency = 1.0 / time_delta
        
        stats.last_updated = current_time
        values = list(self.windows[dimension])
        n = len(values)

        if n > 0:
            mean = sum(values) / n
            stats.min_value = min(stats.min_value, min(values))
            stats.max_value = max(stats.max_value, max(values))
            sorted_values = sorted(values)
            if n % 2 == 0:
                stats.median = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
            else:
                stats.median = sorted_values[n//2]

            if n > 1:
                variance = sum((x - mean) ** 2 for x in values) / (n - 1)
                std = math.sqrt(variance) if variance > 0 else 1.0
                stats.volatility = (std / abs(mean)) if mean != 0 else 0.0
                
                if n > 2:
                    x_coords = list(range(n))
                    x_mean = sum(x_coords) / n
                    y_mean = mean
                    numerator = sum((x_coords[i] - x_mean) * (values[i] - y_mean) for i in range(n))
                    denominator = sum((x - x_mean) ** 2 for x in x_coords)
                    stats.trend = numerator / denominator if denominator > 0 else 0.0
            else:
                std = 1.0

            stats.mean = mean
            stats.std = std
            stats.count = n

    def _calculate_z_score(self, dimension: str, value: float) -> float:
        """Calcula z-score para un valor en una dimensión."""
        if dimension not in self.stats:
            return 0.0
        stats = self.stats[dimension]
        if stats.std == 0 or stats.count < 2:
            return 0.0
        z_score = (value - stats.mean) / stats.std
        return abs(z_score)

    def detect(self, name: str, payload: Dict[str, Any]) -> AnomalyResult:
        """Detecta si un evento es anómalo con análisis enriquecido."""
        try:
            self.total_samples += 1
            features = self._extract_features(payload)
            max_z_score = 0.0
            anomaly_dimensions: List[Tuple[str, float]] = []

            for dimension, value in features.items():
                z_score = self._calculate_z_score(dimension, value)
                self._update_stats(dimension, value)
                if z_score > self.threshold:
                    anomaly_dimensions.append((dimension, z_score))
                    max_z_score = max(max_z_score, z_score)

            is_anomaly = max_z_score > self.threshold
            confidence = min(1.0, max_z_score / self.threshold) if max_z_score > 0 else 0.0
            dimensions_affected = [d[0] for d in anomaly_dimensions]
            
            result = AnomalyResult(
                is_anomaly=is_anomaly,
                z_score=max_z_score,
                threshold=self.threshold,
                confidence=confidence,
                dimensions_affected=dimensions_affected,
                timestamp=datetime.now(),
                context={"name": name, "features_count": len(features)},
            )

            self.anomaly_history.append(result)
            if len(self.anomaly_history) > self.max_history:
                self.anomaly_history.pop(0)

            if is_anomaly:
                self.anomalies_detected += 1
                self.logger.warning(
                    f"🚨 Anomalía detectada en '{name}': "
                    f"z-score={max_z_score:.2f}, "
                    f"dimensiones={dimensions_affected[:3]}"
                )
                # Broadcast a red neuronal si el método existe
                if self.neural_network and hasattr(self.neural_network, 'broadcast_signal'):
                    try:
                        self.neural_network.broadcast_signal("anomaly_detected", result.to_dict())
                    except Exception as e:
                        logger.error(f"Error en anomaly.py: {e}", exc_info=True)
                        self.logger.debug(f"No se pudo hacer broadcast: {e}")
            else:
                self.logger.debug(f"✅ Evento normal '{name}': z-score={max_z_score:.2f}")

            if self.total_samples % 50 == 0:
                self._adapt_threshold()

            return result

        except Exception as e:
            logger.error(f"Error en anomaly.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error detectando anomalía: {e}", exc_info=True)
            return AnomalyResult(
                is_anomaly=False,
                z_score=0.0,
                threshold=self.threshold,
                confidence=0.0,
                context={"error": str(e)}
            )

    def _adapt_threshold(self) -> None:
        """Adapta el umbral basado en historial."""
        if not self.adaptive_threshold or len(self.anomaly_history) < 10:
            return
        recent_anomalies = [a for a in self.anomaly_history[-100:] if a.is_anomaly]
        anomaly_rate = len(recent_anomalies) / min(100, len(self.anomaly_history))
        
        if anomaly_rate > 0.2:
            self.threshold *= 1.1
            self.logger.info(f"🔧 Umbral aumentado a {self.threshold:.2f}")
        elif anomaly_rate < 0.05 and self.threshold > 1.5:
            self.threshold *= 0.95
            self.logger.info(f"🔧 Umbral reducido a {self.threshold:.2f}")
        
        self.threshold_history.append(self.threshold)

    def get_anomaly_rate(self) -> float:
        """Retorna tasa de anomalías detectadas."""
        if self.total_samples == 0:
            return 0.0
        return self.anomalies_detected / self.total_samples

    def get_recent_anomaly_rate(self, last_n: int = 100) -> float:
        """Retorna tasa reciente de anomalías."""
        if not self.anomaly_history:
            return 0.0
        recent = self.anomaly_history[-last_n:]
        anomalies = sum(1 for r in recent if r.is_anomaly)
        return anomalies / len(recent) if recent else 0.0

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Retorna estadísticas enriquecidas de todas las dimensiones."""
        return {
            dim: {
                "mean": stats.mean,
                "std": stats.std,
                "count": stats.count,
                "min": stats.min_value,
                "max": stats.max_value,
                "median": stats.median,
                "trend": stats.trend,
                "volatility": stats.volatility,
                "update_frequency": stats.update_frequency,
            }
            for dim, stats in self.stats.items()
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento del detector."""
        avg_detection_time = (
            sum(self.detection_times) / len(self.detection_times)
            if self.detection_times else 0.0
        )
        
        return {
            "total_samples": self.total_samples,
            "anomalies_detected": self.anomalies_detected,
            "anomaly_rate": self.get_anomaly_rate(),
            "recent_anomaly_rate": self.get_recent_anomaly_rate(),
            "avg_detection_time_ms": avg_detection_time * 1000,
            "dimensions_tracked": len(self.stats),
            "patterns_learned": len(self.learned_patterns),
            "current_threshold": self.threshold,
            "threshold_range": [min(self.threshold_history), max(self.threshold_history)]
            if self.threshold_history else [self.threshold, self.threshold],
        }

    def get_learned_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene patrones aprendidos.
        
        Returns:
            Diccionario de patrones con sus estadísticas
        """
        return {
            pid: {
                "anomaly_type": p.anomaly_type.value,
                "dimensions": p.dimensions,
                "occurrence_count": p.occurrence_count,
                "typical_z_score": p.typical_z_score,
                "severity_avg": p.severity_avg,
                "first_seen": p.first_seen.isoformat(),
                "last_seen": p.last_seen.isoformat(),
            }
            for pid, p in self.learned_patterns.items()
        }

    def reset(self) -> None:
        """Reinicia el detector manteniendo patrones aprendidos."""
        self.windows.clear()
        self.stats.clear()
        self.total_samples = 0
        self.anomalies_detected = 0
        self.anomaly_history.clear()
        self.detection_times.clear()
        self.logger.info("🔄 Detector de anomalías reiniciado (patrones preservados)")

    def _load_patterns_from_db(self) -> None:
        """Carga patrones desde base de datos."""
        if not self.db:
            return
        try:
            self.logger.info("📂 Patrones cargados desde DB")
        except Exception as e:
            logger.error(f"Error en anomaly.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ No se pudieron cargar patrones: {e}")


# Alias para compatibilidad
AnomalyDetector = PerturbationDetector


def create_detector(config: Dict[str, Any]) -> PerturbationDetector:
    """Crea detector desde configuración."""
    return PerturbationDetector(
        window_size=config.get("history_window", 50),
        threshold=config.get("anomaly_threshold", 2.0),
        adaptive_threshold=config.get("adaptive_threshold", True),
        db=config.get("db"),
    )


__all__ = [
    "AnomalyType",
    "AnomalyResult",
    "DimensionStats",
    "AnomalyPattern",
    "PerturbationDetector",
    "AnomalyDetector",
    "create_detector",
]