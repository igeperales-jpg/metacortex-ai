#!/usr/bin/env python3
"""
📊 METRICS LOGGER & ALERT MANAGER - TAREA 14
=============================================

Sistema de logging de métricas e inteligencia de alertas para la red neuronal.

Características:
    pass  # TODO: Implementar
- MetricsLogger: Guarda métricas automáticamente en DB cada vez que se ejecuta un agente
- AlertManager: Detecta condiciones anómalas y genera alertas
- Auto-optimización basada en métricas históricas

Integración con planning.py y db.py
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

from neural_symbiotic_network import get_neural_network
import json
import csv


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Niveles de severidad de alertas."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Tipos de alertas."""

    LOW_SUCCESS_RATE = "low_success_rate"
    SLOW_EXECUTION = "slow_execution"
    AGENT_INACTIVE = "agent_inactive"
    HIGH_FAILURE_RATE = "high_failure_rate"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class AlertThresholds:
    """Thresholds para generación de alertas."""

    min_success_rate: float = 0.5  # 50%
    max_avg_execution_time: float = 10.0  # 10 segundos
    max_inactive_days: int = 7  # 7 días
    max_failure_rate: float = 0.5  # 50%
    min_executions_for_alert: int = 5  # Mínimo de ejecuciones antes de alertar


class MetricsLogger:
    """
    Logger de métricas de agentes.

    Guarda automáticamente métricas en base de datos cada vez que:
    - Se ejecuta un agente (record_success/record_failure)
    - Se actualizan conexiones
    """

    def __init__(self, db: Any):
        """
        Args:
            db: Instancia de MetacortexDB
        """
        self.db = db
        self.last_save_time: Dict[str, float] = {}  # agent_name -> timestamp
        self.save_interval: float = 60.0  # Guardar cada 60 segundos como máximo
        self.metrics_buffer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.real_time_metrics: Dict[str, Any] = {}
        self.aggregation_window: int = 300  # 5 minutos

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            self.neural_network = get_neural_network()
            self.neural_network.register_module("metrics_system", self)
            logger.info("✅ 'metrics_system' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

    def log_agent_metrics(self, agent_name: str, connection: Any) -> bool:
        """
        Guarda métricas de un agente en la base de datos.

        Args:
            agent_name: Nombre del agente
            connection: Instancia de NeuralConnection

        Returns:
            True si se guardó, False si se skippeó (muy reciente)
        """
        try:
            # Verificar si es necesario guardar (no guardar muy frecuentemente)
            now = time.time()
            last_save = self.last_save_time.get(agent_name, 0)

            if now - last_save < self.save_interval:
                return False  # Muy reciente, skip

            # Preparar métricas
            metrics = {
                "timestamp": now,
                "connection_strength": connection.connection_strength,
                "usage_count": connection.usage_count,
                "success_count": connection.success_count,
                "failure_count": connection.failure_count,
                "avg_execution_time": connection.avg_execution_time,
                "success_rate": connection.success_rate,
                "total_executions": connection.total_executions,
                "active": connection.active,
                "metadata": {
                    "capabilities": connection.capabilities,
                    "agent_path": connection.agent_path,
                    "last_used": connection.last_used.isoformat()
                    if connection.last_used
                    else None,
                },
            }

            # Guardar en DB
            self.db.save_agent_metrics(agent_name, metrics)
            self.last_save_time[agent_name] = now

            logger.debug(f"📊 Métricas guardadas: {agent_name}")
            return True

        except Exception as e:
            logger.error(f"Error guardando métricas de {agent_name}: {e}")
            return False

    def force_log_all_agents(self, neural_network: Dict[str, Any]) -> int:
        """
        Fuerza el guardado de métricas de todos los agentes.

        Args:
            neural_network: Diccionario de NeuralConnection

        Returns:
            Número de agentes guardados
        """
        count = 0
        # Temporalmente deshabilitar save_interval
        original_interval = self.save_interval
        self.save_interval = 0

        try:
            for agent_name, connection in neural_network.items():
                if self.log_agent_metrics(agent_name, connection):
                    count += 1
        finally:
            self.save_interval = original_interval

            logger.info(f"📊 Métricas forzadas guardadas: {count} agentes")
        return count

    def get_real_time_metrics(self, agent_name: str) -> Dict[str, Any]:
        """Obtiene métricas en tiempo real de un agente."""
        return self.real_time_metrics.get(agent_name, {})

    def get_aggregated_metrics(
        self, agent_name: str, window_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Obtiene métricas agregadas en ventana temporal.
        
        Args:
            agent_name: Nombre del agente
            window_seconds: Ventana de tiempo en segundos
            
        Returns:
            Métricas agregadas (avg, min, max, stddev)
        """
        buffer = self.metrics_buffer.get(agent_name, [])
        if not buffer:
            return {}

        now = time.time()
        recent_metrics = [
            m for m in buffer if now - m.get("timestamp", 0) <= window_seconds
        ]

        if not recent_metrics:
            return {}

        # Calcular agregaciones
        success_rates = [m.get("success_rate", 0) for m in recent_metrics]
        exec_times = [m.get("avg_execution_time", 0) for m in recent_metrics]

        return {
            "count": len(recent_metrics),
            "success_rate": {
                "avg": statistics.mean(success_rates) if success_rates else 0,
                "min": min(success_rates) if success_rates else 0,
                "max": max(success_rates) if success_rates else 0,
                "stddev": statistics.stdev(success_rates) if len(success_rates) > 1 else 0,
            },
            "execution_time": {
                "avg": statistics.mean(exec_times) if exec_times else 0,
                "min": min(exec_times) if exec_times else 0,
                "max": max(exec_times) if exec_times else 0,
                "stddev": statistics.stdev(exec_times) if len(exec_times) > 1 else 0,
            },
        }
class AlertManager:
    """
    Manager de alertas inteligentes.

    Detecta condiciones anómalas y genera alertas:
    - Success rate bajo
    - Ejecuciones muy lentas
    - Agentes inactivos
    - Degradación de rendimiento
    """

    def __init__(self, db: Any, thresholds: Optional[AlertThresholds] = None):
        """
        Args:
            db: Instancia de MetacortexDB
            thresholds: Thresholds personalizados (opcional)
        """
        self.db = db
        self.thresholds = thresholds or AlertThresholds()
        self.checked_agents: Set[str] = set()  # Para evitar alertas duplicadas en ciclo actual
        self.alert_history: List[Dict[str, Any]] = []
        self.predictive_alerts_enabled: bool = True

    def check_agent(self, agent_name: str, connection: Any) -> List[Dict[str, Any]]:
        """
        Verifica un agente y genera alertas si es necesario.

        Args:
            agent_name: Nombre del agente
            connection: Instancia de NeuralConnection

        Returns:
            Lista de alertas generadas
        """
        alerts: List[Dict[str, Any]] = []

        # No alertar si ya se checó en este ciclo
        if agent_name in self.checked_agents:
            return alerts

        # Marcar como checado
        self.checked_agents.add(agent_name)

        # 1. Check success rate (solo si tiene suficientes ejecuciones)
        if connection.total_executions >= self.thresholds.min_executions_for_alert:
            if connection.success_rate < self.thresholds.min_success_rate:
                severity = self._calculate_severity(
                    connection.success_rate,
                    self.thresholds.min_success_rate,
                    lower_is_worse=True,
                )

                alert_id = self.db.save_alert(
                    agent_name=agent_name,
                    alert_type=AlertType.LOW_SUCCESS_RATE.value,
                    severity=severity.value,
                    message=f"Success rate bajo: {connection.success_rate:.1%} (threshold: {self.thresholds.min_success_rate:.1%})",
                    metric_value=connection.success_rate,
                    threshold_value=self.thresholds.min_success_rate,
                    metadata={
                        "total_executions": connection.total_executions,
                        "success_count": connection.success_count,
                        "failure_count": connection.failure_count,
                    },
                )

                alerts.append(
                    {
                        "id": alert_id,
                        "type": AlertType.LOW_SUCCESS_RATE.value,
                        "severity": severity.value,
                        "agent_name": agent_name,
                    }
                )

                logger.warning(
                    f"🚨 Alerta LOW_SUCCESS_RATE: {agent_name} ({connection.success_rate:.1%})"
                )

        # 2. Check ejecución lenta (solo si tiene ejecuciones)
        if connection.total_executions > 0:
            if connection.avg_execution_time > self.thresholds.max_avg_execution_time:
                severity = self._calculate_severity(
                    connection.avg_execution_time,
                    self.thresholds.max_avg_execution_time,
                    lower_is_worse=False,
                )

                alert_id = self.db.save_alert(
                    agent_name=agent_name,
                    alert_type=AlertType.SLOW_EXECUTION.value,
                    severity=severity.value,
                    message=f"Ejecución lenta: {connection.avg_execution_time:.2f}s (threshold: {self.thresholds.max_avg_execution_time:.2f}s)",
                    metric_value=connection.avg_execution_time,
                    threshold_value=self.thresholds.max_avg_execution_time,
                    metadata={"total_executions": connection.total_executions},
                )

                alerts.append(
                    {
                        "id": alert_id,
                        "type": AlertType.SLOW_EXECUTION.value,
                        "severity": severity.value,
                        "agent_name": agent_name,
                    }
                )

                logger.warning(
                    f"🚨 Alerta SLOW_EXECUTION: {agent_name} ({connection.avg_execution_time:.2f}s)"
                )

        # 3. Check inactividad
        if connection.last_used:
            days_inactive = (datetime.now() - connection.last_used).days
            if days_inactive > self.thresholds.max_inactive_days:
                severity = (
                    AlertSeverity.LOW if days_inactive < 30 else AlertSeverity.MEDIUM
                )

                alert_id = self.db.save_alert(
                    agent_name=agent_name,
                    alert_type=AlertType.AGENT_INACTIVE.value,
                    severity=severity.value,
                    message=f"Agente inactivo: {days_inactive} días sin uso (threshold: {self.thresholds.max_inactive_days} días)",
                    metric_value=float(days_inactive),
                    threshold_value=float(self.thresholds.max_inactive_days),
                    metadata={"last_used": connection.last_used.isoformat()},
                )

                alerts.append(
                    {
                        "id": alert_id,
                        "type": AlertType.AGENT_INACTIVE.value,
                        "severity": severity.value,
                        "agent_name": agent_name,
                    }
                )

                logger.warning(
                    f"🚨 Alerta AGENT_INACTIVE: {agent_name} ({days_inactive} días)"
                )

        return alerts

    def _calculate_severity(
        self, value: float, threshold: float, lower_is_worse: bool = True
    ) -> AlertSeverity:
        """
        Calcula la severidad basándose en qué tan lejos está el valor del threshold.

        Args:
            value: Valor actual
            threshold: Valor del threshold
            lower_is_worse: Si True, valores menores son peores

        Returns:
            Severidad calculada
        """
        if lower_is_worse:
            ratio = value / threshold  # < 1 es malo
            if ratio < 0.3:
                return AlertSeverity.CRITICAL
            elif ratio < 0.5:
                return AlertSeverity.HIGH
            elif ratio < 0.75:
                return AlertSeverity.MEDIUM
            else:
                return AlertSeverity.LOW
        else:
            ratio = value / threshold  # > 1 es malo
            if ratio > 3.0:
                return AlertSeverity.CRITICAL
            elif ratio > 2.0:
                return AlertSeverity.HIGH
            elif ratio > 1.5:
                return AlertSeverity.MEDIUM
            else:
                return AlertSeverity.LOW

    def check_all_agents(self, neural_network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verifica todos los agentes y genera alertas.

        Args:
            neural_network: Diccionario de NeuralConnection

        Returns:
            Lista de todas las alertas generadas
        """
        all_alerts = []

        # Reset checked agents para nuevo ciclo
        self.checked_agents.clear()

        for agent_name, connection in neural_network.items():
            alerts = self.check_agent(agent_name, connection)
            all_alerts.extend(alerts)

        if all_alerts:
            logger.info(f"🚨 Total alertas generadas: {len(all_alerts)}")

        return all_alerts

    def get_active_alerts_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de alertas activas."""
        try:
            stats = self.db.get_alert_statistics()
            active_alerts = self.db.get_active_alerts(limit=10)

            return {
                "statistics": stats,
                "recent_alerts": active_alerts,
                "critical_count": stats["by_severity"].get("critical", 0),
                "high_count": stats["by_severity"].get("high", 0),
            }
        except Exception as e:
            logger.error(f"Error obteniendo resumen de alertas: {e}")
            return {"error": str(e)}

    def predict_future_alerts(
        self, agent_name: str, metrics_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Predice alertas futuras basándose en tendencias.
        
        Args:
            agent_name: Nombre del agente
            metrics_history: Historial de métricas
            
        Returns:
            Lista de alertas predictivas
        """
        if not self.predictive_alerts_enabled or len(metrics_history) < 10:
            return []

        predictive_alerts: List[Dict[str, Any]] = []

        # Analizar tendencia de success_rate
        recent_success_rates = [
            m.get("success_rate", 1.0) for m in metrics_history[-10:]
        ]
        
        if len(recent_success_rates) >= 3:
            # Calcular tendencia (slope)
            avg_recent = statistics.mean(recent_success_rates[-3:])
            avg_older = statistics.mean(recent_success_rates[:3])
            trend = avg_recent - avg_older

            # Si tendencia negativa significativa
            if trend < -0.15:  # Bajando más del 15%
                predictive_alerts.append({
                    "type": "PREDICTED_DEGRADATION",
                    "agent_name": agent_name,
                    "severity": "medium",
                    "message": f"Tendencia negativa detectada en success_rate: {trend:.1%}",
                    "prediction_confidence": abs(trend),
                })

        return predictive_alerts

    def get_alert_trends(
        self, days: int = 7
    ) -> Dict[str, Any]:
        """
        Analiza tendencias de alertas en los últimos días.
        
        Args:
            days: Número de días a analizar
            
        Returns:
            Análisis de tendencias
        """
        cutoff_time = time.time() - (days * 86400)
        recent_alerts = [
            a for a in self.alert_history if a.get("timestamp", 0) >= cutoff_time
        ]

        if not recent_alerts:
            return {"trend": "stable", "total_alerts": 0}

        # Contar por tipo
        by_type: Dict[str, int] = defaultdict(int)
        by_severity: Dict[str, int] = defaultdict(int)
        
        for alert in recent_alerts:
            by_type[alert.get("type", "unknown")] += 1
            by_severity[alert.get("severity", "unknown")] += 1

        return {
            "total_alerts": len(recent_alerts),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "avg_per_day": len(recent_alerts) / days,
            "most_common_type": max(by_type.items(), key=lambda x: x[1])[0] if by_type else None,
        }


class AgentOptimizer:
    """
    Optimizador automático de asignaciones de agentes.

    Analiza métricas históricas y ajusta:
    - Priorización basada en success_rate
    - Penalización por lentitud
    - Rotación de agentes poco usados
    """

    def __init__(self, db: Any):
        """
        Args:
            db: Instancia de MetacortexDB
        """
        self.db = db
        self.optimization_history: List[Dict[str, Any]] = []
        self.agent_scores: Dict[str, float] = {}
        self.score_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def calculate_agent_score(self, connection: Any) -> float:
        """
        Calcula un score de optimización para un agente.

        Score = success_rate * 100 - (avg_execution_time * 10) + bonus_usage

        Args:
            connection: Instancia de NeuralConnection

        Returns:
            Score (higher is better)
        """
        # Base: success rate (0-100)
        score = connection.success_rate * 100

        # Penalización por lentitud (-0 a -100)
        time_penalty = min(connection.avg_execution_time * 10, 100)
        score -= time_penalty

        # Bonus por uso frecuente (+0 a +20)
        usage_bonus = min(connection.usage_count * 0.5, 20)
        score += usage_bonus

        # Bonus por alta fortaleza de conexión (+0 a +10)
        strength_bonus = connection.connection_strength * 10
        score += strength_bonus

        return max(score, 0)  # No scores negativos

    def get_best_agent_for_capability(
        self, neural_network: Dict[str, Any], required_capability: str
    ) -> Optional[str]:
        """
        Encuentra el mejor agente para una capacidad específica.

        Args:
            neural_network: Diccionario de NeuralConnection
            required_capability: Capacidad requerida

        Returns:
            Nombre del mejor agente o None
        """
        # Filtrar agentes con la capacidad
        candidates = [
            (name, conn)
            for name, conn in neural_network.items()
            if required_capability.lower() in [cap.lower() for cap in conn.capabilities]
        ]

        if not candidates:
            return None

        # Calcular scores
        scored_candidates = [
            (name, self.calculate_agent_score(conn)) for name, conn in candidates
        ]

        # Ordenar por score (desc)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Retornar el mejor
        best_agent = scored_candidates[0][0]
        logger.debug(
            f"🎯 Mejor agente para '{required_capability}': {best_agent} (score: {scored_candidates[0][1]:.2f})"
        )

        return best_agent

    def optimize_agent_selection(
        self, neural_network: Dict[str, Any], task_description: str
    ) -> List[str]:
        """
        Optimiza la selección de agentes para una tarea.

        Args:
            neural_network: Diccionario de NeuralConnection
            task_description: Descripción de la tarea

        Returns:
            Lista de agentes ordenados por score (mejores primero)
        """
        # Buscar agentes que pueden ayudar
        candidates = [
            (name, conn)
            for name, conn in neural_network.items()
            if conn.can_help_with(task_description)
        ]

        if not candidates:
            return []

        # Calcular scores
        scored_candidates = [
            (name, self.calculate_agent_score(conn)) for name, conn in candidates
        ]

        # Ordenar por score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Retornar lista de nombres
        return [name for name, score in scored_candidates]

    def get_optimization_report(self, neural_network: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera reporte de optimización de todos los agentes.

        Args:
            neural_network: Diccionario de NeuralConnection

        Returns:
            Reporte con rankings y recomendaciones
        """
        # Calcular scores para todos
        scored_agents = [
            {
                "agent_name": name,
                "score": self.calculate_agent_score(conn),
                "success_rate": conn.success_rate,
                "avg_execution_time": conn.avg_execution_time,
                "usage_count": conn.usage_count,
                "connection_strength": conn.connection_strength,
            }
            for name, conn in neural_network.items()
        ]

        # Ordenar por score
        scored_agents.sort(key=lambda x: x["score"], reverse=True)

        # Top 10 y Bottom 10
        top_10 = scored_agents[:10]
        bottom_10 = scored_agents[-10:]

        # Estadísticas
        avg_score = sum(a["score"] for a in scored_agents) / len(scored_agents)

        return {
            "total_agents": len(scored_agents),
            "avg_score": avg_score,
            "top_10_agents": top_10,
            "bottom_10_agents": bottom_10,
            "recommendations": self._generate_recommendations(scored_agents),
        }

    def _generate_recommendations(
        self, scored_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """Genera recomendaciones de optimización."""
        recommendations = []

        # Recomendación 1: Agentes con score muy bajo
        low_score_agents = [a for a in scored_agents if a["score"] < 20]
        if low_score_agents:
            recommendations.append(
                f"⚠️  {len(low_score_agents)} agentes con score muy bajo. "
                f"Considerar desactivar: {', '.join([a['agent_name'] for a in low_score_agents[:3]])}"
            )

        # Recomendación 2: Agentes lentos con buen success rate
        slow_but_good = [
            a
            for a in scored_agents
            if a["avg_execution_time"] > 5.0 and a["success_rate"] > 0.8
        ]
        if slow_but_good:
            recommendations.append(
                f"🐌 {len(slow_but_good)} agentes lentos pero confiables. "
                f"Optimizar: {', '.join([a['agent_name'] for a in slow_but_good[:3]])}"
            )

        # Recomendación 3: Agentes poco usados con buen potencial
        underused = [
            a for a in scored_agents if a["usage_count"] < 5 and a["success_rate"] > 0.7
        ]
        if underused:
            recommendations.append(
                f"💎 {len(underused)} agentes poco usados con potencial. "
                f"Considerar promover: {', '.join([a['agent_name'] for a in underused[:3]])}"
            )

        return recommendations

    def track_score_evolution(
        self, agent_name: str, score: float
    ) -> Dict[str, Any]:
        """
        Rastrea evolución de score de un agente.
        
        Args:
            agent_name: Nombre del agente
            score: Score actual
            
        Returns:
            Análisis de evolución
        """
        self.score_history[agent_name].append({
            "score": score,
            "timestamp": time.time()
        })

        history = list(self.score_history[agent_name])
        if len(history) < 2:
            return {"trend": "insufficient_data"}

        scores = [h["score"] for h in history]
        
        return {
            "current_score": score,
            "avg_score": statistics.mean(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "stddev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "trend": "improving" if scores[-1] > scores[0] else "declining",
            "samples": len(scores),
        }

    def get_performance_dashboard(
        self, neural_network: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Genera dashboard completo de performance.
        
        Args:
            neural_network: Diccionario de NeuralConnection
            
        Returns:
            Dashboard con métricas clave
        """
        total_agents = len(neural_network)
        active_agents = sum(1 for conn in neural_network.values() if getattr(conn, 'active', False))
        
        all_scores = []
        high_performers = []
        low_performers = []
        
        for name, conn in neural_network.items():
            score = self.calculate_agent_score(conn)
            all_scores.append(score)
            
            if score > 80:
                high_performers.append(name)
            elif score < 30:
                low_performers.append(name)

        return {
            "summary": {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "inactive_agents": total_agents - active_agents,
                "avg_score": statistics.mean(all_scores) if all_scores else 0,
                "median_score": statistics.median(all_scores) if all_scores else 0,
            },
            "performance_distribution": {
                "high_performers": len(high_performers),
                "medium_performers": total_agents - len(high_performers) - len(low_performers),
                "low_performers": len(low_performers),
            },
            "top_performers": high_performers[:10],
            "needs_attention": low_performers[:10],
            "timestamp": time.time(),
        }


# === EXPORTACIÓN MULTI-FORMATO ===


class MetricsExporter:
    """
    Exportador de métricas a múltiples formatos (JSON, CSV, Prometheus, Grafana).
    """

    def __init__(self):
        self.export_history: List[Dict[str, Any]] = []

    def export_to_json(
        self, metrics: Dict[str, Any], filepath: Optional[str] = None
    ) -> str:
        """
        Exporta métricas a formato JSON.
        
        Args:
            metrics: Diccionario de métricas
            filepath: Ruta opcional para guardar archivo
            
        Returns:
            JSON string
        """
        
        json_data = json.dumps(metrics, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_data)
            logger.info(f"📄 Métricas exportadas a JSON: {filepath}")
        
        return json_data

    def export_to_csv(
        self, metrics_list: List[Dict[str, Any]], filepath: str
    ) -> None:
        """
        Exporta métricas a formato CSV.
        
        Args:
            metrics_list: Lista de diccionarios de métricas
            filepath: Ruta del archivo CSV
        """
        
        if not metrics_list:
            logger.warning("No hay métricas para exportar a CSV")
            return
        
        # Obtener todas las claves
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        fieldnames = sorted(list(all_keys))
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_list)
        
        logger.info(f"📊 Métricas exportadas a CSV: {filepath}")

    def export_to_prometheus(
        self, metrics: Dict[str, Any], agent_name: str
    ) -> str:
        """
        Exporta métricas a formato Prometheus.
        
        Args:
            metrics: Diccionario de métricas
            agent_name: Nombre del agente
            
        Returns:
            Texto en formato Prometheus
        """
        lines: List[str] = []
        
        # Formato Prometheus: metric_name{labels} value timestamp
        timestamp_ms = int(time.time() * 1000)
        
        # Success rate
        if "success_rate" in metrics:
            lines.append(
                f'metacortex_agent_success_rate{{agent="{agent_name}"}} '
                f'{metrics["success_rate"]:.4f} {timestamp_ms}'
            )
        
        # Execution time
        if "avg_execution_time" in metrics:
            lines.append(
                f'metacortex_agent_execution_time_seconds{{agent="{agent_name}"}} '
                f'{metrics["avg_execution_time"]:.4f} {timestamp_ms}'
            )
        
        # Total executions
        if "total_executions" in metrics:
            lines.append(
                f'metacortex_agent_total_executions{{agent="{agent_name}"}} '
                f'{metrics["total_executions"]} {timestamp_ms}'
            )
        
        # Connection strength
        if "connection_strength" in metrics:
            lines.append(
                f'metacortex_agent_connection_strength{{agent="{agent_name}"}} '
                f'{metrics["connection_strength"]:.4f} {timestamp_ms}'
            )
        
        return '\n'.join(lines)

    def export_to_grafana_json(
        self, dashboard_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Exporta métricas a formato compatible con Grafana dashboard.
        
        Args:
            dashboard_metrics: Métricas del dashboard
            
        Returns:
            Diccionario en formato Grafana
        """
        return {
            "dashboard": {
                "title": "METACORTEX Metrics",
                "panels": [
                    {
                        "id": 1,
                        "title": "Agent Success Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "metacortex_agent_success_rate",
                                "legendFormat": "{{agent}}",
                            }
                        ],
                    },
                    {
                        "id": 2,
                        "title": "Execution Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "metacortex_agent_execution_time_seconds",
                                "legendFormat": "{{agent}}",
                            }
                        ],
                    },
                    {
                        "id": 3,
                        "title": "Total Executions",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "metacortex_agent_total_executions",
                                "legendFormat": "{{agent}}",
                            }
                        ],
                    },
                ],
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now",
                },
            }
        }


# === FUNCIONES DE UTILIDAD ===


def init_metrics_system(db: Any) -> Tuple[MetricsLogger, AlertManager, AgentOptimizer]:
    """
    Inicializa el sistema completo de métricas.

    Args:
        db: Instancia de MetacortexDB

    Returns:
        Tupla (MetricsLogger, AlertManager, AgentOptimizer)
    """
    metrics_logger = MetricsLogger(db)
    alert_manager = AlertManager(db)
    agent_optimizer = AgentOptimizer(db)

    logger.info("📊 Sistema de métricas inicializado")
    return metrics_logger, alert_manager, agent_optimizer