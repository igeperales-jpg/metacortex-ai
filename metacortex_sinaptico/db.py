#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - Capa de Persistencia SQLite
=========================================

Manejo de base de datos SQLite para persistencia de episodios, hechos,
grafo de conocimiento y métricas del sistema cognitivo.

Tablas:
    pass  # TODO: Implementar
- episodes: eventos y experiencias
- facts: conocimiento semántico
- graph_edges: estructura del grafo
- metrics: métricas de rendimiento
"""
from __future__ import annotations

import logging
import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING, Union, Set
from contextlib import contextmanager
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

from .utils import DatabaseError, setup_logging

if TYPE_CHECKING:
    from neural_symbiotic_network import MetacortexNeuralSymbioticNetworkV2


logger = setup_logging()


# === DDL IDEMPOTENTE ===

CREATE_TABLES_SQL = """
-- Tabla de episodios (memoria episódica)
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    name TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    anomaly_flag INTEGER DEFAULT 0,
    z_score REAL NULL,
    created_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Tabla de hechos (memoria semántica)
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Tabla de aristas del grafo
CREATE TABLE IF NOT EXISTS graph_edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    edge_type TEXT DEFAULT 'association',
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now')),
    PRIMARY KEY(src, dst)
);

-- Tabla de métricas del sistema
CREATE TABLE IF NOT EXISTS metrics (
    ts REAL PRIMARY KEY,
    homeo_var REAL NOT NULL,
    anomaly_rate REAL NOT NULL,
    edge_delta INTEGER NOT NULL,
    goal_progress REAL NOT NULL,
    wellbeing REAL NOT NULL,
    energy REAL NOT NULL,
    valence REAL NOT NULL,
    activation REAL NOT NULL
);

-- 📊 Tabla de histórico de métricas de agentes (NUEVA - TAREA 14)
CREATE TABLE IF NOT EXISTS agent_metrics_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    timestamp REAL NOT NULL,
    connection_strength REAL NOT NULL,
    usage_count INTEGER NOT NULL,
    success_count INTEGER NOT NULL,
    failure_count INTEGER NOT NULL,
    avg_execution_time REAL NOT NULL,
    success_rate REAL NOT NULL,
    total_executions INTEGER NOT NULL,
    active INTEGER NOT NULL,
    metadata_json TEXT DEFAULT '{}'
);

-- 🚨 Tabla de alertas inteligentes (NUEVA - TAREA 14)
CREATE TABLE IF NOT EXISTS agent_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    metric_value REAL NOT NULL,
    threshold_value REAL NOT NULL,
    timestamp REAL NOT NULL,
    acknowledged INTEGER DEFAULT 0,
    resolved INTEGER DEFAULT 0,
    metadata_json TEXT DEFAULT '{}'
);

-- Índices para rendimiento
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS idx_episodes_name ON episodes(name);
CREATE INDEX IF NOT EXISTS idx_episodes_anomaly ON episodes(anomaly_flag);
CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence);
CREATE INDEX IF NOT EXISTS idx_graph_weight ON graph_edges(weight);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent_name ON agent_metrics_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_timestamp ON agent_metrics_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_agent_alerts_agent_name ON agent_alerts(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_alerts_timestamp ON agent_alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_agent_alerts_resolved ON agent_alerts(resolved);
"""


# === CLASE PRINCIPAL ===


class MetacortexDB:
    """Capa de abstracción para la base de datos SQLite con pooling y optimizaciones."""

    def __init__(self, db_path: str = "metacortex.sqlite"):
        """
        Inicializa la conexión a la base de datos.

        Args:
            db_path: Ruta al archivo SQLite
        """
        self.db_path = Path(db_path)
        self.logger = logger.getChild("db")
        
        # Crear directorio si no existe
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connection pooling simple
        self._connection_pool: List[sqlite3.Connection] = []
        self._pool_size: int = 5
        self._active_connections: int = 0
        
        # Cache para queries frecuentes
        self._query_cache: Dict[str, Any] = {}
        self._cache_max_size: int = 100
        self._cache_ttl: float = 300.0  # 5 minutos
        
        # Métricas de rendimiento
        self.query_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "errors": 0
        })

        # Inicializar DB
        self._init_database()

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        self.neural_network: Optional[MetacortexNeuralSymbioticNetworkV2] = None
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("metacortex_db", self)
            logger.info("✅ 'metacortex_db' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")

        self.logger.info(f"Base de datos inicializada: {self.db_path}")

    def _init_database(self):
        """Crea las tablas si no existen."""
        try:
            with self._get_connection() as conn:
                conn.executescript(CREATE_TABLES_SQL)
                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error inicializando base de datos: {e}")

    @contextmanager
    def _get_connection(self):
        """
        Context manager para conexiones SQLite con retry logic y optimización.

        🔥 SOLUCIÓN DE RAÍZ:
        - Retry con backoff exponencial
        - WAL mode para mejor concurrencia
        - Timeouts aumentados
        - Error handling robusto
        """
        conn = None
        max_retries = 5
        retry_delay = 0.1  # Segundos

        for attempt in range(max_retries):
            try:
                # 🔥 Configuración optimizada para concurrencia
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=60.0,  # Aumentado de 30 a 60 segundos
                    check_same_thread=False,
                    isolation_level=None,  # Autocommit mode
                )
                conn.row_factory = sqlite3.Row  # Acceso por nombre de columna

                # 🔥 Optimizaciones críticas
                conn.execute(
                    "PRAGMA journal_mode=WAL"
                )  # Write-Ahead Logging para concurrencia
                conn.execute("PRAGMA synchronous=NORMAL")  # Balance seguridad/velocidad
                conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
                conn.execute("PRAGMA temp_store=MEMORY")  # Temp en RAM
                conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped

                yield conn
                break  # Éxito, salir del retry loop

            except sqlite3.OperationalError as e:
                logger.error(f"Error: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    # Retry con backoff exponencial
                    wait_time = retry_delay * (2**attempt)
                    self.logger.warning(
                        f"Error DB (intento {attempt + 1}/{max_retries}): {e}. "
                        f"Reintentando en {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                    if conn:
                        try:
                            conn.close()
                        except Exception as close_error:
                            logger.exception(f"Error closing connection during retry: {close_error}")
                    conn = None
                else:
                    # Último intento fallido
                    if conn:
                        conn.rollback()
                    raise DatabaseError(
                        f"Error de base de datos después de {max_retries} intentos: {e}"
                    )

            except sqlite3.Error as e:
                logger.error(f"Error: {e}", exc_info=True)
                if conn:
                    conn.rollback()
                raise DatabaseError(f"Error de base de datos: {e}")

        # Finally block para cerrar conexión
        try:
            if conn:
                conn.close()
        except Exception as e:
            logger.error(f"Error en db.py: {e}", exc_info=True)
            self.logger.debug(f"Error cerrando conexión: {e}")

    # === EPISODIOS ===

    def store_episode(
        self,
        name: str,
        payload: Dict[str, Any],
        anomaly: bool = False,
        z_score: Optional[float] = None,
    ) -> int:
        """
        Almacena un episodio en memoria episódica.

        Args:
            name: Nombre del evento
            payload: Datos del evento
            anomaly: Si es una anomalía
            z_score: Puntuación z si es anomalía

        Returns:
            ID del episodio almacenado
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """INSERT INTO episodes (ts, name, payload_json, anomaly_flag, z_score)
                       VALUES (?, ?, ?, ?, ?)""",
                    (time.time(), name, json.dumps(payload), int(anomaly), z_score),
                )
                episode_id = cursor.lastrowid or 0
                conn.commit()

                self.logger.debug(f"Episodio almacenado: ID={episode_id}, name={name}")
                return episode_id

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error almacenando episodio: {e}")

    def get_recent_episodes(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene episodios recientes."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """SELECT * FROM episodes
                       ORDER BY ts DESC LIMIT ?""",
                    (limit,),
                )

                episodes: List[Dict[str, Any]] = []
                for row in cursor:
                    episode = dict(row)
                    episode["payload"] = json.loads(episode["payload_json"])
                    episodes.append(episode)

                return episodes

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error obteniendo episodios: {e}")

    def get_anomaly_count(self, since_hours: int = 24) -> int:
        """Cuenta anomalías en las últimas N horas."""
        try:
            since_ts = time.time() - (since_hours * 3600)

            with self._get_connection() as conn:
                cursor = conn.execute(
                    """SELECT COUNT(*) FROM episodes
                       WHERE anomaly_flag = 1 AND ts > ?""",
                    (since_ts,),
                )
                return cursor.fetchone()[0]

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error contando anomalías: {e}")

    # === HECHOS (MEMORIA SEMÁNTICA) ===

    def store_fact(self, key: str, value: Any, confidence: float = 1.0):
        """Almacena un hecho en memoria semántica."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO facts (key, value_json, confidence, updated_at)
                       VALUES (?, ?, ?, ?)""",
                    (key, json.dumps(value), confidence, time.time()),
                )
                conn.commit()

                self.logger.debug(f"Hecho almacenado: {key} = {value}")

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error almacenando hecho: {e}")

    def get_fact(self, key: str) -> Optional[Tuple[Any, float]]:
        """Obtiene un hecho y su confianza."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """SELECT value_json, confidence FROM facts WHERE key = ?""", (key,)
                )
                row = cursor.fetchone()

                if row:
                    return json.loads(row["value_json"]), row["confidence"]
                return None

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error obteniendo hecho: {e}")

    def get_all_facts(self) -> Dict[str, Tuple[Any, float]]:
        """Obtiene todos los hechos."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT key, value_json, confidence FROM facts")

                facts: Dict[str, Tuple[Any, float]] = {}
                for row in cursor:
                    key = str(row["key"])
                    value = json.loads(row["value_json"])
                    confidence = float(row["confidence"])
                    facts[key] = (value, confidence)

                return facts

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error obteniendo hechos: {e}")

    # === GRAFO ===

    def store_edge(
        self, src: str, dst: str, weight: float = 1.0, edge_type: str = "association"
    ):
        """Almacena una arista del grafo."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO graph_edges
                       (src, dst, weight, edge_type, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (src, dst, weight, edge_type, time.time()),
                )
                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error almacenando arista: {e}")

    def get_all_edges(
        self, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Obtiene aristas del grafo con paginación opcional."""
        try:
            with self._get_connection() as conn:
                edges: List[Dict[str, Any]] = []
                
                if limit is not None:
                    query = """
                        SELECT src, dst, weight, edge_type 
                        FROM graph_edges
                        ORDER BY weight DESC
                        LIMIT ? OFFSET ?
                    """
                    cursor = conn.execute(query, (limit, offset))
                else:
                    query = """
                        SELECT src, dst, weight, edge_type 
                        FROM graph_edges
                        ORDER BY weight DESC
                    """
                    cursor = conn.execute(query)

                for row in cursor:
                    edges.append(dict(row))
                return edges

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error obteniendo aristas: {e}")

    def get_edge_count(self) -> int:
        """Obtiene cantidad total de aristas."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM graph_edges")
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error contando aristas: {e}")

    def update_edge_weight(self, src: str, dst: str, weight: float):
        """Actualiza el peso de una arista."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """UPDATE graph_edges SET weight = ?, updated_at = ?
                       WHERE src = ? AND dst = ?""",
                    (weight, time.time(), src, dst),
                )
                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error actualizando peso de arista: {e}")

    def get_graph_snapshot(self) -> Dict[str, Any]:
        """Obtiene snapshot completo del grafo."""
        try:
            edges = self.get_all_edges()

            # Extraer nodos únicos
            nodes: Set[str] = set()
            for edge in edges:
                nodes.add(str(edge["src"]))
                nodes.add(str(edge["dst"]))

            # Métricas del grafo
            metrics: Dict[str, Union[int, float]] = {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "avg_weight": sum(float(e["weight"]) for e in edges) / len(edges)
                if edges
                else 0.0,
                "max_weight": max(float(e["weight"]) for e in edges) if edges else 0.0,
                "min_weight": min(float(e["weight"]) for e in edges) if edges else 0.0,
            }

            return {
                "nodes": list(nodes),
                "edges": edges,
                "metrics": metrics,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise DatabaseError(f"Error creando snapshot: {e}")

    # === MÉTRICAS ===

    def store_metrics(
        self,
        ts: Optional[float] = None,
        homeo_var: float = 0.0,
        anomaly_rate: float = 0.0,
        edge_delta: int = 0,
        goal_progress: float = 0.0,
        wellbeing: float = 0.0,
        energy: float = 0.0,
        valence: float = 0.0,
        activation: float = 0.0,
    ):
        """Almacena métricas del sistema."""
        try:
            if ts is None:
                ts = time.time()

            with self._get_connection() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO metrics
                       (ts, homeo_var, anomaly_rate, edge_delta, goal_progress,
                        wellbeing, energy, valence, activation)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ts,
                        homeo_var,
                        anomaly_rate,
                        edge_delta,
                        goal_progress,
                        wellbeing,
                        energy,
                        valence,
                        activation,
                    ),
                )
                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error almacenando métricas: {e}")

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Obtiene las últimas métricas."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """SELECT * FROM metrics ORDER BY ts DESC LIMIT 1"""
                )
                row = cursor.fetchone()

                if row:
                    return dict(row)
                return None

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error obteniendo métricas: {e}")

    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Obtiene historial de métricas."""
        try:
            since_ts = time.time() - (hours * 3600)

            with self._get_connection() as conn:
                cursor = conn.execute(
                    """SELECT * FROM metrics WHERE ts > ? ORDER BY ts ASC""",
                    (since_ts,),
                )

                metrics: List[Dict[str, Any]] = []
                for row in cursor:
                    metrics.append(dict(row))

                return metrics

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error obteniendo historial de métricas: {e}")

    # === UTILIDADES ===

    def cleanup_old_data(self, days: int = 30):
        """Limpia datos antiguos para mantener rendimiento."""
        try:
            cutoff_ts = time.time() - (days * 24 * 3600)

            with self._get_connection() as conn:
                # Limpiar episodios antiguos (excepto anomalías)
                conn.execute(
                    """DELETE FROM episodes
                       WHERE ts < ? AND anomaly_flag = 0""",
                    (cutoff_ts,),
                )

                # Limpiar métricas antiguas
                conn.execute("""DELETE FROM metrics WHERE ts < ?""", (cutoff_ts,))

                conn.commit()

                self.logger.info(f"Limpieza completada: datos > {days} días")

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error en limpieza: {e}")

    def get_database_stats(self) -> Dict[str, int]:
        """Obtiene estadísticas de la base de datos."""
        try:
            with self._get_connection() as conn:
                stats: Dict[str, int] = {}

                # Contar registros por tabla
                tables = [
                    "episodes",
                    "facts",
                    "graph_edges",
                    "metrics",
                    "agent_metrics_history",
                    "agent_alerts",
                ]

                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = int(cursor.fetchone()[0])

                # Tamaño del archivo
                stats["db_size_bytes"] = int(self.db_path.stat().st_size)

                return stats

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error obteniendo estadísticas: {e}")

    # === MÉTODOS PARA AGENT METRICS HISTORY (TAREA 14) ===

    def save_agent_metrics(self, agent_name: str, metrics: Dict[str, Any]) -> int:
        """
        Guarda métricas de un agente en el histórico.

        Args:
            agent_name: Nombre del agente
            metrics: Diccionario con métricas (connection_strength, usage_count, etc.)

        Returns:
            ID del registro insertado
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO agent_metrics_history (
                        agent_name, timestamp, connection_strength, usage_count,
                        success_count, failure_count, avg_execution_time,
                        success_rate, total_executions, active, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        agent_name,
                        metrics.get("timestamp", time.time()),
                        metrics.get("connection_strength", 0.0),
                        metrics.get("usage_count", 0),
                        metrics.get("success_count", 0),
                        metrics.get("failure_count", 0),
                        metrics.get("avg_execution_time", 0.0),
                        metrics.get("success_rate", 0.0),
                        metrics.get("total_executions", 0),
                        1 if metrics.get("active", True) else 0,
                        json.dumps(metrics.get("metadata", {})),
                    ),
                )
                return cursor.lastrowid or 0

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error guardando métricas de agente: {e}")

    def get_agent_metrics_history(
        self, agent_name: str, since_ts: Optional[float] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtiene histórico de métricas de un agente.

        Args:
            agent_name: Nombre del agente
            since_ts: Timestamp desde donde obtener (opcional)
            limit: Máximo de registros a retornar

        Returns:
            Lista de registros de métricas
        """
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row

                if since_ts:
                    cursor = conn.execute(
                        """
                        SELECT * FROM agent_metrics_history
                        WHERE agent_name = ? AND timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (agent_name, since_ts, limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM agent_metrics_history
                        WHERE agent_name = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (agent_name, limit),
                    )

                metrics: List[Dict[str, Any]] = []
                for row in cursor:
                    metric = dict(row)
                    metric["metadata"] = json.loads(metric["metadata_json"])
                    del metric["metadata_json"]
                    metrics.append(metric)

                return metrics

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error obteniendo histórico de métricas: {e}")

    def get_metrics_trends(
        self,
        agent_name: str,
        metric_name: str,
        since_ts: Optional[float] = None,
        limit: int = 100,
    ) -> List[Tuple[float, float]]:
        """
        Obtiene tendencias de una métrica específica de un agente.

        Args:
            agent_name: Nombre del agente
            metric_name: Nombre de la métrica (e.g., 'success_rate', 'avg_execution_time')
            since_ts: Timestamp desde donde obtener
            limit: Máximo de registros

        Returns:
            Lista de tuplas (timestamp, valor)
        """
        try:
            with self._get_connection() as conn:
                if since_ts:
                    cursor = conn.execute(
                        f"""
                        SELECT timestamp, {metric_name}
                        FROM agent_metrics_history
                        WHERE agent_name = ? AND timestamp >= ?
                        ORDER BY timestamp ASC
                        LIMIT ?
                        """,
                        (agent_name, since_ts, limit),
                    )
                else:
                    cursor = conn.execute(
                        f"""
                        SELECT timestamp, {metric_name}
                        FROM agent_metrics_history
                        WHERE agent_name = ?
                        ORDER BY timestamp ASC
                        LIMIT ?
                        """,
                        (agent_name, limit),
                    )

                return [(row[0], row[1]) for row in cursor]

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error obteniendo tendencias: {e}")

    # === MÉTODOS PARA AGENT ALERTS (TAREA 14) ===

    def save_alert(
        self,
        agent_name: str,
        alert_type: str,
        severity: str,
        message: str,
        metric_value: float,
        threshold_value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Guarda una alerta en la base de datos.

        Args:
            agent_name: Nombre del agente
            alert_type: Tipo de alerta (e.g., 'low_success_rate', 'slow_execution')
            severity: Severidad ('low', 'medium', 'high', 'critical')
            message: Mensaje descriptivo
            metric_value: Valor de la métrica que disparó la alerta
            threshold_value: Valor del threshold configurado
            metadata: Información adicional (opcional)

        Returns:
            ID de la alerta creada
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO agent_alerts (
                        agent_name, alert_type, severity, message,
                        metric_value, threshold_value, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        agent_name,
                        alert_type,
                        severity,
                        message,
                        metric_value,
                        threshold_value,
                        time.time(),
                        json.dumps(metadata or {}),
                    ),
                )
                return cursor.lastrowid or 0

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error guardando alerta: {e}")

    def get_active_alerts(
        self,
        agent_name: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Obtiene alertas activas (no resueltas).

        Args:
            agent_name: Filtrar por agente (opcional)
            severity: Filtrar por severidad (opcional)
            limit: Máximo de alertas a retornar

        Returns:
            Lista de alertas activas
        """
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row

                query = "SELECT * FROM agent_alerts WHERE resolved = 0"
                params: List[Union[str, int]] = []

                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)

                if severity:
                    query += " AND severity = ?"
                    params.append(severity)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, tuple(params))

                alerts: List[Dict[str, Any]] = []
                for row in cursor:
                    alert = dict(row)
                    alert["metadata"] = json.loads(alert["metadata_json"])
                    del alert["metadata_json"]
                    alerts.append(alert)

                return alerts

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error obteniendo alertas activas: {e}")

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Marca una alerta como reconocida."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE agent_alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
                )
                return True

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error reconociendo alerta: {e}")

    def resolve_alert(self, alert_id: int) -> bool:
        """Marca una alerta como resuelta."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE agent_alerts SET resolved = 1 WHERE id = ?", (alert_id,)
                )
                return True

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error resolviendo alerta: {e}")

    # === MÉTODOS AVANZADOS DE OPTIMIZACIÓN ===

    def _generate_cache_key(self, query: str, params: Tuple[Any, ...]) -> str:
        """Genera clave de cache para query."""
        combined = f"{query}:{str(params)}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Obtiene resultado del cache si existe y es válido."""
        if cache_key in self._query_cache:
            cached_data, timestamp = self._query_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_data
            else:
                # Cache expirado
                del self._query_cache[cache_key]
        return None

    def _put_in_cache(self, cache_key: str, data: Any) -> None:
        """Almacena resultado en cache."""
        # Límite de tamaño de cache
        if len(self._query_cache) >= self._cache_max_size:
            # Eliminar entrada más antigua
            oldest_key = min(self._query_cache, key=lambda k: self._query_cache[k][1])
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = (data, time.time())

    def clear_cache(self) -> None:
        """Limpia el cache de queries."""
        self._query_cache.clear()
        self.logger.info("Cache de queries limpiado")

    def vacuum_database(self) -> None:
        """Ejecuta VACUUM para optimizar el archivo SQLite."""
        try:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
                self.logger.info("VACUUM completado exitosamente")
        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error ejecutando VACUUM: {e}")

    def analyze_database(self) -> None:
        """Ejecuta ANALYZE para actualizar estadísticas del optimizador."""
        try:
            with self._get_connection() as conn:
                conn.execute("ANALYZE")
                self.logger.info("ANALYZE completado exitosamente")
        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error ejecutando ANALYZE: {e}")

    def get_query_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de queries ejecutadas."""
        total_queries = sum(stat["count"] for stat in self.query_stats.values())
        total_time = sum(stat["total_time"] for stat in self.query_stats.values())
        
        return {
            "total_queries": total_queries,
            "total_time": total_time,
            "avg_time_overall": total_time / max(1, total_queries),
            "cache_size": len(self._query_cache),
            "cache_max_size": self._cache_max_size,
            "cache_ttl": self._cache_ttl,
            "by_operation": dict(self.query_stats)
        }

    def optimize_database(self) -> Dict[str, Any]:
        """Ejecuta optimizaciones completas en la base de datos."""
        start_time = time.time()
        
        try:
            # ANALYZE para actualizar estadísticas
            self.analyze_database()
            
            # VACUUM para desfragmentar
            self.vacuum_database()
            
            # Limpiar cache
            self.clear_cache()
            
            # Limpiar datos antiguos (30 días)
            self.cleanup_old_data(days=30)
            
            elapsed = time.time() - start_time
            
            stats = self.get_database_stats()
            
            return {
                "success": True,
                "elapsed_time": elapsed,
                "database_stats": stats,
                "optimizations_applied": [
                    "ANALYZE",
                    "VACUUM", 
                    "cache_clear",
                    "old_data_cleanup"
                ]
            }
            
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }

    # === MÉTODOS DE EXPORT/IMPORT ===

    def export_to_json(self, output_file: str) -> bool:
        """Exporta toda la base de datos a JSON."""
        try:
            data = {
                "episodes": self.get_recent_episodes(limit=1000),
                "facts": {k: {"value": v[0], "confidence": v[1]} 
                         for k, v in self.get_all_facts().items()},
                "edges": self.get_all_edges(limit=1000),
                "metrics": self.get_metrics_history(hours=168),  # 1 semana
                "export_timestamp": time.time()
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Base de datos exportada a {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error en db.py: {e}", exc_info=True)
            self.logger.error(f"Error exportando base de datos: {e}")
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """Retorna estado de salud de la base de datos."""
        try:
            stats = self.get_database_stats()
            query_stats = self.get_query_statistics()
            
            # Calcular métricas de salud
            db_size_mb = stats["db_size_bytes"] / (1024 * 1024)
            
            health_score = 100.0
            issues = []
            
            # Verificar tamaño
            if db_size_mb > 1000:  # >1GB
                health_score -= 20
                issues.append("database_too_large")
            
            # Verificar tasa de errores en queries
            error_rate = sum(s.get("errors", 0) for s in self.query_stats.values()) / max(1, query_stats["total_queries"])
            if error_rate > 0.05:  # >5% errores
                health_score -= 30
                issues.append("high_query_error_rate")
            
            # Verificar cache hit rate (aproximado)
            cache_utilization = len(self._query_cache) / max(1, self._cache_max_size)
            if cache_utilization > 0.9:
                issues.append("cache_nearly_full")
            
            return {
                "health_score": max(0, health_score),
                "status": "healthy" if health_score > 70 else "degraded" if health_score > 40 else "unhealthy",
                "issues": issues,
                "db_size_mb": db_size_mb,
                "total_records": sum(v for k, v in stats.items() if k.endswith("_count")),
                "query_performance": {
                    "total_queries": query_stats["total_queries"],
                    "avg_time_ms": query_stats["avg_time_overall"] * 1000,
                    "error_rate": error_rate
                },
                "cache_metrics": {
                    "size": query_stats["cache_size"],
                    "utilization": cache_utilization
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return {
                "health_score": 0,
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }


# === FUNCIONES DE UTILIDAD ===
        """Obtiene estadísticas de alertas."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END) as active,
                        SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved,
                        SUM(CASE WHEN acknowledged = 1 THEN 1 ELSE 0 END) as acknowledged
                    FROM agent_alerts
                    """
                )
                row = cursor.fetchone()

                cursor_by_severity = conn.execute(
                    """
                    SELECT severity, COUNT(*) as count
                    FROM agent_alerts
                    WHERE resolved = 0
                    GROUP BY severity
                    """
                )

                by_severity = {row[0]: row[1] for row in cursor_by_severity}

                return {
                    "total_alerts": row[0],
                    "active_alerts": row[1],
                    "resolved_alerts": row[2],
                    "acknowledged_alerts": row[3],
                    "by_severity": by_severity,
                }

        except sqlite3.Error as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise DatabaseError(f"Error obteniendo estadísticas de alertas: {e}")


# === FUNCIONES DE UTILIDAD ===


def init_database(db_path: str = "metacortex.sqlite") -> MetacortexDB:
    """Inicializa y retorna instancia de base de datos."""
    return MetacortexDB(db_path)


def backup_database(db_path: str, backup_path: str):
    """Crea backup de la base de datos."""
    try:
        import shutil

        shutil.copy2(db_path, backup_path)
        logger.info(f"Backup creado: {backup_path}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise DatabaseError(f"Error creando backup: {e}")