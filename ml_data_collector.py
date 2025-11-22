#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 ML DATA COLLECTOR - Recolector de Datos Reales del Sistema
==============================================================

Recolecta datos reales de METACORTEX para entrenar modelos ML:
    pass  # TODO: Implementar
- Interacciones de usuarios
- Métricas del sistema
- Patrones de caché
- Consultas y respuestas
- Rendimiento de agentes

Autor: METACORTEX Team
Fecha: 2025-10-17
"""

import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from memory_system import get_memory

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

try:
    from advanced_cache_system import get_global_cache

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLDataCollector:
    """
    Recolecta datos reales del sistema para entrenar modelos ML
    """

    def __init__(self, data_dir: str = "ml_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Subdirectorios
        self.training_dir = self.data_dir / "training"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        for dir_path in [self.training_dir, self.raw_dir, self.processed_dir]:
            dir_path.mkdir(exist_ok=True)

        # Base de datos SQLite para datos raw
        self.db_path = self.raw_dir / "system_data.db"
        self._init_database()

        # Conexiones a sistemas
        self.memory = get_memory() if MEMORY_AVAILABLE else None
        self.cache = get_global_cache() if CACHE_AVAILABLE else None

        logger.info("✅ ML Data Collector inicializado")
        logger.info(f"   Directorio: {self.data_dir}")

    def _init_database(self):
        """Inicializa base de datos SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabla: Interacciones de usuarios
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT,
                intent TEXT,
                response_time_ms REAL,
                tokens_used INTEGER,
                success BOOLEAN,
                agent_used TEXT,
                context_length INTEGER
            )
        """)

        # Tabla: Métricas del sistema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cpu_percent REAL,
                memory_percent REAL,
                disk_io_read_mb REAL,
                disk_io_write_mb REAL,
                network_sent_mb REAL,
                network_recv_mb REAL,
                active_processes INTEGER,
                load_category TEXT
            )
        """)

        # Tabla: Patrones de caché
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cache_key TEXT,
                hit BOOLEAN,
                access_time_ms REAL,
                data_size_kb REAL,
                ttl_seconds INTEGER,
                access_count INTEGER
            )
        """)

        # Tabla: Rendimiento de agentes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                agent_name TEXT,
                task_type TEXT,
                execution_time_ms REAL,
                success BOOLEAN,
                tokens_used INTEGER,
                error_type TEXT
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Base de datos SQLite inicializada")

    # ═══════════════════════════════════════════════════════════════════════
    # RECOLECCIÓN DE DATOS
    # ═══════════════════════════════════════════════════════════════════════

    def collect_user_interaction(
        self,
        user_query: str,
        intent: str,
        response_time_ms: float,
        tokens_used: int,
        success: bool,
        agent_used: str = "unknown",
        context_length: int = 0,
    ):
        """Registra una interacción de usuario"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO user_interactions 
            (user_query, intent, response_time_ms, tokens_used, success, agent_used, context_length)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_query,
                intent,
                response_time_ms,
                tokens_used,
                success,
                agent_used,
                context_length,
            ),
        )

        conn.commit()
        conn.close()

    def collect_system_metrics(self):
        """Recolecta métricas actuales del sistema"""
        if not PSUTIL_AVAILABLE:
            return

        try:
            # Obtener métricas
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()

            # Categorizar carga
            avg_load = (cpu + memory) / 2
            if avg_load < 30:
                load_category = "low"
            elif avg_load < 70:
                load_category = "medium"
            else:
                load_category = "high"

            # Guardar
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO system_metrics 
                (cpu_percent, memory_percent, disk_io_read_mb, disk_io_write_mb,
                 network_sent_mb, network_recv_mb, active_processes, load_category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    cpu,
                    memory,
                    disk_io.read_bytes / (1024**2),
                    disk_io.write_bytes / (1024**2),
                    net_io.bytes_sent / (1024**2),
                    net_io.bytes_recv / (1024**2),
                    len(psutil.pids()),
                    load_category,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error recolectando métricas del sistema: {e}")

    def collect_cache_pattern(
        self,
        cache_key: str,
        hit: bool,
        access_time_ms: float,
        data_size_kb: float = 0,
        ttl_seconds: int = 3600,
        access_count: int = 1,
    ):
        """Registra un patrón de acceso a caché"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO cache_patterns 
            (cache_key, hit, access_time_ms, data_size_kb, ttl_seconds, access_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (cache_key, hit, access_time_ms, data_size_kb, ttl_seconds, access_count),
        )

        conn.commit()
        conn.close()

    def collect_agent_performance(
        self,
        agent_name: str,
        task_type: str,
        execution_time_ms: float,
        success: bool,
        tokens_used: int = 0,
        error_type: Optional[str] = None,
    ):
        """Registra rendimiento de un agente"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO agent_performance 
            (agent_name, task_type, execution_time_ms, success, tokens_used, error_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                agent_name,
                task_type,
                execution_time_ms,
                success,
                tokens_used,
                error_type,
            ),
        )

        conn.commit()
        conn.close()

    # ═══════════════════════════════════════════════════════════════════════
    # GENERACIÓN DE DATASETS PARA ENTRENAMIENTO
    # ═══════════════════════════════════════════════════════════════════════

    def generate_intention_classifier_dataset(
        self, min_samples: int = 100
    ) -> Optional[Path]:
        """
        Genera dataset para clasificador de intenciones

        Características:
        - Longitud de consulta
        - Palabras clave presentes
        - Complejidad sintáctica

        Target: intent (coding, search, analysis, chat, etc.)
        """
        conn = sqlite3.connect(self.db_path)

        # Verificar cantidad de datos
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_interactions")
        count = cursor.fetchone()[0]

        if count < min_samples:
            logger.warning(f"⚠️ Insuficientes datos: {count}/{min_samples} muestras")
            conn.close()
            return None

        # Extraer datos
        df = pd.read_sql_query(
            """
            SELECT 
                LENGTH(user_query) as query_length,
                (user_query LIKE '%código%' OR user_query LIKE '%code%') as has_coding_keywords,
                (user_query LIKE '%buscar%' OR user_query LIKE '%search%') as has_search_keywords,
                (user_query LIKE '%analizar%' OR user_query LIKE '%analyze%') as has_analysis_keywords,
                context_length,
                tokens_used,
                intent as target
            FROM user_interactions
            WHERE intent IS NOT NULL
        """,
            conn,
        )

        conn.close()

        # Guardar
        output_path = self.training_dir / "intention_classifier.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"✅ Dataset creado: {output_path} ({len(df)} muestras)")
        return output_path

    def generate_load_predictor_dataset(self, min_samples: int = 100) -> Optional[Path]:
        """
        Genera dataset para predictor de carga del sistema

        Características:
        - CPU actual
        - Memoria actual
        - I/O disco
        - Procesos activos
        - Hora del día

        Target: load_category (low, medium, high)
        """
        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM system_metrics")
        count = cursor.fetchone()[0]

        if count < min_samples:
            logger.warning(f"⚠️ Insuficientes datos: {count}/{min_samples} muestras")
            conn.close()
            return None

        df = pd.read_sql_query(
            """
            SELECT 
                cpu_percent,
                memory_percent,
                disk_io_read_mb,
                disk_io_write_mb,
                active_processes,
                CAST(strftime('%H', timestamp) AS INTEGER) as hour_of_day,
                load_category as target
            FROM system_metrics
            WHERE load_category IS NOT NULL
        """,
            conn,
        )

        conn.close()

        output_path = self.training_dir / "load_predictor.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"✅ Dataset creado: {output_path} ({len(df)} muestras)")
        return output_path

    def generate_cache_optimizer_dataset(
        self, min_samples: int = 100
    ) -> Optional[Path]:
        """
        Genera dataset para optimizador de caché

        Características:
        - Tamaño de datos
        - Frecuencia de acceso
        - Tiempo de acceso
        - TTL actual

        Target: hit (1=cache hit, 0=cache miss)
        """
        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache_patterns")
        count = cursor.fetchone()[0]

        if count < min_samples:
            logger.warning(f"⚠️ Insuficientes datos: {count}/{min_samples} muestras")
            conn.close()
            return None

        df = pd.read_sql_query(
            """
            SELECT 
                data_size_kb,
                access_count,
                access_time_ms,
                ttl_seconds,
                CAST(hit AS INTEGER) as target
            FROM cache_patterns
        """,
            conn,
        )

        conn.close()

        output_path = self.training_dir / "cache_optimizer.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"✅ Dataset creado: {output_path} ({len(df)} muestras)")
        return output_path

    def generate_agent_performance_dataset(
        self, min_samples: int = 100
    ) -> Optional[Path]:
        """
        Genera dataset para predictor de rendimiento de agentes

        Características:
        - Tipo de tarea
        - Tokens usados
        - Agente usado

        Target: execution_time_ms
        """
        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM agent_performance")
        count = cursor.fetchone()[0]

        if count < min_samples:
            logger.warning(f"⚠️ Insuficientes datos: {count}/{min_samples} muestras")
            conn.close()
            return None

        df = pd.read_sql_query(
            """
            SELECT 
                agent_name,
                task_type,
                tokens_used,
                CAST(success AS INTEGER) as success,
                execution_time_ms as target
            FROM agent_performance
        """,
            conn,
        )

        # One-hot encoding para agent_name y task_type
        df = pd.get_dummies(df, columns=["agent_name", "task_type"])

        conn.close()

        output_path = self.training_dir / "agent_performance.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"✅ Dataset creado: {output_path} ({len(df)} muestras)")
        return output_path

    def generate_all_datasets(
        self, min_samples: int = 100
    ) -> Dict[str, Optional[Path]]:
        """Genera todos los datasets disponibles"""
        logger.info("📊 Generando todos los datasets...")

        datasets = {
            "intention_classifier": self.generate_intention_classifier_dataset(
                min_samples
            ),
            "load_predictor": self.generate_load_predictor_dataset(min_samples),
            "cache_optimizer": self.generate_cache_optimizer_dataset(min_samples),
            "agent_performance": self.generate_agent_performance_dataset(min_samples),
        }

        # Resumen
        created = sum(1 for path in datasets.values() if path is not None)
        logger.info(f"✅ Datasets generados: {created}/{len(datasets)}")

        return datasets

    # ═══════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ═══════════════════════════════════════════════════════════════════════

    def get_data_stats(self) -> Dict[str, int]:
        """Obtiene estadísticas de datos recolectados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        for table in [
            "user_interactions",
            "system_metrics",
            "cache_patterns",
            "agent_performance",
        ]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        conn.close()

        return stats

    def clear_old_data(self, days: int = 30):
        """Elimina datos antiguos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = datetime.now() - timedelta(days=days)

        for table in [
            "user_interactions",
            "system_metrics",
            "cache_patterns",
            "agent_performance",
        ]:
            cursor.execute(
                f"DELETE FROM {table} WHERE timestamp < ?", (cutoff.isoformat(),)
            )

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"🗑️ Eliminados {deleted} registros antiguos (>{days} días)")

    def export_to_json(self, output_path: Optional[Path] = None) -> Path:
        """Exporta todos los datos a JSON"""
        if output_path is None:
            output_path = (
                self.raw_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        conn = sqlite3.connect(self.db_path)

        data = {}
        for table in [
            "user_interactions",
            "system_metrics",
            "cache_patterns",
            "agent_performance",
        ]:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            data[table] = df.to_dict(orient="records")

        conn.close()

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"📦 Datos exportados: {output_path}")
        return output_path


# ═══════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════

_global_data_collector = None


def get_data_collector(**kwargs) -> MLDataCollector:
    """Obtiene instancia global del data collector"""
    global _global_data_collector
    if _global_data_collector is None:
        _global_data_collector = MLDataCollector(**kwargs)
    return _global_data_collector


# ═══════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🤖 ML Data Collector - Test")
    print("=" * 60)

    collector = get_data_collector()

    # Generar datos de ejemplo
    print("\n📊 Generando datos de ejemplo...")

    # Interacciones de usuario
    intents = ["coding", "search", "analysis", "chat", "debug"]
    queries = [
        "Crea una función para calcular fibonacci",
        "Busca información sobre Python",
        "Analiza el rendimiento de este código",
        "Hola, ¿cómo estás?",
        "Este código tiene un error",
    ]

    for i in range(50):
        intent = intents[i % len(intents)]
        query = queries[i % len(queries)]
        collector.collect_user_interaction(
            user_query=query,
            intent=intent,
            response_time_ms=100 + (i * 10),
            tokens_used=50 + i,
            success=True,
            agent_used="test_agent",
            context_length=200,
        )

    # Métricas del sistema
    for _ in range(50):
        collector.collect_system_metrics()

    # Patrones de caché
    for i in range(50):
        collector.collect_cache_pattern(
            cache_key=f"key_{i}",
            hit=i % 2 == 0,
            access_time_ms=5 + i,
            data_size_kb=10 + i,
            ttl_seconds=3600,
            access_count=i,
        )

    # Rendimiento de agentes
    agents = ["programming_agent", "search_agent", "analysis_agent"]
    tasks = ["code_generation", "web_search", "data_analysis"]

    for i in range(50):
        collector.collect_agent_performance(
            agent_name=agents[i % len(agents)],
            task_type=tasks[i % len(tasks)],
            execution_time_ms=100 + (i * 5),
            success=True,
            tokens_used=50 + i,
        )

    print("✅ Datos de ejemplo generados")

    # Estadísticas
    print("\n📊 Estadísticas:")
    stats = collector.get_data_stats()
    for table, count in stats.items():
        print(f"   {table}: {count} registros")

    # Generar datasets
    print("\n📊 Generando datasets para entrenamiento...")
    datasets = collector.generate_all_datasets(min_samples=10)

    for name, path in datasets.items():
        if path:
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ⚠️ {name}: Insuficientes datos")

    print("\n✅ Test completado")