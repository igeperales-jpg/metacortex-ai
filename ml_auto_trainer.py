#!/usr/bin/env python3
"""
🤖 ML AUTO TRAINER - Sistema de Entrenamientos y Re-entrenamientos Automáticos
===============================================================================

CONSOLIDADO: Gestiona entrenamientos automáticos de MÚLTIPLES modelos + Re-entrenamiento continuo
1. Recolecta datos reales del sistema
2. Genera datasets automáticamente
3. Entrena múltiples modelos en paralelo
4. Re-entrena periódicamente (reemplaza ml_auto_retrainer.py OBSOLETO)
5. Despliega modelos exitosos

FEATURES v2.0:
    pass  # TODO: Implementar
- Entrenamiento programado + bajo demanda
- Re-entrenamiento incremental automático
- Cola de prioridad para entrenamientos
- Métricas consolidadas de entrenamientos/re-entrenamientos
- Circuit breakers para prevención de fallos

Autor: METACORTEX Team
Fecha: 2025-11-04 (Consolidado)
"""

import json
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

try:
    from ml_pipeline import ModelType, TrainingConfig, TrainingStatus, get_ml_pipeline

    ML_PIPELINE_AVAILABLE = True
except ImportError:
    ML_PIPELINE_AVAILABLE = False
    logging.warning("⚠️ ml_pipeline no disponible")

try:
    from ml_data_collector import get_data_collector

    DATA_COLLECTOR_AVAILABLE = True
except ImportError:
    DATA_COLLECTOR_AVAILABLE = False
    logging.warning("⚠️ ml_data_collector no disponible")

logger = logging.getLogger(__name__)


class MLAutoTrainer:
    """
    Sistema de entrenamientos automáticos para METACORTEX

    Funcionalidades:
    - Recolección continua de datos
    - Entrenamiento automático de múltiples modelos
    - Re-entrenamiento periódico
    - Despliegue automático de modelos exitosos
    """

    def __init__(
        self,
        retraining_interval_hours: int = 24,
        min_samples_threshold: int = 100,
        enable_auto_collection: bool = True,
    ):
        if not ML_PIPELINE_AVAILABLE:
            raise RuntimeError("ml_pipeline no disponible")

        if not DATA_COLLECTOR_AVAILABLE:
            raise RuntimeError("ml_data_collector no disponible")

        self.ml_pipeline = get_ml_pipeline()
        self.data_collector = get_data_collector()

        self.retraining_interval = timedelta(hours=retraining_interval_hours)
        self.min_samples_threshold = min_samples_threshold
        self.enable_auto_collection = enable_auto_collection

        # Control de threads
        self.running = False
        self.collection_thread = None
        self.training_thread = None

        # Estado
        self.last_training = {}
        self.training_schedule = self._create_training_schedule()

        logger.info("✅ ML Auto Trainer inicializado")
        logger.info(f"   Re-entrenamiento cada: {retraining_interval_hours}h")
        logger.info(f"   Mínimo de muestras: {min_samples_threshold}")
        logger.info(f"   Modelos programados: {len(self.training_schedule)}")

    def _create_training_schedule(self) -> list[dict]:
        """
        Define todos los modelos a entrenar automáticamente
        """
        schedule = [
            {
                "name": "intention_classifier",
                "model_type": ModelType.CLASSIFICATION,
                "algorithm": "random_forest",
                "dataset_generator": "generate_intention_classifier_dataset",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                },
                "auto_deploy": True,
                "min_accuracy": 0.85,
                "description": "Clasificador de intenciones de usuario",
            },
            {
                "name": "load_predictor",
                "model_type": ModelType.CLASSIFICATION,
                "algorithm": "gradient_boosting",
                "dataset_generator": "generate_load_predictor_dataset",
                "hyperparameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                },
                "auto_deploy": True,
                "min_accuracy": 0.80,
                "description": "Predictor de carga del sistema",
            },
            {
                "name": "cache_optimizer",
                "model_type": ModelType.CLASSIFICATION,
                "algorithm": "logistic_regression",
                "dataset_generator": "generate_cache_optimizer_dataset",
                "hyperparameters": {"max_iter": 1000, "C": 1.0},
                "auto_deploy": True,
                "min_accuracy": 0.75,
                "description": "Optimizador de patrones de caché",
            },
            {
                "name": "agent_performance",
                "model_type": ModelType.REGRESSION,
                "algorithm": "gradient_boosting",
                "dataset_generator": "generate_agent_performance_dataset",
                "hyperparameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                },
                "auto_deploy": True,
                "min_accuracy": 0.70,  # R² para regresión
                "description": "Predictor de rendimiento de agentes",
            },
        ]

        return schedule

    # ═══════════════════════════════════════════════════════════════════════
    # RECOLECCIÓN DE DATOS AUTOMÁTICA
    # ═══════════════════════════════════════════════════════════════════════

    def start_auto_collection(self):
        """Inicia recolección automática de datos del sistema"""
        if not self.enable_auto_collection:
            logger.info("⚠️ Recolección automática deshabilitada")
            return

        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, daemon=True, name="MLAutoCollector"
        )
        self.collection_thread.start()
        logger.info("✅ Recolección automática iniciada")

    def _collection_loop(self):
        """Loop de recolección de datos"""
        while self.running:
            try:
                # Recolectar métricas del sistema cada 5 minutos
                self.data_collector.collect_system_metrics()
                logger.debug("📊 Métricas del sistema recolectadas")

                # Esperar 5 minutos
                time.sleep(300)

            except Exception:
                logger.exception("❌ Error en recolección automática")
                time.sleep(60)  # Esperar 1 min si hay error

    # ═══════════════════════════════════════════════════════════════════════
    # ENTRENAMIENTO AUTOMÁTICO
    # ═══════════════════════════════════════════════════════════════════════

    def start_auto_training(self):
        """Inicia entrenamientos automáticos"""
        self.running = True
        self.training_thread = threading.Thread(
            target=self._training_loop, daemon=True, name="MLAutoTrainer"
        )
        self.training_thread.start()
        logger.info("✅ Entrenamientos automáticos iniciados")

    def _training_loop(self):
        """Loop de entrenamientos automáticos"""
        while self.running:
            try:
                # Verificar cada modelo en el schedule
                for model_config in self.training_schedule:
                    self._check_and_train_model(model_config)

                # Esperar 1 hora antes de verificar de nuevo
                time.sleep(3600)

            except Exception:
                logger.exception("❌ Error en training loop")
                time.sleep(300)  # Esperar 5 min si hay error

    def _check_and_train_model(self, model_config: dict):
        """
        Verifica si un modelo necesita entrenamiento y lo ejecuta
        """
        model_name = model_config["name"]

        # Verificar si necesita re-entrenamiento
        if model_name in self.last_training:
            time_since_last = datetime.now(UTC) - self.last_training[model_name]
            if time_since_last < self.retraining_interval:
                logger.debug(f"⏭️ {model_name}: No requiere re-entrenamiento aún")
                return

        # Generar dataset
        logger.info(f"📊 Generando dataset para {model_name}...")
        generator_method = getattr(self.data_collector, model_config["dataset_generator"])
        dataset_path = generator_method(min_samples=self.min_samples_threshold)

        if dataset_path is None:
            logger.warning(f"⚠️ {model_name}: Insuficientes datos para entrenar")
            return

        # Crear configuración de entrenamiento
        training_config = TrainingConfig(
            model_type=model_config["model_type"],
            model_name=model_name,
            algorithm=model_config["algorithm"],
            train_data_path=str(dataset_path),
            hyperparameters=model_config["hyperparameters"],
            auto_deploy=model_config["auto_deploy"],
            min_accuracy=model_config["min_accuracy"],
            validation_split=0.2,
        )

        # Encolar entrenamiento
        logger.info(f"🚀 Encolando entrenamiento: {model_name}")
        logger.info(f"   Descripción: {model_config['description']}")
        logger.info(f"   Algoritmo: {model_config['algorithm']}")
        logger.info(f"   Dataset: {dataset_path}")

        self.ml_pipeline.enqueue_training(training_config)

        # Actualizar última vez entrenado
        self.last_training[model_name] = datetime.now(UTC)

    def train_all_models_now(self) -> dict[str, bool]:
        """
        Fuerza entrenamiento inmediato de todos los modelos

        Returns:
            Dict con resultado de cada modelo
        """
        logger.info("🚀 Iniciando entrenamiento de TODOS los modelos...")

        results = {}

        for model_config in self.training_schedule:
            model_name = model_config["name"]

            try:
                # Generar dataset
                generator_method = getattr(self.data_collector, model_config["dataset_generator"])
                dataset_path = generator_method(min_samples=self.min_samples_threshold)

                if dataset_path is None:
                    logger.warning(f"⚠️ {model_name}: Insuficientes datos")
                    results[model_name] = False
                    continue

                # Crear configuración
                training_config = TrainingConfig(
                    model_type=model_config["model_type"],
                    model_name=model_name,
                    algorithm=model_config["algorithm"],
                    train_data_path=str(dataset_path),
                    hyperparameters=model_config["hyperparameters"],
                    auto_deploy=model_config["auto_deploy"],
                    min_accuracy=model_config["min_accuracy"],
                    validation_split=0.2,
                )

                # Encolar
                self.ml_pipeline.enqueue_training(training_config)
                self.last_training[model_name] = datetime.now(UTC)

                results[model_name] = True
                logger.info(f"✅ {model_name} encolado para entrenamiento")

            except Exception:
                logger.exception("❌ Error entrenando {model_name}")
                results[model_name] = False

        # Resumen
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"✅ Entrenamiento iniciado: {successful}/{total} modelos")

        return results

    # ═══════════════════════════════════════════════════════════════════════
    # CONTROL
    # ═══════════════════════════════════════════════════════════════════════

    def start(self):
        """Inicia sistema completo de auto-entrenamiento"""
        logger.info("🚀 Iniciando ML Auto Trainer...")

        # Iniciar recolección
        self.start_auto_collection()

        # Iniciar entrenamientos
        self.start_auto_training()

        # Guardar estado en archivo
        self._save_status_file()

        logger.info("✅ ML Auto Trainer completamente iniciado")

    def stop(self):
        """Detiene todos los threads"""
        logger.info("🛑 Deteniendo ML Auto Trainer...")
        self.running = False

        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
            logger.info("✅ Recolección detenida")

        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
            logger.info("✅ Entrenamientos detenidos")

    # ═══════════════════════════════════════════════════════════════════════
    # RE-ENTRENAMIENTO AUTOMÁTICO (consolidado de ml_auto_retrainer.py OBSOLETO)
    # ═══════════════════════════════════════════════════════════════════════

    def trigger_retraining(self, model_name: str) -> bool:
        """
        Dispara re-entrenamiento inmediato de un modelo específico

        Args:
            model_name: Nombre del modelo a re-entrenar

        Returns:
            True si el re-entrenamiento fue programado exitosamente
        """
        logger.info(f"🔄 Re-entrenamiento disparado para: {model_name}")

        # Buscar configuración del modelo
        model_config = next((m for m in self.training_schedule if m["name"] == model_name), None)

        if not model_config:
            logger.exception(f"❌ Modelo no encontrado: {model_name}")
            return False

        try:
            self._check_and_train_model(model_config)
            logger.info(f"✅ Re-entrenamiento de {model_name} programado")
            return True
        except Exception:
            logger.exception("❌ Error disparando re-entrenamiento de {model_name}")
            return False

    def retrain_all_models(self) -> dict[str, bool]:
        """
        Re-entrena todos los modelos inmediatamente

        Returns:
            Dict con resultado de cada re-entrenamiento
        """
        logger.info("🔄 Re-entrenando TODOS los modelos...")
        return self.train_all_models_now()

    def get_retraining_metrics(self) -> dict:
        """
        Obtiene métricas de re-entrenamientos

        Returns:
            Dict con métricas de re-entrenamientos
        """
        total_retrainings = len(self.last_training)

        return {
            "total_models": len(self.training_schedule),
            "models_trained": total_retrainings,
            "last_trainings": {
                name: timestamp.isoformat() for name, timestamp in self.last_training.items()
            },
            "retraining_interval_hours": self.retraining_interval.total_seconds() / 3600,
            "next_retraining_due": {
                name: (timestamp + self.retraining_interval).isoformat()
                for name, timestamp in self.last_training.items()
            },
        }

    # ═══════════════════════════════════════════════════════════════════════
    # GESTIÓN Y ESTADO
    # ═══════════════════════════════════════════════════════════════════════

    def _save_status_file(self):
        """Guarda estado en archivo para lectura externa"""
        try:
            status_file = Path("ml_data") / "auto_trainer_status.json"
            status_file.parent.mkdir(exist_ok=True)

            with open(status_file, "w") as f:
                json.dump(
                    {
                        "running": self.running,
                        "collection_thread_alive": self.collection_thread.is_alive()
                        if self.collection_thread
                        else False,
                        "training_thread_alive": self.training_thread.is_alive()
                        if self.training_thread
                        else False,
                        "last_update": datetime.now(UTC).isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            logger.exception("Error guardando estado")

    def get_status(self) -> dict:
        """Obtiene estado del auto-trainer"""
        stats = self.data_collector.get_data_stats()

        # Intentar leer estado de archivo si existe
        running_status = self.running
        try:
            status_file = Path("ml_data") / "auto_trainer_status.json"
            if status_file.exists():
                with open(status_file) as f:
                    file_status = json.load(f)
                    running_status = file_status.get("running", self.running)
        except Exception:
            logger.error(f"Error: {e}", exc_info=True)
        return {
            "running": running_status,
            "models_scheduled": len(self.training_schedule),
            "models_trained": len(self.last_training),
            "data_collected": stats,
            "next_retraining": {
                model: (self.last_training[model] + self.retraining_interval).isoformat()
                for model in self.last_training
            },
            "ml_pipeline_stats": self.ml_pipeline.get_stats(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════

_global_auto_trainer = None


def get_auto_trainer(**kwargs) -> MLAutoTrainer:
    """Obtiene instancia global del auto-trainer"""
    global _global_auto_trainer
    if _global_auto_trainer is None:
        _global_auto_trainer = MLAutoTrainer(**kwargs)
    return _global_auto_trainer


# ═══════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("🤖 ML Auto Trainer - Test")
    print("=" * 60)

    # Crear auto-trainer
    trainer = get_auto_trainer(
        retraining_interval_hours=24,
        min_samples_threshold=10,  # Bajo para test
        enable_auto_collection=False,  # Deshabilitado para test
    )

    # Primero generar datos de prueba
    print("\n📊 Generando datos de prueba...")
    collector = trainer.data_collector

    # Datos de usuario
    intents = ["coding", "search", "analysis", "chat", "debug"]
    queries = [
        "Crea una función",
        "Busca información",
        "Analiza el código",
        "Hola",
        "Error en el código",
    ]

    for i in range(50):
        collector.collect_user_interaction(
            user_query=queries[i % len(queries)],
            intent=intents[i % len(intents)],
            response_time_ms=100 + i * 10,
            tokens_used=50 + i,
            success=True,
            agent_used="test_agent",
        )

    # Métricas del sistema
    for _ in range(50):
        collector.collect_system_metrics()

    # Caché
    for i in range(50):
        collector.collect_cache_pattern(
            cache_key=f"key_{i}",
            hit=i % 2 == 0,
            access_time_ms=5 + i,
            data_size_kb=10 + i,
        )

    # Agentes
    for i in range(50):
        collector.collect_agent_performance(
            agent_name="test_agent",
            task_type="test_task",
            execution_time_ms=100 + i * 5,
            success=True,
        )

    print("✅ Datos de prueba generados")

    # Entrenar todos los modelos
    print("\n🚀 Entrenando todos los modelos...")
    results = trainer.train_all_models_now()

    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {model}")

    # Ver estado
    print("\n📊 Estado del sistema:")
    status = trainer.get_status()
    print(f"   Modelos programados: {status['models_scheduled']}")
    print(f"   Modelos encolados: {status['models_trained']}")
    print("   Datos recolectados:")
    for table, count in status["data_collected"].items():
        print(f"      {table}: {count}")

    # Ver estado del pipeline
    print("\n📊 Estado del ML Pipeline:")
    ml_stats = status["ml_pipeline_stats"]
    print(f"   Modelos entrenados: {ml_stats['total_models_trained']}")
    print(f"   Modelos activos: {ml_stats['active_models']}")
    print(f"   Cola de entrenamiento: {ml_stats['queue_size']}")
    print(f"   Modo perpetuo: {ml_stats['perpetual_mode']}")

    print("\n✅ Test completado")
    print("\n💡 Para ver el progreso del entrenamiento:")
    print("   tail -f ml_models/training_history.json")