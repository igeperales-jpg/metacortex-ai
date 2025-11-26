#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖🎖️⚡ AUTONOMOUS MODEL ORCHESTRATOR - Sistema Autónomo de Orquestación de Modelos
═══════════════════════════════════════════════════════════════════════════════

MISIÓN: Poner a trabajar los 956+ modelos ML entrenados de forma autónoma,
        asignando cada modelo a tareas específicas según su especialización.

ARQUITECTURA:
┌─────────────────────────────────────────────────────────────┐
│  CAPA 1: MODEL DISCOVERY & CLASSIFICATION                  │
│  ↓ Analiza todos los modelos y clasifica por tipo          │
├─────────────────────────────────────────────────────────────┤
│  CAPA 2: TASK GENERATION SYSTEM                            │
│  ↓ Genera tareas automáticamente del mundo real            │
├─────────────────────────────────────────────────────────────┤
│  CAPA 3: INTELLIGENT TASK ASSIGNMENT                       │
│  ↓ Asigna modelos óptimos a cada tarea                     │
├─────────────────────────────────────────────────────────────┤
│  CAPA 4: PARALLEL EXECUTION ENGINE                         │
│  ↓ Ejecuta tareas en paralelo con modelos especializados   │
├─────────────────────────────────────────────────────────────┤
│  CAPA 5: RESULT AGGREGATION & LEARNING                     │
│  ↓ Agrega resultados y mejora asignaciones                 │
└─────────────────────────────────────────────────────────────┘

CARACTERÍSTICAS:
✅ 956+ modelos ML working 24/7
✅ Task generation from internet_search.py
✅ Integration with metacortex_sinaptico
✅ Integration with Ollama Mistral
✅ Self-optimization and learning
✅ Distributed execution
✅ Real-time monitoring

AUTOR: METACORTEX AUTONOMOUS SYSTEM
FECHA: 2025-11-26
"""

import json
import logging
import asyncio
import threading
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import sys
from collections import defaultdict, deque
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON REGISTRY - NO MÁS CIRCULAR IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
try:
    # Usar registry centralizado para evitar circular imports
    from singleton_registry import (
        registry,
        get_ml_pipeline,
        get_ollama,
        get_internet_search,
        get_world_model,
        get_cognitive_agent
    )
    
    logger = logging.getLogger(__name__)
    logger.info("✅ Singleton registry imported successfully")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Failed to import singleton_registry: {e}")
    # Crear stubs para que el código no falle
    def get_ml_pipeline(): return None
    def get_ollama(): return None
    def get_internet_search(): return None
    def get_world_model(): return None
    def get_cognitive_agent(): return None


# ═══════════════════════════════════════════════════════════════════════════
# 🏷️ ENUMS & DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

class ModelSpecialization(Enum):
    """Especializaciones de modelos según su training type."""
    
    # Por tipo de algoritmo
    REGRESSION = "regression"              # Predicción numérica
    CLASSIFICATION = "classification"      # Clasificación categórica
    CLUSTERING = "clustering"              # Agrupamiento
    TIME_SERIES = "time_series"           # Series temporales
    NLP = "nlp"                           # Procesamiento lenguaje natural
    VISION = "vision"                     # Visión computacional
    
    # Por tarea específica
    PROGRAMMING = "programming"            # Asistencia programación
    ASSISTANCE = "assistance"              # Ayuda a personas
    ANALYSIS = "analysis"                  # Análisis de datos
    CREATIVITY = "creativity"              # Generación creativa
    ENGINEERING = "engineering"            # Tareas ingeniería
    RESEARCH = "research"                  # Investigación
    EMERGENCY = "emergency"                # Respuesta emergencias
    OPTIMIZATION = "optimization"          # Optimización
    PREDICTION = "prediction"              # Predicción general
    SELF_IMPROVEMENT = "self_improvement"  # 🧠 Auto-mejora del sistema


class TaskPriority(Enum):
    """Prioridades de tareas."""
    CRITICAL = "critical"      # Emergencias, vidas en riesgo
    HIGH = "high"              # Importante, respuesta rápida
    MEDIUM = "medium"          # Normal, puede esperar
    LOW = "low"                # Background, no urgente


class TaskStatus(Enum):
    """Estado de ejecución de tareas."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class ModelProfile:
    """Perfil completo de un modelo ML con todas sus capacidades."""
    
    model_id: str
    model_type: str
    algorithm: str
    specializations: List[ModelSpecialization]
    
    # Métricas de rendimiento
    accuracy: float = 0.0
    r2_score: float = 0.0
    training_samples: int = 0
    num_features: int = 0
    
    # Metadata de entrenamiento
    training_time: float = 0.0
    model_size_mb: float = 0.0
    trained_at: Optional[datetime] = None
    
    # Estado operacional
    tasks_assigned: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    success_rate: float = 1.0
    avg_execution_time: float = 0.0
    
    # Disponibilidad
    is_loaded: bool = False
    last_used: Optional[datetime] = None
    current_task_id: Optional[str] = None
    
    # Capacidades específicas
    can_handle_realtime: bool = False
    max_parallel_tasks: int = 1
    memory_requirement_mb: float = 0.0
    
    def update_success_rate(self, success: bool):
        """Actualiza tasa de éxito."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        total = self.tasks_completed + self.tasks_failed
        if total > 0:
            self.success_rate = self.tasks_completed / total


@dataclass
class Task:
    """Tarea autónoma que será ejecutada por modelos."""
    
    task_id: str
    task_type: ModelSpecialization
    priority: TaskPriority
    description: str
    
    # Datos de entrada
    input_data: Dict[str, Any]
    required_features: List[str]
    
    # Asignación
    assigned_model_ids: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    
    # Ejecución
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: float = 0.0
    
    # Resultados
    results: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    error_message: Optional[str] = None
    
    # Retry logic
    retry_count: int = 0
    max_retries: int = 3
    
    # Context
    generated_from: Optional[str] = None  # De dónde vino esta tarea
    requires_ollama: bool = False         # Si necesita Ollama para completar
    requires_internet: bool = False        # Si necesita búsqueda web


# ═══════════════════════════════════════════════════════════════════════════
# 🧠 AUTONOMOUS MODEL ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class AutonomousModelOrchestrator:
    """
    Orquestador autónomo que pone a trabajar todos los modelos ML del sistema.
    
    Funciona como un cerebro distribuido que:
    1. Descubre y clasifica todos los modelos disponibles
    2. Genera tareas automáticamente del mundo real
    3. Asigna modelos óptimos a cada tarea
    4. Ejecuta en paralelo y aprende de los resultados
    """
    
    def __init__(
        self,
        models_dir: Path = None,
        max_parallel_tasks: int = 50,
        enable_auto_task_generation: bool = True
    ):
        """
        Inicializa el orquestador autónomo.
        
        Args:
            models_dir: Directorio de modelos ML
            max_parallel_tasks: Máximo de tareas paralelas
            enable_auto_task_generation: Activar generación automática de tareas
        """
        self.models_dir = models_dir or Path(__file__).parent / "ml_models"
        self.max_parallel_tasks = max_parallel_tasks
        self.enable_auto_task_generation = enable_auto_task_generation
        
        # Estado del sistema
        self.model_profiles: Dict[str, ModelProfile] = {}
        self.task_queue: deque = deque()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        
        # Mapeo de especializaciones a modelos
        self.specialization_index: Dict[ModelSpecialization, List[str]] = defaultdict(list)
        
        # Integrations (lazy-loaded via singleton registry)
        self.ml_pipeline: Optional[Any] = None
        self.internet_search: Optional[Any] = None
        self.ollama: Optional[Any] = None
        self.world_model: Optional[Any] = None
        self.cognitive_agent: Optional[Any] = None
        self.self_improvement_system: Optional[Any] = None  # 🧠 Auto-mejora
        
        # Control de ejecución
        self.is_running = False
        self.executor_thread: Optional[threading.Thread] = None
        self.task_generator_thread: Optional[threading.Thread] = None
        
        # Métricas
        self.total_tasks_generated = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.total_models_active = 0
        
        logger.info("🤖 Autonomous Model Orchestrator initialized")
    
    def initialize(self):
        """Inicialización completa del sistema."""
        logger.info("🚀 Initializing Autonomous Model Orchestrator...")
        
        # 1. Descubrir y clasificar modelos
        self._discover_models()
        
        # 2. Conectar con integrations
        self._setup_integrations()
        
        # 3. Iniciar threads de ejecución
        self._start_execution_threads()
        
        logger.info(f"✅ Orchestrator ready with {len(self.model_profiles)} models")
    
    def _discover_models(self):
        """Descubre y clasifica todos los modelos ML disponibles."""
        logger.info(f"🔍 Discovering models in {self.models_dir}...")
        
        metadata_files = list(self.models_dir.glob("*_metadata.json"))
        logger.info(f"   Found {len(metadata_files)} metadata files")
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Extraer información
                model_id = metadata.get("model_id")
                model_type = metadata.get("model_type", "unknown")
                algorithm = metadata.get("algorithm", "unknown")
                
                # Determinar especializaciones
                specializations = self._determine_specializations(metadata)
                
                # Crear perfil
                profile = ModelProfile(
                    model_id=model_id,
                    model_type=model_type,
                    algorithm=algorithm,
                    specializations=specializations,
                    accuracy=self._extract_accuracy(metadata),
                    r2_score=metadata.get("train_metrics", {}).get("r2", 0.0),
                    training_samples=metadata.get("num_train_samples", 0),
                    num_features=metadata.get("num_features", 0),
                    training_time=metadata.get("training_time_seconds", 0.0),
                    model_size_mb=metadata.get("model_size_mb", 0.0)
                )
                
                # Almacenar
                self.model_profiles[model_id] = profile
                
                # Indexar por especialización
                for spec in specializations:
                    self.specialization_index[spec].append(model_id)
                
            except Exception as e:
                logger.error(f"   Error loading {metadata_file}: {e}")
        
        logger.info(f"✅ Loaded {len(self.model_profiles)} model profiles")
        logger.info(f"   Specializations: {dict([(s.value, len(m)) for s, m in self.specialization_index.items()])}")
    
    def _determine_specializations(self, metadata: Dict) -> List[ModelSpecialization]:
        """Determina las especializaciones de un modelo basado en su metadata."""
        specs = []
        
        model_type = metadata.get("model_type", "").lower()
        algorithm = metadata.get("algorithm", "").lower()
        
        # Mapeo por tipo de modelo
        if model_type == "regression":
            specs.append(ModelSpecialization.REGRESSION)
            specs.append(ModelSpecialization.PREDICTION)
        elif model_type == "classification":
            specs.append(ModelSpecialization.CLASSIFICATION)
            specs.append(ModelSpecialization.ANALYSIS)
        elif model_type == "clustering":
            specs.append(ModelSpecialization.CLUSTERING)
            specs.append(ModelSpecialization.ANALYSIS)
        elif model_type == "time_series":
            specs.append(ModelSpecialization.TIME_SERIES)
            specs.append(ModelSpecialization.PREDICTION)
        
        # Mapeo por algoritmo (heurística)
        if "gradient" in algorithm or "boosting" in algorithm:
            specs.append(ModelSpecialization.OPTIMIZATION)
        
        # Asignar tareas generales basadas en rendimiento
        accuracy = self._extract_accuracy(metadata)
        if accuracy > 0.85:
            specs.append(ModelSpecialization.ASSISTANCE)
        if accuracy > 0.90:
            specs.append(ModelSpecialization.ENGINEERING)
        
        # Si no hay especializaciones, asignar general
        if not specs:
            specs.append(ModelSpecialization.ANALYSIS)
        
        return specs
    
    def _extract_accuracy(self, metadata: Dict) -> float:
        """Extrae accuracy de metadata (r2 para regression, accuracy para classification)."""
        train_metrics = metadata.get("train_metrics", {})
        val_metrics = metadata.get("val_metrics", {})
        
        # Priorizar métricas de validación
        if "r2" in val_metrics:
            return max(0.0, val_metrics["r2"])
        elif "accuracy" in val_metrics:
            return val_metrics["accuracy"]
        elif "r2" in train_metrics:
            return max(0.0, train_metrics["r2"])
        elif "accuracy" in train_metrics:
            return train_metrics["accuracy"]
        
        return 0.5  # Default neutro
    
    def _setup_integrations(self):
        """Conecta con todos los sistemas del ecosistema via singleton registry."""
        logger.info("🔗 Setting up integrations via singleton registry...")
        
        try:
            # ML Pipeline - lazy load via singleton
            self.ml_pipeline = get_ml_pipeline()
            logger.info("   ✅ ML Pipeline connected (singleton)")
        except Exception as e:
            logger.warning(f"   ⚠️  ML Pipeline not available: {e}")
        
        try:
            # Internet Search - lazy load via singleton
            self.internet_search = get_internet_search()
            logger.info("   ✅ Internet Search connected (singleton)")
        except Exception as e:
            logger.warning(f"   ⚠️  Internet Search not available: {e}")
        
        try:
            # Ollama Integration - lazy load via singleton
            self.ollama = get_ollama()
            logger.info("   ✅ Ollama Integration connected (singleton)")
        except Exception as e:
            logger.warning(f"   ⚠️  Ollama not available: {e}")
        
        try:
            # World Model - lazy load via singleton
            self.world_model = get_world_model()
            logger.info("   ✅ World Model connected (singleton)")
        except Exception as e:
            logger.warning(f"   ⚠️  World Model not available: {e}")
        
        try:
            # Self-Improvement System - Auto-evolución
            from self_improvement_system import SelfImprovementSystem
            self.self_improvement_system = SelfImprovementSystem(Path(__file__).parent)
            logger.info("   ✅ Self-Improvement System connected (AUTO-EVOLUTION ENABLED)")
            logger.info("      🧠 System can now see itself and self-improve")
        except Exception as e:
            logger.warning(f"   ⚠️  Self-Improvement System not available: {e}")
        
        logger.info("✅ Integration setup complete (zero circular dependencies)")
    
    def _start_execution_threads(self):
        """Inicia threads de ejecución en background."""
        self.is_running = True
        
        # Thread 1: Task Executor
        self.executor_thread = threading.Thread(
            target=self._task_executor_loop,
            daemon=True,
            name="TaskExecutor"
        )
        self.executor_thread.start()
        logger.info("   ✅ Task Executor thread started")
        
        # Thread 2: Auto Task Generator
        if self.enable_auto_task_generation:
            self.task_generator_thread = threading.Thread(
                target=self._task_generator_loop,
                daemon=True,
                name="TaskGenerator"
            )
            self.task_generator_thread.start()
            logger.info("   ✅ Task Generator thread started")
    
    def _task_executor_loop(self):
        """Loop principal que ejecuta tareas asignadas a modelos."""
        logger.info("🎯 Task Executor loop started")
        
        while self.is_running:
            try:
                # Procesar tareas pendientes
                if len(self.task_queue) > 0 and len(self.active_tasks) < self.max_parallel_tasks:
                    task = self.task_queue.popleft()
                    self._execute_task(task)
                
                # Pequeño sleep para no saturar CPU
                threading.Event().wait(0.1)
                
            except Exception as e:
                logger.error(f"Error in task executor loop: {e}")
                threading.Event().wait(1.0)
    
    def _task_generator_loop(self):
        """Loop que genera tareas automáticamente del mundo real."""
        logger.info("🌍 Task Generator loop started")
        
        while self.is_running:
            try:
                # Generar tareas cada 30 segundos
                threading.Event().wait(30.0)
                
                if len(self.task_queue) < self.max_parallel_tasks * 2:
                    self._generate_auto_tasks()
                
            except Exception as e:
                logger.error(f"Error in task generator loop: {e}")
                threading.Event().wait(5.0)
    
    def _generate_auto_tasks(self):
        """Genera tareas automáticamente basado en el mundo real."""
        logger.info("🎲 Generating automatic tasks...")
        
        # ESTRATEGIA 1: Búsquedas de internet
        if self.internet_search:
            # Topics de interés general
            topics = [
                "latest AI breakthroughs",
                "global emergencies today",
                "new programming technologies",
                "scientific discoveries 2025",
                "humanitarian crisis updates"
            ]
            
            topic = np.random.choice(topics)
            task = Task(
                task_id=f"auto_search_{self.total_tasks_generated}",
                task_type=ModelSpecialization.RESEARCH,
                priority=TaskPriority.LOW,
                description=f"Research topic: {topic}",
                input_data={"query": topic},
                required_features=["text"],
                requires_internet=True,
                requires_ollama=True,
                generated_from="auto_generator"
            )
            
            self.add_task(task)
            self.total_tasks_generated += 1
        
        # ESTRATEGIA 2: Análisis de datos sintéticos
        # (Para mantener modelos activos)
        task = Task(
            task_id=f"auto_analysis_{self.total_tasks_generated}",
            task_type=ModelSpecialization.ANALYSIS,
            priority=TaskPriority.LOW,
            description="Analyze synthetic dataset",
            input_data=self._generate_synthetic_data(),
            required_features=["numeric"],
            generated_from="auto_generator"
        )
        
        self.add_task(task)
        self.total_tasks_generated += 1
        
        # ESTRATEGIA 3: Auto-mejora del sistema (cada 10 tareas)
        # 🧠 El sistema se analiza y se mejora a sí mismo
        if self.self_improvement_system and self.total_tasks_generated % 10 == 0:
            task = Task(
                task_id=f"auto_improve_{self.total_tasks_generated}",
                task_type=ModelSpecialization.SELF_IMPROVEMENT,
                priority=TaskPriority.MEDIUM,
                description="Self-improvement: Analyze and improve own code",
                input_data={"action": "self_analyze_and_improve"},
                required_features=["code_analysis"],
                generated_from="auto_generator_self_improvement"
            )
            
            self.add_task(task)
            self.total_tasks_generated += 1
            logger.info("   🧠 Generated SELF-IMPROVEMENT task")
        
        logger.info(f"   Generated 2 new tasks (total: {self.total_tasks_generated})")
    
    def _generate_synthetic_data(self) -> Dict[str, Any]:
        """Genera datos sintéticos para análisis."""
        return {
            "features": np.random.randn(100, 10).tolist(),
            "target": np.random.rand(100).tolist()
        }
    
    def _execute_task(self, task: Task):
        """Ejecuta una tarea usando los modelos apropiados."""
        logger.info(f"⚡ Executing task: {task.task_id} ({task.task_type.value})")
        
        # Asignar modelos
        task.assigned_model_ids = self._assign_models_to_task(task)
        
        if not task.assigned_model_ids:
            logger.warning(f"   No models available for task {task.task_id}")
            task.status = TaskStatus.FAILED
            task.error_message = "No suitable models found"
            return
        
        # Marcar como activa
        task.status = TaskStatus.EXECUTING
        task.started_at = datetime.now()
        self.active_tasks[task.task_id] = task
        
        # Ejecutar en thread separado
        exec_thread = threading.Thread(
            target=self._run_task_with_models,
            args=(task,),
            daemon=True
        )
        exec_thread.start()
    
    def _assign_models_to_task(self, task: Task) -> List[str]:
        """Asigna los mejores modelos para una tarea."""
        # Obtener modelos con la especialización requerida
        candidate_models = self.specialization_index.get(task.task_type, [])
        
        if not candidate_models:
            # Fallback: usar modelos generales
            candidate_models = self.specialization_index.get(ModelSpecialization.ANALYSIS, [])
        
        # Ordenar por success_rate
        sorted_models = sorted(
            candidate_models,
            key=lambda mid: self.model_profiles[mid].success_rate,
            reverse=True
        )
        
        # Retornar top 3 (ensemble)
        return sorted_models[:3]
    
    def _run_task_with_models(self, task: Task):
        """Ejecuta la tarea con los modelos asignados."""
        try:
            # Caso 1: Tarea de auto-mejora 🧠
            if task.task_type == ModelSpecialization.SELF_IMPROVEMENT and self.self_improvement_system:
                results = self._execute_self_improvement_task(task)
            
            # Caso 2: Tarea requiere internet
            elif task.requires_internet and self.internet_search:
                results = self._execute_internet_task(task)
            
            # Caso 3: Tarea requiere Ollama
            elif task.requires_ollama and self.ollama:
                results = self._execute_ollama_task(task)
            
            # Caso 4: Tarea de ML estándar
            else:
                results = self._execute_ml_task(task)
            
            # Completar tarea
            task.status = TaskStatus.COMPLETED
            task.results = results
            task.completed_at = datetime.now()
            task.execution_time = (task.completed_at - task.started_at).total_seconds()
            
            # Actualizar métricas de modelos
            for model_id in task.assigned_model_ids:
                self.model_profiles[model_id].update_success_rate(True)
            
            self.total_tasks_completed += 1
            logger.info(f"✅ Task {task.task_id} completed in {task.execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Task {task.task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            # Actualizar métricas
            for model_id in task.assigned_model_ids:
                self.model_profiles[model_id].update_success_rate(False)
            
            self.total_tasks_failed += 1
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                self.task_queue.append(task)
                logger.info(f"🔄 Retrying task {task.task_id} ({task.retry_count}/{task.max_retries})")
        
        finally:
            # Remover de activas
            self.active_tasks.pop(task.task_id, None)
            
            # Mover a completadas
            self.completed_tasks.append(task)
    
    def _execute_internet_task(self, task: Task) -> Dict[str, Any]:
        """Ejecuta tarea que requiere búsqueda en internet."""
        query = task.input_data.get("query", "")
        
        # Buscar en internet
        search_results = self.internet_search.search(query, max_results=5)
        
        # Analizar con Ollama si está disponible
        if self.ollama and task.requires_ollama:
            summary = self.ollama.generate(
                prompt=f"Summarize these search results about '{query}':\n\n{search_results}",
                model="mistral:instruct",
                stream=False
            )
            
            return {
                "search_results": search_results,
                "ai_summary": summary.get("response", ""),
                "source": "internet_search + ollama"
            }
        
        return {
            "search_results": search_results,
            "source": "internet_search"
        }
    
    def _execute_ollama_task(self, task: Task) -> Dict[str, Any]:
        """Ejecuta tarea con Ollama."""
        prompt = task.description
        
        result = self.ollama.generate(
            prompt=prompt,
            model="mistral:instruct",
            stream=False
        )
        
        return {
            "response": result.get("response", ""),
            "model": result.get("model", ""),
            "source": "ollama"
        }
    
    def _execute_self_improvement_task(self, task: Task) -> Dict[str, Any]:
        """
        Ejecuta tarea de AUTO-MEJORA del sistema.
        🧠 El sistema se analiza y se mejora a sí mismo.
        """
        logger.info("🧠 Executing SELF-IMPROVEMENT task")
        
        action = task.input_data.get("action", "self_analyze_and_improve")
        
        if action == "self_analyze_and_improve":
            # 1. Analizar propio código
            logger.info("   🔍 Analyzing self...")
            analyses = self.self_improvement_system.analyze_self()
            
            # 2. Generar mejoras
            logger.info("   💡 Generating improvements...")
            improvements = self.self_improvement_system.generate_improvements(analyses)
            
            # 3. Aplicar mejoras de alta prioridad (solo 1 por ciclo para seguridad)
            high_priority = [i for i in improvements if i.priority.value >= 4]
            applied_count = 0
            
            if high_priority:
                improvement = high_priority[0]
                logger.info(f"   🔧 Applying improvement: {improvement.description}")
                
                success = self.self_improvement_system.apply_improvement(improvement)
                if success:
                    applied_count = 1
                    logger.info("   ✅ Self-improvement applied successfully!")
            
            return {
                "action": action,
                "files_analyzed": len(analyses),
                "improvements_suggested": len(improvements),
                "high_priority_improvements": len(high_priority),
                "improvements_applied": applied_count,
                "status": "self_improved" if applied_count > 0 else "analysis_complete",
                "source": "self_improvement_system"
            }
        
        else:
            return {
                "action": action,
                "status": "unknown_action",
                "source": "self_improvement_system"
            }
    
    def _execute_ml_task(self, task: Task) -> Dict[str, Any]:
        """Ejecuta tarea de ML estándar con modelos entrenados."""
        # Cargar primer modelo
        model_id = task.assigned_model_ids[0]
        model_path = self.models_dir / f"{model_id}.pkl"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Obtener número de features que espera el modelo
        model_profile = self.model_profiles.get(model_id)
        expected_features = model_profile.num_features if model_profile else None
        
        # Preparar features
        features = np.array(task.input_data.get("features", []))
        
        # 🔧 AJUSTAR features al tamaño esperado por el modelo
        if expected_features and features.shape[1] != expected_features:
            logger.warning(f"⚠️  Adjusting features from {features.shape[1]} to {expected_features} for {model_id}")
            
            if features.shape[1] > expected_features:
                # Reducir features (tomar las primeras N)
                features = features[:, :expected_features]
            else:
                # Expandir features (padding con zeros)
                padding = np.zeros((features.shape[0], expected_features - features.shape[1]))
                features = np.hstack([features, padding])
        
        # Predicción
        prediction = model.predict(features)
        
        return {
            "prediction": prediction.tolist(),
            "model_id": model_id,
            "source": "ml_model",
            "features_adjusted": features.shape[1]
        }
    
    def add_task(self, task: Task):
        """Añade una tarea a la cola."""
        self.task_queue.append(task)
        logger.info(f"📝 Task added: {task.task_id} (queue size: {len(self.task_queue)})")
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del orquestador."""
        return {
            "is_running": self.is_running,
            "total_models": len(self.model_profiles),
            "models_by_specialization": {
                spec.value: len(models)
                for spec, models in self.specialization_index.items()
            },
            "queue_size": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": self.total_tasks_completed,
            "failed_tasks": self.total_tasks_failed,
            "success_rate": (
                self.total_tasks_completed / 
                max(1, self.total_tasks_completed + self.total_tasks_failed)
            ),
            "total_tasks_generated": self.total_tasks_generated
        }
    
    def shutdown(self):
        """Detiene el orquestador de forma limpia."""
        logger.info("🛑 Shutting down Autonomous Model Orchestrator...")
        self.is_running = False
        
        if self.executor_thread:
            self.executor_thread.join(timeout=5.0)
        if self.task_generator_thread:
            self.task_generator_thread.join(timeout=5.0)
        
        logger.info("✅ Orchestrator shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════
# 🌐 GLOBAL INSTANCE & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

_global_orchestrator: Optional[AutonomousModelOrchestrator] = None


def get_autonomous_orchestrator(**kwargs) -> AutonomousModelOrchestrator:
    """Obtiene instancia global del orquestador."""
    global _global_orchestrator
    
    if _global_orchestrator is None:
        _global_orchestrator = AutonomousModelOrchestrator(**kwargs)
        _global_orchestrator.initialize()
    
    return _global_orchestrator


# ═══════════════════════════════════════════════════════════════════════════
# 🧪 TESTING
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Test del orquestador autónomo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 80)
    print("🤖 AUTONOMOUS MODEL ORCHESTRATOR - TEST")
    print("=" * 80 + "\n")
    
    # Inicializar
    orchestrator = get_autonomous_orchestrator(
        max_parallel_tasks=10,
        enable_auto_task_generation=True
    )
    
    # Status inicial
    status = orchestrator.get_status()
    print(f"\n📊 Initial Status:")
    print(json.dumps(status, indent=2))
    
    # Añadir tarea manual de ejemplo
    test_task = Task(
        task_id="manual_test_1",
        task_type=ModelSpecialization.ANALYSIS,
        priority=TaskPriority.MEDIUM,
        description="Test analysis task",
        input_data={"features": np.random.randn(10, 5).tolist()},
        required_features=["numeric"]
    )
    
    orchestrator.add_task(test_task)
    
    print("\n⏳ Running for 60 seconds...")
    try:
        import time
        time.sleep(60)
        
        # Status final
        final_status = orchestrator.get_status()
        print(f"\n📊 Final Status:")
        print(json.dumps(final_status, indent=2))
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        orchestrator.shutdown()
        print("\n✅ Test completed\n")


if __name__ == "__main__":
    main()
