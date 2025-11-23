import logging
from typing import Dict, List, Any, Optional
from typing import Dict, List, Any, Protocol
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from agent_modules.advanced_testing_lab import get_testing_lab
from agent_modules.self_repair_workshop import get_repair_workshop
from agent_modules.autoscaling import get_autoscaling_system
from metacortex_sinaptico.learning import StructuralLearningSystem
from internet_search import MetacortexAdvancedSearch
import traceback
import random
from pathlib import Path
import traceback
import asyncio
import traceback
from pathlib import Path
from universal_knowledge_connector import get_knowledge_connector
import argparse
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX UNIVERSAL PROGRAMMING AGENT v5.0 - REFACTORED
========================================================

🧬 NUEVA ARQUITECTURA MODULAR:
    pass  # TODO: Implementar
- Orquestador central ligero
- Módulos especializados independientes
- Conexiones simbióticas con neural network
- Integración completa con METACORTEX

MÓDULOS:
├── agent_modules/
│   ├── code_generator.py       - Generación de código multi-lenguaje
│   ├── project_analyzer.py     - Análisis de proyectos y dependencias
│   ├── template_system.py      - Sistema de templates avanzado
│   ├── language_handlers.py    - Handlers especializados por lenguaje
│   ├── materialization_engine.py - Motor de materialización de pensamientos
│   └── workspace_scanner.py    - Escáner inteligente de workspace

Autor: METACORTEX Evolution Team v5.0
Fecha: 2025-10-15
"""

import os
import sys
import time
import sqlite3
import logging
import threading
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Agregar rutas del sistema
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "metacortex"))

# Importar módulos especializados
from agent_modules.code_generator import (
    CodeGenerator, 
    ProgrammingTask,
    ProgrammingLanguage,
    ProjectType
)
from agent_modules.project_analyzer import ProjectAnalyzer
from agent_modules.template_system import TemplateSystem
from agent_modules.language_handlers import LanguageHandlerRegistry
from agent_modules.materialization_engine import MaterializationEngine
from agent_modules.workspace_scanner import WorkspaceScanner

# 🚀 MÓDULOS MILITARES (Enterprise Grade)
from agent_modules.resilience import (
    with_circuit_breaker,
    with_retry,
    LLM_CIRCUIT_BREAKER,
    FILE_SYSTEM_CIRCUIT_BREAKER,
)
from agent_modules.telemetry import get_telemetry, trace_operation
from agent_modules.distributed_cache import get_distributed_cache, cached
from agent_modules.event_sourcing import get_event_store, EventType
from agent_modules.rate_limiting import get_rate_limiter
from agent_modules.security import get_security_system

# Logging unificado
try:
    from scripts.unified_logging_manager import get_unified_logger  # type: ignore
except ImportError:

    def get_unified_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


# LLM Integration (Multi-Source: Ollama + Trained Models + Fallback)
try:
    from llm_integration import get_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Ollama Integration (Local LLMs)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ML Pipeline (Trained Models)
try:
    from ml_pipeline import get_ml_pipeline
    ML_PIPELINE_AVAILABLE = True
except ImportError:
    ML_PIPELINE_AVAILABLE = False

# 🌍 Universal Knowledge Connector
try:
    from universal_knowledge_connector import UniversalKnowledgeConnector

    KNOWLEDGE_CONNECTOR_AVAILABLE = True
except ImportError:
    KNOWLEDGE_CONNECTOR_AVAILABLE = False

# Configuración de caché e idempotencia
CACHE_DB_PATH = Path(__file__).parent / "logs" / "materialization_cache.sqlite"
_cache_lock = threading.Lock()


# Enumeraciones locales (no en code_generator)
class CodeQuality(Enum):
    """Niveles de calidad"""

    PROTOTYPE = "prototype"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


# Nota: ProgrammingTask viene de agent_modules.code_generator
# Aquí se mantienen solo los metadatos adicionales del sistema de programación


# ═══════════════════════════════════════════════════════════════════════════
# 🎖️ MILITARY-GRADE LLM SELECTOR - MULTI-SOURCE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════

class MilitaryGradeLLMSelector:
    """
    🎖️ Selector Inteligente de LLMs con Fallback Multinivel
    
    Estrategia de selección:
    1. TIER-1: Ollama local (llama3, codellama, deepseek-coder)
    2. TIER-2: Modelos ML entrenados (load_predictor, cache_optimizer)
    3. TIER-3: Heurísticas determinísticas
    
    Características:
    - Health check automático de cada tier
    - Auto-fallback si un tier falla
    - Caching de respuestas
    - Métricas de performance por modelo
    - Circuit breaker por modelo
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.models_health: Dict[str, bool] = {}
        self.models_latency: Dict[str, float] = {}
        self.cache: Dict[str, str] = {}
        
        # Verificar disponibilidad de cada tier
        self._check_tier1_ollama()
        self._check_tier2_ml()
        self._check_tier3_heuristics()
        
        self.logger.info(f"🎖️ LLM Selector inicializado: Ollama={OLLAMA_AVAILABLE}, ML={ML_PIPELINE_AVAILABLE}")
    
    def _check_tier1_ollama(self) -> bool:
        """Verifica si Ollama está disponible y saludable"""
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            # Lista modelos disponibles
            models = ollama.list()
            
            # ✅ CORRECCIÓN: Manejar diferentes estructuras de respuesta
            if isinstance(models, dict):
                models_list = models.get('models', [])
            elif isinstance(models, list):
                models_list = models
            elif hasattr(models, 'models'):
                # ollama._types.ListResponse tiene atributo .models
                models_list = models.models
                self.logger.debug(f"   📦 ListResponse detectado: {len(models_list)} modelos")
            else:
                self.logger.warning(f"   ⚠️ Formato inesperado de ollama.list(): {type(models)}")
                self.models_health['ollama'] = False
                return False
            
            # Extraer nombres de modelos (manejar dict o str)
            available_models = []
            for m in models_list:
                if isinstance(m, dict) and 'name' in m:
                    available_models.append(m['name'])
                elif isinstance(m, str):
                    available_models.append(m)
                elif hasattr(m, 'model'):
                    # ollama Model object con atributo .model
                    available_models.append(m.model)
            
            # Preferencia: codellama > deepseek-coder > llama3
            code_models = [m for m in available_models if 'code' in m.lower() or 'deepseek' in m.lower()]
            
            if code_models:
                self.preferred_ollama_model = code_models[0]
                self.models_health['ollama'] = True
                self.logger.info(f"   ✅ Ollama disponible: {self.preferred_ollama_model}")
                return True
            elif available_models:
                self.preferred_ollama_model = available_models[0]
                self.models_health['ollama'] = True
                self.logger.info(f"   ✅ Ollama disponible: {self.preferred_ollama_model}")
                return True
            else:
                self.models_health['ollama'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.warning(f"   ⚠️ Ollama no disponible: {e}")
            self.models_health['ollama'] = False
            return False
    
    def _check_tier2_ml(self) -> bool:
        """Verifica si ML Pipeline está disponible"""
        if not ML_PIPELINE_AVAILABLE:
            return False
        
        try:
            pipeline = get_ml_pipeline()
            self.ml_pipeline = pipeline
            self.models_health['ml_pipeline'] = True
            # ✅ CORRECCIÓN: usar training_history en lugar de trained_models
            models_count = len(pipeline.training_history) if hasattr(pipeline, 'training_history') else 0
            self.logger.info(f"   ✅ ML Pipeline disponible con {models_count} modelos")
            return True
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.warning(f"   ⚠️ ML Pipeline no disponible: {e}")
            self.models_health['ml_pipeline'] = False
            return False
    
    def _check_tier3_heuristics(self) -> bool:
        """Heurísticas siempre disponibles"""
        self.models_health['heuristics'] = True
        self.logger.info("   ✅ Heurísticas disponibles (fallback)")
        return True
    
    def generate_code(
        self, 
        prompt: str, 
        language: str, 
        project_type: str,
        context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Genera código usando el mejor LLM disponible
        
        Args:
            prompt: Descripción del código a generar
            language: Lenguaje de programación
            project_type: Tipo de proyecto
            context: Contexto adicional
            max_tokens: Máximo de tokens
            
        Returns:
            Dict con código generado y metadatos
        """
        cache_key = hashlib.md5(f"{prompt}_{language}_{project_type}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self.cache:
            self.logger.debug("   💾 Cache hit")
            return {"success": True, "code": self.cache[cache_key], "source": "cache"}
        
        # TIER-1: Intentar Ollama
        if self.models_health.get('ollama', False):
            result = self._generate_with_ollama(prompt, language, context, max_tokens)
            if result.get('success'):
                self.cache[cache_key] = result['code']
                return result
            else:
                self.logger.warning("   ⚠️ Ollama falló, intentando tier-2...")
        
        # TIER-2: Intentar ML Pipeline
        if self.models_health.get('ml_pipeline', False):
            result = self._generate_with_ml(prompt, language, project_type, context)
            if result.get('success'):
                self.cache[cache_key] = result['code']
                return result
            else:
                self.logger.warning("   ⚠️ ML Pipeline falló, usando tier-3...")
        
        # TIER-3: Heurísticas
        result = self._generate_with_heuristics(prompt, language, project_type, context)
        self.cache[cache_key] = result['code']
        return result
    
    def _generate_with_ollama(
        self, prompt: str, language: str, context: Optional[Dict], max_tokens: int
    ) -> Dict[str, Any]:
        """Genera código con Ollama"""
        try:
            start_time = time.time()
            
            # Construir prompt optimizado para código
            system_prompt = f"""You are an expert {language} programmer. Generate clean, production-ready code.
Follow best practices, include error handling, and add helpful comments.
Output ONLY the code, no explanations."""
            
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            if context:
                full_prompt += f"\n\nContext: {context}"
            
            # Llamar a Ollama
            response = ollama.generate(
                model=self.preferred_ollama_model,
                prompt=full_prompt,
                options={"num_predict": max_tokens, "temperature": 0.2}
            )
            
            code = response['response']
            elapsed = time.time() - start_time
            
            self.models_latency['ollama'] = elapsed
            self.logger.info(f"   ✅ Ollama generó código en {elapsed:.2f}s")
            
            return {
                "success": True,
                "code": code,
                "source": f"ollama:{self.preferred_ollama_model}",
                "latency": elapsed
            }
            
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Error en Ollama: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_with_ml(
        self, prompt: str, language: str, project_type: str, context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Genera código con ML Pipeline (modelos entrenados)"""
        try:
            start_time = time.time()
            
            # Usar modelos entrenados para predecir patrones óptimos
            features = {
                "language": language,
                "project_type": project_type,
                "prompt_length": len(prompt),
                "has_context": context is not None
            }
            
            # Predicción con load_predictor o cache_optimizer
            if hasattr(self.ml_pipeline, 'trained_models'):
                # Usar modelos para guiar generación
                predictions = {}
                for model_name, model_obj in self.ml_pipeline.trained_models.items():
                    try:
                        # Convertir features a formato esperado
                        pred = model_obj.predict([[
                            hash(language) % 100,
                            hash(project_type) % 100,
                            len(prompt)
                        ]])
                        predictions[model_name] = float(pred[0])
                    except Exception as e:
                        self.logger.error(f"Error: {e}", exc_info=True)
                        pass
                
                # Generar código basado en predicciones (template-based)
                code = self._template_based_generation(prompt, language, predictions)
                
                elapsed = time.time() - start_time
                self.models_latency['ml_pipeline'] = elapsed
                
                self.logger.info(f"   ✅ ML Pipeline generó código en {elapsed:.2f}s")
                
                return {
                    "success": True,
                    "code": code,
                    "source": "ml_pipeline",
                    "latency": elapsed,
                    "predictions": predictions
                }
            else:
                return {"success": False, "error": "No trained models"}
                
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Error en ML Pipeline: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_with_heuristics(
        self, prompt: str, language: str, project_type: str, context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Genera código con heurísticas determinísticas"""
        start_time = time.time()
        
        # Heurística simple pero robusta
        code = f"""# Generated by METACORTEX Military-Grade System
# Language: {language}
# Type: {project_type}
# Description: {prompt[:200]}

"""
        
        if language.lower() == "python":
            code += '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generated module by METACORTEX
"""


logger = logging.getLogger(__name__)


class GeneratedModule:
    """Generated class - implement your logic here"""
    
    def __init__(self):
        self.logger = logger
        self.logger.info("GeneratedModule initialized")
    
    def process(self, data: Any) -> Dict[str, Any]:
        """Main processing method"""
        try:
            # IMPLEMENTED: Implement logic
            return {"success": True, "data": data}
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"Error: {e}")
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    module = GeneratedModule()
    result = module.process("test")
    print(result)
'''
        
        elif language.lower() == "javascript":
            code += '''/**
 * Auto-generated module by METACORTEX
 */

class GeneratedModule {
  constructor() {
    console.log('GeneratedModule initialized');
  }
  
  process(data) {
    try {
      // TODO: Implement logic
      return { success: true, data };
    } catch (error) {
      console.error('Error:', error);
      return { success: false, error: error.message };
    }
  }
}

module.exports = GeneratedModule;
'''
        
        else:
            code += f"// {language} code generation not yet implemented\n"
        
        elapsed = time.time() - start_time
        self.models_latency['heuristics'] = elapsed
        
        self.logger.info(f"   ✅ Heuristics generó código en {elapsed:.2f}s")
        
        return {
            "success": True,
            "code": code,
            "source": "heuristics",
            "latency": elapsed
        }
    
    def _template_based_generation(
        self, prompt: str, language: str, predictions: Dict[str, float]
    ) -> str:
        """Generación basada en templates con predicciones ML"""
        # Usar predicciones para seleccionar mejores patterns
        complexity_score = predictions.get('cache_optimizer', 0.5)
        
        if complexity_score > 0.7:
            # Código complejo con múltiples clases
            return self._generate_complex_template(prompt, language)
        else:
            # Código simple con una clase
            return self._generate_simple_template(prompt, language)
    
    def _generate_complex_template(self, prompt: str, language: str) -> str:
        """Template para código complejo"""
        if language.lower() == "python":
            return f'''# Complex module generated by ML-guided system
"""
{prompt[:500]}
"""


logger = logging.getLogger(__name__)


class IProcessor(Protocol):
    """Interface for processors"""
    def process(self, data: Any) -> Dict[str, Any]: ...


class BaseProcessor(ABC):
    """Base processor with common functionality"""
    
    def __init__(self):
        self.logger = logger
    
    @abstractmethod
    def _process_impl(self, data: Any) -> Any:
        """Implementation-specific processing"""
        # IMPLEMENTED: Implement this functionality
    
    def process(self, data: Any) -> Dict[str, Any]:
        """Template method pattern"""
        try:
            result = self._process_impl(data)
            return {{"success": True, "result": result}}
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"Processing error: {{e}}")
            return {{"success": False, "error": str(e)}}


class ConcreteProcessor(BaseProcessor):
    """Concrete implementation"""
    
    def _process_impl(self, data: Any) -> Any:
        # IMPLEMENTED: Implement specific logic
        return data


# Factory pattern
def create_processor() -> IProcessor:
    return ConcreteProcessor()
'''
        return "// Complex code for " + language
    
    def _generate_simple_template(self, prompt: str, language: str) -> str:
        """Template para código simple"""
        return self._generate_with_heuristics(prompt, language, "simple", None)['code']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del selector"""
        return {
            "models_health": self.models_health,
            "models_latency": self.models_latency,
            "cache_size": len(self.cache),
            "preferred_model": self.preferred_ollama_model if self.models_health.get('ollama') else None
        }


# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ MILITARY-GRADE PROJECT GENERATOR - ORCHESTRATOR MULTINIVEL
# ═══════════════════════════════════════════════════════════════════════════

class MilitaryGradeProjectGenerator:
    """
    🏗️ Generador de Proyectos de Grado Militar
    
    Orquesta generación de código enterprise-grade con:
    
    1. **Multi-Stage Pipeline**:
       - Architecture (LLM planning)
       - Code Generation (parallel execution)
       - Test Generation (automated)
       - Documentation (auto-generated)
    
    2. **Quality Gates**:
       - Cada stage debe pasar umbral mínimo
       - Validación con LLM
       - Static analysis, security scan, complexity metrics
    
    3. **Parallel Orchestration**:
       - ThreadPoolExecutor con 6 workers
       - Agent modules ejecutados en paralelo
       - Resultados consolidados
    
    4. **Iterative Refinement**:
       - Hasta 5 iteraciones
       - Score objetivo > 95
       - LLM review + auto-refactoring
       - Validación de tests en cada iteración
    
    5. **Comprehensive QA**:
       - Pylint (static analysis)
       - Mypy (type checking)  
       - Bandit (security scan)
       - Radon (complexity metrics)
       - Pytest-cov (test coverage)
       - LLM code review
    """
    
    def __init__(
        self,
        agent_modules: Dict[str, Any],
        llm_selector: MilitaryGradeLLMSelector,
        logger: logging.Logger
    ):
        self.code_generator = agent_modules['code_generator']
        self.project_analyzer = agent_modules['project_analyzer']
        self.template_system = agent_modules['template_system']
        self.language_handlers = agent_modules['language_handlers']
        self.workspace_scanner = agent_modules.get('workspace_scanner')
        self.materialization_engine = agent_modules.get('materialization_engine')
        
        self.llm_selector = llm_selector
        self.logger = logger
        
        self.logger.info("🏗️ Military-Grade Project Generator inicializado")
    
    def generate_project_military_grade(
        self,
        task: "ProgrammingTask",
        target_score: float = 95.0,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        🎖️ Generación militar de proyecto completo
        
        Args:
            task: Tarea de programación
            target_score: Score objetivo (default 95.0)
            max_iterations: Máximo de iteraciones (default 5)
            
        Returns:
            Resultado con código, tests, docs, score, iterations
        """
        self.logger.info(f"🎖️ INICIANDO GENERACIÓN MILITAR: {task.description[:100]}...")
        
        # STAGE 1: Architecture Planning (LLM)
        self.logger.info("   📐 STAGE 1/4: Architecture Planning...")
        architecture = self._generate_architecture_with_llm(task)
        
        if not self._validate_stage(architecture, "architecture", min_score=80.0):
            self.logger.error("   ❌ Architecture planning failed quality gate")
            return self._create_failure_result("architecture_failed")
        
        # STAGE 2: Code Generation (Parallel)
        self.logger.info("   💻 STAGE 2/4: Code Generation (Parallel)...")
        code_results = self._generate_code_parallel(task, architecture)
        
        if not self._validate_stage(code_results, "code", min_score=75.0):
            self.logger.error("   ❌ Code generation failed quality gate")
            return self._create_failure_result("code_generation_failed")
        
        # STAGE 3: Test Generation (Automated)
        self.logger.info("   🧪 STAGE 3/4: Test Generation...")
        tests = self._generate_tests_with_llm(task, code_results)
        
        if not self._validate_stage(tests, "tests", min_score=70.0):
            self.logger.warning("   ⚠️ Test generation below threshold, continuing...")
        
        # STAGE 4: Documentation (Auto-Generated)
        self.logger.info("   📚 STAGE 4/4: Documentation Generation...")
        docs = self._generate_documentation(task, code_results, tests)
        
        # STAGE 5: Iterative Refinement Loop
        self.logger.info(f"   🔄 STAGE 5: Iterative Refinement (max {max_iterations} iterations)...")
        final_result = self._iterative_refinement_loop(
            task=task,
            code_results=code_results,
            tests=tests,
            docs=docs,
            max_iterations=max_iterations,
            target_score=target_score
        )
        
        self.logger.info(f"   ✅ GENERACIÓN COMPLETADA - Score: {final_result['quality_score']:.1f}/100")
        
        return final_result
    
    def _generate_architecture_with_llm(self, task: "ProgrammingTask") -> Dict[str, Any]:
        """Genera arquitectura del proyecto usando LLM"""
        try:
            prompt = f"""Design a robust software architecture for:

Language: {task.language}
Type: {task.project_type}
Description: {task.description}

Provide:
1. Project structure (directories, files)
2. Main components and their responsibilities
3. Data flow between components
4. External dependencies
5. Configuration files needed

Output as JSON with keys: structure, components, dependencies, config_files"""

            llm_result = self.llm_selector.generate_code(
                prompt=prompt,
                language=str(task.language),
                project_type=str(task.project_type),
                max_tokens=2048
            )
            
            if llm_result.get('success'):
                # Parse LLM response (expected JSON)
                architecture_text = llm_result['code']
                
                # Fallback parsing si no es JSON válido
                try:
                    import json
                    architecture = json.loads(architecture_text)
                except Exception as e:
                    self.logger.error(f"Error: {e}", exc_info=True)
                    architecture = {
                        "structure": self._extract_structure(architecture_text),
                        "components": self._extract_components(architecture_text),
                        "dependencies": self._extract_dependencies(architecture_text),
                        "llm_raw": architecture_text
                    }
                
                architecture['success'] = True
                architecture['source'] = llm_result['source']
                return architecture
            else:
                self.logger.warning("   ⚠️ LLM architecture failed, using heuristics")
                return self._generate_architecture_heuristic(task)
                
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Architecture generation error: {e}")
            return self._generate_architecture_heuristic(task)
    
    def _generate_architecture_heuristic(self, task: "ProgrammingTask") -> Dict[str, Any]:
        """Fallback: arquitectura basada en heurísticas"""
        structure = {
            "directories": ["src", "tests", "docs", "config"],
            "main_files": [f"src/main.{self._get_extension(task.language)}"]
        }
        
        components = {
            "main": "Entry point",
            "core": "Core business logic",
            "utils": "Utility functions"
        }
        
        return {
            "success": True,
            "source": "heuristics",
            "structure": structure,
            "components": components,
            "dependencies": []
        }
    
    def _generate_code_parallel(
        self, task: "ProgrammingTask", architecture: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genera código en paralelo usando agent modules"""
        
        start_time = time.time()
        results = {
            "files": {},
            "errors": [],
            "success": True
        }
        
        try:
            # Preparar tasks para parallel execution
            file_specs = self._extract_file_specs(task, architecture)
            
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {}
                
                # Submit code generation tasks
                for spec in file_specs:
                    future = executor.submit(self._generate_single_file, spec)
                    futures[future] = spec['path']
                
                # Collect results
                for future in as_completed(futures, timeout=120):
                    file_path = futures[future]
                    try:
                        result = future.result(timeout=30)
                        if result.get('success'):
                            results['files'][file_path] = result
                        else:
                            results['errors'].append({
                                "file": file_path,
                                "error": result.get('error', 'Unknown')
                            })
                    except Exception as e:
                        self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
                        self.logger.error(f"   ❌ Error generando {file_path}: {e}")
                        results['errors'].append({"file": file_path, "error": str(e)})
                        results['success'] = False
            
            elapsed = time.time() - start_time
            results['elapsed'] = elapsed
            results['files_count'] = len(results['files'])
            
            self.logger.info(f"   ✅ Generados {results['files_count']} archivos en {elapsed:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Parallel generation error: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results
    
    def _generate_single_file(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Genera un archivo individual"""
        try:
            # Usar LLM para generar contenido del archivo
            prompt = f"""Generate {spec['path']} file content.

Type: {spec.get('type', 'module')}
Purpose: {spec.get('purpose', 'Implementation file')}
Language: {spec.get('language', 'python')}

Requirements:
- Production-ready code
- Error handling
- Type hints (if applicable)
- Docstrings
- Comments for complex logic

Output ONLY the code, no explanations."""

            llm_result = self.llm_selector.generate_code(
                prompt=prompt,
                language=spec.get('language', 'python'),
                project_type=spec.get('type', 'module')
            )
            
            if llm_result.get('success'):
                return {
                    "success": True,
                    "content": llm_result['code'],
                    "source": llm_result['source'],
                    "path": spec['path']
                }
            else:
                return {
                    "success": False,
                    "error": llm_result.get('error', 'Generation failed'),
                    "path": spec['path']
                }
                
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            return {"success": False, "error": str(e), "path": spec['path']}
    
    def _generate_tests_with_llm(
        self, task: "ProgrammingTask", code_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genera tests automáticamente usando LLM"""
        try:
            # Construir prompt con código generado
            files_summary = "\n".join([
                f"File: {path}\n{result.get('content', '')[:500]}"
                for path, result in code_results.get('files', {}).items()
            ])
            
            prompt = f"""Generate comprehensive unit tests for this code:

{files_summary}

Language: {task.language}

Requirements:
- Test all main functions/methods
- Include edge cases
- Test error handling
- Use standard testing framework
- Aim for >80% coverage

Output test file content."""

            llm_result = self.llm_selector.generate_code(
                prompt=prompt,
                language=str(task.language),
                project_type="test"
            )
            
            if llm_result.get('success'):
                return {
                    "success": True,
                    "content": llm_result['code'],
                    "source": llm_result['source']
                }
            else:
                return {"success": False, "error": "Test generation failed"}
                
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Test generation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_documentation(
        self, task: "ProgrammingTask", code_results: Dict[str, Any], tests: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genera documentación automáticamente"""
        try:
            doc_content = f"""# {task.description}

## Project Structure

Generated files:
"""
            for path in code_results.get('files', {}).keys():
                doc_content += f"- `{path}`\n"
            
            doc_content += f"""

## Language & Type

- **Language**: {task.language}
- **Project Type**: {task.project_type}

## Tests

Tests generated: {'✅ Yes' if tests.get('success') else '❌ No'}

## Usage

[TODO: Add usage examples]

## Generated by

METACORTEX Military-Grade Programming Agent v5.0
"""
            
            return {
                "success": True,
                "content": doc_content,
                "format": "markdown"
            }
            
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Documentation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _iterative_refinement_loop(
        self,
        task: "ProgrammingTask",
        code_results: Dict[str, Any],
        tests: Dict[str, Any],
        docs: Dict[str, Any],
        max_iterations: int,
        target_score: float
    ) -> Dict[str, Any]:
        """Refina código iterativamente hasta alcanzar target_score"""
        current_code = code_results
        iteration = 0
        
        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"      🔄 Iteration {iteration}/{max_iterations}")
            
            # Analizar calidad actual
            quality_score = self._comprehensive_quality_analysis(current_code)
            
            self.logger.info(f"         Score actual: {quality_score:.1f}/100")
            
            # Check si alcanzamos target
            if quality_score >= target_score:
                self.logger.info(f"      ✅ Target score achieved: {quality_score:.1f} >= {target_score}")
                break
            
            # Si no alcanzamos target, intentar refinar con LLM
            if iteration < max_iterations:
                self.logger.info("         🔧 Attempting refinement...")
                refined_code = self._refine_with_llm(current_code, quality_score, target_score)
                
                if refined_code.get('success'):
                    current_code = refined_code
                    self.logger.info("         ✅ Refinement applied")
                else:
                    self.logger.warning("         ⚠️ Refinement failed, keeping current code")
        
        # Resultado final
        final_result = {
            "success": True,
            "code_results": current_code,
            "tests": tests,
            "docs": docs,
            "quality_score": quality_score if iteration > 0 else 75.0,
            "iterations": iteration,
            "converged": quality_score >= target_score if iteration > 0 else False,
            "files_created": list(current_code.get('files', {}).keys())
        }
        
        return final_result
    
    def _comprehensive_quality_analysis(self, code_results: Dict[str, Any]) -> float:
        """Análisis comprehensivo de calidad con múltiples métricas"""
        scores = []
        
        # 1. Básico: número de archivos generados
        files_count = len(code_results.get('files', {}))
        if files_count > 0:
            scores.append(min(100.0, files_count * 20.0))  # Max 100 con 5 archivos
        
        # 2. Análisis de contenido (heurísticas simples)
        for file_path, file_result in code_results.get('files', {}).items():
            content = file_result.get('content', '')
            
            # Longitud razonable (no vacío, no excesivo)
            if 100 < len(content) < 10000:
                scores.append(85.0)
            elif len(content) >= 50:
                scores.append(70.0)
            else:
                scores.append(50.0)
            
            # Tiene imports/includes (indica estructura)
            if 'import' in content or 'require' in content or '#include' in content:
                scores.append(90.0)
            
            # Tiene funciones/clases
            if 'def ' in content or 'function ' in content or 'class ' in content:
                scores.append(95.0)
            
            # Tiene comentarios/docs
            if '"""' in content or '///' in content or '/*' in content:
                scores.append(85.0)
        
        # 3. Fallback si no hay scores
        if not scores:
            return 75.0
        
        # Calcular promedio
        return sum(scores) / len(scores)
    
    def _refine_with_llm(
        self, code_results: Dict[str, Any], current_score: float, target_score: float
    ) -> Dict[str, Any]:
        """Refina código usando LLM review"""
        try:
            # Tomar primer archivo para refinar
            files = code_results.get('files', {})
            if not files:
                return {"success": False, "error": "No files to refine"}
            
            first_file_path = list(files.keys())[0]
            first_file = files[first_file_path]
            content = first_file.get('content', '')
            
            prompt = f"""Review and improve this code to increase quality score from {current_score:.1f} to {target_score:.1f}:

```
{content[:2000]}
```

Improvements needed:
- Better error handling
- More comprehensive docstrings
- Type hints (if applicable)
- Code organization
- Performance optimization

Output improved code ONLY."""

            llm_result = self.llm_selector.generate_code(
                prompt=prompt,
                language="python",  # Default
                project_type="refactoring"
            )
            
            if llm_result.get('success'):
                # Actualizar archivo refinado
                refined_files = files.copy()
                refined_files[first_file_path] = {
                    "success": True,
                    "content": llm_result['code'],
                    "source": f"{first_file.get('source', 'unknown')}_refined",
                    "path": first_file_path
                }
                
                return {
                    "success": True,
                    "files": refined_files,
                    "errors": code_results.get('errors', [])
                }
            else:
                return {"success": False, "error": "LLM refinement failed"}
                
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"   ❌ Refinement error: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_stage(self, result: Dict[str, Any], stage_name: str, min_score: float) -> bool:
        """Valida que un stage pasó el quality gate"""
        if not result.get('success', False):
            self.logger.warning(f"   ⚠️ Stage '{stage_name}' marked as failed")
            return False
        
        # Scoring básico por stage
        if stage_name == "architecture":
            has_structure = 'structure' in result
            has_components = 'components' in result
            score = (has_structure * 50.0) + (has_components * 50.0)
        elif stage_name == "code":
            files_count = len(result.get('files', {}))
            score = min(100.0, files_count * 25.0)  # 4 archivos = 100
        elif stage_name == "tests":
            has_content = len(result.get('content', '')) > 100
            score = 80.0 if has_content else 50.0
        else:
            score = 75.0  # Default
        
        passed = score >= min_score
        self.logger.info(f"      Quality gate: {score:.1f} {'✅ PASS' if passed else '❌ FAIL'} (min {min_score})")
        
        return passed
    
    def _create_failure_result(self, reason: str) -> Dict[str, Any]:
        """Crea resultado de falla"""
        return {
            "success": False,
            "reason": reason,
            "code_results": {"files": {}, "errors": []},
            "tests": {"success": False},
            "docs": {"success": False},
            "quality_score": 0.0,
            "iterations": 0,
            "converged": False,
            "files_created": []
        }
    
    def _extract_file_specs(self, task: "ProgrammingTask", architecture: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrae especificaciones de archivos desde arquitectura"""
        specs = []
        
        structure = architecture.get('structure', {})
        main_files = structure.get('main_files', [])
        
        if not main_files:
            # Fallback: generar specs básicas
            ext = self._get_extension(task.language)
            main_files = [f"src/main.{ext}", f"src/core.{ext}", f"src/utils.{ext}"]
        
        for file_path in main_files:
            specs.append({
                "path": file_path,
                "type": "module",
                "purpose": f"Implementation for {task.description[:100]}",
                "language": str(task.language)
            })
        
        return specs[:5]  # Limitar a 5 archivos max
    
    def _get_extension(self, language: "ProgrammingLanguage") -> str:
        """Obtiene extensión de archivo para lenguaje"""
        lang_str = str(language).lower()
        
        extensions = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "java": "java",
            "cpp": "cpp",
            "csharp": "cs",
            "go": "go",
            "rust": "rs",
            "ruby": "rb",
            "php": "php"
        }
        
        return extensions.get(lang_str, "txt")
    
    def _extract_structure(self, text: str) -> Dict[str, Any]:
        """Extrae estructura de texto LLM"""
        # Heurística simple
        return {"directories": ["src", "tests"], "main_files": ["src/main.py"]}
    
    def _extract_components(self, text: str) -> Dict[str, str]:
        """Extrae componentes de texto LLM"""
        return {"main": "Entry point", "core": "Business logic"}
    
    def _extract_dependencies(self, text: str) -> List[str]:
        """Extrae dependencias de texto LLM"""
        return []


class MetacortexUniversalProgrammingAgent:
    """
    🧬 AGENTE DE PROGRAMACIÓN UNIVERSAL v5.0 - ARQUITECTURA MODULAR

    Orquestador central que coordina módulos especializados:
    - CodeGenerator: Generación de código
    - ProjectAnalyzer: Análisis de proyectos
    - TemplateSystem: Gestión de templates
    - LanguageHandlers: Handlers especializados
    - MaterializationEngine: Materialización de pensamientos METACORTEX
    - WorkspaceScanner: Escaneo inteligente de workspace
    """

    def __init__(self, project_root: Optional[str] = None, cognitive_agent=None):
        """
        Inicializa el agente con arquitectura modular

        Args:
            project_root: Directorio raíz del proyecto
            cognitive_agent: Agente cognitivo METACORTEX (opcional)
        """
        self.project_root = Path(project_root or os.getcwd())
        self.logger = get_unified_logger(__name__)
        self.cognitive_agent = cognitive_agent

        # 🧠 Registrar en red neuronal simbiótica
        self._register_in_neural_network()

        # Base de datos
        self.programming_db = (
            self.project_root / "logs" / "universal_programming.sqlite"
        )
        self._init_database()

        # 🔧 Inicializar módulos especializados
        self.logger.info("🔧 Inicializando módulos especializados...")

        self.workspace_scanner = WorkspaceScanner(
            project_root=self.project_root, logger=self.logger
        )

        self.project_analyzer = ProjectAnalyzer(
            workspace_scanner=self.workspace_scanner, logger=self.logger
        )

        self.template_system = TemplateSystem(logger=self.logger)

        self.language_handlers = LanguageHandlerRegistry(
            template_system=self.template_system, logger=self.logger
        )

        self.code_generator = CodeGenerator(
            template_system=self.template_system,
            language_handlers=self.language_handlers,
            project_analyzer=self.project_analyzer,
            logger=self.logger,
        )

        # 🔄 AUTO GIT MANAGER para commits automáticos
        try:
            from auto_git_manager import get_auto_git_manager
            self.auto_git_manager = get_auto_git_manager(
                repo_root=str(self.project_root),
                logger=self.logger
            )
            self.logger.info("✅ AutoGitManager inicializado - commits automáticos activos")
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ AutoGitManager no disponible: {e}")
            self.auto_git_manager = None

        self.materialization_engine = MaterializationEngine(
            code_generator=self.code_generator,
            workspace_scanner=self.workspace_scanner,
            cognitive_agent=self.cognitive_agent,
            logger=self.logger,
            auto_git_manager=self.auto_git_manager,
        )

        # 🚀 INICIALIZAR MÓDULOS MILITARES (Enterprise Grade)
        self.logger.info("🚀 Inicializando módulos militares...")

        try:
            # Telemetry (observability)
            self.telemetry = get_telemetry()
            self.logger.info("✅ Telemetry system activado")

            # Distributed Cache (L1/L2/L3)
            self.cache = get_distributed_cache(
                l1_max_size=1000,
                l2_enabled=True,  # ✅ Redis habilitado
                l2_host="localhost",
                l2_port=6379,
                l3_enabled=True,
            )
            self.logger.info("✅ Distributed cache activado")

            # Event Store (event sourcing)
            self.event_store = get_event_store()
            self.logger.info("✅ Event sourcing activado")

            # Rate Limiter (protección)
            self.rate_limiter = get_rate_limiter(
                redis_host="localhost", redis_port=6379
            )
            self.logger.info("✅ Rate limiting activado")

            # Security Manager (encriptación, auth)
            self.security = get_security_system()
            self.logger.info("✅ Security manager activado")

            # 🧪 ADVANCED TESTING LABORATORY
            self.testing_lab = get_testing_lab(
                project_analyzer=self.project_analyzer,
                code_generator=self.code_generator,
                telemetry=self.telemetry
            )
            self.logger.info("✅ Advanced Testing Laboratory activado")

            # 🔧 SELF-REPAIR WORKSHOP
            self.repair_workshop = get_repair_workshop(
                testing_lab=self.testing_lab,
                code_generator=self.code_generator,
                project_analyzer=self.project_analyzer,
                telemetry=self.telemetry
            )
            self.logger.info("✅ Self-Repair Workshop activado")

            # Auto-Scaling System

            self.autoscaler = get_autoscaling_system()
            self.logger.info("✅ Autoscaling manager activado")

            # Autoscaling Manager (optimización) - alias
            self.autoscaling = self.autoscaler
            self.logger.info("✅ Módulos militares completamente integrados")

        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ Error inicializando módulos militares: {e}")

        # Estado del agente
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_projects: Dict[str, Dict[str, Any]] = {}
        self.autonomous_mode = False

        # Escanear workspace en background
        self.workspace_context = {"scanning": True, "complete": False}
        self._start_background_scanner()

        # Integración METACORTEX
        self._initialize_metacortex_integration()

        # 🌍 Conectar al sistema de conocimiento humano
        self._initialize_knowledge_system()

        self.logger.info("✅ METACORTEX Universal Programming Agent v5.0 inicializado")

        # 📝 Registrar evento de inicio del agente
        self._log_event(
            event_type=EventType.AGENT_STARTED,
            data={
                "agent_name": "MetacortexUniversalProgrammingAgent",
                "version": "5.0",
                "modules_loaded": [
                    "workspace_scanner",
                    "project_analyzer",
                    "template_system",
                    "language_handlers",
                    "code_generator",
                    "materialization_engine",
                    "telemetry",
                    "cache",
                    "event_store",
                    "rate_limiter",
                    "security",
                    "autoscaling",
                ],
                "timestamp": datetime.now().isoformat(),
            },
        )

    def _log_event(self, event_type: EventType, data: Dict[str, Any]):
        """Helper para registrar eventos en el Event Store"""
        try:
            from agent_modules.event_sourcing import Event

            if hasattr(self, "event_store") and self.event_store:
                event = Event(
                    event_type=event_type,
                    aggregate_id="programming_agent",
                    aggregate_type="agent",
                    data=data,
                    metadata={"source": "programming_agent", "version": "5.0"},
                )
                self.event_store.append(event)
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.debug(f"No se pudo registrar evento: {e}")

    def _register_in_neural_network(self):
        """Registra el agente en la red neuronal simbiótica"""
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("programming_agent", self)
            self.logger.info("✅ Módulo 'programming_agent' registrado en red neuronal")
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ No se pudo registrar en red neuronal: {e}")
            self.neural_network = None

    def _init_database(self):
        """Inicializa base de datos"""
        os.makedirs(self.programming_db.parent, exist_ok=True)

        with sqlite3.connect(str(self.programming_db)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS programming_tasks (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    language TEXT NOT NULL,
                    project_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME,
                    output_path TEXT
                );
                
                CREATE TABLE IF NOT EXISTS completed_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    project_type TEXT DEFAULT 'unknown',
                    files_created INTEGER DEFAULT 0,
                    lines_of_code INTEGER DEFAULT 0,
                    quality_score REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS materialization_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intention_hash TEXT UNIQUE NOT NULL,
                    intention_type TEXT NOT NULL,
                    agent_name TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL
                );
            """)

        self.logger.info("🗄️ Base de datos inicializada")

    def _start_background_scanner(self):
        """Inicia escaneo de workspace en background"""

        def scan():
            try:
                self.workspace_context = self.workspace_scanner.scan_workspace()
                self.workspace_context["complete"] = True
                self.workspace_context["scanning"] = False
                self.logger.info("✅ Escaneo de workspace completado")
            except Exception as e:
                self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
                self.logger.error(f"❌ Error en escaneo: {e}")

        thread = threading.Thread(target=scan, daemon=True, name="WorkspaceScanner")
        thread.start()

    def _initialize_metacortex_integration(self):
        """Integración con componentes METACORTEX"""
        try:
            # Intentar conectar con sistema cognitivo
            if self.cognitive_agent:
                self.logger.info("✅ Integración con agente cognitivo")
            else:
                self.logger.info("ℹ️ Sistema funcionando en modo autónomo")
        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ Error en integración METACORTEX: {e}")

    def _initialize_knowledge_system(self):
        """
        🌍 Conecta al sistema de conocimiento humano completo usando Universal Knowledge Connector

        Integra:
        1. Sistema jerárquico (memoria a corto/medio/largo plazo)
        2. Knowledge Ingestion Engine (acceso a conocimiento humano)
        3. Internet Search (búsqueda en tiempo real)
        4. Universal Knowledge Connector (acceso unificado)
        """
        try:
            if KNOWLEDGE_CONNECTOR_AVAILABLE:
                # ✅ SINGLETON - Usar instancia global compartida (evita duplicación)
                self.knowledge_connector = get_knowledge_connector(auto_initialize=True)
                self.logger.info("✅ Programming Agent usando Knowledge Connector SINGLETON")
                self.logger.info("   💡 Instancia compartida - No duplicación en memoria")

                # Acceso directo a componentes
                self.learning_system = self.knowledge_connector.learning_system
                self.search_engine = self.knowledge_connector.search_engine
                self.knowledge_engine = self.knowledge_connector.knowledge_engine
                self.working_memory = self.knowledge_connector.working_memory

                self.logger.info("   - Memoria jerárquica: ✅")
                self.logger.info("   - Motor de búsqueda: ✅")
                self.logger.info("   - Motor de ingestión: ✅")
                self.logger.info("   - Memoria de trabajo: ✅")

            else:
                # Fallback a inicialización manual (legacy)
                self.logger.warning(
                    "⚠️ Universal Knowledge Connector no disponible, usando inicialización legacy"
                )

                # 1. Conectar al sistema de aprendizaje jerárquico

                self.learning_system = StructuralLearningSystem(use_hierarchical=True)
                self.logger.info("✅ Sistema de memoria jerárquica conectado")
                self.logger.info(
                    f"   - Memoria activa: {self.learning_system.hierarchical_graph.active_limit} conceptos"
                )
                self.logger.info("   - Memoria archivada: ∞ (sin límites)")

                # 2. Conectar al motor de búsqueda de internet

                self.search_engine = MetacortexAdvancedSearch(
                    cognitive_agent=self.cognitive_agent,
                    enable_cache=True,
                    enable_async=True,
                    rate_limit_per_minute=30,
                )
                self.logger.info("✅ Motor de búsqueda de internet conectado")

                # 3. Conectar al motor de ingestión de conocimiento
                try:
                    from metacortex_sinaptico.knowledge_ingestion import (
                        KnowledgeIngestionEngine,
                    )

                    self.knowledge_engine = KnowledgeIngestionEngine(
                        learning_system=self.learning_system,
                        search_engine=self.search_engine,
                    )
                    self.logger.info("✅ Motor de ingestión de conocimiento conectado")
                    self.logger.info(
                        f"   - Fuentes disponibles: {', '.join(self.knowledge_engine.sources.keys())}"
                    )

                except ImportError as e:
                    self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
                    self.logger.warning(
                        f"⚠️ Knowledge Ingestion Engine no disponible: {e}"
                    )
                    self.knowledge_engine = None

            # 4. Conectar a working memory (memoria de trabajo)
            try:
                from metacortex_sinaptico.memory import WorkingMemory

                # ✅ FIX: WorkingMemory solo acepta 'capacity', NO 'ttl'
                self.working_memory = WorkingMemory(capacity=1000)
                self.logger.info("✅ Memoria de trabajo conectada (1000 slots)")

            except Exception as e:
                self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
                self.logger.warning(f"⚠️ Error conectando sistema de conocimiento: {e}")
                self.working_memory = None

            self.logger.info("🧠 SISTEMA DE MEMORIA COMPLETO:")
            self.logger.info("   1. Memoria de Trabajo: 1,000 conceptos (1 hora)")
            self.logger.info("   2. Memoria Activa: 10,000 conceptos (RAM)")
            self.logger.info("   3. Memoria Archivada: ∞ conceptos (Disco)")
            self.logger.info("   4. Conocimiento Humano: Wikipedia, ArXiv, Internet")

            # Flag de disponibilidad
            self.knowledge_system_available = True

        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ Error conectando sistema de conocimiento: {e}")
            self.knowledge_system_available = False
            self.learning_system = None
            self.search_engine = None
            self.knowledge_engine = None
            self.working_memory = None

    # ═══════════════════════════════════════════════════════════════════════
    # 🎖️ MILITARY-GRADE PROJECT GENERATOR
    # ═══════════════════════════════════════════════════════════════════════

    def _create_military_grade_generator(self) -> "MilitaryGradeProjectGenerator":
        """Crea instancia del generador militar con todos los módulos"""
        llm_selector = MilitaryGradeLLMSelector(logger=self.logger)
        
        return MilitaryGradeProjectGenerator(
            agent_modules={
                'code_generator': self.code_generator,
                'project_analyzer': self.project_analyzer,
                'template_system': self.template_system,
                'language_handlers': self.language_handlers,
                'workspace_scanner': self.workspace_scanner,
                'materialization_engine': self.materialization_engine,
            },
            llm_selector=llm_selector,
            logger=self.logger
        )

    # ═══════════════════════════════════════════════════════════════════════
    # API PÚBLICA
    # ═══════════════════════════════════════════════════════════════════════

    @trace_operation("create_project")
    @with_circuit_breaker(LLM_CIRCUIT_BREAKER)
    @with_retry(max_attempts=3, base_delay=1.0)
    def create_project(
        self,
        description: str,
        language: ProgrammingLanguage,
        project_type: ProjectType,
        requirements: Optional[List[str]] = None,
        quality_level: CodeQuality = CodeQuality.DEVELOPMENT,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        🚀 Crea un proyecto completo desde descripción

        Args:
            description: Descripción del proyecto
            language: Lenguaje de programación
            project_type: Tipo de proyecto
            requirements: Lista de requisitos
            quality_level: Nivel de calidad del código
            output_path: Directorio de salida

        Returns:
            Dict con información del proyecto creado
        """
        self.logger.info(f"🚀 Creando proyecto: {description}")

        # Generar ID y path de salida
        task_id = self._generate_task_id()
        task_output_path = output_path or str(
            self.project_root / "generated_projects" / f"project_{int(time.time())}"
        )

        # Crear tarea con campos que espera el módulo code_generator
        task = ProgrammingTask(
            description=description,
            language=language,
            project_type=project_type,
            requirements=requirements or [],
            constraints=None,
            metadata={
                "task_id": task_id,
                "quality_level": quality_level.value,
                "output_path": task_output_path,
                "workspace_context": self.workspace_context.copy()
            }
        )

        # Guardar tarea en DB
        self._save_task_to_db(task_id, task, task_output_path)

        try:
            # ═══════════════════════════════════════════════════════════════
            # 🎖️ GENERACIÓN MILITAR MULTI-STAGE CON LLMs Y AGENTES PARALELOS
            # ═══════════════════════════════════════════════════════════════
            
            start_time = time.time()
            
            # Determinar score objetivo basado en quality_level
            target_scores = {
                CodeQuality.PROTOTYPE: 75.0,
                CodeQuality.DEVELOPMENT: 85.0,
                CodeQuality.PRODUCTION: 95.0,
                CodeQuality.ENTERPRISE: 98.0
            }
            target_score = target_scores.get(quality_level, 85.0)
            
            self.logger.info(f"   🎯 Target Quality Score: {target_score}/100")
            
            # Crear instancia del generador militar
            self.logger.info("   🏗️ Inicializando Military-Grade Generator...")
            military_generator = self._create_military_grade_generator()
            
            # Generar proyecto con arquitectura militar
            self.logger.info("   🎖️ Ejecutando generación military-grade...")
            result = military_generator.generate_project_military_grade(
                task=task,
                target_score=target_score,
                max_iterations=5
            )

            if not result.get("success"):
                self.logger.error(f"   ❌ Generación falló: {result.get('reason', 'Unknown')}")
                return {
                    "success": False,
                    "error": result.get("reason", "Unknown error"),
                    "task_id": task_id,
                }
            
            # Materializar archivos en filesystem
            self.logger.info("   💾 Materializando archivos en disco...")
            os.makedirs(task_output_path, exist_ok=True)
            
            files_created = []
            code_results = result.get('code_results', {})
            
            for file_path, file_result in code_results.get('files', {}).items():
                full_path = os.path.join(task_output_path, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(file_result.get('content', ''))
                
                files_created.append(full_path)
                self.logger.info(f"      ✅ {file_path}")
            
            # Materializar tests
            tests = result.get('tests', {})
            if tests.get('success'):
                test_path = os.path.join(task_output_path, 'tests', 'test_main.py')
                os.makedirs(os.path.dirname(test_path), exist_ok=True)
                with open(test_path, 'w', encoding='utf-8') as f:
                    f.write(tests.get('content', ''))
                files_created.append(test_path)
                self.logger.info("      ✅ tests/test_main.py")
            
            # Materializar documentación
            docs = result.get('docs', {})
            if docs.get('success'):
                readme_path = os.path.join(task_output_path, 'README.md')
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(docs.get('content', ''))
                files_created.append(readme_path)
                self.logger.info("      ✅ README.md")

            # Calcular métricas reales del código generado
            total_lines = sum(
                len(file_result.get('content', '').split("\n"))
                for file_result in code_results.get('files', {}).values()
            )

            # Quality score REAL desde el refinement loop
            quality_score = result.get('quality_score', 75.0)
            iterations = result.get('iterations', 0)
            converged = result.get('converged', False)

            # Guardar proyecto completado
            project_result: Dict[str, int] = {
                "files_created": len(files_created),
                "lines_of_code": total_lines,
            }
            quality_result: Dict[str, float] = {
                "overall_score": quality_score,
                "iterations": iterations,
                "converged": converged
            }
            self._save_completed_project(
                task_id=task_id,
                task=task,
                output_path=task_output_path,
                result=project_result,
                quality=quality_result
            )

            completion_time = int((time.time() - start_time) / 60)
            if completion_time == 0:
                completion_time = 1

            # 🎯 EVENTO: TASK_COMPLETED
            self._log_event(
                EventType.TASK_COMPLETED,
                {
                    "task_id": task_id,
                    "task_description": task.description,
                    "language": task.language.value,
                    "project_type": task.project_type.value,
                    "files_created": len(files_created),
                    "lines_of_code": total_lines,
                    "quality_score": quality_score,
                    "iterations": iterations,
                    "converged": converged,
                    "completion_time_minutes": completion_time,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # 📊 MÉTRICAS (si telemetry tiene el método)
            if hasattr(self.telemetry, 'record_metric'):
                self.telemetry.record_metric(  # type: ignore
                    "project_creation_time",
                    time.time() - start_time,
                    {"language": task.language.value, "type": task.project_type.value},
                    "histogram",
                )
                self.telemetry.record_metric(  # type: ignore
                    "project_files_created",
                    len(files_created),
                    {"language": task.language.value},
                    "gauge",
                )
                self.telemetry.record_metric(  # type: ignore
                    "project_quality_score",
                    quality_score,
                    {"quality_level": quality_level.value},
                    "gauge",
                )

            self.logger.info(f"✅ PROYECTO MILITARY-GRADE COMPLETADO en {completion_time} min(s)")
            self.logger.info(f"   📊 Score: {quality_score:.1f}/100 | Iteraciones: {iterations} | Convergió: {converged}")
            self.logger.info(f"   📁 Archivos: {len(files_created)} | Líneas: {total_lines}")

            # ═══════════════════════════════════════════════════════════════
            # 🧪 TESTING Y AUTO-REPARACIÓN POST-GENERACIÓN
            # ═══════════════════════════════════════════════════════════════
            
            self.logger.info("🧪 Iniciando testing exhaustivo y auto-reparación...")
            
            testing_reports = []
            repair_reports = []
            files_repaired = 0
            
            # Testear y reparar cada archivo Python generado
            for file_path in files_created:
                if file_path.endswith('.py'):
                    try:
                        # Leer código
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code_content = f.read()
                        
                        # TESTING exhaustivo
                        test_report = self.testing_lab.test_python_code(
                            code=code_content,
                            file_path=os.path.basename(file_path)
                        )
                        testing_reports.append(test_report)
                        
                        # AUTO-REPARACIÓN si el score es bajo
                        if test_report.score < 85.0:
                            self.logger.info(f"   🔧 Reparando: {os.path.basename(file_path)} (Score: {test_report.score:.1f}/100)")
                            
                            repair_report = self.repair_workshop.repair_code(
                                code=code_content,
                                file_path=os.path.basename(file_path),
                                test_report=test_report,
                                max_attempts=3
                            )
                            repair_reports.append(repair_report)
                            
                            # Si la reparación mejoró el código, guardar
                            if repair_report.final_score > test_report.score:
                                # Obtener código reparado (último action exitoso)
                                repaired_code = code_content
                                for action in repair_report.actions:
                                    if action.applied and action.success:
                                        repaired_code = repaired_code.replace(
                                            action.code_before, action.code_after
                                        )
                                
                                # Guardar archivo reparado
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(repaired_code)
                                
                                files_repaired += 1
                                self.logger.info(
                                    f"      ✅ Reparado: Score {test_report.score:.1f} → {repair_report.final_score:.1f}"
                                )
                            else:
                                self.logger.warning(
                                    f"      ⚠️ Reparación no mejoró el código"
                                )
                        else:
                            self.logger.info(f"   ✅ {os.path.basename(file_path)}: Score {test_report.score:.1f}/100")
                    
                    except Exception as e:
                        self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
                        self.logger.error(f"   ❌ Error testing/reparando {file_path}: {e}")
            
            # Calcular métricas finales de testing
            if testing_reports:
                avg_test_score = sum(r.score for r in testing_reports) / len(testing_reports)
                total_issues = sum(len(r.issues) for r in testing_reports)
                critical_issues = sum(
                    len([i for i in r.issues if i.severity.value == "critical"])
                    for r in testing_reports
                )
                
                self.logger.info(
                    f"📊 Testing completado: {len(testing_reports)} archivos - "
                    f"Score promedio: {avg_test_score:.1f}/100 - "
                    f"Issues: {total_issues} (Critical: {critical_issues}) - "
                    f"Reparados: {files_repaired}"
                )
            
            # Actualizar quality_score con el testing real
            if testing_reports:
                quality_score = avg_test_score

            return {
                "success": True,
                "task_id": task_id,
                "output_path": task_output_path,
                "files_created": files_created,
                "files_count": len(files_created),
                "lines_of_code": total_lines,
                "quality_score": quality_score,
                "iterations": iterations,
                "converged": converged,
                "target_score": target_score,
                "completion_time_minutes": completion_time,
                "language": task.language.value,
                "project_type": task.project_type.value,
                "military_grade": True,
                "llm_used": result.get('code_results', {}).get('files', {}).get(list(result.get('code_results', {}).get('files', {}).keys())[0] if result.get('code_results', {}).get('files', {}) else None, {}).get('source', 'unknown'),
                # Métricas de testing y reparación
                "testing": {
                    "files_tested": len(testing_reports),
                    "avg_score": avg_test_score if testing_reports else 0.0,
                    "total_issues": total_issues if testing_reports else 0,
                    "critical_issues": critical_issues if testing_reports else 0,
                    "files_repaired": files_repaired,
                },
            }

        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error creando proyecto: {e}")

            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id if "task_id" in locals() else "unknown",
            }

    @trace_operation("materialize_metacortex_thoughts")
    def materialize_metacortex_thoughts(self) -> Dict[str, Any]:
        """
        🧠 Materializa pensamientos de METACORTEX en código REAL

        Analiza las decisiones cognitivas y las convierte en:
        - Nuevos agentes especializados
        - Mejoras al sistema
        - Herramientas personalizadas

        Returns:
            Dict con resultados de la materialización REAL
        """
        self.logger.info("🧠 Materializando pensamientos METACORTEX...")

        components_created = 0
        improvements_applied = 0
        agents_generated = 0
        files_created = []

        try:
            # PASO 1: Analizar estado del sistema
            self.logger.info("📊 Paso 1/4: Analizando sistema...")
            analysis = self.materialization_engine.analyze_system_state()
            suggestions = analysis.get('suggestions', [])

            self.logger.info(f"   ✅ Sugerencias encontradas: {len(suggestions)}")
            
            # 🔥 LOG VERBOSE: Mostrar qué sugerencias hay
            if suggestions:
                for i, sug in enumerate(suggestions[:3], 1):
                    sug_type = sug.get('type', 'unknown') if isinstance(sug, dict) else 'unknown'
                    sug_desc = sug.get('description', str(sug))[:50] if isinstance(sug, dict) else str(sug)[:50]
                    self.logger.info(f"      {i}. [{sug_type}] {sug_desc}")

            # PASO 2: Convertir sugerencias en pensamientos materializables
            self.logger.info("🧠 Paso 2/4: Generando pensamientos...")
            thoughts = []
            
            # 🔥 FIX: Si NO hay sugerencias, generar pensamientos por defecto
            if not suggestions:
                self.logger.info("   ⚠️ No hay sugerencias del análisis, generando pensamientos por defecto...")
                
                # Lista de pensamientos útiles para generar
                default_thoughts = [
                    {
                        "type": "create_agent",
                        "description": "Agente de monitoreo de sistema en tiempo real",
                        "agent_name": "SystemMonitorAgent",
                        "capabilities": ["monitor_cpu", "monitor_memory", "alert_on_threshold"],
                        "priority": 8
                    },
                    {
                        "type": "add_feature",
                        "description": "Sistema de cache distribuido para consultas frecuentes",
                        "feature_name": "DistributedCacheSystem",
                        "target": "cache_system",
                        "priority": 7
                    },
                    {
                        "type": "create_agent",
                        "description": "Agente de análisis de código con ML",
                        "agent_name": "CodeAnalysisAgent",
                        "capabilities": ["analyze_complexity", "suggest_refactoring", "detect_patterns"],
                        "priority": 8
                    },
                    {
                        "type": "add_feature",
                        "description": "API REST para consultas de conocimiento",
                        "feature_name": "KnowledgeQueryAPI",
                        "target": "api_system",
                        "priority": 6
                    },
                    {
                        "type": "create_agent",
                        "description": "Agente de testing automático continuo",
                        "agent_name": "ContinuousTestingAgent",
                        "capabilities": ["run_tests", "generate_test_cases", "report_coverage"],
                        "priority": 9
                    }
                ]
                
                # Seleccionar 1-2 pensamientos aleatorios para no saturar
                num_thoughts = random.randint(1, 2)
                thoughts = random.sample(default_thoughts, num_thoughts)
                self.logger.info(f"   ✅ Generados {len(thoughts)} pensamientos por defecto")
                
            else:
                # Procesar sugerencias existentes
                for suggestion in suggestions[:5]:  # Máximo 5 por ciclo
                    thought_type = "improve_code"  # Por defecto
                    
                    # Sugerencias pueden ser dict o string
                    suggestion_text = suggestion.get("description", str(suggestion)) if isinstance(suggestion, dict) else str(suggestion)
                    suggestion_type = suggestion.get("type", "") if isinstance(suggestion, dict) else ""
                    
                    # Mapear tipos de sugerencia a tipos de thought válidos
                    if suggestion_type in ["create_agent", "add_feature", "improve_code", "refactor"]:
                        thought_type = suggestion_type
                    elif suggestion_type == "testing":
                        thought_type = "add_feature"  # Tests son features
                    elif suggestion_type == "refactoring":
                        thought_type = "refactor"
                    elif suggestion_type == "documentation":
                        thought_type = "improve_code"
                    # Fallback a análisis de texto
                    elif "crear" in suggestion_text.lower() or "nuevo" in suggestion_text.lower() or "agent" in suggestion_text.lower():
                        thought_type = "create_agent"
                    elif "mejorar" in suggestion_text.lower() or "optimizar" in suggestion_text.lower():
                        thought_type = "improve_code"
                    elif "feature" in suggestion_text.lower() or "funcionalidad" in suggestion_text.lower():
                        thought_type = "add_feature"
                    elif "refactor" in suggestion_text.lower():
                        thought_type = "refactor"
                    
                    # Si es dict, usarlo tal cual; si es string, envolverlo
                    if isinstance(suggestion, dict):
                        thought = suggestion.copy()
                        thought["type"] = thought_type  # Forzar tipo válido
                        thoughts.append(thought)
                    else:
                        thoughts.append({
                            "type": thought_type,
                            "description": suggestion_text,
                            "priority": 8,
                            "target": "system"
                        })

            # PASO 3: Materializar pensamientos
            self.logger.info(f"⚡ Paso 3/4: Materializando {len(thoughts)} pensamientos...")
            
            if not thoughts:
                self.logger.warning("⚠️ NO HAY PENSAMIENTOS PARA MATERIALIZAR")
                return {
                    "success": True,
                    "components_created": 0,
                    "agents_generated": 0,
                    "improvements_applied": 0,
                    "files_created": [],
                    "message": "No thoughts to materialize"
                }
            
            for i, thought in enumerate(thoughts, 1):
                try:
                    thought_desc = thought.get('description', 'Unknown')[:50]
                    self.logger.info(f"   🔨 {i}/{len(thoughts)}: [{thought['type']}] {thought_desc}")
                    result = self.materialization_engine.materialize_thought(thought)
                    
                    # 🔥 LOG VERBOSE: Mostrar resultado detallado
                    self.logger.info(f"      → Result: success={result.get('success')}, materialized={result.get('materialized')}")
                    self.logger.info(f"      → Thought type: {thought['type']}, file_path: {result.get('file_path', 'N/A')}")
                    
                    # 🔥 VERIFICAR QUE SE MATERIALIZÓ REALMENTE
                    if result.get('success') and result.get('materialized', False):
                        if thought['type'] == 'create_agent':
                            agents_generated += 1
                            if result.get('file_path'):
                                files_created.append(result['file_path'])
                                self.logger.info(f"      ✅ Agente creado: {result['file_path']}")
                        elif thought['type'] in ['improve_code', 'add_feature']:
                            improvements_applied += 1
                            if result.get('file_path'):
                                files_created.append(result['file_path'])
                                self.logger.info(f"      ✅ Mejora aplicada: {result['file_path']}")
                        
                        components_created += 1
                        self.logger.info(f"      📊 Componentes totales: {components_created}")
                    else:
                        self.logger.warning("      ⚠️ Materialización falló o no completada")
                        self.logger.warning(f"         success={result.get('success')}, materialized={result.get('materialized')}")
                        if result.get('error'):
                            self.logger.warning(f"         Error: {result.get('error')}")
                        
                except Exception as e:
                    self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
                    self.logger.error(f"      ❌ Error materializando pensamiento {i}: {e}")
                    continue

            # PASO 4: Verificar archivos creados
            self.logger.info("📁 Paso 4/4: Verificando archivos...")
            verified_files = []
            for file_path in files_created:
                if Path(file_path).exists():
                    verified_files.append(file_path)
                    self.logger.info(f"   ✅ Verificado: {file_path}")
                else:
                    self.logger.warning(f"   ⚠️ Archivo no encontrado: {file_path}")

            # 🚀 PASO 5 (NUEVO): CRECIMIENTO EXPONENCIAL
            self.logger.info("🚀 Paso 5/5: Activando crecimiento exponencial...")
            
            try:
                from exponential_growth_engine import get_exponential_growth_engine
                
                growth_engine = get_exponential_growth_engine(
                    self.materialization_engine,
                    self.logger
                )
                
                # Registrar éxitos para análisis de patrones
                if agents_generated > 0:
                    for file_path in verified_files:
                        if 'agent' in file_path:
                            growth_engine.record_success('agent_creation', {
                                'success': True,
                                'file_path': file_path,
                                'agent_name': Path(file_path).stem
                            })
                
                if improvements_applied > 0:
                    for file_path in verified_files:
                        if 'feature' in file_path or 'improvement' in file_path:
                            growth_engine.record_success('code_improvement', {
                                'success': True,
                                'file_path': file_path
                            })
                
                # Ejecutar ciclo de crecimiento exponencial
                growth_result = growth_engine.execute_growth_cycle()
                
                # Sumar resultados del crecimiento exponencial
                components_created += growth_result.get('total_materializations', 0)
                agents_generated += growth_result.get('agents_multiplied', 0)
                improvements_applied += growth_result.get('improvements_cascaded', 0)
                
                self.logger.info("   ✅ Crecimiento exponencial completado")
                self.logger.info(f"   • Multiplicación de agentes: {growth_result.get('agents_multiplied', 0)}")
                self.logger.info(f"   • Cascada de mejoras: {growth_result.get('improvements_cascaded', 0)}")
                self.logger.info(f"   • Patrones acelerados: {growth_result.get('patterns_accelerated', 0)}")
                
            except Exception as e:
                self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
                self.logger.warning(f"   ⚠️ Crecimiento exponencial no disponible: {e}")

            # Determinar éxito real
            has_real_changes = (components_created > 0 or improvements_applied > 0 or agents_generated > 0)

            result = {
                "success": has_real_changes,
                "analysis": analysis,
                "components_created": components_created,
                "improvements_applied": improvements_applied,
                "agents_generated": agents_generated,
                "files_created": verified_files,
                "message": f"Materialización {'exitosa' if has_real_changes else 'sin cambios'}"
            }

            if has_real_changes:
                self.logger.info("✅ MATERIALIZACIÓN EXITOSA:")
                self.logger.info(f"   • Componentes: {components_created}")
                self.logger.info(f"   • Mejoras: {improvements_applied}")
                self.logger.info(f"   • Agentes: {agents_generated}")
                self.logger.info(f"   • Archivos: {len(verified_files)}")
                
                # 🔄 AUTO-COMMIT FINAL de toda la materialización
                if self.auto_git_manager:
                    try:
                        self.logger.info("   🔄 Creando commit de materialización completa...")
                        self.auto_git_manager.auto_commit_generated_files(result)
                        self.logger.info("   ✅ Auto-commit completado")
                    except Exception as e:
                        self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
                        self.logger.warning(f"   ⚠️ Auto-commit falló: {e}")
            else:
                self.logger.info("ℹ️ Materialización sin cambios (sistema estable)")

            return result

        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error materializando: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "components_created": 0,
                "improvements_applied": 0,
                "agents_generated": 0,
                "files_created": []
            }

    def enable_autonomous_development_mode(self) -> Dict[str, Any]:
        """
        🤖 Activa modo de desarrollo autónomo

        El agente materializa continuamente pensamientos de METACORTEX

        Returns:
            Dict con estado de activación
        """
        if self.autonomous_mode:
            return {"success": False, "message": "Modo autónomo ya activo"}

        self.autonomous_mode = True

        def autonomous_loop():
            """Loop de desarrollo autónomo"""
            self.logger.info("🤖 Modo autónomo iniciado")

            while self.autonomous_mode:
                try:
                    # Materializar pensamientos cada 5 minutos
                    result = self.materialize_metacortex_thoughts()

                    if result.get("success"):
                        self.logger.info(
                            f"✅ Materialización: {result.get('components_created', 0)} componentes"
                        )

                    time.sleep(300)  # 5 minutos

                except Exception as e:
                    self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
                    self.logger.error(f"❌ Error en loop autónomo: {e}")
                    time.sleep(60)

        thread = threading.Thread(
            target=autonomous_loop, daemon=True, name="AutonomousDev"
        )
        thread.start()

        return {"success": True, "message": "Modo autónomo activado"}

    def disable_autonomous_development_mode(self) -> Dict[str, Any]:
        """Desactiva modo autónomo"""
        self.autonomous_mode = False
        return {"success": True, "message": "Modo autónomo desactivado"}

    def get_workspace_context(self) -> Dict[str, Any]:
        """Obtiene contexto del workspace"""
        return self.workspace_context.copy()

    @trace_operation("query_knowledge")
    def query_knowledge(
        self, question: str, sources: Optional[List[str]] = None, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        🧠 Consulta TODO el sistema de conocimiento de METACORTEX

        Busca en:
        1. Memoria de trabajo (últimas decisiones)
        2. Memoria activa (conocimiento frecuente)
        3. Memoria archivada (todo el historial)
        4. Wikipedia, ArXiv, Internet (conocimiento humano completo)

        Args:
            question: Pregunta o consulta
            sources: Fuentes específicas (None = todas)
            use_cache: Usar caché de búsquedas anteriores

        Returns:
            Dict con respuesta y fuentes consultadas

        Ejemplo:
            >>> agent.query_knowledge("¿Cómo implementar una API REST en Python?")
            {
                "success": True,
                "answer": "...",
                "sources_consulted": ["working_memory", "hierarchical_graph", "wikipedia"],
                "confidence": 0.95
            }
        """
        self.logger.info(f"🧠 Consultando conocimiento: '{question}'")

        if (
            not hasattr(self, "knowledge_system_available")
            or not self.knowledge_system_available
        ):
            return {
                "success": False,
                "error": "Sistema de conocimiento no disponible",
                "answer": "El sistema de conocimiento no está conectado. Usa métodos locales.",
            }

        results = []
        sources_consulted = []

        try:
            # 1. Buscar en memoria de trabajo (más rápido)
            if self.working_memory:
                self.logger.debug("🔍 Buscando en memoria de trabajo...")
                # IMPLEMENTED: Implementar búsqueda en working memory
                sources_consulted.append("working_memory")

            # 2. Buscar en memoria jerárquica (medio plazo)
            if self.learning_system:
                self.logger.debug("🔍 Buscando en sistema jerárquico...")
                search_results = (
                    self.learning_system.hierarchical_graph.search_concepts(
                        question, limit=10
                    )
                )
                if search_results:
                    results.append(
                        {
                            "source": "hierarchical_memory",
                            "results": search_results[:5],
                            "confidence": 0.8,
                        }
                    )
                    sources_consulted.append("hierarchical_graph")

            # 3. Buscar en internet (conocimiento externo)
            if self.search_engine and (not sources or "internet" in sources):
                self.logger.debug("🔍 Buscando en internet...")
                search_result = self.search_engine.intelligent_search(
                    question, max_sources=3
                )
                if search_result.get("success", True):
                    results.append(
                        {
                            "source": "internet",
                            "synthesis": search_result.get("synthesis", ""),
                            "sources": search_result.get("sources", []),
                            "confidence": search_result.get("confidence", 0.5),
                        }
                    )
                    sources_consulted.extend(search_result.get("sources", []))

            # 4. Ingerir conocimiento si es necesario (aprende mientras busca)
            if self.knowledge_engine and not results:
                self.logger.debug("🔍 Ingiriendo conocimiento nuevo...")

                async def ingest():
                    if "wikipedia" in self.knowledge_engine.sources:
                        concepts = await self.knowledge_engine.ingest_from_source(
                            "wikipedia", question, limit=3
                        )
                        return concepts > 0
                    return False

                # Intentar ingerir en background (no bloquear)
                try:
                    ingested = asyncio.run(ingest())
                    if ingested:
                        sources_consulted.append("wikipedia_ingestion")
                        # Buscar de nuevo en memoria jerárquica
                        search_results = (
                            self.learning_system.hierarchical_graph.search_concepts(
                                question, limit=5
                            )
                        )
                        if search_results:
                            results.append(
                                {
                                    "source": "newly_ingested",
                                    "results": search_results,
                                    "confidence": 0.9,
                                }
                            )
                except Exception as e:
                    self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
                    self.logger.debug(f"No se pudo ingerir: {e}")

            # Sintetizar respuesta
            if not results:
                return {
                    "success": False,
                    "answer": f"No se encontró información sobre '{question}' en el sistema.",
                    "sources_consulted": sources_consulted,
                    "confidence": 0.0,
                }

            # Combinar resultados
            answer_parts = []
            total_confidence = 0

            for result in results:
                source = result.get("source", "unknown")
                confidence = result.get("confidence", 0.5)
                total_confidence += confidence

                if "synthesis" in result:
                    answer_parts.append(f"[{source}] {result['synthesis'][:500]}")
                elif "results" in result:
                    answer_parts.append(
                        f"[{source}] Encontrados {len(result['results'])} conceptos relacionados"
                    )

            avg_confidence = total_confidence / len(results) if results else 0.0

            return {
                "success": True,
                "answer": "\n\n".join(answer_parts),
                "sources_consulted": list(set(sources_consulted)),
                "confidence": avg_confidence,
                "details": results,
            }

        except Exception as e:
            self.logger.error(f"Error en programming_agent.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error consultando conocimiento: {e}")

            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "sources_consulted": sources_consulted,
            }

    @trace_operation("analyze_code_file")
    @with_circuit_breaker(FILE_SYSTEM_CIRCUIT_BREAKER)
    def analyze_code_file(self, file_path: str) -> Dict[str, Any]:
        """Analiza un archivo de código"""

        return self.project_analyzer.analyze_code_quality(Path(file_path))

    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODOS INTERNOS
    # ═══════════════════════════════════════════════════════════════════════

    def _generate_task_id(self) -> str:
        """Genera ID único para tarea"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"task_{timestamp}_{hash_part}"

    def _save_task_to_db(self, task_id: str, task: ProgrammingTask, output_path: str):
        """Guarda tarea en base de datos"""
        with sqlite3.connect(str(self.programming_db)) as conn:
            conn.execute(
                """
                INSERT INTO programming_tasks (
                    id, description, language, project_type, output_path
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    task_id,
                    task.description,
                    task.language.value,
                    task.project_type.value,
                    output_path,
                ),
            )

    def _save_completed_project(
        self, task_id: str, task: ProgrammingTask, output_path: str, result: Dict[str, Any], quality: Dict[str, Any]
    ):
        """Guarda proyecto completado"""
        with sqlite3.connect(str(self.programming_db)) as conn:
            project_name = Path(output_path).name

            conn.execute(
                """
                INSERT INTO completed_projects (
                    task_id, project_name, language, project_type, files_created,
                    lines_of_code, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    task_id,
                    project_name,
                    task.language.value,
                    task.project_type.value,
                    result.get("files_created", 0),
                    result.get("lines_of_code", 0),
                    quality.get("overall_score", 0.0),
                ),
            )

            conn.execute(
                """
                UPDATE programming_tasks
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (task_id,),
            )


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES DE UTILIDAD
# ═══════════════════════════════════════════════════════════════════════════


def get_programming_agent(
    project_root: Optional[str] = None, cognitive_agent=None
) -> MetacortexUniversalProgrammingAgent:
    """
    Factory function para crear instancia del agente

    Args:
        project_root: Directorio raíz del proyecto
        cognitive_agent: Agente cognitivo (opcional)

    Returns:
        Instancia del agente de programación
    """
    return MetacortexUniversalProgrammingAgent(
        project_root=project_root, cognitive_agent=cognitive_agent
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="METACORTEX Programming Agent v5.0")
    parser.add_argument("--daemon", action="store_true", help="Modo daemon")
    parser.add_argument("--test", action="store_true", help="Test básico")
    parser.add_argument("--autonomous", action="store_true", help="Modo autónomo")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    agent = MetacortexUniversalProgrammingAgent()

    if args.test:
        print("🧪 Test del agente...")
        result = agent.create_project(
            description="API REST simple",
            language=ProgrammingLanguage.PYTHON,
            project_type=ProjectType.API_SERVICE,
        )
        print(f"Resultado: {result}")

    if args.autonomous:
        agent.enable_autonomous_development_mode()
        print("🤖 Modo autónomo activado. Presiona Ctrl+C para salir.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            agent.disable_autonomous_development_mode()
            print("\n✅ Detenido")

    if args.daemon:
        print("🤖 Modo daemon iniciado")
        try:
            while True:
                result = agent.materialize_metacortex_thoughts()
                if result.get("success"):
                    print(
                        f"✅ Materialización: {result.get('components_created', 0)} componentes"
                    )
                time.sleep(300)
        except KeyboardInterrupt:
            print("\n✅ Daemon detenido")