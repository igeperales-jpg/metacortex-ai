#!/usr/bin/env python3
"""
🎖️ QUALITY INTEGRATION SYSTEM
Sistema integrador de los 5 subsistemas de calidad militar

Integra:
    pass  # TODO: Implementar
1. Advanced Testing Laboratory
2. Self-Repair Workshop
3. Code Quality Enforcer
4. AI Programming Evolution
5. Continuous Validation System

Autor: METACORTEX v3.0
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging


@dataclass
class IntegratedQualityResult:
    """Resultado integrado de todos los sistemas de calidad"""
    
    # Identificación
    filepath: str
    timestamp: float = field(default_factory=time.time)
    
    # Scores individuales
    testing_score: float = 0.0
    quality_enforcement_score: float = 0.0
    continuous_validation_score: float = 0.0
    
    # Score final (promedio ponderado)
    final_score: float = 0.0
    
    # Reparación
    repaired: bool = False
    repair_strategies: List[str] = field(default_factory=list)
    
    # Aprendizaje
    patterns_found: List[str] = field(default_factory=list)
    patterns_learned: List[str] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)
    
    # Issues agregados
    total_issues: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    
    # Detalles por sistema
    testing_details: Optional[Dict[str, Any]] = None
    quality_details: Optional[Dict[str, Any]] = None
    validation_details: Optional[Dict[str, Any]] = None
    evolution_details: Optional[Dict[str, Any]] = None
    
    # Tiempo de ejecución
    execution_time: float = 0.0
    
    # Estado militar
    military_grade: bool = False  # >= 90
    professional_grade: bool = False  # >= 80
    acceptable: bool = False  # >= 70


class QualityIntegrationSystem:
    """Sistema integrador de calidad militar"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.logger = logging.getLogger("quality_integration")
        
        # Lazy loading de sistemas
        self._testing_lab = None
        self._repair_workshop = None
        self._quality_enforcer = None
        self._evolution_system = None
        self._validation_system = None
        
        # Configuración
        self.config = {
            "military_threshold": 90.0,
            "professional_threshold": 80.0,
            "acceptable_threshold": 70.0,
            "auto_repair_threshold": 85.0,
            "max_repair_attempts": 3,
            
            # Pesos para score final
            "weights": {
                "testing": 0.35,
                "quality_enforcement": 0.30,
                "continuous_validation": 0.35
            }
        }
        
        # Métricas
        self.metrics = {
            "total_analyzed": 0,
            "total_repaired": 0,
            "military_grade_count": 0,
            "professional_grade_count": 0,
            "avg_score": 0.0,
            "avg_execution_time": 0.0
        }
        
        self._initialized = True
        self.logger.info("✅ Quality Integration System inicializado")
    
    def _load_testing_lab(self):
        """Carga lazy del Testing Lab"""
        if self._testing_lab is None:
            try:
                from agent_modules.advanced_testing_lab import get_testing_lab
                self._testing_lab = get_testing_lab()
                self.logger.info("✅ Testing Lab cargado")
            except Exception as e:
                logger.error(f"Error en quality_integration_system.py: {e}", exc_info=True)
                self.logger.warning(f"⚠️ No se pudo cargar Testing Lab: {e}")
        return self._testing_lab
    
    def _load_repair_workshop(self):
        """Carga lazy del Repair Workshop"""
        if self._repair_workshop is None:
            try:
                from agent_modules.self_repair_workshop import get_repair_workshop
                self._repair_workshop = get_repair_workshop()
                self.logger.info("✅ Repair Workshop cargado")
            except Exception as e:
                logger.error(f"Error en quality_integration_system.py: {e}", exc_info=True)
                self.logger.warning(f"⚠️ No se pudo cargar Repair Workshop: {e}")
        return self._repair_workshop
    
    def _load_quality_enforcer(self):
        """Carga lazy del Quality Enforcer"""
        if self._quality_enforcer is None:
            try:
                from agent_modules.code_quality_enforcer import get_enforcer
                self._quality_enforcer = get_enforcer()
                self.logger.info("✅ Quality Enforcer cargado")
            except Exception as e:
                logger.error(f"Error en quality_integration_system.py: {e}", exc_info=True)
                self.logger.warning(f"⚠️ No se pudo cargar Quality Enforcer: {e}")
        return self._quality_enforcer
    
    def _load_evolution_system(self):
        """Carga lazy del Evolution System"""
        if self._evolution_system is None:
            try:
                from agent_modules.ai_programming_evolution import get_evolution_system
                self._evolution_system = get_evolution_system()
                self.logger.info("✅ Evolution System cargado")
            except Exception as e:
                logger.error(f"Error en quality_integration_system.py: {e}", exc_info=True)
                self.logger.warning(f"⚠️ No se pudo cargar Evolution System: {e}")
        return self._evolution_system
    
    def _load_validation_system(self):
        """Carga lazy del Validation System"""
        if self._validation_system is None:
            try:
                from agent_modules.continuous_validation import get_validation_system
                self._validation_system = get_validation_system()
                self.logger.info("✅ Validation System cargado")
            except Exception as e:
                logger.error(f"Error en quality_integration_system.py: {e}", exc_info=True)
                self.logger.warning(f"⚠️ No se pudo cargar Validation System: {e}")
        return self._validation_system
    
    async def analyze_code(
        self,
        code: str,
        filepath: str,
        description: str = "",
        auto_repair: bool = True
    ) -> IntegratedQualityResult:
        """
        Análisis integrado de código con todos los sistemas
        
        Args:
            code: Código a analizar
            filepath: Ruta del archivo
            description: Descripción del proyecto/tarea
            auto_repair: Si debe intentar reparar automáticamente
        
        Returns:
            IntegratedQualityResult con todos los resultados
        """
        start_time = time.time()
        result = IntegratedQualityResult(filepath=filepath)
        
        self.logger.info(f"🎖️ Iniciando análisis militar integrado: {filepath}")
        
        current_code = code
        all_issues = []
        
        # 1️⃣ TESTING LABORATORY
        testing_lab = self._load_testing_lab()
        if testing_lab:
            try:
                test_result = testing_lab.test_code(current_code, filepath)
                
                result.testing_score = test_result.score  # score no overall_score
                result.testing_details = {
                    "passed": test_result.passed,
                    "issues": len(test_result.issues),
                    "execution_time": test_result.execution_time_ms
                }
                
                all_issues.extend(test_result.issues)  # issues no all_issues
                
                self.logger.info(f"   ✅ Testing: {result.testing_score:.1f}/100")
                
            except Exception as e:
                logger.error(f"Error en quality_integration_system.py: {e}", exc_info=True)
                self.logger.error(f"   ❌ Testing Lab error: {e}")
        
        # 2️⃣ CODE QUALITY ENFORCER
        enforcer = self._load_quality_enforcer()
        if enforcer:
            try:
                metrics = enforcer.enforce(current_code, filepath)
                
                result.quality_enforcement_score = metrics.overall_score
                result.quality_details = {
                    "level": metrics.level.name,
                    "complexity_score": metrics.complexity_score,
                    "maintainability_score": metrics.maintainability_score,
                    "security_score": metrics.security_score,
                    "documentation_score": metrics.documentation_score,
                    "type_safety_score": metrics.type_safety_score,
                    "architecture_score": metrics.architecture_score,
                    "violations": len(metrics.violations)
                }
                
                all_issues.extend(metrics.violations)
                
                self.logger.info(f"   ✅ Quality: {result.quality_enforcement_score:.1f}/100 ({metrics.level.name})")
                
            except Exception as e:
                logger.error(f"Error en quality_integration_system.py: {e}", exc_info=True)
                self.logger.error(f"   ❌ Quality Enforcer error: {e}")
        
        # 3️⃣ CONTINUOUS VALIDATION (async)
        validation_system = self._load_validation_system()
        if validation_system:
            try:
                validation_result = await validation_system.validate_continuously(
                    current_code, filepath
                )
                
                result.continuous_validation_score = validation_result['avg_score']
                result.validation_details = {
                    "passed": validation_result['passed'],
                    "total_issues": validation_result['total_issues'],
                    "critical_issues": validation_result['critical_issues'],
                    "validators_passed": validation_result['validators_passed'],
                    "validators_failed": validation_result['validators_failed']
                }
                
                # Agregar issues de validación
                for val_result in validation_result['results']:
                    all_issues.extend(val_result.issues)
                
                self.logger.info(f"   ✅ Validation: {result.continuous_validation_score:.1f}/100")
                
            except Exception as e:
                logger.error(f"Error en quality_integration_system.py: {e}", exc_info=True)
                self.logger.error(f"   ❌ Continuous Validation error: {e}")
        
        # Calcular score final (promedio ponderado)
        weights = self.config['weights']
        result.final_score = (
            result.testing_score * weights['testing'] +
            result.quality_enforcement_score * weights['quality_enforcement'] +
            result.continuous_validation_score * weights['continuous_validation']
        )
        
        # Contar issues por severidad
        for issue in all_issues:
            # Manejar tanto objetos TestIssue como dicts
            if hasattr(issue, 'severity'):
                severity = str(issue.severity.value if hasattr(issue.severity, 'value') else issue.severity).lower()
            else:
                severity = str(issue.get('severity', 'low')).lower()
            
            if 'critical' in severity:
                result.critical_issues += 1
            elif 'high' in severity:
                result.high_issues += 1
            elif 'medium' in severity or 'moderate' in severity:
                result.medium_issues += 1
            else:
                result.low_issues += 1
        
        result.total_issues = len(all_issues)
        
        # 4️⃣ AUTO-REPAIR (si score bajo y auto_repair activado)
        if auto_repair and result.final_score < self.config['auto_repair_threshold']:
            repair_workshop = self._load_repair_workshop()
            
            if repair_workshop and all_issues:
                self.logger.warning(f"   ⚠️ Score bajo ({result.final_score:.1f}) - Iniciando auto-reparación...")
                
                repair_attempts = 0
                max_attempts = self.config['max_repair_attempts']
                
                while repair_attempts < max_attempts and result.final_score < self.config['auto_repair_threshold']:
                    try:
                        repair_result = repair_workshop.repair_code(
                            code=current_code,
                            file_path=filepath  # file_path no filepath
                        )
                        
                        if repair_result.repairs_successful > 0:
                            current_code = repair_result.code if hasattr(repair_result, 'code') else current_code
                            result.repaired = True
                            # RepairReport tiene actions no strategies
                            result.repair_strategies.extend([a.strategy_type for a in repair_result.actions])
                            
                            # Re-validar código reparado
                            old_score = result.final_score
                            result.final_score = repair_result.final_score
                            
                            self.logger.info(f"   ✅ Reparado (intento {repair_attempts + 1}): {old_score:.1f} → {result.final_score:.1f}")
                            
                            if result.final_score >= self.config['auto_repair_threshold']:
                                break
                        else:
                            self.logger.warning(f"   ⚠️ Reparación fallida (intento {repair_attempts + 1})")
                            break
                        
                        repair_attempts += 1
                        
                    except Exception as e:
                        logger.error(f"Error en quality_integration_system.py: {e}", exc_info=True)
                        self.logger.error(f"   ❌ Repair error: {e}")
                        break
        
        # 5️⃣ AI EVOLUTION (aprendizaje)
        evolution = self._load_evolution_system()
        if evolution:
            try:
                project_id = f"gen_{int(time.time())}_{Path(filepath).stem}"
                
                learning_result = evolution.learn_from_project(
                    project_id=project_id,
                    description=description or f"Code generation: {filepath}",
                    code={filepath: current_code},
                    quality_score=result.final_score,
                    success=result.final_score >= self.config['acceptable_threshold'],
                    errors=[
                        i.message if hasattr(i, 'message') else str(i.get('message', ''))
                        for i in all_issues
                        if (hasattr(i, 'severity') and 'critical' in str(i.severity).lower()) or
                           (hasattr(i, 'get') and i.get('severity') == 'critical')
                    ]
                )
                
                # Los valores son int, no listas
                result.patterns_found = []
                result.patterns_learned = []
                result.anti_patterns = []
                result.evolution_details = learning_result
                
                patterns_count = learning_result.get('new_patterns_learned', 0)
                self.logger.info(f"   ✅ Evolution: {patterns_count} patrones aprendidos")
                
            except Exception as e:
                logger.error(f"Error en quality_integration_system.py: {e}", exc_info=True)
                self.logger.error(f"   ❌ Evolution error: {e}")
        
        # Determinar grado militar
        result.military_grade = result.final_score >= self.config['military_threshold']
        result.professional_grade = result.final_score >= self.config['professional_threshold']
        result.acceptable = result.final_score >= self.config['acceptable_threshold']
        
        # Tiempo de ejecución
        result.execution_time = time.time() - start_time
        
        # Actualizar métricas
        self._update_metrics(result)
        
        # Log final
        grade = "🏆 MILITARY" if result.military_grade else "⭐ PROFESSIONAL" if result.professional_grade else "✅ ACCEPTABLE" if result.acceptable else "❌ NEEDS WORK"
        
        self.logger.info(f"   {grade} - Score final: {result.final_score:.1f}/100 ({result.execution_time:.2f}s)")
        
        if result.repaired:
            self.logger.info(f"   🔧 Reparaciones aplicadas: {len(result.repair_strategies)}")
        
        if result.critical_issues > 0:
            self.logger.warning(f"   ⚠️ Issues críticos pendientes: {result.critical_issues}")
        
        return result
    
    def _update_metrics(self, result: IntegratedQualityResult):
        """Actualiza métricas del sistema"""
        self.metrics['total_analyzed'] += 1
        
        if result.repaired:
            self.metrics['total_repaired'] += 1
        
        if result.military_grade:
            self.metrics['military_grade_count'] += 1
        
        if result.professional_grade:
            self.metrics['professional_grade_count'] += 1
        
        # Promedio móvil del score
        n = self.metrics['total_analyzed']
        old_avg = self.metrics['avg_score']
        self.metrics['avg_score'] = (old_avg * (n - 1) + result.final_score) / n
        
        # Promedio móvil del tiempo de ejecución
        old_time_avg = self.metrics['avg_execution_time']
        self.metrics['avg_execution_time'] = (old_time_avg * (n - 1) + result.execution_time) / n
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema"""
        return {
            **self.metrics,
            "military_rate": self.metrics['military_grade_count'] / max(self.metrics['total_analyzed'], 1),
            "professional_rate": self.metrics['professional_grade_count'] / max(self.metrics['total_analyzed'], 1),
            "repair_rate": self.metrics['total_repaired'] / max(self.metrics['total_analyzed'], 1)
        }
    
    def generate_report(self) -> str:
        """Genera reporte completo del sistema"""
        metrics = self.get_metrics()
        
        report = f"""
═══════════════════════════════════════════════════════════════
🎖️ QUALITY INTEGRATION SYSTEM - MILITARY REPORT
═══════════════════════════════════════════════════════════════

📊 MÉTRICAS GENERALES:
  • Total archivos analizados: {metrics['total_analyzed']}
  • Archivos reparados: {metrics['total_repaired']} ({metrics['repair_rate']*100:.1f}%)
  • Score promedio: {metrics['avg_score']:.1f}/100
  • Tiempo promedio: {metrics['avg_execution_time']:.2f}s

🏆 GRADOS DE CALIDAD:
  • Military Grade (≥90): {metrics['military_grade_count']} ({metrics['military_rate']*100:.1f}%)
  • Professional Grade (≥80): {metrics['professional_grade_count']} ({metrics['professional_rate']*100:.1f}%)

🔧 SISTEMAS INTEGRADOS:
  • Testing Laboratory: {'✅' if self._testing_lab else '❌'}
  • Self-Repair Workshop: {'✅' if self._repair_workshop else '❌'}
  • Code Quality Enforcer: {'✅' if self._quality_enforcer else '❌'}
  • AI Programming Evolution: {'✅' if self._evolution_system else '❌'}
  • Continuous Validation: {'✅' if self._validation_system else '❌'}

═══════════════════════════════════════════════════════════════
"""
        return report


# Singleton global
_integration_system_instance = None

def get_integration_system() -> QualityIntegrationSystem:
    """Obtiene instancia singleton del Integration System"""
    global _integration_system_instance
    if _integration_system_instance is None:
        _integration_system_instance = QualityIntegrationSystem()
    return _integration_system_instance


# Función de conveniencia para usar con asyncio
def analyze_code_sync(
    code: str,
    filepath: str,
    description: str = "",
    auto_repair: bool = True
) -> IntegratedQualityResult:
    """Versión sync de analyze_code (ejecuta el loop asyncio internamente)"""
    system = get_integration_system()
    return asyncio.run(system.analyze_code(code, filepath, description, auto_repair))


if __name__ == "__main__":
    # Test del sistema
    logging.basicConfig(level=logging.INFO)
    
    test_code = '''
def calculate_sum(a, b):
    """Calculate sum of two numbers"""
    return a + b

def divide(a, b):
    """Divide two numbers"""
    return a / b  # No error handling!

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x):
        self.result += x
'''
    
    result = analyze_code_sync(
        code=test_code,
        filepath="test_calculator.py",
        description="Simple calculator module",
        auto_repair=True
    )
    
    print(f"\n{'='*60}")
    print(f"Final Score: {result.final_score:.1f}/100")
    print(f"Military Grade: {result.military_grade}")
    print(f"Repaired: {result.repaired}")
    print(f"Total Issues: {result.total_issues}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"{'='*60}\n")
    
    system = get_integration_system()
    print(system.generate_report())
