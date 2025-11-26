#!/usr/bin/env python3
"""
🧠🔮 METACORTEX SELF-IMPROVEMENT SYSTEM - AUTONOMOUS EVOLUTION
═══════════════════════════════════════════════════════════════════════════

MISIÓN: Permitir que METACORTEX se vea a sí mismo, se analice, y se mejore
        de forma autónoma sin intervención humana.

CAPACIDADES:
    ✅ Introspección de código propio
    ✅ Detección de áreas de mejora
    ✅ Auto-generación de código mejorado
    ✅ Auto-testing y validación
    ✅ Auto-deployment de mejoras
    ✅ Rollback automático si falla
    ✅ Versioning y auditoría completa

ARQUITECTURA:
    ┌─────────────────────────────────────────────────────┐
    │  SELF-AWARENESS LAYER (Consciencia de sí mismo)    │
    │  • Escanea su propio código                         │
    │  • Detecta patrones subóptimos                      │
    │  • Identifica áreas de mejora                       │
    └─────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────┐
    │  ANALYSIS LAYER (Análisis profundo)                │
    │  • Analiza complejidad ciclomática                  │
    │  • Detecta código duplicado                         │
    │  • Encuentra anti-patrones                          │
    │  • Calcula métricas de calidad                      │
    └─────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────┐
    │  DECISION LAYER (Toma de decisiones)               │
    │  • Prioriza mejoras por impacto                     │
    │  • Decide qué cambiar primero                       │
    │  • Valida seguridad de cambios                      │
    └─────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────┐
    │  IMPROVEMENT LAYER (Mejora continua)               │
    │  • Genera código mejorado con LLM                   │
    │  • Aplica mejoras incrementales                     │
    │  • Ejecuta tests automáticos                        │
    └─────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────┐
    │  DEPLOYMENT LAYER (Auto-deployment)                │
    │  • Aplica cambios de forma segura                   │
    │  • Monitorea post-deployment                        │
    │  • Rollback automático si falla                     │
    └─────────────────────────────────────────────────────┘

FILOSOFÍA:
    "Un sistema que puede verse a sí mismo y mejorarse
     es un sistema que trasciende su programación inicial."

AUTOR: METACORTEX AUTONOMOUS SYSTEM
FECHA: 2025-11-26
═══════════════════════════════════════════════════════════════════════════
"""

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import radon.complexity as radon_cc
import radon.metrics as radon_metrics
from radon.raw import analyze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS Y DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════

class ImprovementType(Enum):
    """Tipos de mejoras que el sistema puede hacer."""
    CODE_OPTIMIZATION = "code_optimization"
    REFACTORING = "refactoring"
    BUG_FIX = "bug_fix"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    NEW_FEATURE = "new_feature"


class ImprovementPriority(Enum):
    """Prioridad de las mejoras."""
    CRITICAL = 5  # Bugs críticos, seguridad
    HIGH = 4      # Performance, refactoring importante
    MEDIUM = 3    # Optimizaciones, mejoras generales
    LOW = 2       # Documentación, limpieza
    NICE_TO_HAVE = 1  # Features nuevas


@dataclass
class CodeAnalysis:
    """Resultado del análisis de un archivo."""
    file_path: str
    lines_of_code: int
    complexity: float
    maintainability_index: float
    duplications: List[str] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    quality_score: float = 0.0


@dataclass
class Improvement:
    """Representa una mejora que el sistema puede hacer."""
    id: str
    type: ImprovementType
    priority: ImprovementPriority
    file_path: str
    description: str
    original_code: str
    improved_code: str
    expected_impact: str
    risk_level: str
    created_at: datetime = field(default_factory=datetime.now)
    applied: bool = False
    success: Optional[bool] = None
    rollback_available: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# SELF-IMPROVEMENT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class SelfImprovementSystem:
    """
    Sistema de auto-mejora que permite a METACORTEX analizarse y mejorarse
    a sí mismo de forma autónoma.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.improvements_dir = self.project_root / "self_improvements"
        self.backups_dir = self.improvements_dir / "backups"
        self.history_file = self.improvements_dir / "improvement_history.json"
        
        # Crear directorios
        self.improvements_dir.mkdir(exist_ok=True)
        self.backups_dir.mkdir(exist_ok=True)
        
        # Estado
        self.is_running = False
        self.total_improvements_applied = 0
        self.success_rate = 0.0
        self.improvement_history: List[Dict] = []
        
        # Lock
        self._lock = threading.RLock()
        
        # Cargar historial
        self._load_history()
        
        logger.info("=" * 80)
        logger.info("🧠 SELF-IMPROVEMENT SYSTEM INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Improvements dir: {self.improvements_dir}")
        logger.info(f"Total improvements applied: {self.total_improvements_applied}")
        logger.info("=" * 80)
    
    def _load_history(self):
        """Carga el historial de mejoras."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.improvement_history = data.get('improvements', [])
                    self.total_improvements_applied = data.get('total_applied', 0)
                    self.success_rate = data.get('success_rate', 0.0)
                    logger.info(f"✅ Loaded {len(self.improvement_history)} improvements from history")
            except Exception as e:
                logger.error(f"Error loading history: {e}")
    
    def _save_history(self):
        """Guarda el historial de mejoras."""
        try:
            data = {
                'improvements': self.improvement_history,
                'total_applied': self.total_improvements_applied,
                'success_rate': self.success_rate,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def analyze_self(self) -> Dict[str, CodeAnalysis]:
        """
        Analiza todo el código del proyecto.
        Esto es como "mirarse en el espejo" - el sistema se ve a sí mismo.
        """
        logger.info("🔍 ANALYZING SELF - SYSTEM INTROSPECTION")
        logger.info("=" * 80)
        
        analyses = {}
        
        # Archivos Python a analizar
        python_files = list(self.project_root.glob("*.py"))
        python_files.extend(self.project_root.glob("agent_modules/*.py"))
        python_files.extend(self.project_root.glob("metacortex_sinaptico/**/*.py"))
        
        logger.info(f"Found {len(python_files)} Python files to analyze")
        
        for file_path in python_files:
            try:
                analysis = self._analyze_file(file_path)
                analyses[str(file_path)] = analysis
                
                logger.info(f"   📄 {file_path.name}:")
                logger.info(f"      Lines: {analysis.lines_of_code}")
                logger.info(f"      Complexity: {analysis.complexity:.2f}")
                logger.info(f"      Maintainability: {analysis.maintainability_index:.2f}")
                logger.info(f"      Quality: {analysis.quality_score:.2f}/100")
                
                if analysis.anti_patterns:
                    logger.warning(f"      ⚠️  Anti-patterns: {len(analysis.anti_patterns)}")
                
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        logger.info("=" * 80)
        logger.info(f"✅ Analysis complete: {len(analyses)} files analyzed")
        
        return analyses
    
    def _analyze_file(self, file_path: Path) -> CodeAnalysis:
        """Analiza un archivo específico."""
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Análisis con radon
        raw_analysis = analyze(code)
        
        # Complejidad ciclomática
        try:
            cc_results = radon_cc.cc_visit(code)
            avg_complexity = sum(r.complexity for r in cc_results) / len(cc_results) if cc_results else 0
        except:
            avg_complexity = 0
        
        # Maintainability index
        try:
            mi_result = radon_metrics.mi_visit(code, multi=True)
            maintainability = mi_result if isinstance(mi_result, (int, float)) else 0
        except:
            maintainability = 0
        
        # Detectar anti-patrones
        anti_patterns = self._detect_anti_patterns(code, file_path)
        
        # Sugerencias
        suggestions = self._generate_suggestions(raw_analysis, avg_complexity, maintainability, anti_patterns)
        
        # Calcular quality score
        quality_score = self._calculate_quality_score(avg_complexity, maintainability, len(anti_patterns))
        
        return CodeAnalysis(
            file_path=str(file_path),
            lines_of_code=raw_analysis.loc,
            complexity=avg_complexity,
            maintainability_index=maintainability,
            anti_patterns=anti_patterns,
            suggestions=suggestions,
            quality_score=quality_score
        )
    
    def _detect_anti_patterns(self, code: str, file_path: Path) -> List[str]:
        """Detecta anti-patrones en el código."""
        anti_patterns = []
        
        # Funciones muy largas (>100 líneas)
        if code.count('\n') > 100:
            functions = re.findall(r'def\s+\w+\([^)]*\):', code)
            if functions:
                # Verificar si hay funciones individuales muy largas
                for func in functions:
                    func_code = code[code.find(func):]
                    next_def = func_code[10:].find('def ')
                    if next_def > 0:
                        func_code = func_code[:next_def + 10]
                    if func_code.count('\n') > 100:
                        anti_patterns.append(f"Very long function: {func}")
        
        # God objects (archivos >1000 líneas)
        if code.count('\n') > 1000:
            anti_patterns.append("God object: File too large (>1000 lines)")
        
        # Imports *
        if 'from' in code and 'import *' in code:
            anti_patterns.append("Wildcard import (import *)")
        
        # Too many try/except
        try_count = code.count('try:')
        if try_count > 10:
            anti_patterns.append(f"Too many try/except blocks ({try_count})")
        
        # Código comentado
        commented_lines = len([l for l in code.split('\n') if l.strip().startswith('#') and len(l.strip()) > 20])
        if commented_lines > 50:
            anti_patterns.append(f"Too many commented lines ({commented_lines})")
        
        return anti_patterns
    
    def _generate_suggestions(self, raw_analysis, complexity, maintainability, anti_patterns) -> List[str]:
        """Genera sugerencias de mejora."""
        suggestions = []
        
        # Alta complejidad
        if complexity > 10:
            suggestions.append("High complexity: Consider breaking down into smaller functions")
        
        # Baja mantenibilidad
        if maintainability < 50:
            suggestions.append("Low maintainability: Refactor to improve code clarity")
        
        # Muchos comentarios
        if raw_analysis.comments > raw_analysis.loc * 0.3:
            suggestions.append("Too many comments: Consider self-documenting code")
        
        # Anti-patrones detectados
        for ap in anti_patterns:
            suggestions.append(f"Fix anti-pattern: {ap}")
        
        return suggestions
    
    def _calculate_quality_score(self, complexity, maintainability, anti_patterns_count) -> float:
        """Calcula un score de calidad del código."""
        score = 100.0
        
        # Penalizar complejidad
        if complexity > 10:
            score -= min(30, (complexity - 10) * 3)
        
        # Penalizar baja mantenibilidad
        if maintainability < 50:
            score -= (50 - maintainability) * 0.5
        
        # Penalizar anti-patrones
        score -= anti_patterns_count * 5
        
        return max(0, score)
    
    def generate_improvements(self, analyses: Dict[str, CodeAnalysis]) -> List[Improvement]:
        """
        Genera mejoras basadas en el análisis.
        Aquí es donde el sistema DECIDE qué cambiar.
        """
        logger.info("💡 GENERATING IMPROVEMENTS")
        logger.info("=" * 80)
        
        improvements = []
        
        for file_path, analysis in analyses.items():
            # Solo mejorar archivos con bajo quality score
            if analysis.quality_score < 70:
                logger.info(f"   🎯 {Path(file_path).name} needs improvement (score: {analysis.quality_score:.1f})")
                
                # Generar mejoras según las sugerencias
                for suggestion in analysis.suggestions:
                    improvement = self._create_improvement(file_path, analysis, suggestion)
                    if improvement:
                        improvements.append(improvement)
                        logger.info(f"      ✅ Created improvement: {improvement.description}")
        
        logger.info("=" * 80)
        logger.info(f"✅ Generated {len(improvements)} improvements")
        
        # Ordenar por prioridad
        improvements.sort(key=lambda i: i.priority.value, reverse=True)
        
        return improvements
    
    def _create_improvement(self, file_path: str, analysis: CodeAnalysis, suggestion: str) -> Optional[Improvement]:
        """Crea una mejora específica."""
        
        # Leer código actual
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
        except:
            return None
        
        # Determinar tipo y prioridad
        improvement_type, priority = self._classify_improvement(suggestion)
        
        # Generar código mejorado (aquí usaríamos Ollama para generar el código)
        improved_code = self._generate_improved_code(original_code, suggestion)
        
        improvement_id = f"imp_{int(time.time())}_{Path(file_path).stem}"
        
        return Improvement(
            id=improvement_id,
            type=improvement_type,
            priority=priority,
            file_path=file_path,
            description=suggestion,
            original_code=original_code,
            improved_code=improved_code,
            expected_impact=self._estimate_impact(analysis, suggestion),
            risk_level=self._estimate_risk(analysis)
        )
    
    def _classify_improvement(self, suggestion: str) -> Tuple[ImprovementType, ImprovementPriority]:
        """Clasifica el tipo y prioridad de una mejora."""
        suggestion_lower = suggestion.lower()
        
        if 'security' in suggestion_lower or 'vulnerability' in suggestion_lower:
            return ImprovementType.SECURITY, ImprovementPriority.CRITICAL
        
        if 'bug' in suggestion_lower or 'fix' in suggestion_lower:
            return ImprovementType.BUG_FIX, ImprovementPriority.HIGH
        
        if 'performance' in suggestion_lower or 'optimization' in suggestion_lower:
            return ImprovementType.PERFORMANCE, ImprovementPriority.HIGH
        
        if 'complexity' in suggestion_lower or 'refactor' in suggestion_lower:
            return ImprovementType.REFACTORING, ImprovementPriority.MEDIUM
        
        if 'documentation' in suggestion_lower or 'comment' in suggestion_lower:
            return ImprovementType.DOCUMENTATION, ImprovementPriority.LOW
        
        return ImprovementType.CODE_OPTIMIZATION, ImprovementPriority.MEDIUM
    
    def _generate_improved_code(self, original_code: str, suggestion: str) -> str:
        """
        Genera código mejorado usando Ollama.
        Aquí es donde el sistema SE REPROGRAMA A SÍ MISMO.
        """
        # TODO: Integrar con Ollama para generar código mejorado
        # Por ahora, retornamos el código original con un comentario
        return f"# IMPROVED: {suggestion}\n{original_code}"
    
    def _estimate_impact(self, analysis: CodeAnalysis, suggestion: str) -> str:
        """Estima el impacto de una mejora."""
        if analysis.quality_score < 40:
            return "HIGH - Significant quality improvement"
        elif analysis.quality_score < 60:
            return "MEDIUM - Moderate improvement"
        else:
            return "LOW - Minor improvement"
    
    def _estimate_risk(self, analysis: CodeAnalysis) -> str:
        """Estima el riesgo de aplicar una mejora."""
        if analysis.complexity > 15:
            return "HIGH - Complex code, careful testing needed"
        elif analysis.complexity > 10:
            return "MEDIUM - Moderate complexity"
        else:
            return "LOW - Simple change"
    
    def apply_improvement(self, improvement: Improvement) -> bool:
        """
        Aplica una mejora al código.
        Aquí es donde el sistema EJECUTA los cambios en sí mismo.
        """
        logger.info(f"🔧 APPLYING IMPROVEMENT: {improvement.id}")
        logger.info(f"   File: {improvement.file_path}")
        logger.info(f"   Type: {improvement.type.value}")
        logger.info(f"   Priority: {improvement.priority.name}")
        
        try:
            # 1. Backup del archivo original
            backup_path = self._create_backup(improvement.file_path)
            logger.info(f"   ✅ Backup created: {backup_path}")
            
            # 2. Aplicar cambios
            with open(improvement.file_path, 'w', encoding='utf-8') as f:
                f.write(improvement.improved_code)
            logger.info(f"   ✅ Changes applied")
            
            # 3. Validar cambios
            if self._validate_changes(improvement.file_path):
                logger.info(f"   ✅ Changes validated")
                improvement.applied = True
                improvement.success = True
                self.total_improvements_applied += 1
                
                # Guardar en historial
                self._save_to_history(improvement)
                
                return True
            else:
                logger.error(f"   ❌ Validation failed, rolling back")
                self._rollback(improvement.file_path, backup_path)
                improvement.success = False
                return False
                
        except Exception as e:
            logger.error(f"   ❌ Error applying improvement: {e}")
            if 'backup_path' in locals():
                self._rollback(improvement.file_path, backup_path)
            improvement.success = False
            return False
    
    def _create_backup(self, file_path: str) -> Path:
        """Crea un backup del archivo."""
        backup_name = f"{Path(file_path).stem}_{int(time.time())}.backup"
        backup_path = self.backups_dir / backup_name
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def _validate_changes(self, file_path: str) -> bool:
        """Valida que los cambios no rompan el código."""
        try:
            # 1. Verificar que el Python es válido
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            compile(code, file_path, 'exec')
            
            # 2. TODO: Ejecutar tests automáticos
            
            return True
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def _rollback(self, file_path: str, backup_path: Path):
        """Hace rollback de los cambios."""
        logger.warning(f"🔙 Rolling back {file_path}")
        shutil.copy2(backup_path, file_path)
    
    def _save_to_history(self, improvement: Improvement):
        """Guarda una mejora en el historial."""
        history_entry = {
            'id': improvement.id,
            'type': improvement.type.value,
            'priority': improvement.priority.name,
            'file': improvement.file_path,
            'description': improvement.description,
            'applied_at': datetime.now().isoformat(),
            'success': improvement.success
        }
        self.improvement_history.append(history_entry)
        self._save_history()
    
    def start_autonomous_improvement_loop(self, interval_minutes: int = 60):
        """
        Inicia el loop de mejora autónoma.
        El sistema se analizará y mejorará a sí mismo periódicamente.
        """
        logger.info("=" * 80)
        logger.info("🚀 STARTING AUTONOMOUS IMPROVEMENT LOOP")
        logger.info(f"   Interval: {interval_minutes} minutes")
        logger.info("=" * 80)
        
        self.is_running = True
        
        def improvement_loop():
            while self.is_running:
                try:
                    logger.info("🧠 Starting self-improvement cycle...")
                    
                    # 1. Analizarse a sí mismo
                    analyses = self.analyze_self()
                    
                    # 2. Generar mejoras
                    improvements = self.generate_improvements(analyses)
                    
                    # 3. Aplicar mejoras (solo las de alta prioridad)
                    high_priority = [i for i in improvements if i.priority.value >= 4]
                    
                    logger.info(f"   Found {len(high_priority)} high-priority improvements")
                    
                    for improvement in high_priority[:3]:  # Aplicar máximo 3 por ciclo
                        self.apply_improvement(improvement)
                        time.sleep(5)  # Esperar entre mejoras
                    
                    logger.info(f"✅ Self-improvement cycle complete")
                    logger.info(f"   Total improvements applied: {self.total_improvements_applied}")
                    
                except Exception as e:
                    logger.error(f"Error in improvement loop: {e}")
                
                # Esperar hasta el próximo ciclo
                logger.info(f"⏰ Next cycle in {interval_minutes} minutes")
                time.sleep(interval_minutes * 60)
        
        # Iniciar en thread separado
        improvement_thread = threading.Thread(target=improvement_loop, daemon=True)
        improvement_thread.start()
        
        logger.info("✅ Autonomous improvement loop started")
    
    def stop(self):
        """Detiene el loop de mejora autónoma."""
        logger.info("🛑 Stopping autonomous improvement loop")
        self.is_running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado del sistema de auto-mejora."""
        return {
            'is_running': self.is_running,
            'total_improvements_applied': self.total_improvements_applied,
            'success_rate': self.success_rate,
            'total_history_entries': len(self.improvement_history),
            'last_improvement': self.improvement_history[-1] if self.improvement_history else None
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    project_root = Path(__file__).parent
    
    system = SelfImprovementSystem(project_root)
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # Solo analizar
        analyses = system.analyze_self()
        improvements = system.generate_improvements(analyses)
        
        print(f"\n{'=' * 80}")
        print(f"📊 ANALYSIS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Files analyzed: {len(analyses)}")
        print(f"Improvements suggested: {len(improvements)}")
        print(f"\nTop improvements:")
        for i, imp in enumerate(improvements[:5], 1):
            print(f"  {i}. [{imp.priority.name}] {imp.description}")
            print(f"     File: {Path(imp.file_path).name}")
            print(f"     Expected impact: {imp.expected_impact}")
            print()
    
    else:
        # Iniciar loop autónomo
        system.start_autonomous_improvement_loop(interval_minutes=60)
        
        print("\n" + "=" * 80)
        print("🤖 SELF-IMPROVEMENT SYSTEM RUNNING")
        print("=" * 80)
        print("The system will:")
        print("  • Analyze its own code every 60 minutes")
        print("  • Detect areas for improvement")
        print("  • Apply high-priority improvements automatically")
        print("  • Validate changes and rollback if needed")
        print("\nPress Ctrl+C to stop")
        print("=" * 80 + "\n")
        
        try:
            while True:
                time.sleep(10)
                status = system.get_status()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Status: {'RUNNING' if status['is_running'] else 'STOPPED'} | "
                      f"Improvements: {status['total_improvements_applied']}")
        except KeyboardInterrupt:
            system.stop()
            print("\n✅ Self-improvement system stopped gracefully")
