#!/usr/bin/env python3
"""
🧠 AI Programming Evolution - Sistema Evolutivo de Programación
Sistema que aprende de éxitos/fracasos y evoluciona estrategias de programación
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProgrammingPattern:
    """Patrón de programación aprendido"""
    pattern_id: str
    name: str
    description: str
    success_rate: float
    times_used: int
    avg_quality_score: float
    code_template: str
    applicable_contexts: List[str]
    learned_from: List[str]  # IDs de proyectos donde se aprendió
    last_used: str
    created_at: str


@dataclass
class EvolutionMetrics:
    """Métricas de evolución del sistema"""
    total_projects: int
    successful_projects: int
    failed_projects: int
    patterns_learned: int
    avg_improvement_rate: float
    best_practices_discovered: int
    anti_patterns_identified: int


class AIProgrammingEvolution:
    """Sistema evolutivo que aprende a programar mejor"""
    
    def __init__(self, db_path: str = "ai_evolution.db"):
        self.db_path = Path(db_path)
        self.logger = logger
        self._init_database()
        
        # Métricas en memoria
        self.session_metrics = {
            'projects_generated': 0,
            'avg_quality': 0.0,
            'patterns_applied': 0,
            'improvements_made': 0
        }
        
        self.logger.info("✅ AI Programming Evolution inicializado")
    
    def _init_database(self):
        """Inicializar base de datos de evolución"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de patrones aprendidos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                success_rate REAL DEFAULT 0.0,
                times_used INTEGER DEFAULT 0,
                avg_quality_score REAL DEFAULT 0.0,
                code_template TEXT,
                applicable_contexts TEXT,
                learned_from TEXT,
                last_used TEXT,
                created_at TEXT
            )
        """)
        
        # Tabla de proyectos generados
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                description TEXT,
                quality_score REAL,
                patterns_used TEXT,
                success BOOLEAN,
                errors TEXT,
                created_at TEXT,
                metadata TEXT
            )
        """)
        
        # Tabla de mejores prácticas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS best_practices (
                practice_id TEXT PRIMARY KEY,
                category TEXT,
                description TEXT,
                impact_score REAL,
                times_applied INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                examples TEXT,
                created_at TEXT
            )
        """)
        
        # Tabla de anti-patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anti_patterns (
                anti_pattern_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                severity TEXT,
                times_detected INTEGER DEFAULT 0,
                fix_strategies TEXT,
                examples TEXT,
                created_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"✅ Base de datos inicializada: {self.db_path}")
    
    def learn_from_project(
        self,
        project_id: str,
        description: str,
        code: Dict[str, str],
        quality_score: float,
        success: bool,
        errors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Aprender de un proyecto generado
        
        Args:
            project_id: ID del proyecto
            description: Descripción del proyecto
            code: Diccionario {filepath: code}
            quality_score: Score de calidad (0-100)
            success: Si fue exitoso
            errors: Lista de errores si hubo
            
        Returns:
            Dict con patrones aprendidos
        """
        self.logger.info(f"🧠 Aprendiendo de proyecto: {project_id}")
        
        # Extraer patrones del código
        patterns_found = self._extract_patterns(code)
        
        # Guardar proyecto
        self._save_project(project_id, description, quality_score, patterns_found, success, errors)
        
        # Aprender patrones exitosos
        if success and quality_score >= 80:
            new_patterns = self._learn_successful_patterns(code, patterns_found, quality_score)
        else:
            new_patterns = []
        
        # Identificar anti-patterns si falló
        if not success or quality_score < 60:
            anti_patterns = self._identify_anti_patterns(code, errors or [])
        else:
            anti_patterns = []
        
        # Actualizar métricas
        self.session_metrics['projects_generated'] += 1
        self.session_metrics['avg_quality'] = (
            (self.session_metrics['avg_quality'] * (self.session_metrics['projects_generated'] - 1) + quality_score) /
            self.session_metrics['projects_generated']
        )
        
        return {
            'patterns_found': len(patterns_found),
            'new_patterns_learned': len(new_patterns),
            'anti_patterns_identified': len(anti_patterns),
            'learning_applied': True
        }
    
    def _extract_patterns(self, code: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extraer patrones del código generado"""
        patterns = []
        
        for filepath, content in code.items():
            # Pattern: Class structure
            if 'class ' in content and '__init__' in content:
                patterns.append({
                    'type': 'class_structure',
                    'file': filepath,
                    'confidence': 0.9
                })
            
            # Pattern: Async/await
            if 'async def' in content or 'await ' in content:
                patterns.append({
                    'type': 'async_programming',
                    'file': filepath,
                    'confidence': 0.95
                })
            
            # Pattern: Context manager
            if 'with ' in content and '__enter__' in content:
                patterns.append({
                    'type': 'context_manager',
                    'file': filepath,
                    'confidence': 0.9
                })
            
            # Pattern: Decorator usage
            if '@' in content and 'def ' in content:
                patterns.append({
                    'type': 'decorator_usage',
                    'file': filepath,
                    'confidence': 0.85
                })
            
            # Pattern: Type hints
            if ' -> ' in content or ': str' in content or ': int' in content:
                patterns.append({
                    'type': 'type_hints',
                    'file': filepath,
                    'confidence': 0.95
                })
            
            # Pattern: Error handling
            if 'try:' in content and 'except' in content:
                patterns.append({
                    'type': 'error_handling',
                    'file': filepath,
                    'confidence': 0.9
                })
            
            # Pattern: Logging
            if 'logger.' in content or 'logging.' in content:
                patterns.append({
                    'type': 'logging',
                    'file': filepath,
                    'confidence': 0.95
                })
            
            # Pattern: Dependency injection
            if '__init__' in content and 'self.' in content:
                patterns.append({
                    'type': 'dependency_injection',
                    'file': filepath,
                    'confidence': 0.7
                })
        
        return patterns
    
    def _learn_successful_patterns(
        self,
        code: Dict[str, str],
        patterns: List[Dict[str, Any]],
        quality_score: float
    ) -> List[str]:
        """Aprender patrones de código exitoso"""
        learned = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pattern in patterns:
            pattern_type = pattern['type']
            pattern_id = f"pattern_{pattern_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Check si ya existe patrón similar
            cursor.execute(
                "SELECT pattern_id, times_used, success_rate, avg_quality_score FROM patterns WHERE name = ?",
                (pattern_type,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Actualizar patrón existente
                old_id, times_used, success_rate, avg_quality = existing
                
                new_times = times_used + 1
                new_success = ((success_rate * times_used) + 1.0) / new_times
                new_avg_quality = ((avg_quality * times_used) + quality_score) / new_times
                
                cursor.execute("""
                    UPDATE patterns 
                    SET times_used = ?, 
                        success_rate = ?,
                        avg_quality_score = ?,
                        last_used = ?
                    WHERE pattern_id = ?
                """, (new_times, new_success, new_avg_quality, datetime.now().isoformat(), old_id))
                
                self.logger.info(f"📊 Patrón actualizado: {pattern_type} (usado {new_times} veces)")
            else:
                # Crear nuevo patrón
                cursor.execute("""
                    INSERT INTO patterns (
                        pattern_id, name, description, success_rate, times_used,
                        avg_quality_score, applicable_contexts, learned_from,
                        last_used, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_id,
                    pattern_type,
                    f"Pattern learned from high-quality code (score: {quality_score:.1f})",
                    1.0,
                    1,
                    quality_score,
                    json.dumps([pattern.get('file', 'unknown')]),
                    json.dumps([]),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                
                learned.append(pattern_id)
                self.logger.info(f"✨ Nuevo patrón aprendido: {pattern_type}")
        
        conn.commit()
        conn.close()
        
        return learned
    
    def _identify_anti_patterns(self, code: Dict[str, str], errors: List[str]) -> List[str]:
        """Identificar anti-patterns en código fallido"""
        anti_patterns = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for filepath, content in code.items():
            # Anti-pattern: Bare except
            if 'except:' in content:
                anti_patterns.append('bare_except')
            
            # Anti-pattern: Global variables
            if content.count('global ') > 2:
                anti_patterns.append('excessive_globals')
            
            # Anti-pattern: Long functions
            functions = content.split('def ')
            for func in functions[1:]:
                if func.count('\n') > 100:
                    anti_patterns.append('long_function')
                    break
            
            # Anti-pattern: No type hints
            if 'def ' in content and ' -> ' not in content:
                anti_patterns.append('missing_type_hints')
        
        # Guardar anti-patterns
        for ap in set(anti_patterns):
            ap_id = f"anti_{ap}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            cursor.execute(
                "SELECT anti_pattern_id, times_detected FROM anti_patterns WHERE name = ?",
                (ap,)
            )
            existing = cursor.fetchone()
            
            if existing:
                old_id, times = existing
                cursor.execute(
                    "UPDATE anti_patterns SET times_detected = ? WHERE anti_pattern_id = ?",
                    (times + 1, old_id)
                )
            else:
                cursor.execute("""
                    INSERT INTO anti_patterns (
                        anti_pattern_id, name, description, severity,
                        times_detected, examples, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    ap_id,
                    ap,
                    f"Anti-pattern detected in failed code",
                    'high',
                    1,
                    json.dumps([filepath]),
                    datetime.now().isoformat()
                ))
        
        conn.commit()
        conn.close()
        
        return anti_patterns
    
    def _save_project(
        self,
        project_id: str,
        description: str,
        quality_score: float,
        patterns: List[Dict[str, Any]],
        success: bool,
        errors: Optional[List[str]]
    ):
        """Guardar proyecto en base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO projects (
                project_id, description, quality_score, patterns_used,
                success, errors, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_id,
            description,
            quality_score,
            json.dumps([p['type'] for p in patterns]),
            success,
            json.dumps(errors or []),
            datetime.now().isoformat(),
            json.dumps({
                'num_files': len(patterns),
                'avg_confidence': sum(p.get('confidence', 0) for p in patterns) / max(len(patterns), 1)
            })
        ))
        
        conn.commit()
        conn.close()
    
    def get_best_patterns_for_context(self, context: str, limit: int = 5) -> List[ProgrammingPattern]:
        """
        Obtener mejores patrones para un contexto
        
        Args:
            context: Contexto (ej: "web_api", "ml_model", "cli_tool")
            limit: Número máximo de patrones
            
        Returns:
            Lista de patrones recomendados
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Buscar patrones con alto success rate
        cursor.execute("""
            SELECT * FROM patterns
            WHERE success_rate >= 0.8
            ORDER BY success_rate DESC, avg_quality_score DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        patterns = []
        for row in rows:
            patterns.append(ProgrammingPattern(
                pattern_id=row[0],
                name=row[1],
                description=row[2] or "",
                success_rate=row[3],
                times_used=row[4],
                avg_quality_score=row[5],
                code_template=row[6] or "",
                applicable_contexts=json.loads(row[7] or "[]"),
                learned_from=json.loads(row[8] or "[]"),
                last_used=row[9] or "",
                created_at=row[10] or ""
            ))
        
        self.logger.info(f"📚 Encontrados {len(patterns)} patrones para contexto: {context}")
        return patterns
    
    def get_evolution_metrics(self) -> EvolutionMetrics:
        """Obtener métricas de evolución del sistema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total projects
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) FROM projects")
        total, successful = cursor.fetchone()
        failed = total - (successful or 0)
        
        # Patterns learned
        cursor.execute("SELECT COUNT(*) FROM patterns")
        patterns_count = cursor.fetchone()[0]
        
        # Best practices
        cursor.execute("SELECT COUNT(*) FROM best_practices")
        best_practices = cursor.fetchone()[0]
        
        # Anti-patterns
        cursor.execute("SELECT COUNT(*) FROM anti_patterns")
        anti_patterns = cursor.fetchone()[0]
        
        # Avg improvement (comparar proyectos recientes vs antiguos)
        cursor.execute("""
            SELECT AVG(quality_score) FROM projects
            ORDER BY created_at DESC
            LIMIT 10
        """)
        recent_avg = cursor.fetchone()[0] or 0.0
        
        cursor.execute("""
            SELECT AVG(quality_score) FROM projects
            ORDER BY created_at ASC
            LIMIT 10
        """)
        old_avg = cursor.fetchone()[0] or 0.0
        
        improvement = ((recent_avg - old_avg) / max(old_avg, 1)) * 100 if old_avg > 0 else 0.0
        
        conn.close()
        
        return EvolutionMetrics(
            total_projects=total or 0,
            successful_projects=successful or 0,
            failed_projects=failed or 0,
            patterns_learned=patterns_count,
            avg_improvement_rate=improvement,
            best_practices_discovered=best_practices,
            anti_patterns_identified=anti_patterns
        )
    
    def generate_evolution_report(self) -> str:
        """Generar reporte de evolución del sistema"""
        metrics = self.get_evolution_metrics()
        
        report = []
        report.append("=" * 80)
        report.append("🧠 AI PROGRAMMING EVOLUTION REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"📊 Total Projects: {metrics.total_projects}")
        report.append(f"   ✅ Successful: {metrics.successful_projects}")
        report.append(f"   ❌ Failed: {metrics.failed_projects}")
        report.append(f"   Success Rate: {(metrics.successful_projects / max(metrics.total_projects, 1) * 100):.1f}%")
        report.append("")
        report.append(f"🎓 Learning Progress:")
        report.append(f"   Patterns Learned: {metrics.patterns_learned}")
        report.append(f"   Best Practices: {metrics.best_practices_discovered}")
        report.append(f"   Anti-Patterns Identified: {metrics.anti_patterns_identified}")
        report.append(f"   Avg Improvement Rate: {metrics.avg_improvement_rate:.1f}%")
        report.append("")
        
        # Session metrics
        report.append("📈 Session Metrics:")
        report.append(f"   Projects Generated: {self.session_metrics['projects_generated']}")
        report.append(f"   Avg Quality: {self.session_metrics['avg_quality']:.1f}/100")
        report.append(f"   Patterns Applied: {self.session_metrics['patterns_applied']}")
        report.append(f"   Improvements Made: {self.session_metrics['improvements_made']}")
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


# Global singleton
_evolution_instance: Optional[AIProgrammingEvolution] = None


def get_evolution_system() -> AIProgrammingEvolution:
    """Get singleton instance"""
    global _evolution_instance
    if _evolution_instance is None:
        _evolution_instance = AIProgrammingEvolution()
    return _evolution_instance
