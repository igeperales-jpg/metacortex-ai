#!/usr/bin/env python3
"""
🏗️ Code Quality Enforcer - Sistema de Calidad de Código Militar
Sistema de enforcement de estándares de código con métricas militares
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Niveles de calidad militar"""
    UNACCEPTABLE = 0  # < 50
    BASIC = 1  # 50-69
    STANDARD = 2  # 70-79
    PROFESSIONAL = 3  # 80-89
    MILITARY = 4  # 90-94
    ELITE = 5  # 95-100


@dataclass
class QualityMetrics:
    """Métricas de calidad del código"""
    complexity_score: float
    maintainability_score: float
    security_score: float
    documentation_score: float
    type_safety_score: float
    architecture_score: float
    overall_score: float
    level: QualityLevel
    violations: List[Dict[str, Any]]
    recommendations: List[str]


class CodeQualityEnforcer:
    """Sistema de enforcement de calidad de código militar"""
    
    def __init__(self):
        self.logger = logger
        
        # Estándares militares
        self.standards = {
            'max_function_length': 50,
            'max_function_complexity': 10,
            'max_class_methods': 15,
            'max_nesting_depth': 4,
            'min_docstring_coverage': 90,
            'min_type_hint_coverage': 85,
            'max_file_length': 500,
            'max_duplicate_lines': 5,
        }
        
        # Patrones obligatorios
        self.required_patterns = {
            'error_handling': r'try:|except:|raise',
            'logging': r'logger\.(debug|info|warning|error)',
            'type_hints': r':\s*(str|int|float|bool|List|Dict|Optional|Any)',
            'docstrings': r'""".*?"""',
        }
        
        # Patrones prohibidos (anti-patterns)
        self.forbidden_patterns = {
            'bare_except': r'except\s*:',
            'global_vars': r'^[A-Z_]+\s*=',
            'print_debug': r'print\(',
            'hardcoded_credentials': r'(password|api_key|secret)\s*=\s*["\']',
            'eval_exec': r'\b(eval|exec)\s*\(',
            'shell_injection': r'shell\s*=\s*True',
        }
        
        self.logger.info("✅ Code Quality Enforcer inicializado")
    
    def enforce(self, code: str, filepath: str) -> QualityMetrics:
        """
        Enforce military-grade quality standards
        
        Args:
            code: Código fuente a analizar
            filepath: Path del archivo
            
        Returns:
            QualityMetrics con análisis completo
        """
        violations = []
        
        # 1. Complejidad ciclomática
        complexity_score = self._check_complexity(code, violations)
        
        # 2. Mantenibilidad
        maintainability_score = self._check_maintainability(code, violations)
        
        # 3. Seguridad
        security_score = self._check_security(code, violations)
        
        # 4. Documentación
        documentation_score = self._check_documentation(code, violations)
        
        # 5. Type safety
        type_safety_score = self._check_type_safety(code, violations)
        
        # 6. Arquitectura
        architecture_score = self._check_architecture(code, violations)
        
        # Score general (promedio ponderado)
        overall_score = (
            complexity_score * 0.2 +
            maintainability_score * 0.15 +
            security_score * 0.25 +
            documentation_score * 0.15 +
            type_safety_score * 0.15 +
            architecture_score * 0.1
        )
        
        # Determinar nivel
        level = self._determine_level(overall_score)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(violations)
        
        metrics = QualityMetrics(
            complexity_score=complexity_score,
            maintainability_score=maintainability_score,
            security_score=security_score,
            documentation_score=documentation_score,
            type_safety_score=type_safety_score,
            architecture_score=architecture_score,
            overall_score=overall_score,
            level=level,
            violations=violations,
            recommendations=recommendations
        )
        
        self.logger.info(f"🏗️ Quality enforced: {overall_score:.1f}/100 ({level.name})")
        
        return metrics
    
    def _check_complexity(self, code: str, violations: List) -> float:
        """Analizar complejidad ciclomática"""
        score = 100.0
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = self._calculate_complexity(node)
                    
                    if complexity > self.standards['max_function_complexity']:
                        violations.append({
                            'type': 'complexity',
                            'severity': 'high',
                            'message': f"Función '{node.name}' tiene complejidad {complexity} (máx: {self.standards['max_function_complexity']})",
                            'line': node.lineno
                        })
                        score -= 10
                    
                    # Check function length
                    func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    if func_lines > self.standards['max_function_length']:
                        violations.append({
                            'type': 'complexity',
                            'severity': 'medium',
                            'message': f"Función '{node.name}' tiene {func_lines} líneas (máx: {self.standards['max_function_length']})",
                            'line': node.lineno
                        })
                        score -= 5
                
                elif isinstance(node, ast.ClassDef):
                    method_count = sum(1 for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
                    if method_count > self.standards['max_class_methods']:
                        violations.append({
                            'type': 'complexity',
                            'severity': 'medium',
                            'message': f"Clase '{node.name}' tiene {method_count} métodos (máx: {self.standards['max_class_methods']})",
                            'line': node.lineno
                        })
                        score -= 5
        
        except Exception as e:
            logger.error(f"Error en code_quality_enforcer.py: {e}", exc_info=True)
            self.logger.warning(f"Error checking complexity: {e}")
            score = 50.0
        
        return max(0.0, min(100.0, score))
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calcular complejidad ciclomática de una función"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _check_maintainability(self, code: str, violations: List) -> float:
        """Analizar mantenibilidad del código"""
        score = 100.0
        lines = code.split('\n')
        
        # Check duplicated lines
        line_counts = {}
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
        
        for line, count in line_counts.items():
            if count > self.standards['max_duplicate_lines']:
                violations.append({
                    'type': 'maintainability',
                    'severity': 'low',
                    'message': f"Línea duplicada {count} veces: {line[:50]}...",
                    'line': 0
                })
                score -= 2
        
        # Check file length
        if len(lines) > self.standards['max_file_length']:
            violations.append({
                'type': 'maintainability',
                'severity': 'medium',
                'message': f"Archivo muy largo: {len(lines)} líneas (máx: {self.standards['max_file_length']})",
                'line': 0
            })
            score -= 10
        
        # Check magic numbers
        magic_numbers = re.findall(r'\b\d{3,}\b', code)
        if len(magic_numbers) > 5:
            violations.append({
                'type': 'maintainability',
                'severity': 'low',
                'message': f"Múltiples magic numbers encontrados: {len(magic_numbers)}",
                'line': 0
            })
            score -= 5
        
        return max(0.0, min(100.0, score))
    
    def _check_security(self, code: str, violations: List) -> float:
        """Analizar seguridad del código"""
        score = 100.0
        
        for pattern_name, pattern in self.forbidden_patterns.items():
            matches = list(re.finditer(pattern, code, re.MULTILINE))
            if matches:
                severity = 'critical' if pattern_name in ['eval_exec', 'shell_injection', 'hardcoded_credentials'] else 'high'
                
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    violations.append({
                        'type': 'security',
                        'severity': severity,
                        'message': f"Anti-pattern detectado: {pattern_name}",
                        'line': line_num,
                        'code_snippet': match.group(0)
                    })
                    
                    if severity == 'critical':
                        score -= 20
                    else:
                        score -= 10
        
        # Check SQL injection risks
        sql_patterns = re.findall(r'(execute|cursor|query)\s*\(.*%s.*\)', code)
        if sql_patterns:
            violations.append({
                'type': 'security',
                'severity': 'high',
                'message': f"Posible SQL injection risk: {len(sql_patterns)} ocurrencias",
                'line': 0
            })
            score -= 15
        
        return max(0.0, min(100.0, score))
    
    def _check_documentation(self, code: str, violations: List) -> float:
        """Analizar documentación del código"""
        score = 100.0
        
        try:
            tree = ast.parse(code)
            
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            
            total_items = len(functions) + len(classes)
            if total_items == 0:
                return 100.0
            
            documented_items = 0
            
            for node in functions + classes:
                has_docstring = (
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)
                ) if node.body else False
                
                if has_docstring:
                    documented_items += 1
                else:
                    item_type = 'Función' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'Clase'
                    violations.append({
                        'type': 'documentation',
                        'severity': 'medium',
                        'message': f"{item_type} '{node.name}' sin docstring",
                        'line': node.lineno
                    })
            
            coverage = (documented_items / total_items) * 100
            
            if coverage < self.standards['min_docstring_coverage']:
                score = coverage
                violations.append({
                    'type': 'documentation',
                    'severity': 'high',
                    'message': f"Cobertura de docstrings baja: {coverage:.1f}% (mín: {self.standards['min_docstring_coverage']}%)",
                    'line': 0
                })
        
        except Exception as e:
            logger.error(f"Error en code_quality_enforcer.py: {e}", exc_info=True)
            self.logger.warning(f"Error checking documentation: {e}")
            score = 50.0
        
        return max(0.0, min(100.0, score))
    
    def _check_type_safety(self, code: str, violations: List) -> float:
        """Analizar type safety del código"""
        score = 100.0
        
        try:
            tree = ast.parse(code)
            
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            
            if not functions:
                return 100.0
            
            typed_functions = 0
            
            for func in functions:
                has_return_type = func.returns is not None
                has_param_types = all(arg.annotation is not None for arg in func.args.args if arg.arg != 'self')
                
                if has_return_type and has_param_types:
                    typed_functions += 1
                else:
                    missing = []
                    if not has_return_type:
                        missing.append('return type')
                    if not has_param_types:
                        missing.append('parameter types')
                    
                    violations.append({
                        'type': 'type_safety',
                        'severity': 'medium',
                        'message': f"Función '{func.name}' sin {', '.join(missing)}",
                        'line': func.lineno
                    })
            
            coverage = (typed_functions / len(functions)) * 100
            
            if coverage < self.standards['min_type_hint_coverage']:
                score = coverage
                violations.append({
                    'type': 'type_safety',
                    'severity': 'high',
                    'message': f"Cobertura de type hints baja: {coverage:.1f}% (mín: {self.standards['min_type_hint_coverage']}%)",
                    'line': 0
                })
        
        except Exception as e:
            logger.error(f"Error en code_quality_enforcer.py: {e}", exc_info=True)
            self.logger.warning(f"Error checking type safety: {e}")
            score = 50.0
        
        return max(0.0, min(100.0, score))
    
    def _check_architecture(self, code: str, violations: List) -> float:
        """Analizar arquitectura del código"""
        score = 100.0
        
        try:
            tree = ast.parse(code)
            
            # Check SOLID principles violations
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            
            for cls in classes:
                # Single Responsibility: clase con demasiados métodos públicos
                public_methods = [
                    m for m in cls.body 
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                    and not m.name.startswith('_')
                ]
                
                if len(public_methods) > 10:
                    violations.append({
                        'type': 'architecture',
                        'severity': 'medium',
                        'message': f"Clase '{cls.name}' con {len(public_methods)} métodos públicos (posible violación de Single Responsibility)",
                        'line': cls.lineno
                    })
                    score -= 5
                
                # Check God Object anti-pattern
                total_lines = cls.end_lineno - cls.lineno if hasattr(cls, 'end_lineno') else 0
                if total_lines > 300:
                    violations.append({
                        'type': 'architecture',
                        'severity': 'high',
                        'message': f"Clase '{cls.name}' muy grande: {total_lines} líneas (posible God Object)",
                        'line': cls.lineno
                    })
                    score -= 10
            
            # Check excessive nesting
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    max_depth = self._calculate_nesting_depth(node)
                    if max_depth > self.standards['max_nesting_depth']:
                        violations.append({
                            'type': 'architecture',
                            'severity': 'medium',
                            'message': f"Función '{node.name}' con anidamiento excesivo: nivel {max_depth} (máx: {self.standards['max_nesting_depth']})",
                            'line': node.lineno
                        })
                        score -= 5
        
        except Exception as e:
            logger.error(f"Error en code_quality_enforcer.py: {e}", exc_info=True)
            self.logger.warning(f"Error checking architecture: {e}")
            score = 50.0
        
        return max(0.0, min(100.0, score))
    
    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calcular profundidad máxima de anidamiento"""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                child_depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _determine_level(self, score: float) -> QualityLevel:
        """Determinar nivel de calidad según score"""
        if score >= 95:
            return QualityLevel.ELITE
        elif score >= 90:
            return QualityLevel.MILITARY
        elif score >= 80:
            return QualityLevel.PROFESSIONAL
        elif score >= 70:
            return QualityLevel.STANDARD
        elif score >= 50:
            return QualityLevel.BASIC
        else:
            return QualityLevel.UNACCEPTABLE
    
    def _generate_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generar recomendaciones basadas en violaciones"""
        recommendations = []
        
        # Agrupar por tipo
        by_type = {}
        for v in violations:
            vtype = v['type']
            if vtype not in by_type:
                by_type[vtype] = []
            by_type[vtype].append(v)
        
        # Generar recomendaciones
        if 'security' in by_type:
            recommendations.append("🔒 CRÍTICO: Resolver issues de seguridad antes de deployment")
            recommendations.append("   - Eliminar anti-patterns (eval, shell=True, hardcoded credentials)")
            recommendations.append("   - Usar prepared statements para queries SQL")
        
        if 'complexity' in by_type:
            recommendations.append("🔧 Reducir complejidad:")
            recommendations.append("   - Extraer funciones auxiliares de funciones complejas")
            recommendations.append("   - Usar early returns para reducir anidamiento")
            recommendations.append("   - Aplicar principio de Single Responsibility")
        
        if 'documentation' in by_type:
            recommendations.append("📝 Mejorar documentación:")
            recommendations.append("   - Agregar docstrings a todas las funciones y clases")
            recommendations.append("   - Incluir Args, Returns, Raises en docstrings")
            recommendations.append("   - Documentar parámetros complejos")
        
        if 'type_safety' in by_type:
            recommendations.append("🎯 Mejorar type safety:")
            recommendations.append("   - Agregar type hints a todas las funciones")
            recommendations.append("   - Usar Optional para valores que pueden ser None")
            recommendations.append("   - Considerar usar mypy para validación estática")
        
        if 'architecture' in by_type:
            recommendations.append("🏗️ Refactorizar arquitectura:")
            recommendations.append("   - Dividir clases grandes en componentes más pequeños")
            recommendations.append("   - Reducir anidamiento usando guard clauses")
            recommendations.append("   - Aplicar patrones de diseño apropiados")
        
        if 'maintainability' in by_type:
            recommendations.append("🔄 Mejorar mantenibilidad:")
            recommendations.append("   - Extraer magic numbers como constantes")
            recommendations.append("   - Eliminar código duplicado")
            recommendations.append("   - Dividir archivos muy largos en módulos")
        
        return recommendations
    
    def get_military_compliance_report(self, metrics: QualityMetrics) -> str:
        """Generar reporte de compliance militar"""
        report = []
        report.append("=" * 80)
        report.append("🎖️  MILITARY-GRADE CODE QUALITY REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"📊 Overall Score: {metrics.overall_score:.1f}/100")
        report.append(f"🏆 Quality Level: {metrics.level.name}")
        report.append("")
        report.append("📈 Detailed Scores:")
        report.append(f"   Complexity:      {metrics.complexity_score:.1f}/100")
        report.append(f"   Maintainability: {metrics.maintainability_score:.1f}/100")
        report.append(f"   Security:        {metrics.security_score:.1f}/100")
        report.append(f"   Documentation:   {metrics.documentation_score:.1f}/100")
        report.append(f"   Type Safety:     {metrics.type_safety_score:.1f}/100")
        report.append(f"   Architecture:    {metrics.architecture_score:.1f}/100")
        report.append("")
        
        if metrics.violations:
            report.append(f"⚠️  Violations Found: {len(metrics.violations)}")
            report.append("")
            
            # Agrupar por severidad
            critical = [v for v in metrics.violations if v['severity'] == 'critical']
            high = [v for v in metrics.violations if v['severity'] == 'high']
            medium = [v for v in metrics.violations if v['severity'] == 'medium']
            low = [v for v in metrics.violations if v['severity'] == 'low']
            
            if critical:
                report.append(f"🔴 CRITICAL ({len(critical)}):")
                for v in critical[:5]:
                    report.append(f"   Line {v['line']}: {v['message']}")
                report.append("")
            
            if high:
                report.append(f"🟠 HIGH ({len(high)}):")
                for v in high[:5]:
                    report.append(f"   Line {v['line']}: {v['message']}")
                report.append("")
            
            if medium:
                report.append(f"🟡 MEDIUM ({len(medium)}):")
                for v in medium[:3]:
                    report.append(f"   Line {v['line']}: {v['message']}")
                report.append("")
            
            if low:
                report.append(f"⚪ LOW ({len(low)}):")
                for v in low[:3]:
                    report.append(f"   Line {v['line']}: {v['message']}")
                report.append("")
        else:
            report.append("✅ No Violations Found!")
            report.append("")
        
        if metrics.recommendations:
            report.append("💡 Recommendations:")
            for rec in metrics.recommendations:
                report.append(f"   {rec}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# Global singleton
_enforcer_instance: Optional[CodeQualityEnforcer] = None


def get_enforcer() -> CodeQualityEnforcer:
    """Get singleton instance"""
    global _enforcer_instance
    if _enforcer_instance is None:
        _enforcer_instance = CodeQualityEnforcer()
    return _enforcer_instance
