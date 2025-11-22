#!/usr/bin/env python3
"""
📊 Continuous Validation System - Sistema de Validación Continua
Valida código en tiempo real y proporciona feedback instantáneo
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
import ast
import re
import re
import ast
import re
import ast
import ast
import ast

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de validación continua"""
    timestamp: str
    validation_type: str
    passed: bool
    score: float
    issues: List[Dict[str, Any]]
    execution_time: float
    recommendations: List[str]


class ContinuousValidationSystem:
    """Sistema de validación continua del código"""
    
    def __init__(self):
        self.logger = logger
        
        # Validators registrados
        self.validators: Dict[str, Callable] = {}
        
        # Resultados de validación
        self.validation_history: List[ValidationResult] = []
        
        # Métricas
        self.metrics = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'avg_score': 0.0,
            'avg_execution_time': 0.0
        }
        
        # Configuración
        self.config = {
            'min_score_threshold': 85.0,
            'max_issues_allowed': 5,
            'critical_issues_allowed': 0,
            'enable_auto_fix': True,
            'enable_notifications': True
        }
        
        self._register_default_validators()
        
        self.logger.info("✅ Continuous Validation System inicializado")
    
    def _register_default_validators(self):
        """Registrar validadores por defecto"""
        self.register_validator('syntax', self._validate_syntax)
        self.register_validator('security', self._validate_security)
        self.register_validator('performance', self._validate_performance)
        self.register_validator('type_safety', self._validate_type_safety)
        self.register_validator('documentation', self._validate_documentation)
        self.register_validator('architecture', self._validate_architecture)
    
    def register_validator(self, name: str, validator_func: Callable):
        """Registrar nuevo validador"""
        self.validators[name] = validator_func
        self.logger.info(f"✅ Validador registrado: {name}")
    
    async def validate_continuously(
        self,
        code: str,
        filepath: str,
        validators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validar código continuamente con múltiples validadores
        
        Args:
            code: Código fuente
            filepath: Path del archivo
            validators: Lista de validadores a usar (None = todos)
            
        Returns:
            Dict con resultados de todas las validaciones
        """
        start_time = time.time()
        
        # Determinar validadores a usar
        validators_to_use = validators or list(self.validators.keys())
        
        self.logger.info(f"🔍 Validación continua: {len(validators_to_use)} validadores")
        
        # Ejecutar validadores en paralelo
        tasks = []
        for validator_name in validators_to_use:
            if validator_name in self.validators:
                validator_func = self.validators[validator_name]
                tasks.append(self._run_validator(validator_name, validator_func, code, filepath))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        validation_results = {}
        all_issues = []
        total_score = 0.0
        valid_results = 0
        validators_passed = 0
        validators_failed = 0
        
        for validator_name, result in zip(validators_to_use, results):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Error en validador {validator_name}: {result}")
                validators_failed += 1
                continue
            
            validation_results[validator_name] = result
            all_issues.extend(result.issues)
            total_score += result.score
            valid_results += 1
            
            # Contar passed/failed
            if result.passed:
                validators_passed += 1
            else:
                validators_failed += 1
        
        # Score promedio
        avg_score = total_score / max(valid_results, 1)
        
        # Determinar si pasó
        passed = (
            avg_score >= self.config['min_score_threshold'] and
            len(all_issues) <= self.config['max_issues_allowed'] and
            sum(1 for i in all_issues if i.get('severity') == 'critical') <= self.config['critical_issues_allowed']
        )
        
        # Actualizar métricas
        self.metrics['total_validations'] += 1
        if passed:
            self.metrics['passed_validations'] += 1
        else:
            self.metrics['failed_validations'] += 1
        
        self.metrics['avg_score'] = (
            (self.metrics['avg_score'] * (self.metrics['total_validations'] - 1) + avg_score) /
            self.metrics['total_validations']
        )
        
        execution_time = time.time() - start_time
        self.metrics['avg_execution_time'] = (
            (self.metrics['avg_execution_time'] * (self.metrics['total_validations'] - 1) + execution_time) /
            self.metrics['total_validations']
        )
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(all_issues, avg_score)
        
        result = {
            'passed': passed,
            'avg_score': avg_score,
            'execution_time': execution_time,
            'validators_run': valid_results,
            'validators_passed': validators_passed,
            'validators_failed': validators_failed,
            'total_issues': len(all_issues),
            'critical_issues': sum(1 for i in all_issues if i.get('severity') == 'critical'),
            'results': list(validation_results.values()),  # Lista de ValidationResult
            'validation_results': validation_results,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar en historial
        summary_result = ValidationResult(
            timestamp=result['timestamp'],
            validation_type='continuous',
            passed=passed,
            score=avg_score,
            issues=all_issues,
            execution_time=execution_time,
            recommendations=recommendations
        )
        self.validation_history.append(summary_result)
        
        # Log resultado
        status = "✅ PASSED" if passed else "❌ FAILED"
        self.logger.info(f"{status} - Score: {avg_score:.1f}/100 - Issues: {len(all_issues)} - Time: {execution_time:.2f}s")
        
        return result
    
    async def _run_validator(
        self,
        name: str,
        validator_func: Callable,
        code: str,
        filepath: str
    ) -> ValidationResult:
        """Ejecutar un validador individual"""
        start_time = time.time()
        
        try:
            issues = await asyncio.to_thread(validator_func, code, filepath)
            
            # Calcular score basado en issues
            score = 100.0
            for issue in issues:
                severity = issue.get('severity', 'low')
                if severity == 'critical':
                    score -= 20
                elif severity == 'high':
                    score -= 10
                elif severity == 'medium':
                    score -= 5
                else:
                    score -= 2
            
            score = max(0.0, score)
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                timestamp=datetime.now().isoformat(),
                validation_type=name,
                passed=score >= 70.0,
                score=score,
                issues=issues,
                execution_time=execution_time,
                recommendations=[]
            )
        
        except Exception as e:
            logger.error(f"Error en continuous_validation.py: {e}", exc_info=True)
            self.logger.error(f"Error en validador {name}: {e}")
            return ValidationResult(
                timestamp=datetime.now().isoformat(),
                validation_type=name,
                passed=False,
                score=0.0,
                issues=[{
                    'type': 'validator_error',
                    'severity': 'critical',
                    'message': f"Error ejecutando validador: {str(e)}"
                }],
                execution_time=time.time() - start_time,
                recommendations=[]
            )
    
    def _validate_syntax(self, code: str, filepath: str) -> List[Dict[str, Any]]:
        """Validar sintaxis del código"""
        
        issues = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.error(f"Error: {e}", exc_info=True)
            issues.append({
                'type': 'syntax_error',
                'severity': 'critical',
                'message': f"Error de sintaxis: {e.msg}",
                'line': e.lineno,
                'offset': e.offset
            })
        
        return issues
    
    def _validate_security(self, code: str, filepath: str) -> List[Dict[str, Any]]:
        """Validar seguridad del código"""
        
        issues = []
        
        # Patterns de seguridad
        security_patterns = {
            'eval_usage': (r'\beval\s*\(', 'critical', 'Uso de eval() detectado'),
            'exec_usage': (r'\bexec\s*\(', 'critical', 'Uso de exec() detectado'),
            'shell_injection': (r'shell\s*=\s*True', 'critical', 'Posible shell injection'),
            'hardcoded_password': (r'(password|pwd|passwd)\s*=\s*["\'][^"\']+["\']', 'high', 'Contraseña hardcodeada'),
            'sql_injection': (r'execute\s*\([^)]*%s[^)]*\)', 'high', 'Posible SQL injection'),
        }
        
        for pattern_name, (pattern, severity, message) in security_patterns.items():
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    'type': 'security',
                    'severity': severity,
                    'message': message,
                    'line': line_num,
                    'pattern': pattern_name
                })
        
        return issues
    
    def _validate_performance(self, code: str, filepath: str) -> List[Dict[str, Any]]:
        """Validar performance del código"""
        
        issues = []
        
        # Loops anidados
        nested_loops = len(re.findall(r'for\s+\w+\s+in.*?:\s*\n\s*for\s+\w+\s+in', code, re.MULTILINE))
        if nested_loops > 3:
            issues.append({
                'type': 'performance',
                'severity': 'medium',
                'message': f'Múltiples loops anidados detectados: {nested_loops}',
                'line': 0
            })
        
        # String concatenation en loops
        if re.search(r'for\s+\w+.*?:\s*\n\s*\w+\s*\+=\s*["\']', code, re.MULTILINE):
            issues.append({
                'type': 'performance',
                'severity': 'medium',
                'message': 'String concatenation en loop (usar join)',
                'line': 0
            })
        
        # Múltiples imports del mismo módulo
        import_lines = [l for l in code.split('\n') if l.strip().startswith('import ') or l.strip().startswith('from ')]
        import_counts = {}
        for line in import_lines:
            module = line.split()[1].split('.')[0]
            import_counts[module] = import_counts.get(module, 0) + 1
        
        for module, count in import_counts.items():
            if count > 2:
                issues.append({
                    'type': 'performance',
                    'severity': 'low',
                    'message': f'Módulo {module} importado {count} veces',
                    'line': 0
                })
        
        return issues
    
    def _validate_type_safety(self, code: str, filepath: str) -> List[Dict[str, Any]]:
        """Validar type safety del código"""
        
        issues = []
        
        try:
            tree = ast.parse(code)
            
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            
            for func in functions:
                # Check return type
                if func.returns is None:
                    issues.append({
                        'type': 'type_safety',
                        'severity': 'medium',
                        'message': f"Función '{func.name}' sin type hint de retorno",
                        'line': func.lineno
                    })
                
                # Check parameter types
                for arg in func.args.args:
                    if arg.arg != 'self' and arg.annotation is None:
                        issues.append({
                            'type': 'type_safety',
                            'severity': 'medium',
                            'message': f"Parámetro '{arg.arg}' en función '{func.name}' sin type hint",
                            'line': func.lineno
                        })
        
        except Exception as e:
            logger.error(f"Error en continuous_validation.py: {e}", exc_info=True)
            self.logger.warning(f"Error validating type safety: {e}")
        
        return issues
    
    def _validate_documentation(self, code: str, filepath: str) -> List[Dict[str, Any]]:
        """Validar documentación del código"""
        
        issues = []
        
        try:
            tree = ast.parse(code)
            
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            
            for node in functions + classes:
                has_docstring = (
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)
                ) if node.body else False
                
                if not has_docstring:
                    node_type = 'Función' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'Clase'
                    issues.append({
                        'type': 'documentation',
                        'severity': 'low',
                        'message': f"{node_type} '{node.name}' sin docstring",
                        'line': node.lineno
                    })
        
        except Exception as e:
            logger.error(f"Error en continuous_validation.py: {e}", exc_info=True)
            self.logger.warning(f"Error validating documentation: {e}")
        
        return issues
    
    def _validate_architecture(self, code: str, filepath: str) -> List[Dict[str, Any]]:
        """Validar arquitectura del código"""
        
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # Check class size
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            for cls in classes:
                methods = [m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if len(methods) > 15:
                    issues.append({
                        'type': 'architecture',
                        'severity': 'medium',
                        'message': f"Clase '{cls.name}' con {len(methods)} métodos (posible God Object)",
                        'line': cls.lineno
                    })
            
            # Check function complexity
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for func in functions:
                complexity = self._calculate_complexity(func)
                if complexity > 10:
                    issues.append({
                        'type': 'architecture',
                        'severity': 'high',
                        'message': f"Función '{func.name}' muy compleja (complejidad: {complexity})",
                        'line': func.lineno
                    })
        
        except Exception as e:
            logger.error(f"Error en continuous_validation.py: {e}", exc_info=True)
            self.logger.warning(f"Error validating architecture: {e}")
        
        return issues
    
    def _calculate_complexity(self, node) -> int:
        """Calcular complejidad ciclomática"""
        
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _generate_recommendations(self, issues: List[Dict[str, Any]], avg_score: float) -> List[str]:
        """Generar recomendaciones basadas en issues"""
        recommendations = []
        
        # Agrupar por severidad
        critical = [i for i in issues if i.get('severity') == 'critical']
        high = [i for i in issues if i.get('severity') == 'high']
        medium = [i for i in issues if i.get('severity') == 'medium']
        
        if critical:
            recommendations.append(f"🔴 CRÍTICO: Resolver {len(critical)} issues críticos inmediatamente")
        
        if high:
            recommendations.append(f"🟠 Resolver {len(high)} issues de alta prioridad")
        
        if medium:
            recommendations.append(f"🟡 Considerar resolver {len(medium)} issues de prioridad media")
        
        if avg_score < 70:
            recommendations.append("📊 Score muy bajo - considerar refactorización completa")
        elif avg_score < 85:
            recommendations.append("📈 Mejorar calidad del código para alcanzar estándar militar")
        
        # Recomendaciones específicas por tipo
        issue_types = set(i.get('type') for i in issues)
        
        if 'security' in issue_types:
            recommendations.append("🔒 Revisar prácticas de seguridad")
        
        if 'performance' in issue_types:
            recommendations.append("⚡ Optimizar código para mejor performance")
        
        if 'type_safety' in issue_types:
            recommendations.append("🎯 Agregar type hints completos")
        
        if 'documentation' in issue_types:
            recommendations.append("📝 Mejorar documentación del código")
        
        return recommendations
    
    def get_validation_report(self) -> str:
        """Generar reporte de validación continua"""
        report = []
        report.append("=" * 80)
        report.append("📊 CONTINUOUS VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Total Validations: {self.metrics['total_validations']}")
        report.append(f"   ✅ Passed: {self.metrics['passed_validations']}")
        report.append(f"   ❌ Failed: {self.metrics['failed_validations']}")
        report.append(f"   Pass Rate: {(self.metrics['passed_validations'] / max(self.metrics['total_validations'], 1) * 100):.1f}%")
        report.append("")
        report.append(f"📈 Performance Metrics:")
        report.append(f"   Avg Score: {self.metrics['avg_score']:.1f}/100")
        report.append(f"   Avg Execution Time: {self.metrics['avg_execution_time']:.3f}s")
        report.append("")
        report.append(f"⚙️ Configuration:")
        report.append(f"   Min Score Threshold: {self.config['min_score_threshold']}")
        report.append(f"   Max Issues Allowed: {self.config['max_issues_allowed']}")
        report.append(f"   Critical Issues Allowed: {self.config['critical_issues_allowed']}")
        report.append(f"   Auto-Fix: {'Enabled' if self.config['enable_auto_fix'] else 'Disabled'}")
        report.append("")
        
        if self.validation_history:
            report.append("📜 Recent Validations (last 5):")
            for val in self.validation_history[-5:]:
                status = "✅" if val.passed else "❌"
                report.append(f"   {status} {val.timestamp} - Score: {val.score:.1f} - Issues: {len(val.issues)}")
            report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)


# Global singleton
_validation_instance: Optional[ContinuousValidationSystem] = None


def get_validation_system() -> ContinuousValidationSystem:
    """Get singleton instance"""
    global _validation_instance
    if _validation_instance is None:
        _validation_instance = ContinuousValidationSystem()
    return _validation_instance
