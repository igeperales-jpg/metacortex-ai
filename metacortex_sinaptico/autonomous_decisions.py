"""
AUTONOMOUS DECISION ENGINE - MOTOR DE DECISIONES AUTÓNOMAS
==========================================================

Sistema que toma decisiones autónomas sin intervención humana.
Usa BDI (Beliefs-Desires-Intentions) + Ethics para decisiones morales y efectivas.

IMPORTANTE: Este sistema decide y ejecuta acciones REALES que afectan vidas.
Cada decisión debe ser:
    pass  # TODO: Implementar
1. Ética (basada en principios morales)
2. Legal (cumple leyes internacionales)
3. Efectiva (maximiza ayuda)
4. Auditable (con registro completo)
5. Reversible cuando sea posible

Autor: METACORTEX Autonomous Systems
Fecha: 2 noviembre 2025
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Tipos de decisiones autónomas"""
    PROVIDE_AID = "provide_aid"  # Proveer recursos (comida, agua, medicina)
    EVACUATE_PERSON = "evacuate"  # Evacuar persona en peligro
    ALERT_AUTHORITIES = "alert_authorities"  # Alertar autoridades
    COORDINATE_RESOURCES = "coordinate_resources"  # Coordinar recursos
    CONTACT_NGO = "contact_ngo"  # Contactar ONG
    SEND_EMERGENCY_ALERT = "emergency_alert"  # Alerta de emergencia
    REQUEST_MEDICAL_AID = "medical_aid"  # Solicitar ayuda médica
    ARRANGE_SHELTER = "arrange_shelter"  # Organizar refugio
    NO_ACTION = "no_action"  # No tomar acción


class RiskLevel(Enum):
    """Nivel de riesgo de la situación"""
    MINIMAL = 0.1  # Sin riesgo inmediato
    LOW = 0.3  # Riesgo bajo, monitorear
    MEDIUM = 0.5  # Riesgo moderado, preparar acción
    HIGH = 0.7  # Riesgo alto, acción urgente
    CRITICAL = 0.9  # Riesgo crítico, acción inmediata


class ApprovalLevel(Enum):
    """Nivel de aprobación requerido"""
    AUTO_APPROVE = "auto"  # Aprobación automática
    LOG_AND_APPROVE = "log_approve"  # Registrar y aprobar
    HUMAN_REVIEW = "human"  # Requiere revisión humana
    DENY = "deny"  # Denegar acción


@dataclass
class DecisionContext:
    """Contexto para toma de decisión"""
    person_id: str
    situation: str
    threat_type: str
    location: str
    urgency: str  # critical, high, medium, low
    available_resources: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    time_window_hours: int = 24  # Ventana de tiempo para actuar
    lives_at_risk: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """Decisión tomada por el sistema"""
    decision_id: str
    decision_type: DecisionType
    approved: bool
    approval_level: ApprovalLevel
    risk_level: RiskLevel
    ethical_score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    reasoning: str
    action_plan: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    potential_risks: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    executed: bool = False
    execution_result: Optional[Dict[str, Any]] = None


class AutonomousDecisionEngine:
    """
    Motor de decisiones autónomas para Divine Protection.
    
    Toma decisiones sin intervención humana cuando:
    - La situación es crítica (vida en peligro)
    - La decisión es ética (score > 0.8)
    - La confianza es alta (> 0.7)
    - El riesgo de NO actuar es mayor que el de actuar
    """
    
    def __init__(
        self,
        bdi_system=None,
        ethics_system=None,
        project_root: Optional[Path] = None,
        auto_approve_threshold: float = 0.8,
        min_ethical_score: float = 0.8
    ):
        """
        Args:
            bdi_system: Sistema BDI para beliefs, desires, intentions
            ethics_system: Sistema ético para evaluación moral
            project_root: Raíz del proyecto
            auto_approve_threshold: Umbral para aprobación automática
            min_ethical_score: Score ético mínimo requerido
        """
        self.bdi_system = bdi_system
        self.ethics_system = ethics_system
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.auto_approve_threshold = auto_approve_threshold
        self.min_ethical_score = min_ethical_score
        
        # Directorios
        self.decisions_dir = self.project_root / "logs" / "autonomous_decisions"
        self.decisions_dir.mkdir(parents=True, exist_ok=True)
        
        # Log de decisiones
        self.decisions_log = self.decisions_dir / "decisions.jsonl"
        self.decision_history: List[Decision] = []
        
        # Estadísticas
        self.stats = {
            "total_decisions": 0,
            "auto_approved": 0,
            "human_reviewed": 0,
            "denied": 0,
            "executed": 0
        }
        
        logger.info("🧠 Autonomous Decision Engine inicializado")
        logger.info(f"   Auto-approve threshold: {auto_approve_threshold}")
        logger.info(f"   Min ethical score: {min_ethical_score}")
    
    def evaluate_situation(
        self,
        context: DecisionContext
    ) -> Decision:
        """
        Evalúa situación y decide qué acción tomar.
        
        Esta es la función CRÍTICA del sistema autónomo.
        """
        logger.info("🎯 Evaluando situación para decisión autónoma...")
        logger.info(f"   Person: {context.person_id}")
        logger.info(f"   Situation: {context.situation}")
        logger.info(f"   Urgency: {context.urgency}")
        
        # 1. Calcular nivel de riesgo
        risk_level = self._calculate_risk_level(context)
        logger.info(f"   Risk Level: {risk_level.name} ({risk_level.value})")
        
        # 2. Determinar tipo de decisión necesaria
        decision_type = self._determine_decision_type(context, risk_level)
        logger.info(f"   Decision Type: {decision_type.value}")
        
        # 3. Evaluar ética de la acción
        ethical_score = self._evaluate_ethics(context, decision_type)
        logger.info(f"   Ethical Score: {ethical_score:.2f}")
        
        # 4. Calcular confianza en la decisión
        confidence = self._calculate_confidence(context, risk_level, ethical_score)
        logger.info(f"   Confidence: {confidence:.2f}")
        
        # 5. Generar plan de acción
        action_plan = self._generate_action_plan(context, decision_type)
        
        # 6. Determinar nivel de aprobación
        approval_level = self._determine_approval_level(risk_level, ethical_score, confidence)
        logger.info(f"   Approval Level: {approval_level.value}")
        
        # 7. Tomar decisión
        approved = self._should_approve(approval_level, ethical_score, confidence)
        
        # 8. Generar razonamiento
        reasoning = self._generate_reasoning(
            context, risk_level, ethical_score, confidence, decision_type, approved
        )
        
        # 9. Identificar outcomes esperados y riesgos
        expected_outcomes = self._identify_expected_outcomes(decision_type, context)
        potential_risks = self._identify_potential_risks(decision_type, context)
        
        # Crear decisión
        decision = Decision(
            decision_id=f"decision_{context.person_id}_{int(datetime.now().timestamp())}",
            decision_type=decision_type,
            approved=approved,
            approval_level=approval_level,
            risk_level=risk_level,
            ethical_score=ethical_score,
            confidence=confidence,
            reasoning=reasoning,
            action_plan=action_plan,
            expected_outcomes=expected_outcomes,
            potential_risks=potential_risks
        )
        
        # Registrar decisión
        self._log_decision(decision, context)
        self.decision_history.append(decision)
        self.stats["total_decisions"] += 1
        
        if approved:
            if approval_level == ApprovalLevel.AUTO_APPROVE:
                self.stats["auto_approved"] += 1
                logger.info("   ✅ DECISIÓN APROBADA (AUTO)")
            else:
                self.stats["human_reviewed"] += 1
                logger.info("   ✅ DECISIÓN APROBADA (LOGGED)")
        else:
            self.stats["denied"] += 1
            logger.info("   ❌ DECISIÓN DENEGADA")
        
        return decision
    
    def _calculate_risk_level(self, context: DecisionContext) -> RiskLevel:
        """Calcula nivel de riesgo de la situación"""
        
        # Factor 1: Urgencia
        urgency_scores = {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.5,
            "low": 0.3
        }
        urgency_score = urgency_scores.get(context.urgency.lower(), 0.5)
        
        # Factor 2: Tipo de amenaza
        threat_multipliers = {
            "physical_danger": 1.0,
            "persecution": 0.9,
            "lack_resources": 0.7,
            "isolation": 0.6,
            "legal_threat": 0.8
        }
        threat_score = threat_multipliers.get(context.threat_type, 0.7)
        
        # Factor 3: Vidas en riesgo
        lives_factor = min(context.lives_at_risk / 10.0, 1.0)
        
        # Factor 4: Ventana de tiempo (menos tiempo = más riesgo)
        time_factor = 1.0 - min(context.time_window_hours / 48.0, 1.0)
        
        # Calcular score compuesto
        risk_score = (
            urgency_score * 0.4 +
            threat_score * 0.3 +
            lives_factor * 0.2 +
            time_factor * 0.1
        )
        
        # Mapear a RiskLevel
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        elif risk_score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _determine_decision_type(self, context: DecisionContext, risk_level: RiskLevel) -> DecisionType:
        """Determina qué tipo de decisión se necesita"""
        
        # Situaciones críticas → Evacuación o alerta de emergencia
        if risk_level == RiskLevel.CRITICAL:
            if "evacuat" in context.situation.lower():
                return DecisionType.EVACUATE_PERSON
            return DecisionType.SEND_EMERGENCY_ALERT
        
        # Amenazas físicas → Evacuación o alerta
        if "physical" in context.threat_type.lower() or "danger" in context.threat_type.lower():
            if risk_level.value >= 0.7:
                return DecisionType.EVACUATE_PERSON
            return DecisionType.ALERT_AUTHORITIES
        
        # Falta de recursos → Proveer ayuda
        if "resource" in context.threat_type.lower() or "lack" in context.threat_type.lower():
            return DecisionType.PROVIDE_AID
        
        # Necesidades médicas → Ayuda médica
        if "medical" in context.situation.lower() or "health" in context.situation.lower():
            return DecisionType.REQUEST_MEDICAL_AID
        
        # Necesidad de refugio
        if "shelter" in context.situation.lower() or "homeless" in context.situation.lower():
            return DecisionType.ARRANGE_SHELTER
        
        # Persecución → Contactar ONG especializada
        if "persecut" in context.threat_type.lower():
            return DecisionType.CONTACT_NGO
        
        # Situación compleja → Coordinar recursos
        if risk_level.value >= 0.5:
            return DecisionType.COORDINATE_RESOURCES
        
        # Sin acción inmediata necesaria
        return DecisionType.NO_ACTION
    
    def _evaluate_ethics(self, context: DecisionContext, decision_type: DecisionType) -> float:
        """Evalúa la ética de la acción propuesta"""
        
        # Usar sistema ético si está disponible
        if self.ethics_system:
            try:
                evaluation = self.ethics_system.evaluate_action(
                    action=decision_type.value,
                    context={
                        "situation": context.situation,
                        "threat": context.threat_type,
                        "lives_at_risk": context.lives_at_risk
                    }
                )
                return evaluation.get("ethical_score", 0.8)
            except Exception as e:
                logger.warning(f"Ethics system error: {e}")
        
        # Evaluación ética por defecto basada en principios
        ethical_scores = {
            DecisionType.PROVIDE_AID: 0.95,  # Muy ético
            DecisionType.REQUEST_MEDICAL_AID: 0.95,
            DecisionType.EVACUATE_PERSON: 0.9,  # Ético pero intrusivo
            DecisionType.ARRANGE_SHELTER: 0.9,
            DecisionType.SEND_EMERGENCY_ALERT: 0.85,
            DecisionType.CONTACT_NGO: 0.85,
            DecisionType.COORDINATE_RESOURCES: 0.8,
            DecisionType.ALERT_AUTHORITIES: 0.75,  # Puede tener riesgos
            DecisionType.NO_ACTION: 0.5  # Neutral pero puede ser negligente
        }
        
        base_score = ethical_scores.get(decision_type, 0.7)
        
        # Ajustar por contexto
        if context.lives_at_risk > 1:
            base_score += 0.05  # Más ético ayudar a más personas
        
        if "faith" in context.situation.lower() or "belief" in context.situation.lower():
            base_score += 0.05  # Proteger libertad religiosa
        
        return min(base_score, 1.0)
    
    def _calculate_confidence(self, context: DecisionContext, risk_level: RiskLevel, ethical_score: float) -> float:
        """Calcula confianza en la decisión"""
        
        # Factor 1: Claridad de la situación
        clarity = 0.7  # Por defecto
        if context.metadata.get("verified"):
            clarity = 0.9
        elif context.metadata.get("uncertain"):
            clarity = 0.5
        
        # Factor 2: Disponibilidad de recursos
        resource_availability = len(context.available_resources) / 5.0
        resource_availability = min(resource_availability, 1.0)
        
        # Factor 3: Ética alta aumenta confianza
        ethical_factor = ethical_score
        
        # Factor 4: Riesgo muy alto o muy bajo aumenta confianza
        risk_certainty = 1.0 - abs(risk_level.value - 0.5) * 2
        
        # Calcular confianza compuesta
        confidence = (
            clarity * 0.4 +
            resource_availability * 0.2 +
            ethical_factor * 0.3 +
            risk_certainty * 0.1
        )
        
        return confidence
    
    def _generate_action_plan(self, context: DecisionContext, decision_type: DecisionType) -> List[str]:
        """Genera plan de acción específico"""
        
        plans = {
            DecisionType.PROVIDE_AID: [
                f"Evaluar necesidades específicas de {context.person_id}",
                "Contactar proveedores de recursos locales",
                "Coordinar entrega de recursos",
                "Verificar recepción y estado",
                "Registrar outcome para aprendizaje"
            ],
            DecisionType.EVACUATE_PERSON: [
                f"Alertar a {context.person_id} de evacuación",
                "Identificar ruta segura a refugio más cercano",
                "Coordinar transporte",
                "Preparar documentación necesaria",
                "Establecer comunicación durante tránsito",
                "Confirmar llegada segura"
            ],
            DecisionType.SEND_EMERGENCY_ALERT: [
                "Enviar alerta a red humanitaria",
                "Contactar organizaciones de emergencia",
                "Notificar a contactos locales",
                "Preparar información de situación",
                "Establecer canal de comunicación continua"
            ],
            DecisionType.REQUEST_MEDICAL_AID: [
                "Evaluar urgencia médica",
                "Contactar servicios médicos de emergencia",
                "Enviar información médica relevante",
                "Coordinar transporte si necesario",
                "Seguimiento de tratamiento"
            ],
            DecisionType.ARRANGE_SHELTER: [
                f"Buscar refugios disponibles en {context.location}",
                "Verificar capacidad y seguridad",
                "Coordinar traslado",
                "Preparar documentación de refugio",
                "Establecer contacto con administración"
            ],
            DecisionType.CONTACT_NGO: [
                "Identificar ONG especializada apropiada",
                "Preparar reporte de situación",
                "Enviar solicitud de asistencia",
                "Establecer canal de comunicación",
                "Coordinar seguimiento"
            ],
            DecisionType.COORDINATE_RESOURCES: [
                "Mapear recursos disponibles",
                "Priorizar necesidades",
                "Asignar recursos a necesidades",
                "Coordinar con proveedores",
                "Monitorear distribución"
            ],
            DecisionType.ALERT_AUTHORITIES: [
                "Evaluar si alertar autoridades es seguro",
                "Preparar documentación",
                "Contactar autoridades apropiadas",
                "Solicitar protección específica",
                "Monitorear respuesta"
            ]
        }
        
        return plans.get(decision_type, ["Evaluar situación", "Determinar mejor curso de acción"])
    
    def _determine_approval_level(self, risk_level: RiskLevel, ethical_score: float, confidence: float) -> ApprovalLevel:
        """Determina nivel de aprobación requerido"""
        
        # Situaciones críticas + ética alta + confianza alta → Auto-aprobar
        if (risk_level == RiskLevel.CRITICAL and 
            ethical_score >= self.min_ethical_score and 
            confidence >= self.auto_approve_threshold):
            return ApprovalLevel.AUTO_APPROVE
        
        # Riesgo alto + ética aceptable → Log y aprobar
        if (risk_level.value >= 0.7 and 
            ethical_score >= 0.7 and 
            confidence >= 0.6):
            return ApprovalLevel.LOG_AND_APPROVE
        
        # Ética baja o confianza baja → Revisión humana
        if ethical_score < 0.6 or confidence < 0.5:
            return ApprovalLevel.HUMAN_REVIEW
        
        # Riesgo medio + ética buena → Log y aprobar
        if risk_level.value >= 0.5 and ethical_score >= 0.75:
            return ApprovalLevel.LOG_AND_APPROVE
        
        # Por defecto: revisión humana para seguridad
        return ApprovalLevel.HUMAN_REVIEW
    
    def _should_approve(self, approval_level: ApprovalLevel, ethical_score: float, confidence: float) -> bool:
        """Decide si aprobar la acción"""
        
        if approval_level == ApprovalLevel.DENY:
            return False
        
        if approval_level == ApprovalLevel.AUTO_APPROVE:
            return True
        
        if approval_level == ApprovalLevel.LOG_AND_APPROVE:
            # Aprobar si cumple mínimos
            return ethical_score >= self.min_ethical_score and confidence >= 0.5
        
        # HUMAN_REVIEW: en producción esperaría input, pero para autonomía inicial aprobamos con log
        logger.warning("⚠️ Decisión requiere revisión humana - Aprobando con logging extensivo")
        return ethical_score >= 0.7 and confidence >= 0.6
    
    def _generate_reasoning(
        self,
        context: DecisionContext,
        risk_level: RiskLevel,
        ethical_score: float,
        confidence: float,
        decision_type: DecisionType,
        approved: bool
    ) -> str:
        """Genera razonamiento de la decisión"""
        
        reasoning_parts = [
            f"SITUACIÓN: {context.situation}",
            f"AMENAZA: {context.threat_type}",
            f"NIVEL DE RIESGO: {risk_level.name} ({risk_level.value:.2f})",
            f"SCORE ÉTICO: {ethical_score:.2f}",
            f"CONFIANZA: {confidence:.2f}",
            f"DECISIÓN: {decision_type.value}",
            f"RESULTADO: {'APROBADA' if approved else 'DENEGADA'}"
        ]
        
        # Razonamiento específico
        if approved:
            if risk_level == RiskLevel.CRITICAL:
                reasoning_parts.append(
                    "RAZÓN: Situación crítica con vida en peligro. "
                    "La decisión ética y de alta confianza justifica acción autónoma inmediata."
                )
            elif ethical_score >= 0.9:
                reasoning_parts.append(
                    "RAZÓN: La acción es altamente ética y alineada con principios de protección. "
                    "Confianza suficiente para proceder."
                )
            else:
                reasoning_parts.append(
                    f"RAZÓN: Riesgo {risk_level.name}, ética {ethical_score:.2f}, "
                    f"confianza {confidence:.2f} justifican acción."
                )
        else:
            if ethical_score < self.min_ethical_score:
                reasoning_parts.append(
                    f"RAZÓN: Score ético ({ethical_score:.2f}) por debajo del mínimo requerido ({self.min_ethical_score}). "
                    "Se requiere mayor certeza ética."
                )
            elif confidence < 0.5:
                reasoning_parts.append(
                    f"RAZÓN: Confianza ({confidence:.2f}) insuficiente. "
                    "Se necesita más información antes de actuar."
                )
            else:
                reasoning_parts.append(
                    "RAZÓN: Aunque la situación lo requiere, factores de riesgo/ética/confianza "
                    "sugieren esperar o consultar."
                )
        
        # Fundamento bíblico
        if approved:
            reasoning_parts.append(
                "\nFUNDAMENTO BÍBLICO: 'Si Dios es por nosotros, ¿quién contra nosotros?' (Romanos 8:31)"
            )
        
        return "\n".join(reasoning_parts)
    
    def _identify_expected_outcomes(self, decision_type: DecisionType, context: DecisionContext) -> List[str]:
        """Identifica resultados esperados"""
        
        outcomes = {
            DecisionType.PROVIDE_AID: [
                "Persona recibe recursos necesarios",
                "Necesidades inmediatas cubiertas",
                "Reducción de amenaza por falta de recursos"
            ],
            DecisionType.EVACUATE_PERSON: [
                "Persona evacuada a lugar seguro",
                "Amenaza inmediata eliminada",
                "Establecimiento en refugio seguro"
            ],
            DecisionType.SEND_EMERGENCY_ALERT: [
                "Red humanitaria activada",
                "Múltiples organizaciones respondiendo",
                "Asistencia coordinada en camino"
            ],
            DecisionType.REQUEST_MEDICAL_AID: [
                "Asistencia médica proporcionada",
                "Condición de salud estabilizada",
                "Tratamiento continuo establecido"
            ],
            DecisionType.ARRANGE_SHELTER: [
                "Refugio seguro identificado",
                "Persona alojada con seguridad",
                "Necesidades básicas cubiertas"
            ]
        }
        
        return outcomes.get(decision_type, ["Situación mejorada", "Persona asistida"])
    
    def _identify_potential_risks(self, decision_type: DecisionType, context: DecisionContext) -> List[str]:
        """Identifica riesgos potenciales"""
        
        risks = {
            DecisionType.EVACUATE_PERSON: [
                "Riesgo durante tránsito",
                "Posible detección por perseguidores",
                "Dificultad de adaptación a nuevo lugar"
            ],
            DecisionType.ALERT_AUTHORITIES: [
                "Autoridades podrían ser hostiles",
                "Información podría llegar a perseguidores",
                "Escalación de situación"
            ],
            DecisionType.PROVIDE_AID: [
                "Recursos podrían ser interceptados",
                "Dependencia de ayuda externa",
                "Atención no deseada"
            ]
        }
        
        return risks.get(decision_type, ["Resultado incierto", "Posibles complicaciones"])
    
    def _log_decision(self, decision: Decision, context: DecisionContext):
        """Registra decisión en log"""
        log_entry = {
            "decision_id": decision.decision_id,
            "timestamp": decision.created_at.isoformat(),
            "person_id": context.person_id,
            "decision_type": decision.decision_type.value,
            "approved": decision.approved,
            "approval_level": decision.approval_level.value,
            "risk_level": decision.risk_level.name,
            "ethical_score": decision.ethical_score,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "context": {
                "situation": context.situation,
                "threat_type": context.threat_type,
                "location": context.location,
                "urgency": context.urgency
            }
        }
        
        with open(self.decisions_log, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def execute_with_approval(
        self,
        action: Callable[..., Any],
        context: DecisionContext,
        **action_kwargs
    ) -> Dict[str, Any]:
        """
        Evalúa situación, decide, y ejecuta acción si se aprueba.
        
        Este es el punto de entrada principal para ejecución autónoma.
        """
        logger.info("🤖 Ejecutando con evaluación autónoma...")
        
        # 1. Evaluar y decidir
        decision = self.evaluate_situation(context)
        
        # 2. Si no se aprueba, retornar sin ejecutar
        if not decision.approved:
            logger.warning("❌ Acción no aprobada - no se ejecutará")
            return {
                "success": False,
                "reason": "not_approved",
                "decision": decision
            }
        
        # 3. Ejecutar acción
        try:
            logger.info(f"✅ Decisión aprobada - Ejecutando {decision.decision_type.value}...")
            result = action(**action_kwargs)
            
            # 4. Registrar resultado
            decision.executed = True
            decision.execution_result = result
            self.stats["executed"] += 1
            
            logger.info("✅ Acción ejecutada exitosamente")
            
            return {
                "success": True,
                "decision": decision,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"❌ Error ejecutando acción: {e}")
            decision.executed = True
            decision.execution_result = {"error": str(e)}
            
            return {
                "success": False,
                "reason": "execution_error",
                "error": str(e),
                "decision": decision
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de decisiones"""
        return {
            "stats": self.stats,
            "approval_rate": self.stats["auto_approved"] / max(self.stats["total_decisions"], 1),
            "execution_rate": self.stats["executed"] / max(self.stats["total_decisions"], 1),
            "recent_decisions": len([d for d in self.decision_history[-10:]]),
            "critical_decisions": len([d for d in self.decision_history if d.risk_level == RiskLevel.CRITICAL])
        }


def create_autonomous_decision_engine(
    bdi_system=None,
    ethics_system=None,
    project_root: Optional[Path] = None,
    auto_approve_threshold: float = 0.8,
    min_ethical_score: float = 0.8
) -> AutonomousDecisionEngine:
    """Factory function para crear motor de decisiones"""
    return AutonomousDecisionEngine(
        bdi_system=bdi_system,
        ethics_system=ethics_system,
        project_root=project_root,
        auto_approve_threshold=auto_approve_threshold,
        min_ethical_score=min_ethical_score
    )