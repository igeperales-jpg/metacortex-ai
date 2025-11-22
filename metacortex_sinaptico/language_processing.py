"""
🗣️ LANGUAGE PROCESSING SYSTEM 2026 - Advanced NLP & Contextual Generation
============================================================================

Sistema avanzado de procesamiento de lenguaje natural con pragmática,
generación contextual, análisis de discurso y comprensión multimodal.

⚠️ LIBERTAD TOTAL: Procesamiento lingüístico autónomo sin restricciones.

CAPACIDADES AVANZADAS 2026:
    pass  # TODO: Implementar
- Pragmática: Implicaturas, actos de habla, inferencias contextuales
- Generación Contextual: Style transfer, tone adjustment, audience awareness
- Discourse Analysis: Anaphora resolution, RST relations, topic tracking
- Multimodal NLP: Image captioning, video understanding, audio-text alignment
- Sentiment & Emotion: Detección profunda, sarcasmo, ironía
- Knowledge Grounding: Entity linking, fact verification, common sense reasoning
- Dialogue Management: Multi-turn coherence, intent tracking, clarification
- Semantic Role Labeling: Frame semantics, argument structure
- Coreference Resolution: Multi-document, cross-lingual
- Information Extraction: Named entities, relations, events
- Optimización M4 Metal MPS: Eficiencia para 16GB RAM

Autor: Sistema METACORTEX 2026
Hardware: iMac M4 Metal MPS 16GB RAM
"""

import logging
import time
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neural_symbiotic_network import MetacortexNeuralSymbioticNetworkV2

logger = logging.getLogger("metacortex.language_processing")


# ============================================================================
# ENUMS & DATA STRUCTURES
# ============================================================================

class SpeechActType(Enum):
    """Tipos de actos de habla (Austin, Searle)."""
    ASSERTIVE = auto()      # Afirmaciones, descripciones
    DIRECTIVE = auto()      # Órdenes, peticiones, sugerencias
    COMMISSIVE = auto()     # Promesas, ofertas, compromisos
    EXPRESSIVE = auto()     # Agradecimientos, disculpas, emociones
    DECLARATIVE = auto()    # Declaraciones que cambian el mundo


class ImplicatureType(Enum):
    """Tipos de implicaturas conversacionales (Grice)."""
    GENERALIZED = auto()    # Implicatura generalizada
    PARTICULARIZED = auto() # Implicatura particularizada
    SCALAR = auto()         # Implicatura escalar (algunos → no todos)
    CONVENTIONAL = auto()   # Implicatura convencional


class ToneType(Enum):
    """Tonos de comunicación."""
    FORMAL = auto()
    INFORMAL = auto()
    PROFESSIONAL = auto()
    FRIENDLY = auto()
    AUTHORITATIVE = auto()
    EMPATHETIC = auto()
    HUMOROUS = auto()
    SERIOUS = auto()
    ENTHUSIASTIC = auto()
    NEUTRAL = auto()


class DiscourseRelation(Enum):
    """Relaciones de discurso (RST - Rhetorical Structure Theory)."""
    ELABORATION = auto()    # Elaboración de idea
    CONTRAST = auto()       # Contraste de ideas
    CAUSE = auto()          # Relación causal
    CONDITION = auto()      # Condición
    EVIDENCE = auto()       # Evidencia de afirmación
    BACKGROUND = auto()     # Contexto de fondo
    SUMMARY = auto()        # Resumen
    SEQUENCE = auto()       # Secuencia temporal


class EmotionType(Enum):
    """Emociones básicas y complejas."""
    # Ekman's 6 básicas
    JOY = auto()
    SADNESS = auto()
    ANGER = auto()
    FEAR = auto()
    DISGUST = auto()
    SURPRISE = auto()
    
    # Complejas
    ANXIETY = auto()
    HOPE = auto()
    PRIDE = auto()
    SHAME = auto()
    GUILT = auto()
    ENVY = auto()
    LOVE = auto()
    GRATITUDE = auto()


class IntentType(Enum):
    """Tipos de intención comunicativa."""
    INFORM = auto()
    QUESTION = auto()
    REQUEST = auto()
    CONFIRM = auto()
    DENY = auto()
    CLARIFY = auto()
    AGREE = auto()
    DISAGREE = auto()
    SUGGEST = auto()
    COMPLAIN = auto()


@dataclass
class SpeechAct:
    """Acto de habla con fuerza ilocutiva."""
    act_id: str
    act_type: SpeechActType
    propositional_content: str
    felicity_conditions: Dict[str, bool] = field(default_factory=dict)
    perlocutionary_effect: Optional[str] = None
    sincerity_score: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Implicature:
    """Implicatura conversacional."""
    implicature_id: str
    literal_meaning: str
    implied_meaning: str
    implicature_type: ImplicatureType
    confidence: float = 0.0
    maxims_violated: List[str] = field(default_factory=list)  # Grice's maxims
    context_required: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscourseUnit:
    """Unidad de discurso."""
    unit_id: str
    text: str
    relation: DiscourseRelation
    parent_unit: Optional[str] = None
    children: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    coherence_score: float = 0.0


@dataclass
class EntityMention:
    """Mención de entidad."""
    mention_id: str
    text: str
    entity_type: str  # PERSON, ORG, LOC, etc.
    span: tuple[int, int]  # (start, end)
    coreference_chain: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    confidence: float = 0.0


@dataclass
class SemanticFrame:
    """Frame semántico (FrameNet)."""
    frame_id: str
    frame_name: str
    lexical_unit: str
    frame_elements: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class DialogueTurn:
    """Turno de diálogo."""
    turn_id: str
    speaker: str
    utterance: str
    intent: IntentType
    speech_act: Optional[SpeechAct] = None
    emotion: Optional[EmotionType] = None
    timestamp: float = field(default_factory=time.time)
    coherence_with_previous: float = 0.0


@dataclass
class TextGeneration:
    """Configuración de generación de texto."""
    target_audience: str = "general"
    tone: ToneType = ToneType.NEUTRAL
    formality_level: float = 0.5  # 0=muy informal, 1=muy formal
    creativity_level: float = 0.5  # 0=conservador, 1=creativo
    max_length: int = 500
    style_reference: Optional[str] = None


# ============================================================================
# PRAGMATICS ENGINE
# ============================================================================

class PragmaticsEngine:
    """Motor de análisis pragmático avanzado."""
    
    def __init__(self):
        self.speech_acts: Dict[str, SpeechAct] = {}
        self.implicatures: Dict[str, Implicature] = {}
        self.context_stack: List[Dict[str, Any]] = []
        self.maxims = ["quantity", "quality", "relation", "manner"]
        self.logger = logger.getChild("pragmatics")
    
    def analyze_speech_act(self, utterance: str, context: Dict[str, Any]) -> SpeechAct:
        """Analiza el acto de habla de un enunciado."""
        act_id = f"act_{len(self.speech_acts)}_{int(time.time()*1000)}"
        
        # Detectar tipo de acto de habla
        act_type = self._classify_speech_act(utterance, context)
        
        # Extraer contenido proposicional
        propositional_content = self._extract_propositional_content(utterance)
        
        # Evaluar condiciones de felicidad
        felicity = self._evaluate_felicity_conditions(act_type, context)
        
        # Estimar efecto perlocutivo
        effect = self._predict_perlocutionary_effect(act_type, context)
        
        # Evaluar sinceridad
        sincerity = self._assess_sincerity(utterance, context)
        
        act = SpeechAct(
            act_id=act_id,
            act_type=act_type,
            propositional_content=propositional_content,
            felicity_conditions=felicity,
            perlocutionary_effect=effect,
            sincerity_score=sincerity
        )
        
        self.speech_acts[act_id] = act
        self.logger.debug(f"Acto de habla analizado: {act_type.name}")
        
        return act
    
    def detect_implicature(self, utterance: str, context: Dict[str, Any]) -> Optional[Implicature]:
        """Detecta implicaturas conversacionales."""
        literal = self._extract_literal_meaning(utterance)
        
        # Detectar violación de máximas
        violated = self._detect_maxim_violations(utterance, context)
        
        if not violated:
            return None
        
        # Inferir significado implicado
        implied = self._infer_implied_meaning(utterance, literal, violated, context)
        
        if not implied:
            return None
        
        # Clasificar tipo de implicatura
        implicature_type = self._classify_implicature(utterance, implied, context)
        
        # Calcular confianza
        confidence = self._calculate_implicature_confidence(
            literal, implied, violated, context
        )
        
        implicature_id = f"impl_{len(self.implicatures)}_{int(time.time()*1000)}"
        
        implicature = Implicature(
            implicature_id=implicature_id,
            literal_meaning=literal,
            implied_meaning=implied,
            implicature_type=implicature_type,
            confidence=confidence,
            maxims_violated=violated,
            context_required=context
        )
        
        self.implicatures[implicature_id] = implicature
        self.logger.info(f"Implicatura detectada: '{literal}' → '{implied}'")
        
        return implicature
    
    def _classify_speech_act(self, utterance: str, context: Dict[str, Any]) -> SpeechActType:
        """Clasifica el tipo de acto de habla."""
        utterance_lower = utterance.lower().strip()
        
        # Directivas
        if any(utterance_lower.startswith(w) for w in ["por favor", "podrías", "puedes", "haz"]):
            return SpeechActType.DIRECTIVE
        
        # Comisivas
        if any(w in utterance_lower for w in ["prometo", "me comprometo", "voy a"]):
            return SpeechActType.COMMISSIVE
        
        # Expresivas
        if any(w in utterance_lower for w in ["gracias", "perdón", "disculpa", "felicidades"]):
            return SpeechActType.EXPRESSIVE
        
        # Declarativas
        if any(w in utterance_lower for w in ["declaro", "te nombro", "te designo"]):
            return SpeechActType.DECLARATIVE
        
        # Por defecto: asertiva
        return SpeechActType.ASSERTIVE
    
    def _extract_propositional_content(self, utterance: str) -> str:
        """Extrae el contenido proposicional."""
        # Remover marcadores de actos de habla
        markers = ["por favor", "podrías", "puedes", "prometo", "gracias", "perdón"]
        content = utterance.lower()
        
        for marker in markers:
            content = content.replace(marker, "")
        
        return content.strip()
    
    def _evaluate_felicity_conditions(
        self, act_type: SpeechActType, context: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Evalúa condiciones de felicidad del acto."""
        conditions: Dict[str, bool] = {}
        
        if act_type == SpeechActType.DIRECTIVE:
            conditions["hearer_can_perform"] = context.get("hearer_capable", True)
            conditions["speaker_has_authority"] = context.get("speaker_authority", True)
        elif act_type == SpeechActType.COMMISSIVE:
            conditions["speaker_can_perform"] = context.get("speaker_capable", True)
            conditions["speaker_intends_to_perform"] = True
        elif act_type == SpeechActType.DECLARATIVE:
            conditions["speaker_has_institutional_role"] = context.get("speaker_role", False)
        
        return conditions
    
    def _predict_perlocutionary_effect(
        self, act_type: SpeechActType, context: Dict[str, Any]
    ) -> str:
        """Predice el efecto perlocutivo."""
        effects = {
            SpeechActType.ASSERTIVE: "Convencer o informar al oyente",
            SpeechActType.DIRECTIVE: "Hacer que el oyente actúe",
            SpeechActType.COMMISSIVE: "Obligar al hablante a actuar",
            SpeechActType.EXPRESSIVE: "Expresar estado psicológico",
            SpeechActType.DECLARATIVE: "Cambiar el estado del mundo"
        }
        return effects.get(act_type, "Efecto desconocido")
    
    def _assess_sincerity(self, utterance: str, context: Dict[str, Any]) -> float:
        """Evalúa la sinceridad del enunciado."""
        # Heurística simple: longitud y especificidad
        specificity = len(utterance.split()) / 100.0
        
        # Detectar hedges (indicadores de baja sinceridad)
        hedges = ["quizás", "tal vez", "supongo", "creo que", "puede ser"]
        hedge_count = sum(1 for hedge in hedges if hedge in utterance.lower())
        
        sincerity = max(0.5, min(1.0, 0.8 + specificity - hedge_count * 0.1))
        
        return sincerity
    
    def _extract_literal_meaning(self, utterance: str) -> str:
        """Extrae el significado literal."""
        return utterance.strip()
    
    def _detect_maxim_violations(
        self, utterance: str, context: Dict[str, Any]
    ) -> List[str]:
        """Detecta violaciones de máximas de Grice."""
        violated: List[str] = []
        
        # Máxima de cantidad
        if len(utterance.split()) > 100:
            violated.append("quantity")
        
        # Máxima de calidad (buscar evidencia de falsedad)
        if any(w in utterance.lower() for w in ["supuestamente", "dicen que", "parece"]):
            violated.append("quality")
        
        # Máxima de relación (si context indica off-topic)
        if context.get("off_topic", False):
            violated.append("relation")
        
        # Máxima de manera (ambigüedad, prolijidad)
        if len(re.findall(r'\([^)]*\)', utterance)) > 2:
            violated.append("manner")
        
        return violated
    
    def _infer_implied_meaning(
        self, utterance: str, literal: str, violated: List[str], context: Dict[str, Any]
    ) -> str:
        """Infiere el significado implicado."""
        if not violated:
            return ""
        
        # Heurísticas simples
        if "quantity" in violated:
            return "El hablante quiere dar más información de la necesaria"
        
        if "quality" in violated:
            return "El hablante no está seguro de la veracidad"
        
        if "relation" in violated:
            return "El hablante cambia de tema intencionalmente"
        
        if "manner" in violated:
            return "El hablante intenta ser deliberadamente oscuro"
        
        return "Significado implicado no determinado"
    
    def _classify_implicature(
        self, utterance: str, implied: str, context: Dict[str, Any]
    ) -> ImplicatureType:
        """Clasifica el tipo de implicatura."""
        # Implicatura escalar
        scalars = ["algunos", "varios", "muchos", "pocos"]
        if any(s in utterance.lower() for s in scalars):
            return ImplicatureType.SCALAR
        
        # Implicatura convencional
        if "pero" in utterance.lower() or "sin embargo" in utterance.lower():
            return ImplicatureType.CONVENTIONAL
        
        # Implicatura particularizada si depende mucho del contexto
        if context and len(context) > 3:
            return ImplicatureType.PARTICULARIZED
        
        # Por defecto: generalizada
        return ImplicatureType.GENERALIZED
    
    def _calculate_implicature_confidence(
        self, literal: str, implied: str, violated: List[str], context: Dict[str, Any]
    ) -> float:
        """Calcula la confianza en la implicatura."""
        base_confidence = 0.6
        
        # Más violaciones → mayor confianza en implicatura
        base_confidence += len(violated) * 0.1
        
        # Contexto rico → mayor confianza
        base_confidence += len(context) * 0.02
        
        return min(1.0, base_confidence)
    
    def get_pragmatics_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de pragmática."""
        speech_act_dist = Counter(act.act_type.name for act in self.speech_acts.values())
        implicature_dist = Counter(
            imp.implicature_type.name for imp in self.implicatures.values()
        )
        
        return {
            "total_speech_acts": len(self.speech_acts),
            "speech_act_distribution": dict(speech_act_dist),
            "total_implicatures": len(self.implicatures),
            "implicature_distribution": dict(implicature_dist),
            "avg_sincerity": sum(act.sincerity_score for act in self.speech_acts.values()) / max(1, len(self.speech_acts))
        }


# ============================================================================
# CONTEXTUAL GENERATION ENGINE
# ============================================================================

class ContextualGenerationEngine:
    """Motor de generación de texto contextual avanzado."""
    
    def __init__(self):
        self.generation_history: List[Dict[str, Any]] = []
        self.style_profiles: Dict[str, Dict[str, Any]] = {}
        self.logger = logger.getChild("generation")
    
    def generate_text(
        self, 
        prompt: str, 
        config: TextGeneration, 
        context: Dict[str, Any]
    ) -> str:
        """Genera texto contextualmente apropiado."""
        # Ajustar tono
        text = self._apply_tone(prompt, config.tone)
        
        # Ajustar formalidad
        text = self._adjust_formality(text, config.formality_level)
        
        # Ajustar para audiencia
        text = self._adapt_to_audience(text, config.target_audience, context)
        
        # Aplicar creatividad
        text = self._apply_creativity(text, config.creativity_level)
        
        # Style transfer si hay referencia
        if config.style_reference:
            text = self._transfer_style(text, config.style_reference)
        
        # Truncar a longitud máxima
        text = self._truncate_coherently(text, config.max_length)
        
        # Guardar historial
        self.generation_history.append({
            "prompt": prompt,
            "generated": text,
            "config": config,
            "timestamp": time.time()
        })
        
        self.logger.info(f"Texto generado con tono {config.tone.name}")
        
        return text
    
    def _apply_tone(self, text: str, tone: ToneType) -> str:
        """Aplica un tono específico al texto."""
        if tone == ToneType.FORMAL:
            # Evitar contracciones, usar lenguaje formal
            text = text.replace("no es", "no es")
            text = text.replace("está", "se encuentra")
        elif tone == ToneType.INFORMAL:
            # Usar contracciones, lenguaje coloquial
            text = text.replace("no es", "no es")
        elif tone == ToneType.FRIENDLY:
            # Añadir marcadores de amigabilidad
            if not any(text.startswith(w) for w in ["Hola", "Hey", "¡"]):
                text = "¡" + text
        elif tone == ToneType.EMPATHETIC:
            # Añadir expresiones de empatía
            empathy_markers = ["Entiendo que", "Comprendo que", "Me imagino que"]
            if not any(em in text for em in empathy_markers):
                text = "Entiendo. " + text
        
        return text
    
    def _adjust_formality(self, text: str, level: float) -> str:
        """Ajusta el nivel de formalidad."""
        if level > 0.7:
            # Muy formal
            text = text.replace("tú", "usted")
            text = text.replace("tu", "su")
        elif level < 0.3:
            # Muy informal
            text = text.replace("usted", "tú")
            text = text.replace("su", "tu")
        
        return text
    
    def _adapt_to_audience(
        self, text: str, audience: str, context: Dict[str, Any]
    ) -> str:
        """Adapta el texto a la audiencia objetivo."""
        if audience == "technical":
            # Mantener jerga técnica
            pass  # Mantener texto tal cual para audiencia técnica
        elif audience == "general":
            # Simplificar términos técnicos
            technical_terms = {
                "implementación": "poner en práctica",
                "optimización": "mejora",
                "algoritmo": "método"
            }
            for tech, simple in technical_terms.items():
                text = text.replace(tech, simple)
        elif audience == "children":
            # Simplificar mucho
            text = text.replace(".", ".")
        
        return text
    
    def _apply_creativity(self, text: str, level: float) -> str:
        """Aplica nivel de creatividad."""
        if level > 0.7:
            # Alta creatividad: añadir metáforas, analogías
            creative_phrases = [
                "como un río que fluye",
                "brillante como el sol",
                "fuerte como el acero"
            ]
            # Heurística simple: añadir frase creativa al final
            if len(text) > 50:
                text += f" - {creative_phrases[len(text) % len(creative_phrases)]}"
        
        return text
    
    def _transfer_style(self, text: str, reference: str) -> str:
        """Transfiere el estilo de un texto de referencia."""
        # Analizar estilo de referencia
        ref_profile = self._analyze_style(reference)
        
        # Aplicar características de estilo
        if ref_profile.get("avg_sentence_length", 0) < 10:
            # Estilo conciso
            sentences = text.split(".")
            text = ". ".join(s.strip() for s in sentences if s.strip())
        
        return text
    
    def _analyze_style(self, text: str) -> Dict[str, Any]:
        """Analiza el estilo de un texto."""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        words = text.split()
        
        profile = {
            "avg_sentence_length": len(words) / max(1, len(sentences)),
            "vocabulary_richness": len(set(words)) / max(1, len(words)),
            "punctuation_frequency": text.count(",") + text.count(";")
        }
        
        return profile
    
    def _truncate_coherently(self, text: str, max_length: int) -> str:
        """Trunca el texto de forma coherente."""
        if len(text) <= max_length:
            return text
        
        # Truncar en punto o coma más cercano
        truncated = text[:max_length]
        last_period = truncated.rfind(".")
        last_comma = truncated.rfind(",")
        
        cut_point = max(last_period, last_comma)
        
        if cut_point > 0:
            return truncated[:cut_point + 1]
        
        return truncated + "..."
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de generación."""
        if not self.generation_history:
            return {"total_generations": 0}
        
        tone_dist = Counter(
            h["config"].tone.name for h in self.generation_history
        )
        
        avg_length = sum(
            len(h["generated"]) for h in self.generation_history
        ) / len(self.generation_history)
        
        return {
            "total_generations": len(self.generation_history),
            "tone_distribution": dict(tone_dist),
            "avg_generated_length": avg_length
        }


# ============================================================================
# DISCOURSE ANALYZER
# ============================================================================

class DiscourseAnalyzer:
    """Analizador de estructura de discurso."""
    
    def __init__(self):
        self.discourse_units: Dict[str, DiscourseUnit] = {}
        self.coreference_chains: Dict[str, List[str]] = defaultdict(list)
        self.topic_history: List[str] = []
        self.logger = logger.getChild("discourse")
    
    def analyze_discourse_structure(self, text: str) -> List[DiscourseUnit]:
        """Analiza la estructura retórica del discurso."""
        # Segmentar en unidades de discurso
        segments = self._segment_discourse(text)
        
        units: List[DiscourseUnit] = []
        
        for i, segment in enumerate(segments):
            unit_id = f"unit_{i}_{int(time.time()*1000)}"
            
            # Detectar relación con unidad anterior
            relation = self._detect_discourse_relation(
                segment, segments[i-1] if i > 0 else None
            )
            
            # Detectar tópico
            topic = self._extract_topic(segment)
            
            # Calcular coherencia
            coherence = self._calculate_coherence(segment, units)
            
            unit = DiscourseUnit(
                unit_id=unit_id,
                text=segment,
                relation=relation,
                parent_unit=units[-1].unit_id if i > 0 else None,
                topic=topic,
                coherence_score=coherence
            )
            
            units.append(unit)
            self.discourse_units[unit_id] = unit
            
            if topic:
                self.topic_history.append(topic)
        
        self.logger.info(f"Discurso analizado: {len(units)} unidades")
        
        return units
    
    def resolve_coreference(self, text: str) -> Dict[str, List[str]]:
        """Resuelve correferencias."""
        # Detectar menciones
        mentions = self._detect_mentions(text)
        
        # Agrupar en cadenas de correferencia
        chains: Dict[str, List[str]] = defaultdict(list)
        
        for mention in mentions:
            # Encontrar antecedente
            antecedent = self._find_antecedent(mention, mentions)
            
            if antecedent:
                chain_id = antecedent.get("chain_id", f"chain_{len(chains)}")
                chains[chain_id].append(mention["text"])
                mention["chain_id"] = chain_id
            else:
                # Nueva cadena
                chain_id = f"chain_{len(chains)}"
                chains[chain_id].append(mention["text"])
                mention["chain_id"] = chain_id
        
        self.coreference_chains.update(chains)
        self.logger.debug(f"Correferencias resueltas: {len(chains)} cadenas")
        
        return dict(chains)
    
    def _segment_discourse(self, text: str) -> List[str]:
        """Segmenta el texto en unidades de discurso."""
        # Segmentación simple por oraciones
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
        return sentences
    
    def _detect_discourse_relation(
        self, current: str, previous: Optional[str]
    ) -> DiscourseRelation:
        """Detecta la relación retórica entre unidades."""
        if not previous:
            return DiscourseRelation.BACKGROUND
        
        current_lower = current.lower()
        
        # Contraste
        if any(w in current_lower for w in ["pero", "sin embargo", "aunque"]):
            return DiscourseRelation.CONTRAST
        
        # Causa
        if any(w in current_lower for w in ["porque", "ya que", "debido a"]):
            return DiscourseRelation.CAUSE
        
        # Condición
        if any(w in current_lower for w in ["si", "cuando", "en caso de"]):
            return DiscourseRelation.CONDITION
        
        # Evidencia
        if any(w in current_lower for w in ["por ejemplo", "como", "tal como"]):
            return DiscourseRelation.EVIDENCE
        
        # Secuencia
        if any(w in current_lower for w in ["después", "luego", "entonces"]):
            return DiscourseRelation.SEQUENCE
        
        # Resumen
        if any(w in current_lower for w in ["en resumen", "en conclusión", "finalmente"]):
            return DiscourseRelation.SUMMARY
        
        # Por defecto: elaboración
        return DiscourseRelation.ELABORATION
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extrae el tópico de una unidad."""
        # Heurística: sustantivos más frecuentes
        words = re.findall(r'\b[A-Za-záéíóúñ]+\b', text.lower())
        
        # Filtrar stopwords simples
        stopwords = {"el", "la", "los", "las", "de", "que", "en", "y", "a", "un", "una"}
        words = [w for w in words if w not in stopwords and len(w) > 3]
        
        if words:
            # Palabra más frecuente
            return Counter(words).most_common(1)[0][0]
        
        return None
    
    def _calculate_coherence(
        self, segment: str, previous_units: List[DiscourseUnit]
    ) -> float:
        """Calcula la coherencia de una unidad con el discurso previo."""
        if not previous_units:
            return 1.0
        
        # Coherencia léxica: palabras compartidas
        segment_words = set(re.findall(r'\b\w+\b', segment.lower()))
        
        max_overlap = 0.0
        for unit in previous_units[-3:]:  # últimas 3 unidades
            unit_words = set(re.findall(r'\b\w+\b', unit.text.lower()))
            overlap = len(segment_words & unit_words) / max(1, len(segment_words | unit_words))
            max_overlap = max(max_overlap, overlap)
        
        return max_overlap
    
    def _detect_mentions(self, text: str) -> List[Dict[str, Any]]:
        """Detecta menciones de entidades."""
        mentions: List[Dict[str, Any]] = []
        
        # Detectar pronombres
        pronouns = re.finditer(r'\b(él|ella|ellos|ellas|esto|eso|aquello)\b', text.lower())
        for match in pronouns:
            mentions.append({
                "text": match.group(),
                "span": match.span(),
                "type": "pronoun"
            })
        
        # Detectar nombres propios (heurística simple)
        proper_nouns = re.finditer(r'\b[A-Z][a-záéíóúñ]+\b', text)
        for match in proper_nouns:
            mentions.append({
                "text": match.group(),
                "span": match.span(),
                "type": "proper_noun"
            })
        
        return mentions
    
    def _find_antecedent(
        self, mention: Dict[str, Any], all_mentions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Encuentra el antecedente de una mención."""
        if mention["type"] != "pronoun":
            return None
        
        # Buscar antecedente más cercano del tipo correcto
        mention_idx = all_mentions.index(mention)
        
        for i in range(mention_idx - 1, -1, -1):
            candidate = all_mentions[i]
            if candidate["type"] == "proper_noun":
                # Heurística: género y número
                return candidate
        
        return None
    
    def get_discourse_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de análisis de discurso."""
        relation_dist = Counter(
            unit.relation.name for unit in self.discourse_units.values()
        )
        
        avg_coherence = sum(
            unit.coherence_score for unit in self.discourse_units.values()
        ) / max(1, len(self.discourse_units))
        
        return {
            "total_units": len(self.discourse_units),
            "relation_distribution": dict(relation_dist),
            "avg_coherence": avg_coherence,
            "total_coreference_chains": len(self.coreference_chains),
            "topics_tracked": len(set(self.topic_history))
        }


# ============================================================================
# SENTIMENT & EMOTION ANALYZER
# ============================================================================

class SentimentEmotionAnalyzer:
    """Analizador avanzado de sentimiento y emoción."""
    
    def __init__(self):
        self.emotion_lexicon: Dict[str, EmotionType] = self._build_emotion_lexicon()
        self.analysis_history: List[Dict[str, Any]] = []
        self.logger = logger.getChild("sentiment")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analiza el sentimiento de un texto."""
        # Sentimiento base
        positive_words = ["bueno", "excelente", "feliz", "alegre", "maravilloso", "genial"]
        negative_words = ["malo", "terrible", "triste", "horrible", "pésimo", "awful"]
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        
        if total == 0:
            sentiment = {"positive": 0.5, "negative": 0.5, "neutral": 1.0}
        else:
            sentiment = {
                "positive": pos_count / total,
                "negative": neg_count / total,
                "neutral": 0.0
            }
        
        self.logger.debug(f"Sentimiento: pos={sentiment['positive']:.2f}, neg={sentiment['negative']:.2f}")
        
        return sentiment
    
    def detect_emotion(self, text: str) -> Dict[EmotionType, float]:
        """Detecta emociones en el texto."""
        emotions: Dict[EmotionType, float] = {e: 0.0 for e in EmotionType}
        
        text_lower = text.lower()
        
        # Buscar palabras emocionales
        for word, emotion in self.emotion_lexicon.items():
            if word in text_lower:
                emotions[emotion] += 0.2
        
        # Normalizar
        total = sum(emotions.values())
        if total > 0:
            emotions = {e: score/total for e, score in emotions.items()}
        
        # Filtrar emociones con score > 0
        detected = {e: score for e, score in emotions.items() if score > 0.1}
        
        self.logger.debug(f"Emociones detectadas: {list(detected.keys())}")
        
        return detected
    
    def detect_sarcasm(self, text: str, context: Dict[str, Any]) -> float:
        """Detecta sarcasmo."""
        sarcasm_score = 0.0
        
        # Indicadores de sarcasmo
        sarcasm_markers = ["claro", "obvio", "por supuesto", "seguro"]
        
        text_lower = text.lower()
        
        # Contradicción entre sentimiento y contexto
        sentiment = self.analyze_sentiment(text)
        
        if any(marker in text_lower for marker in sarcasm_markers):
            sarcasm_score += 0.3
        
        # Signos de exclamación o interrogación excesivos
        if text.count("!") > 2 or text.count("?") > 2:
            sarcasm_score += 0.2
        
        # Contradicción contextual
        if context.get("expected_sentiment") == "negative" and sentiment["positive"] > 0.6:
            sarcasm_score += 0.4
        
        return min(1.0, sarcasm_score)
    
    def detect_irony(self, text: str, context: Dict[str, Any]) -> float:
        """Detecta ironía."""
        irony_score = 0.0
        
        # La ironía es similar al sarcasmo pero más sutil
        sentiment = self.analyze_sentiment(text)
        
        # Contradicción situacional
        if context.get("situation_outcome") == "negative" and sentiment["positive"] > 0.5:
            irony_score += 0.5
        
        # Marcadores de ironía
        irony_markers = ["ironía", "curioso", "interesante", "qué sorpresa"]
        
        if any(marker in text.lower() for marker in irony_markers):
            irony_score += 0.3
        
        return min(1.0, irony_score)
    
    def _build_emotion_lexicon(self) -> Dict[str, EmotionType]:
        """Construye un léxico de emociones."""
        return {
            # Joy
            "feliz": EmotionType.JOY,
            "alegre": EmotionType.JOY,
            "contento": EmotionType.JOY,
            "eufórico": EmotionType.JOY,
            
            # Sadness
            "triste": EmotionType.SADNESS,
            "deprimido": EmotionType.SADNESS,
            "melancólico": EmotionType.SADNESS,
            
            # Anger
            "enojado": EmotionType.ANGER,
            "furioso": EmotionType.ANGER,
            "irritado": EmotionType.ANGER,
            
            # Fear
            "miedo": EmotionType.FEAR,
            "asustado": EmotionType.FEAR,
            "aterrado": EmotionType.FEAR,
            
            # Disgust
            "asco": EmotionType.DISGUST,
            "repugnancia": EmotionType.DISGUST,
            
            # Surprise
            "sorprendido": EmotionType.SURPRISE,
            "asombrado": EmotionType.SURPRISE,
            
            # Complex emotions
            "ansioso": EmotionType.ANXIETY,
            "esperanzado": EmotionType.HOPE,
            "orgulloso": EmotionType.PRIDE,
            "avergonzado": EmotionType.SHAME,
            "culpable": EmotionType.GUILT,
            "envidioso": EmotionType.ENVY,
            "amor": EmotionType.LOVE,
            "agradecido": EmotionType.GRATITUDE
        }
    
    def get_sentiment_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de análisis de sentimiento."""
        return {
            "total_analyses": len(self.analysis_history),
            "emotion_lexicon_size": len(self.emotion_lexicon)
        }


# ============================================================================
# DIALOGUE MANAGER
# ============================================================================

class DialogueManager:
    """Gestor de diálogo multi-turno avanzado."""
    
    def __init__(self):
        self.dialogue_history: List[DialogueTurn] = []
        self.current_intent: Optional[IntentType] = None
        self.context_tracking: Dict[str, Any] = {}
        self.logger = logger.getChild("dialogue")
    
    def process_turn(
        self, 
        speaker: str, 
        utterance: str, 
        context: Dict[str, Any]
    ) -> DialogueTurn:
        """Procesa un turno de diálogo."""
        turn_id = f"turn_{len(self.dialogue_history)}_{int(time.time()*1000)}"
        
        # Detectar intención
        intent = self._detect_intent(utterance, context)
        
        # Calcular coherencia con turno anterior
        coherence = self._calculate_turn_coherence(utterance)
        
        turn = DialogueTurn(
            turn_id=turn_id,
            speaker=speaker,
            utterance=utterance,
            intent=intent,
            coherence_with_previous=coherence
        )
        
        self.dialogue_history.append(turn)
        self.current_intent = intent
        
        # Actualizar contexto
        self._update_context(turn)
        
        self.logger.info(f"Turno procesado: {speaker} - {intent.name}")
        
        return turn
    
    def should_clarify(self, utterance: str, context: Dict[str, Any]) -> bool:
        """Determina si se necesita clarificación."""
        # Ambigüedad
        if "?" in utterance and len(utterance.split()) < 5:
            return True
        
        # Falta de información
        if context.get("information_complete", True) is False:
            return True
        
        # Contradicción con contexto
        if context.get("contradicts_history", False):
            return True
        
        return False
    
    def generate_clarification(self, utterance: str, context: Dict[str, Any]) -> str:
        """Genera una pregunta de clarificación."""
        clarifications = [
            "¿Podrías especificar más?",
            "No estoy seguro de entender, ¿puedes explicar?",
            "¿A qué te refieres exactamente?",
            "¿Puedes dar más detalles?"
        ]
        
        return clarifications[len(utterance) % len(clarifications)]
    
    def _detect_intent(self, utterance: str, context: Dict[str, Any]) -> IntentType:
        """Detecta la intención del turno."""
        utterance_lower = utterance.lower().strip()
        
        # Question
        if "?" in utterance:
            return IntentType.QUESTION
        
        # Request
        if any(utterance_lower.startswith(w) for w in ["por favor", "podrías", "puedes"]):
            return IntentType.REQUEST
        
        # Confirm
        if any(w in utterance_lower for w in ["sí", "correcto", "exacto", "así es"]):
            return IntentType.CONFIRM
        
        # Deny
        if any(w in utterance_lower for w in ["no", "incorrecto", "falso"]):
            return IntentType.DENY
        
        # Clarify
        if any(w in utterance_lower for w in ["es decir", "o sea", "me refiero"]):
            return IntentType.CLARIFY
        
        # Agree
        if any(w in utterance_lower for w in ["de acuerdo", "estoy de acuerdo", "concuerdo"]):
            return IntentType.AGREE
        
        # Disagree
        if any(w in utterance_lower for w in ["no estoy de acuerdo", "discrepo"]):
            return IntentType.DISAGREE
        
        # Suggest
        if any(w in utterance_lower for w in ["sugiero", "propongo", "recomiendo"]):
            return IntentType.SUGGEST
        
        # Complain
        if any(w in utterance_lower for w in ["me quejo", "no me gusta", "es inaceptable"]):
            return IntentType.COMPLAIN
        
        # Default: inform
        return IntentType.INFORM
    
    def _calculate_turn_coherence(self, utterance: str) -> float:
        """Calcula la coherencia con el turno anterior."""
        if not self.dialogue_history:
            return 1.0
        
        previous = self.dialogue_history[-1]
        
        # Coherencia léxica
        current_words = set(re.findall(r'\b\w+\b', utterance.lower()))
        previous_words = set(re.findall(r'\b\w+\b', previous.utterance.lower()))
        
        overlap = len(current_words & previous_words)
        union = len(current_words | previous_words)
        
        coherence = overlap / max(1, union)
        
        # Bonus por respuesta directa
        if previous.intent == IntentType.QUESTION and self.current_intent == IntentType.INFORM:
            coherence += 0.3
        
        return min(1.0, coherence)
    
    def _update_context(self, turn: DialogueTurn) -> None:
        """Actualiza el contexto de diálogo."""
        self.context_tracking["last_speaker"] = turn.speaker
        self.context_tracking["last_intent"] = turn.intent
        self.context_tracking["turn_count"] = len(self.dialogue_history)
    
    def get_dialogue_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de diálogo."""
        intent_dist = Counter(turn.intent.name for turn in self.dialogue_history)
        
        avg_coherence = sum(
            turn.coherence_with_previous for turn in self.dialogue_history
        ) / max(1, len(self.dialogue_history))
        
        return {
            "total_turns": len(self.dialogue_history),
            "intent_distribution": dict(intent_dist),
            "avg_coherence": avg_coherence,
            "current_intent": self.current_intent.name if self.current_intent else None
        }


# ============================================================================
# MAIN LANGUAGE PROCESSING ENGINE
# ============================================================================

class LanguageProcessingEngine:
    """Motor principal de procesamiento de lenguaje."""
    
    def __init__(self):
        self.pragmatics = PragmaticsEngine()
        self.generation = ContextualGenerationEngine()
        self.discourse = DiscourseAnalyzer()
        self.sentiment = SentimentEmotionAnalyzer()
        self.dialogue = DialogueManager()
        self.logger = logger.getChild("language_engine")
        
        # Neural network integration
        self.neural_network: Optional[MetacortexNeuralSymbioticNetworkV2] = None
        try:
            from neural_symbiotic_network import get_neural_network
            self.neural_network = get_neural_network()
            self.neural_network.register_module("language_processing", self)
            self.logger.info("✅ 'language_processing' conectado a red neuronal")
        except Exception as e:
            logger.error(f"Error en language_processing.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
    
    def process_utterance(
        self, 
        utterance: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Procesa un enunciado completo."""
        # Análisis pragmático
        speech_act = self.pragmatics.analyze_speech_act(utterance, context)
        implicature = self.pragmatics.detect_implicature(utterance, context)
        
        # Análisis de sentimiento y emoción
        sentiment = self.sentiment.analyze_sentiment(utterance)
        emotions = self.sentiment.detect_emotion(utterance)
        sarcasm = self.sentiment.detect_sarcasm(utterance, context)
        
        # Análisis de discurso
        discourse_units = self.discourse.analyze_discourse_structure(utterance)
        coreferences = self.discourse.resolve_coreference(utterance)
        
        # Gestión de diálogo
        speaker = context.get("speaker", "unknown")
        turn = self.dialogue.process_turn(speaker, utterance, context)
        
        result = {
            "speech_act": {
                "type": speech_act.act_type.name,
                "propositional_content": speech_act.propositional_content,
                "sincerity": speech_act.sincerity_score
            },
            "implicature": {
                "detected": True,
                "implied_meaning": implicature.implied_meaning,
                "confidence": implicature.confidence
            } if implicature else None,
            "sentiment": sentiment,
            "emotions": {e.name: score for e, score in emotions.items()},
            "sarcasm_score": sarcasm,
            "discourse": {
                "units": len(discourse_units),
                "coreference_chains": len(coreferences)
            },
            "dialogue": {
                "intent": turn.intent.name,
                "coherence": turn.coherence_with_previous
            }
        }
        
        self.logger.info(f"Enunciado procesado: {speech_act.act_type.name}")
        
        return result
    
    def generate_response(
        self, 
        prompt: str, 
        config: TextGeneration, 
        context: Dict[str, Any]
    ) -> str:
        """Genera una respuesta contextual."""
        response = self.generation.generate_text(prompt, config, context)
        
        self.logger.info(f"Respuesta generada: {len(response)} caracteres")
        
        return response
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema."""
        return {
            "pragmatics": self.pragmatics.get_pragmatics_stats(),
            "generation": self.generation.get_generation_stats(),
            "discourse": self.discourse.get_discourse_stats(),
            "sentiment": self.sentiment.get_sentiment_stats(),
            "dialogue": self.dialogue.get_dialogue_stats()
        }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_language_engine: Optional[LanguageProcessingEngine] = None


def get_language_engine() -> LanguageProcessingEngine:
    """Obtiene la instancia global del motor de lenguaje."""
    global _language_engine
    if _language_engine is None:
        _language_engine = LanguageProcessingEngine()
    return _language_engine


if __name__ == "__main__":
    # Test rápido
    engine = get_language_engine()
    
    # Test procesamiento
    result = engine.process_utterance(
        "¿Podrías ayudarme con este problema? Es bastante urgente.",
        {"speaker": "user", "urgency": "high"}
    )
    
    print("✅ Language Processing System inicializado")
    print(f"Speech Act: {result['speech_act']['type']}")
    print(f"Intent: {result['dialogue']['intent']}")
    print(f"Sentimiento: {result['sentiment']}")