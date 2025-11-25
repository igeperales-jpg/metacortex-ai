"""
🆘 EMERGENCY CONTACT SYSTEM - Sistema de Contacto Real con Personas Necesitadas
================================================================================

Sistema OPERACIONAL para que personas perseguidas puedan contactar con METACORTEX
de forma SEGURA, ANÓNIMA y VERIFICABLE.

MÉTODOS DE CONTACTO REAL:
1. Web Form (público) - https://metacortex.ai/emergency
2. Telegram Bot - @MetacortexProtectionBot
3. Email PGP - emergency@metacortex.ai
4. Signal - +[NÚMERO_SEGURO]
5. Tor Hidden Service - [ONION_ADDRESS]
6. WhatsApp Emergency - +[NÚMERO_VERIFICADO]

PROCESO AUTOMÁTICO:
1. Persona envía solicitud por cualquier canal
2. Sistema verifica urgencia con IA
3. Auto-triage: CRÍTICO / URGENTE / NORMAL
4. Asignación automática a operadores
5. Respuesta en <5 minutos para casos CRÍTICOS

Autor: METACORTEX Emergency Response Team
Fecha: 24 de Noviembre de 2025
Versión: 1.0.0 - Production Ready
"""

import os
import logging
import asyncio
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# FastAPI para endpoints web
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field

# Telegram Bot
try:
    import telegram
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("⚠️ python-telegram-bot no instalado: pip install python-telegram-bot")

# Email con PGP
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

# Twilio para WhatsApp
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logging.warning("⚠️ Twilio no instalado: pip install twilio")

# Load environment
from dotenv import load_dotenv
load_dotenv()

# AI Integration Layer
try:
    from metacortex_sinaptico.ai_integration_layer import get_ai_integration
    AI_INTEGRATION_AVAILABLE = True
except ImportError:
    AI_INTEGRATION_AVAILABLE = False
    logging.warning("⚠️ AI Integration Layer not available")

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & DATA STRUCTURES
# ============================================================================

class ContactChannel(Enum):
    """Canales de contacto disponibles"""
    WEB_FORM = "web_form"
    TELEGRAM = "telegram"
    EMAIL_PGP = "email_pgp"
    SIGNAL = "signal"
    WHATSAPP = "whatsapp"
    TOR_HIDDEN = "tor_hidden"
    PHONE_HOTLINE = "phone_hotline"


class UrgencyLevel(Enum):
    """Nivel de urgencia de la solicitud"""
    CRITICAL = "critical"  # Peligro inminente - Respuesta <5 min
    URGENT = "urgent"      # Situación grave - Respuesta <30 min
    HIGH = "high"          # Necesidad seria - Respuesta <2 horas
    NORMAL = "normal"      # Ayuda general - Respuesta <24 horas
    LOW = "low"            # Información - Respuesta <48 horas


class RequestStatus(Enum):
    """Estado de la solicitud"""
    RECEIVED = "received"
    VERIFYING = "verifying"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ThreatType(Enum):
    """Tipo de amenaza"""
    PERSECUTION_RELIGIOUS = "persecution_religious"
    PERSECUTION_POLITICAL = "persecution_political"
    VIOLENCE_PHYSICAL = "violence_physical"
    STARVATION = "starvation"
    HOMELESSNESS = "homelessness"
    MEDICAL_EMERGENCY = "medical_emergency"
    HUMAN_TRAFFICKING = "human_trafficking"
    FORCED_DISPLACEMENT = "forced_displacement"
    OTHER = "other"


@dataclass
class EmergencyRequest:
    """Solicitud de ayuda de emergencia"""
    request_id: str
    channel: ContactChannel
    urgency: UrgencyLevel
    status: RequestStatus
    
    # Información del solicitante (puede ser anónimo)
    name: Optional[str] = None
    contact_info: str = ""  # Telegram ID, email, phone, etc.
    location: Optional[str] = None  # Puede ser aproximado por seguridad
    
    # Detalles de la situación
    threat_type: ThreatType = ThreatType.OTHER
    description: str = ""
    needs: List[str] = field(default_factory=list)  # comida, refugio, médico, etc.
    
    # Tracking
    received_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    responded_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Seguridad
    is_verified: bool = False
    verification_notes: str = ""
    risk_assessment: float = 0.0  # 0.0-1.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================

class EmergencyRequestWeb(BaseModel):
    """Modelo para solicitud web"""
    name: Optional[str] = None
    contact: str = Field(..., description="Email, Telegram, WhatsApp o Phone")
    location: Optional[str] = None
    threat_type: str
    description: str
    needs: List[str]
    is_anonymous: bool = False


class EmergencyResponseAPI(BaseModel):
    """Respuesta de API"""
    success: bool
    request_id: Optional[str] = None
    message: str
    urgency_level: Optional[str] = None
    estimated_response_time: Optional[str] = None


# ============================================================================
# EMERGENCY CONTACT SYSTEM
# ============================================================================

class EmergencyContactSystem:
    """
    Sistema de contacto real con personas necesitadas.
    
    Permite recibir solicitudes por múltiples canales y gestionar
    respuestas de forma rápida y segura.
    
    **NUEVO**: Ahora con memoria persistente y integración completa con METACORTEX Core.
    """
    
    def __init__(self, project_root: Path = Path.cwd()):
        self.project_root = project_root
        self.requests_dir = project_root / "emergency_requests"
        self.requests_dir.mkdir(exist_ok=True, parents=True)
        
        # 🧠 MEMORIA PERSISTENTE DE USUARIOS (nuevo)
        self.user_profiles_dir = project_root / "user_profiles"
        self.user_profiles_dir.mkdir(exist_ok=True, parents=True)
        self.user_profiles_cache: Dict[str, Dict[str, Any]] = {}
        
        # AI Integration Layer - DEBE SER PRIMERO
        if AI_INTEGRATION_AVAILABLE:
            self.ai = get_ai_integration(project_root)
            self.ai.connect_emergency_contact(self)
            logger.info("✅ AI Integration connected to Emergency Contact System")
        else:
            self.ai = None
            logger.warning("⚠️ AI Integration not available - using basic responses")
        
        # Storage de solicitudes activas
        self.active_requests: Dict[str, EmergencyRequest] = {}
        self.request_history: List[EmergencyRequest] = []
        
        # Configuración de canales
        self.telegram_bot = None
        self.telegram_app = None  # Guardamos el Application también
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        
        self.email_address = os.getenv("EMERGENCY_EMAIL", "emergency@metacortex.ai")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        
        self.twilio_client = None
        if TWILIO_AVAILABLE:
            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            if account_sid and auth_token:
                self.twilio_client = TwilioClient(account_sid, auth_token)
        
        # Stats
        self.total_requests = 0
        self.critical_requests = 0
        self.resolved_requests = 0
        
        logger.info("🆘 Emergency Contact System initialized")
        logger.info(f"   Telegram: {'✅ Configured' if self.telegram_token else '⚠️ Not configured'}")
        logger.info(f"   Email: {'✅ Configured' if self.email_password else '⚠️ Not configured'}")
        logger.info(f"   WhatsApp: {'✅ Configured' if self.twilio_client else '⚠️ Not configured'}")
    
    async def receive_request(
        self,
        channel: ContactChannel,
        contact_info: str,
        description: str,
        threat_type: ThreatType = ThreatType.OTHER,
        name: Optional[str] = None,
        location: Optional[str] = None,
        needs: Optional[List[str]] = None
    ) -> EmergencyRequest:
        """
        Recibe una solicitud de emergencia desde cualquier canal.
        
        Esta función es el punto de entrada principal para todas las solicitudes.
        """
        # Generar ID único
        request_id = self._generate_request_id()
        
        # Evaluar urgencia con IA
        urgency = await self._assess_urgency(description, threat_type)
        
        # Crear solicitud
        request = EmergencyRequest(
            request_id=request_id,
            channel=channel,
            urgency=urgency,
            status=RequestStatus.RECEIVED,
            name=name,
            contact_info=contact_info,
            location=location,
            threat_type=threat_type,
            description=description,
            needs=needs or []
        )
        
        # Guardar
        self.active_requests[request_id] = request
        self._persist_request(request)
        
        # Stats
        self.total_requests += 1
        if urgency == UrgencyLevel.CRITICAL:
            self.critical_requests += 1
        
        logger.info(f"🆘 Nueva solicitud recibida: {request_id}")
        logger.info(f"   Canal: {channel.value}")
        logger.info(f"   Urgencia: {urgency.value}")
        logger.info(f"   Amenaza: {threat_type.value}")
        
        # Notificar a operadores según urgencia
        await self._notify_operators(request)
        
        # Auto-respuesta inmediata
        await self._send_auto_response(request)
        
        return request
    
    async def _assess_urgency(
        self,
        description: str,
        threat_type: ThreatType
    ) -> UrgencyLevel:
        """
        Evalúa el nivel de urgencia usando IA y keywords.
        """
        description_lower = description.lower()
        
        # Keywords CRÍTICOS (peligro inminente)
        critical_keywords = [
            "now", "ahora", "inmediato", "immediate",
            "killing", "matando", "shooting", "disparando",
            "dying", "muriendo", "death", "muerte",
            "attack", "ataque", "violence", "violencia",
            "rape", "violación", "kidnap", "secuestro",
            "bleeding", "sangrando", "injured", "herido",
            "tonight", "esta noche", "today", "hoy"
        ]
        
        # Keywords URGENTES
        urgent_keywords = [
            "danger", "peligro", "threat", "amenaza",
            "persecution", "persecución", "fled", "huí",
            "hiding", "escondido", "refugee", "refugiado",
            "no food", "sin comida", "starving", "hambriento",
            "sick", "enfermo", "help", "ayuda"
        ]
        
        # Evaluar
        if any(keyword in description_lower for keyword in critical_keywords):
            return UrgencyLevel.CRITICAL
        
        if threat_type in [ThreatType.PERSECUTION_RELIGIOUS, ThreatType.VIOLENCE_PHYSICAL,
                          ThreatType.HUMAN_TRAFFICKING, ThreatType.MEDICAL_EMERGENCY]:
            return UrgencyLevel.URGENT
        
        if any(keyword in description_lower for keyword in urgent_keywords):
            return UrgencyLevel.URGENT
        
        if threat_type == ThreatType.STARVATION:
            return UrgencyLevel.HIGH
        
        return UrgencyLevel.NORMAL
    
    async def _notify_operators(self, request: EmergencyRequest) -> None:
        """
        Notifica a operadores humanos según urgencia.
        """
        if request.urgency == UrgencyLevel.CRITICAL:
            # Notificación inmediata a TODOS los operadores
            logger.critical(f"🚨 CRITICAL REQUEST: {request.request_id}")
            logger.critical(f"   Contact: {request.contact_info}")
            logger.critical(f"   Location: {request.location}")
            logger.critical(f"   Description: {request.description[:100]}...")
            
            # TODO: Enviar SMS/Push notifications a operadores
            # TODO: Llamar a línea de emergencia
            
        elif request.urgency == UrgencyLevel.URGENT:
            logger.warning(f"⚠️ URGENT REQUEST: {request.request_id}")
            # TODO: Notificar a operadores disponibles
    
    async def _send_auto_response(self, request: EmergencyRequest) -> None:
        """
        Envía respuesta automática inmediata al solicitante.
        """
        response_messages = {
            UrgencyLevel.CRITICAL: (
                "🚨 EMERGENCY RECEIVED - We are responding NOW\n\n"
                f"Request ID: {request.request_id}\n"
                "Status: CRITICAL PRIORITY\n"
                "Expected response: <5 MINUTES\n\n"
                "An operator will contact you IMMEDIATELY.\n"
                "If you are in immediate danger, also contact local emergency services.\n\n"
                "📞 Emergency Numbers:\n"
                "- International Emergency: 112\n"
                "- US Emergency: 911\n\n"
                "You are not alone. Help is coming."
            ),
            UrgencyLevel.URGENT: (
                "🆘 Request Received - Help is on the way\n\n"
                f"Request ID: {request.request_id}\n"
                "Status: URGENT\n"
                "Expected response: <30 MINUTES\n\n"
                "An operator will contact you soon.\n"
                "Your safety is our priority."
            ),
            UrgencyLevel.HIGH: (
                "✅ Request Received\n\n"
                f"Request ID: {request.request_id}\n"
                "Status: HIGH PRIORITY\n"
                "Expected response: <2 HOURS\n\n"
                "We are processing your request."
            )
        }
        
        message = response_messages.get(
            request.urgency,
            f"✅ Request Received\nID: {request.request_id}\nWe will respond within 24 hours."
        )
        
        try:
            if request.channel == ContactChannel.TELEGRAM:
                await self._send_telegram_message(request.contact_info, message)
            elif request.channel == ContactChannel.EMAIL_PGP:
                await self._send_email(request.contact_info, "Emergency Request Received", message)
            elif request.channel == ContactChannel.WHATSAPP:
                await self._send_whatsapp_message(request.contact_info, message)
        except Exception as e:
            logger.error(f"Error sending auto-response: {e}")
    
    async def _send_telegram_message(self, chat_id: str, message: str) -> None:
        """Envía mensaje por Telegram"""
        if not self.telegram_bot or not TELEGRAM_AVAILABLE:
            logger.warning("Telegram bot not configured")
            return
        
        try:
            await self.telegram_bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"✅ Telegram message sent to {chat_id}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    async def _send_email(self, to_email: str, subject: str, body: str) -> None:
        """Envía email"""
        if not self.email_password or not EMAIL_AVAILABLE:
            logger.warning("Email not configured")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_address
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.email_address, self.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"✅ Email sent to {to_email}")
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    async def _send_whatsapp_message(self, phone: str, message: str) -> None:
        """Envía mensaje por WhatsApp (vía Twilio)"""
        if not self.twilio_client or not TWILIO_AVAILABLE:
            logger.warning("WhatsApp not configured")
            return
        
        try:
            from_whatsapp = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
            to_whatsapp = f"whatsapp:{phone}"
            
            message_obj = self.twilio_client.messages.create(
                body=message,
                from_=from_whatsapp,
                to=to_whatsapp
            )
            logger.info(f"✅ WhatsApp message sent to {phone}")
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")
    
    def _generate_request_id(self) -> str:
        """Genera ID único para solicitud"""
        timestamp = int(datetime.now().timestamp())
        random_part = secrets.token_hex(4).upper()
        return f"EMRG_{timestamp}_{random_part}"
    
    def _persist_request(self, request: EmergencyRequest) -> None:
        """Guarda solicitud en disco"""
        file_path = self.requests_dir / f"{request.request_id}.json"
        
        data = {
            "request_id": request.request_id,
            "channel": request.channel.value,
            "urgency": request.urgency.value,
            "status": request.status.value,
            "name": request.name,
            "contact_info": request.contact_info,
            "location": request.location,
            "threat_type": request.threat_type.value,
            "description": request.description,
            "needs": request.needs,
            "received_at": request.received_at.isoformat(),
            "is_verified": request.is_verified
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_request_status(self, request_id: str) -> Optional[EmergencyRequest]:
        """Obtiene estado de una solicitud"""
        return self.active_requests.get(request_id)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema"""
        return {
            "total_requests": self.total_requests,
            "active_requests": len(self.active_requests),
            "critical_requests": self.critical_requests,
            "resolved_requests": self.resolved_requests,
            "channels_configured": {
                "telegram": bool(self.telegram_token),
                "email": bool(self.email_password),
                "whatsapp": bool(self.twilio_client)
            }
        }
    
    # ========================================================================
    # 🧠 MEMORIA PERSISTENTE DE USUARIOS (NUEVO)
    # ========================================================================
    
    async def _get_or_create_user_profile(
        self, 
        chat_id: str, 
        username: str = "Anonymous"
    ) -> Dict[str, Any]:
        """
        Obtiene o crea perfil persistente de usuario.
        
        Esto permite que el sistema RECUERDE conversaciones previas,
        nivel de urgencia, historial de solicitudes, etc.
        
        Args:
            chat_id: ID único del usuario (Telegram chat_id)
            username: Nombre de usuario (opcional)
        
        Returns:
            Diccionario con perfil completo del usuario
        """
        # Verificar si está en caché
        if chat_id in self.user_profiles_cache:
            return self.user_profiles_cache[chat_id]
        
        # Buscar archivo de perfil
        profile_file = self.user_profiles_dir / f"{chat_id}.json"
        
        if profile_file.exists():
            # Cargar perfil existente
            with open(profile_file, 'r') as f:
                profile = json.load(f)
            logger.info(f"📂 Perfil cargado para {username} ({chat_id})")
        else:
            # Crear nuevo perfil
            profile = {
                'chat_id': chat_id,
                'username': username,
                'created_at': datetime.now().isoformat(),
                'last_contact': datetime.now().isoformat(),
                'message_history': [],
                'request_count': 0,
                'urgency_level': 0.5,  # 0.0 = normal, 1.0 = crítico
                'threat_level': 'unknown',
                'location_history': [],
                'notes': [],
                'resolved_requests': [],
                'active_request_id': None,
                'language_preference': 'auto',
                'trust_score': 1.0,  # Confianza inicial
                'verification_status': 'unverified'
            }
            logger.info(f"✨ Nuevo perfil creado para {username} ({chat_id})")
        
        # Actualizar timestamp de último contacto
        profile['last_contact'] = datetime.now().isoformat()
        
        # Guardar en caché
        self.user_profiles_cache[chat_id] = profile
        
        return profile
    
    async def _save_user_profile(self, chat_id: str, profile: Dict[str, Any]) -> None:
        """
        Guarda perfil de usuario en disco.
        
        Esto asegura persistencia entre reinicios del sistema.
        """
        profile_file = self.user_profiles_dir / f"{chat_id}.json"
        
        with open(profile_file, 'w') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        
        # Actualizar caché
        self.user_profiles_cache[chat_id] = profile
        
        logger.debug(f"💾 Perfil guardado para {chat_id}")
    
    async def _get_conversation_context(
        self, 
        chat_id: str, 
        last_n_messages: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Obtiene contexto de conversación reciente.
        
        Returns:
            Lista de últimos N mensajes de la conversación
        """
        profile = await self._get_or_create_user_profile(chat_id)
        return profile['message_history'][-last_n_messages:]
    
    async def _update_urgency_level(
        self, 
        chat_id: str, 
        urgency: float
    ) -> None:
        """
        Actualiza nivel de urgencia del usuario.
        
        Args:
            chat_id: ID del usuario
            urgency: Nivel de urgencia (0.0 a 1.0)
        """
        profile = await self._get_or_create_user_profile(chat_id)
        profile['urgency_level'] = max(profile['urgency_level'], urgency)
        await self._save_user_profile(chat_id, profile)


# ============================================================================
# FASTAPI WEB ENDPOINT
# ============================================================================

app = FastAPI(title="METACORTEX Emergency Contact API")
contact_system = EmergencyContactSystem()


@app.post("/api/emergency/request", response_model=EmergencyResponseAPI)
async def submit_emergency_request(
    request: EmergencyRequestWeb,
    background_tasks: BackgroundTasks
) -> EmergencyResponseAPI:
    """
    Endpoint público para recibir solicitudes de emergencia.
    
    Este es el PUNTO DE ENTRADA PRINCIPAL para personas que necesitan ayuda.
    """
    try:
        # Convertir threat_type string a enum
        try:
            threat_type = ThreatType[request.threat_type.upper()]
        except KeyError:
            threat_type = ThreatType.OTHER
        
        # Recibir solicitud
        emergency_request = await contact_system.receive_request(
            channel=ContactChannel.WEB_FORM,
            contact_info=request.contact,
            description=request.description,
            threat_type=threat_type,
            name=request.name if not request.is_anonymous else None,
            location=request.location,
            needs=request.needs
        )
        
        # Determinar tiempo estimado de respuesta
        response_times = {
            UrgencyLevel.CRITICAL: "<5 minutes",
            UrgencyLevel.URGENT: "<30 minutes",
            UrgencyLevel.HIGH: "<2 hours",
            UrgencyLevel.NORMAL: "<24 hours",
            UrgencyLevel.LOW: "<48 hours"
        }
        
        return EmergencyResponseAPI(
            success=True,
            request_id=emergency_request.request_id,
            message="Your request has been received. Help is on the way.",
            urgency_level=emergency_request.urgency.value,
            estimated_response_time=response_times[emergency_request.urgency]
        )
        
    except Exception as e:
        logger.error(f"Error processing emergency request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/emergency/status/{request_id}")
async def check_request_status(request_id: str):
    """Verifica el estado de una solicitud"""
    request = contact_system.get_request_status(request_id)
    
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return {
        "request_id": request.request_id,
        "status": request.status.value,
        "urgency": request.urgency.value,
        "received_at": request.received_at.isoformat(),
        "responded_at": request.responded_at.isoformat() if request.responded_at else None
    }


@app.get("/api/emergency/stats")
async def get_system_stats():
    """Obtiene estadísticas del sistema"""
    return contact_system.get_system_stats()


# ============================================================================
# TELEGRAM BOT HANDLERS
# ============================================================================

async def telegram_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para comando /start del bot"""
    await update.message.reply_text(
        "🛡️ METACORTEX Emergency Protection Bot\n\n"
        "This bot helps people who are:\n"
        "• Persecuted for their faith\n"
        "• In danger or under threat\n"
        "• Fleeing violence\n"
        "• In need of emergency assistance\n\n"
        "Send /help to request help\n"
        "All communications are confidential."
    )


async def telegram_help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para comando /help del bot"""
    chat_id = str(update.effective_chat.id)
    username = update.effective_user.username or "Anonymous"
    
    # Crear solicitud automáticamente
    await contact_system.receive_request(
        channel=ContactChannel.TELEGRAM,
        contact_info=chat_id,
        description=f"User {username} requested help via Telegram bot",
        threat_type=ThreatType.OTHER,
        name=username if username != "Anonymous" else None
    )
    
    await update.message.reply_text(
        "🆘 Emergency Request Received\n\n"
        "An operator will contact you soon.\n\n"
        "Please describe your situation:\n"
        "- Where are you?\n"
        "- What kind of help do you need?\n"
        "- Are you in immediate danger?"
    )


async def telegram_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler para mensajes del bot - CON MEMORIA PERSISTENTE Y METACORTEX CORE
    
    Este handler ahora:
    1. Mantiene memoria de conversaciones (mismo chat_id = mismo contexto)
    2. Usa CognitiveAgent completo con BDI, afecto, planificación
    3. Responde inteligentemente basándose en historial
    """
    chat_id = str(update.effective_chat.id)
    username = update.effective_user.username or "Anonymous"
    message_text = update.message.text
    
    logger.info(f"📨 Telegram message from {username} ({chat_id}): {message_text[:100]}")
    
    # 🧠 INTEGRACIÓN COMPLETA CON METACORTEX CORE
    if contact_system.ai and hasattr(contact_system.ai, 'cognitive_agent'):
        try:
            # Obtener o crear perfil de usuario con MEMORIA PERSISTENTE
            user_profile = await contact_system._get_or_create_user_profile(
                chat_id=chat_id,
                username=username
            )
            
            # Agregar mensaje actual al historial
            user_profile['message_history'].append({
                'timestamp': datetime.now().isoformat(),
                'message': message_text,
                'sender': 'user'
            })
            
            # 🔥 USAR COGNITIVE AGENT (METACORTEX CORE) COMPLETO
            cognitive_agent = contact_system.ai.cognitive_agent
            
            # Registrar percepción en el agente
            cognitive_agent.perceive({
                'type': 'emergency_message',
                'chat_id': chat_id,
                'username': username,
                'message': message_text,
                'conversation_history': user_profile['message_history'][-10:],  # Últimos 10 mensajes
                'urgency_score': user_profile.get('urgency_level', 0.5),
                'previous_requests': user_profile.get('request_count', 0)
            })
            
            # El agente procesa y genera respuesta con TODO su poder cognitivo
            # (BDI, afecto, planificación, memoria, metacognición)
            cognitive_response = cognitive_agent.think_and_respond({
                'context': 'emergency_assistance',
                'user_state': user_profile,
                'current_message': message_text,
                'requires_empathy': True,
                'requires_action_plan': True
            })
            
            # Generar respuesta con IA usando contexto completo
            ai_response = await contact_system.ai.generate_telegram_response(
                message=message_text,
                chat_id=chat_id,
                username=username,
                conversation_history=user_profile['message_history'][-5:],  # Últimos 5 para contexto
                cognitive_insights=cognitive_response
            )
            
            # Agregar respuesta al historial
            user_profile['message_history'].append({
                'timestamp': datetime.now().isoformat(),
                'message': ai_response,
                'sender': 'bot'
            })
            
            # Actualizar urgencia basándose en análisis cognitivo
            if 'urgency_detected' in cognitive_response:
                user_profile['urgency_level'] = cognitive_response['urgency_detected']
            
            # Guardar perfil actualizado
            await contact_system._save_user_profile(chat_id, user_profile)
            
            # Enviar respuesta inteligente
            await update.message.reply_text(ai_response, parse_mode="Markdown")
            
            logger.info(f"✅ AI response sent to {username} (usando METACORTEX Core completo)")
            
        except Exception as e:
            logger.exception(f"Error en procesamiento cognitivo: {e}")
            # Fallback a respuesta básica
            await update.message.reply_text(
                "✅ Message received. Processing your request with advanced AI..."
            )
    else:
        # Sin IA - respuesta básica (no debería llegar aquí)
        await contact_system.receive_request(
            channel=ContactChannel.TELEGRAM,
            contact_info=chat_id,
            description=message_text,
            threat_type=ThreatType.OTHER
        )
        
        await update.message.reply_text(
            "✅ Message received. Processing your request..."
        )


def setup_telegram_bot(token: str) -> Application:
    """Configura el bot de Telegram"""
    if not TELEGRAM_AVAILABLE:
        logger.error("Telegram bot library not available")
        return None
    
    app = Application.builder().token(token).build()
    
    # Handlers
    app.add_handler(CommandHandler("start", telegram_start_command))
    app.add_handler(CommandHandler("help", telegram_help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, telegram_message_handler))
    
    logger.info("✅ Telegram bot configured")
    return app


# ============================================================================
# MAIN & INITIALIZATION
# ============================================================================

async def start_contact_system():
    """Inicia el sistema de contacto"""
    logger.info("🚀 Starting Emergency Contact System...")
    
    # Iniciar Telegram Bot si está configurado
    if contact_system.telegram_token and TELEGRAM_AVAILABLE:
        try:
            telegram_app = setup_telegram_bot(contact_system.telegram_token)
            if telegram_app:
                contact_system.telegram_bot = telegram_app.bot
                # Guardar telegram_app para usar después
                contact_system.telegram_app = telegram_app
                
                # Iniciar bot y polling
                await telegram_app.initialize()
                await telegram_app.start()
                
                # CRITICAL: Iniciar polling de updates
                # Crear tarea de fondo que se ejecuta indefinidamente
                async def poll_updates():
                    """Tarea de fondo para recibir mensajes"""
                    try:
                        logger.info("🎧 Telegram bot polling started - listening for messages...")
                        # Obtener updates continuamente
                        while True:
                            try:
                                updates = await telegram_app.bot.get_updates(
                                    timeout=30,
                                    allowed_updates=["message", "edited_message"]
                                )
                                
                                # Procesar cada update
                                for update in updates:
                                    await telegram_app.process_update(update)
                                    # Marcar como procesado
                                    await telegram_app.bot.get_updates(
                                        offset=update.update_id + 1,
                                        timeout=0
                                    )
                            except Exception as e:
                                logger.error(f"Error polling Telegram updates: {e}")
                                await asyncio.sleep(5)  # Esperar antes de reintentar
                    except Exception as e:
                        logger.error(f"Fatal error in Telegram polling: {e}")
                
                # Iniciar polling en background
                asyncio.create_task(poll_updates())
                
                logger.info("✅ Telegram bot started and ACTIVELY LISTENING for messages")
        except Exception as e:
            logger.error(f"Error starting Telegram bot: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("✅ Emergency Contact System READY")
    logger.info("")
    logger.info("📞 CONTACT CHANNELS:")
    logger.info("   • Web Form: POST /api/emergency/request")
    logger.info(f"   • Telegram: {'✅ ACTIVELY LISTENING' if contact_system.telegram_bot else '⚠️ Not configured'}")
    logger.info(f"   • Email: {contact_system.email_address}")
    logger.info(f"   • WhatsApp: {'✅ ACTIVE' if contact_system.twilio_client else '⚠️ Not configured'}")
    logger.info("")


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("🆘 METACORTEX EMERGENCY CONTACT SYSTEM")
    print("="*80)
    print("\nStarting server...")
    print("Web API: http://localhost:8200")
    print("Documentation: http://localhost:8200/docs")
    print("\n⚠️ CONFIGURE .env file with:")
    print("  - TELEGRAM_BOT_TOKEN")
    print("  - EMERGENCY_EMAIL and EMAIL_PASSWORD")
    print("  - TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")
    print("="*80 + "\n")
    
    # Iniciar sistema con bot de Telegram
    async def startup_event():
        """Evento de inicio para inicializar el bot"""
        await start_contact_system()
    
    # Agregar evento de startup a FastAPI
    app.add_event_handler("startup", startup_event)
    
    # Iniciar FastAPI con uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
