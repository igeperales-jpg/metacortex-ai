"""
📱 WHATSAPP INTEGRATION - Integración de WhatsApp con IA
=========================================================

Sistema para recibir mensajes de WhatsApp a través de Twilio
y responder con inteligencia artificial.

Características:
- Recepción de mensajes vía Twilio
- Respuestas inteligentes con Ollama
- Integración con Divine Protection
- Webhooks para mensajes entrantes

Autor: METACORTEX AI Team
Fecha: 24 de Noviembre de 2025
"""

import logging
import os
from typing import Optional
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import Response
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse

logger = logging.getLogger(__name__)


class WhatsAppBot:
    """Bot de WhatsApp usando Twilio"""
    
    def __init__(self, ai_integration=None):
        self.ai = ai_integration
        
        # Configurar Twilio
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        
        if account_sid and auth_token:
            self.client = TwilioClient(account_sid, auth_token)
            logger.info("✅ WhatsApp bot configured with Twilio")
        else:
            self.client = None
            logger.warning("⚠️ Twilio credentials not found")
    
    async def handle_incoming_message(
        self,
        from_number: str,
        message_body: str
    ) -> str:
        """
        Procesa mensaje entrante y genera respuesta
        
        Args:
            from_number: Número de teléfono del remitente
            message_body: Contenido del mensaje
            
        Returns:
            Respuesta a enviar
        """
        
        logger.info(f"📱 WhatsApp message from {from_number}: {message_body[:100]}")
        
        if self.ai:
            # Generar respuesta con IA
            try:
                response = await self.ai.generate_whatsapp_response(
                    message=message_body,
                    phone_number=from_number
                )
                return response
            except Exception as e:
                logger.exception(f"Error generating AI response: {e}")
                return self._get_fallback_response(message_body)
        else:
            return self._get_fallback_response(message_body)
    
    def _get_fallback_response(self, message: str) -> str:
        """Respuesta de respaldo sin IA"""
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["help", "ayuda", "emergency"]):
            return ("🆘 METACORTEX Divine Protection\n\n"
                   "Your emergency request has been received. "
                   "A team member will contact you shortly. "
                   "You are not alone.")
        else:
            return ("✅ Message received.\n\n"
                   "Thank you for contacting METACORTEX. "
                   "Your message is being processed.")
    
    async def send_message(self, to_number: str, message: str) -> bool:
        """
        Envía mensaje de WhatsApp
        
        Args:
            to_number: Número de destino (formato: whatsapp:+1234567890)
            message: Mensaje a enviar
            
        Returns:
            True si se envió correctamente
        """
        
        if not self.client:
            logger.error("Twilio client not configured")
            return False
        
        try:
            # Asegurar formato correcto
            if not to_number.startswith("whatsapp:"):
                to_number = f"whatsapp:{to_number}"
            
            message_obj = self.client.messages.create(
                from_=self.from_number,
                to=to_number,
                body=message
            )
            
            logger.info(f"✅ WhatsApp message sent to {to_number}: {message_obj.sid}")
            return True
            
        except Exception as e:
            logger.exception(f"Error sending WhatsApp message: {e}")
            return False


def create_whatsapp_webhook_router(whatsapp_bot: WhatsAppBot) -> FastAPI:
    """
    Crea router de FastAPI para webhooks de WhatsApp
    
    Args:
        whatsapp_bot: Instancia de WhatsAppBot
        
    Returns:
        FastAPI app con endpoints de webhook
    """
    
    app = FastAPI()
    
    @app.post("/whatsapp/webhook")
    async def whatsapp_webhook(
        From: str = Form(...),
        Body: str = Form(...),
        MessageSid: str = Form(...)
    ):
        """
        Webhook para mensajes entrantes de WhatsApp
        
        Twilio enviará POST requests aquí cuando lleguen mensajes
        """
        
        logger.info(f"Webhook received: {MessageSid} from {From}")
        
        try:
            # Procesar mensaje con IA
            response_text = await whatsapp_bot.handle_incoming_message(
                from_number=From,
                message_body=Body
            )
            
            # Crear respuesta TwiML
            twiml_response = MessagingResponse()
            twiml_response.message(response_text)
            
            return Response(
                content=str(twiml_response),
                media_type="application/xml"
            )
            
        except Exception as e:
            logger.exception(f"Error processing webhook: {e}")
            # Respuesta de error
            twiml_response = MessagingResponse()
            twiml_response.message("Sorry, an error occurred. Please try again.")
            return Response(
                content=str(twiml_response),
                media_type="application/xml"
            )
    
    @app.get("/whatsapp/status")
    async def whatsapp_status():
        """Estado del bot de WhatsApp"""
        return {
            "status": "active" if whatsapp_bot.client else "not_configured",
            "from_number": whatsapp_bot.from_number
        }
    
    return app


# ============================================================================
# TESTING
# ============================================================================

async def test_whatsapp():
    """Test del sistema de WhatsApp"""
    
    # Crear bot
    bot = WhatsAppBot()
    
    # Test mensaje
    response = await bot.handle_incoming_message(
        from_number="whatsapp:+1234567890",
        message_body="I need help, I'm being persecuted"
    )
    
    print("Response:", response)


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_whatsapp())
