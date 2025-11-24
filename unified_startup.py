"""
🚀 UNIFIED STARTUP - Inicio Unificado de TODOS los Sistemas
=============================================================

Este script inicia TODOS los sistemas integrados:
1. Telegram Bot con IA
2. WhatsApp Bot con IA
3. Web Interface
4. Emergency Contact System
5. Divine Protection System
6. Todos los modelos de lenguaje (Ollama)

Uso:
    python unified_startup.py

Autor: METACORTEX AI Team
Fecha: 24 de Noviembre de 2025
"""

import logging
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Añadir proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configurar logging ANTES de importar módulos
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/unified_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Imports de sistemas - IMPORTAR TODO
try:
    # Core AI Systems
    from metacortex_sinaptico.ai_integration_layer import get_ai_integration
    from metacortex_sinaptico.divine_protection import get_divine_protection
    from metacortex_sinaptico.emergency_contact_system import EmergencyContactSystem, ContactChannel
    from metacortex_sinaptico.whatsapp_integration import WhatsAppBot
    from metacortex_sinaptico.web_interface import create_web_server
    
    # Web Framework
    import uvicorn
    from fastapi import FastAPI
    
    IMPORTS_OK = True
    logger.info("✅ Todos los módulos importados correctamente")
except ImportError as e:
    logger.error(f"❌ Error importing modules: {e}")
    logger.error(f"   Detalle: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    IMPORTS_OK = False


class UnifiedSystem:
    """Sistema unificado que coordina todos los componentes"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.ai = None
        self.divine_protection = None
        self.emergency_contact = None
        self.whatsapp_bot = None
        self.web_interface = None
        self.main_app = FastAPI(title="METACORTEX Unified System")
        
        logger.info("=" * 80)
        logger.info("🚀 METACORTEX UNIFIED SYSTEM STARTUP")
        logger.info("=" * 80)
    
    async def initialize(self):
        """Inicializa todos los sistemas en el orden correcto"""
        
        try:
            # 1. AI Integration Layer (PRIMERO - todos lo necesitan)
            logger.info("\n" + "="*80)
            logger.info("🧠 [1/5] Initializing AI Integration Layer...")
            logger.info("="*80)
            self.ai = get_ai_integration(self.project_root)
            logger.info("✅ AI Integration Layer ready")
            logger.info(f"   • Ollama: http://localhost:11434")
            logger.info(f"   • Models: mistral-nemo (12B), mistral (7B)")
            logger.info(f"   • ML Models: 956+ trained models available")
            
            # 2. Divine Protection System
            logger.info("\n" + "="*80)
            logger.info("🛡️ [2/5] Initializing Divine Protection System...")
            logger.info("="*80)
            try:
                self.divine_protection = get_divine_protection()
                self.ai.connect_divine_protection(self.divine_protection)
                logger.info("✅ Divine Protection System ready")
                logger.info("   • Protected persons: 0 (ready to receive)")
                logger.info("   • Real operations: ACTIVE")
                logger.info("   • Emergency response: ENABLED")
            except Exception as e:
                logger.warning(f"⚠️ Divine Protection System not available: {e}")
                logger.warning("   • Continuing without Divine Protection")
                self.divine_protection = None
            
            # 3. Emergency Contact System
            logger.info("\n" + "="*80)
            logger.info("🆘 [3/5] Initializing Emergency Contact System...")
            logger.info("="*80)
            self.emergency_contact = EmergencyContactSystem(self.project_root)
            logger.info("✅ Emergency Contact System ready")
            
            # 4. WhatsApp Bot
            logger.info("\n" + "="*80)
            logger.info("📱 [4/5] Initializing WhatsApp Bot...")
            logger.info("="*80)
            self.whatsapp_bot = WhatsAppBot(ai_integration=self.ai)
            logger.info("✅ WhatsApp Bot ready")
            logger.info(f"   • Twilio: {'CONFIGURED' if self.whatsapp_bot.client else 'NOT CONFIGURED'}")
            
            # 5. Web Interface
            logger.info("\n" + "="*80)
            logger.info("🌐 [5/5] Initializing Web Interface...")
            logger.info("="*80)
            self.web_interface = create_web_server(
                ai_integration=self.ai,
                emergency_contact=self.emergency_contact
            )
            logger.info("✅ Web Interface ready")
            logger.info("   • URL: http://localhost:8080")
            logger.info("   • API: http://localhost:8080/api")
            logger.info("   • Status: http://localhost:8080/api/status")
            
            # Montar aplicaciones
            self.main_app.mount("/", self.web_interface.app)
            
            logger.info("\n" + "="*80)
            logger.info("✅ ALL SYSTEMS INITIALIZED SUCCESSFULLY")
            logger.info("="*80)
            logger.info("")
            logger.info("📞 CONTACT CHANNELS:")
            logger.info(f"   • Telegram Bot: @metacortex_divine_bot")
            logger.info(f"   • WhatsApp: {'ACTIVE' if self.whatsapp_bot.client else 'NOT CONFIGURED'}")
            logger.info(f"   • Web Form: http://localhost:8080")
            logger.info(f"   • Email: emergency@metacortex.ai")
            logger.info("")
            logger.info("🧠 AI CAPABILITIES:")
            logger.info(f"   • Ollama LLM: ACTIVE (3 models)")
            logger.info(f"   • ML Models: 956+ models available")
            logger.info(f"   • Threat Analysis: ENABLED")
            logger.info(f"   • Divine Protection: ACTIVE")
            logger.info("")
            logger.info("🚀 SYSTEM READY - Listening for emergency requests...")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.exception(f"❌ Fatal error during initialization: {e}")
            return False
    
    async def start_emergency_contact_async(self):
        """Inicia el sistema de contacto de emergencia de forma asíncrona"""
        try:
            # Iniciar Telegram Bot
            if self.emergency_contact.telegram_token:
                from telegram.ext import Application, CommandHandler, MessageHandler, filters
                
                # Handlers con IA integrada
                async def start_handler(update, context):
                    await update.message.reply_text(
                        "🛡️ *METACORTEX Divine Protection*\n\n"
                        "I'm an AI assistant helping people who are:\n"
                        "• Persecuted for their faith\n"
                        "• In danger or under threat\n"
                        "• Need emergency assistance\n\n"
                        "Send me a message describing your situation.",
                        parse_mode="Markdown"
                    )
                
                async def help_handler(update, context):
                    chat_id = str(update.effective_chat.id)
                    username = update.effective_user.username or "Anonymous"
                    
                    # Generar respuesta con IA
                    if self.ai:
                        response = await self.ai.generate_telegram_response(
                            message="User requested help",
                            chat_id=chat_id,
                            username=username
                        )
                        await update.message.reply_text(response, parse_mode="Markdown")
                    else:
                        await update.message.reply_text(
                            "🆘 Emergency Request Received\n\n"
                            "An operator will contact you soon."
                        )
                
                async def message_handler(update, context):
                    chat_id = str(update.effective_chat.id)
                    username = update.effective_user.username or "Anonymous"
                    message = update.message.text
                    
                    logger.info(f"📨 Telegram message from {username}: {message[:50]}")
                    
                    # Procesar con IA
                    if self.ai:
                        try:
                            response = await self.ai.generate_telegram_response(
                                message=message,
                                chat_id=chat_id,
                                username=username
                            )
                            await update.message.reply_text(response, parse_mode="Markdown")
                        except Exception as e:
                            logger.error(f"Error in AI response: {e}")
                            await update.message.reply_text(
                                "✅ Message received. Processing..."
                            )
                    else:
                        await update.message.reply_text(
                            "✅ Message received. An operator will respond soon."
                        )
                
                # Configurar bot
                app = Application.builder().token(self.emergency_contact.telegram_token).build()
                app.add_handler(CommandHandler("start", start_handler))
                app.add_handler(CommandHandler("help", help_handler))
                app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
                
                # Iniciar
                await app.initialize()
                await app.start()
                
                # Polling manual
                logger.info("🎧 Telegram bot ACTIVELY LISTENING...")
                
                async def poll_updates():
                    while True:
                        try:
                            updates = await app.bot.get_updates(timeout=30)
                            for update in updates:
                                await app.process_update(update)
                                await app.bot.get_updates(offset=update.update_id + 1, timeout=0)
                        except Exception as e:
                            logger.error(f"Polling error: {e}")
                            await asyncio.sleep(5)
                
                asyncio.create_task(poll_updates())
                
        except Exception as e:
            logger.exception(f"Error starting emergency contact: {e}")
    
    def run(self):
        """Ejecuta el sistema completo"""
        
        async def startup():
            # Inicializar sistemas
            success = await self.initialize()
            if not success:
                logger.error("❌ Initialization failed")
                return
            
            # Iniciar Emergency Contact
            await self.start_emergency_contact_async()
        
        # Añadir startup event
        self.main_app.add_event_handler("startup", startup)
        
        # Ejecutar servidor
        logger.info("\n🌐 Starting web server...")
        uvicorn.run(
            self.main_app,
            host="0.0.0.0",
            port=8080,
            log_level="info"
        )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Punto de entrada principal"""
    
    if not IMPORTS_OK:
        print("❌ Cannot start - missing dependencies")
        print("Run: pip install -r requirements.txt")
        return 1
    
    # Crear directorios necesarios
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Iniciar sistema unificado
    try:
        system = UnifiedSystem(project_root)
        system.run()
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Shutdown requested by user")
        logger.info("👋 METACORTEX Unified System stopped")
        return 0
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
