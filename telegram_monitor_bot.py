#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
METACORTEX TELEGRAM MONITOR BOT
═══════════════════════════════════════════════════════════════════════════
Bot de Telegram enterprise-grade para monitoreo remoto de:
- 956+ Modelos ML activos
- Sistema Autónomo de Orquestación
- Métricas en tiempo real
- Control remoto del sistema
- Alertas automáticas

Comandos:
- /status - Status completo del sistema
- /models - Lista de modelos activos
- /tasks - Tareas activas y pendientes
- /stats - Estadísticas detalladas
- /help - Ayuda y comandos disponibles

Autor: METACORTEX System
Versión: 1.0.0 - Enterprise Grade
═══════════════════════════════════════════════════════════════════════════
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Telegram
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters
)

# Singleton Registry
try:
    from singleton_registry import get_autonomous_orchestrator
    SINGLETON_AVAILABLE = True
except ImportError:
    SINGLETON_AVAILABLE = False
    print("⚠️  Singleton registry not available - limited functionality")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_orchestrator():
    """Obtiene instancia del orchestrator via singleton registry."""
    if not SINGLETON_AVAILABLE:
        return None
    
    try:
        return get_autonomous_orchestrator()
    except Exception as e:
        logger.error(f"Error getting orchestrator: {e}")
        return None

def format_number(num: int) -> str:
    """Formatea número con separadores de miles."""
    return f"{num:,}"

def format_percentage(value: float) -> str:
    """Formatea porcentaje."""
    return f"{value * 100:.1f}%"

def get_emoji_for_status(status: str) -> str:
    """Obtiene emoji según el status."""
    emojis = {
        "operational": "✅",
        "error": "❌",
        "warning": "⚠️",
        "unavailable": "🔴"
    }
    return emojis.get(status, "❓")

# ═══════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para /start."""
    welcome_message = """
🤖 **METACORTEX Monitor Bot**

Bienvenido al sistema de monitoreo enterprise para el orquestador autónomo de 956+ modelos ML.

**Comandos disponibles:**
/status - Status completo del sistema
/models - Información de modelos activos
/tasks - Tareas en ejecución
/stats - Estadísticas detalladas
/help - Esta ayuda

**Sistema:** METACORTEX Autonomous System
**Versión:** 1.0.0 Enterprise Grade
**Estado:** Operacional 🟢
    """
    
    await update.message.reply_text(welcome_message, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para /help."""
    help_message = """
📚 **Ayuda - METACORTEX Monitor Bot**

**Comandos básicos:**
• `/status` - Muestra el estado actual del sistema
• `/models` - Lista todos los modelos ML activos
• `/tasks` - Muestra tareas activas y pendientes
• `/stats` - Estadísticas detalladas del sistema

**Información técnica:**
• 956+ Modelos ML entrenados y activos
• Sistema de orquestación autónoma
• Ejecución paralela de hasta 50 tareas
• Auto-optimización y self-healing
• Integración con ML Pipeline, Ollama, World Model

**Especialidades de modelos:**
• Regression & Classification
• Time Series & Forecasting
• NLP & Vision
• Programming & Analysis
• Optimization & Prediction

**Support:** @metacortex_divine_bot
    """
    
    await update.message.reply_text(help_message, parse_mode="Markdown")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para /status."""
    orchestrator = get_orchestrator()
    
    if orchestrator is None:
        await update.message.reply_text(
            "❌ **Sistema no disponible**\n\n"
            "El orquestador autónomo no está disponible en este momento.",
            parse_mode="Markdown"
        )
        return
    
    try:
        status = orchestrator.get_status()
        
        # Determinar emoji de status
        status_emoji = "🟢" if status.get("total_models", 0) > 0 else "🔴"
        
        message = f"""
🤖 **METACORTEX System Status**
{status_emoji} **Estado:** Operacional

📊 **Métricas Principales:**
• Modelos Activos: **{format_number(status.get('total_models', 0))}**
• Cola de Tareas: **{format_number(status.get('queue_size', 0))}**
• Tareas Activas: **{format_number(status.get('active_tasks', 0))}**
• Completadas: **{format_number(status.get('completed_tasks', 0))}**
• Fallidas: **{format_number(status.get('failed_tasks', 0))}**
• Success Rate: **{format_percentage(status.get('success_rate', 0))}**

🕐 **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Error in status command: {e}")
        await update.message.reply_text(
            f"❌ Error obteniendo status: {str(e)}",
            parse_mode="Markdown"
        )

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para /models."""
    orchestrator = get_orchestrator()
    
    if orchestrator is None:
        await update.message.reply_text(
            "❌ Sistema no disponible",
            parse_mode="Markdown"
        )
        return
    
    try:
        status = orchestrator.get_status()
        specs = status.get('models_by_specialization', {})
        
        message = "🧠 **Modelos por Especialización**\n\n"
        
        for spec, count in sorted(specs.items(), key=lambda x: x[1], reverse=True):
            message += f"• **{spec}:** {count} modelos\n"
        
        message += f"\n📊 **Total:** {format_number(status.get('total_models', 0))} modelos activos"
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Error in models command: {e}")
        await update.message.reply_text(
            f"❌ Error: {str(e)}",
            parse_mode="Markdown"
        )

async def tasks_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para /tasks."""
    orchestrator = get_orchestrator()
    
    if orchestrator is None:
        await update.message.reply_text(
            "❌ Sistema no disponible",
            parse_mode="Markdown"
        )
        return
    
    try:
        status = orchestrator.get_status()
        
        message = "⚡ **Tareas del Sistema**\n\n"
        message += f"📝 **En Cola:** {status.get('queue_size', 0)}\n"
        message += f"⚡ **Activas:** {status.get('active_tasks', 0)}\n"
        message += f"✅ **Completadas:** {format_number(status.get('completed_tasks', 0))}\n"
        message += f"❌ **Fallidas:** {format_number(status.get('failed_tasks', 0))}\n\n"
        
        # Tareas activas
        active_details = status.get('active_tasks_details', [])
        if active_details:
            message += "🔥 **Tareas Activas Ahora:**\n\n"
            for task in active_details[:5]:  # Solo primeras 5
                message += f"• `{task.get('task_id', 'N/A')[:8]}`\n"
                message += f"  {task.get('description', 'N/A')[:50]}...\n"
                message += f"  📍 {task.get('specialization', 'N/A')}\n\n"
        else:
            message += "✨ No hay tareas activas en este momento\n"
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Error in tasks command: {e}")
        await update.message.reply_text(
            f"❌ Error: {str(e)}",
            parse_mode="Markdown"
        )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para /stats."""
    orchestrator = get_orchestrator()
    
    if orchestrator is None:
        await update.message.reply_text(
            "❌ Sistema no disponible",
            parse_mode="Markdown"
        )
        return
    
    try:
        status = orchestrator.get_status()
        
        # Calcular estadísticas
        total_tasks = status.get('completed_tasks', 0) + status.get('failed_tasks', 0)
        success_rate = status.get('success_rate', 0)
        
        message = f"""
📈 **Estadísticas Detalladas**

🎯 **Performance:**
• Total Tareas Procesadas: **{format_number(total_tasks)}**
• Success Rate: **{format_percentage(success_rate)}**
• Tareas Exitosas: **{format_number(status.get('completed_tasks', 0))}**
• Tareas Fallidas: **{format_number(status.get('failed_tasks', 0))}**

🧠 **Modelos:**
• Modelos Activos: **{format_number(status.get('total_models', 0))}**
• Especializaciones: **{len(status.get('models_by_specialization', {}))}**

⚡ **Sistema:**
• Tareas en Cola: **{status.get('queue_size', 0)}**
• Tareas Activas: **{status.get('active_tasks', 0)}**
• Max Paralelo: **50 tareas**

🕐 **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Error in stats command: {e}")
        await update.message.reply_text(
            f"❌ Error: {str(e)}",
            parse_mode="Markdown"
        )

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para comandos desconocidos."""
    await update.message.reply_text(
        "❓ Comando no reconocido.\n\n"
        "Usa /help para ver comandos disponibles.",
        parse_mode="Markdown"
    )

# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLER
# ═══════════════════════════════════════════════════════════════════════════

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler global de errores."""
    logger.error(f"Update {update} caused error: {context.error}")
    
    if update and update.message:
        await update.message.reply_text(
            "❌ Ocurrió un error procesando tu comando.\n\n"
            "Por favor intenta nuevamente.",
            parse_mode="Markdown"
        )

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Función principal."""
    # Obtener token del bot
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        logger.error("❌ TELEGRAM_BOT_TOKEN no configurado!")
        logger.info("💡 Configura el token con: export TELEGRAM_BOT_TOKEN='tu_token'")
        return
    
    logger.info("🤖 METACORTEX Telegram Monitor Bot iniciando...")
    
    # Crear aplicación
    app = Application.builder().token(token).build()
    
    # Registrar command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(CommandHandler("tasks", tasks_command))
    app.add_handler(CommandHandler("stats", stats_command))
    
    # Handler para mensajes desconocidos
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    # Error handler
    app.add_error_handler(error_handler)
    
    logger.info("✅ Bot configurado correctamente")
    logger.info("📡 Iniciando polling...")
    logger.info("🔗 Telegram: @metacortex_divine_bot")
    
    # Iniciar bot
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
