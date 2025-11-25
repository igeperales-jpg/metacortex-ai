#!/bin/bash

# Script para iniciar el servidor de emergencias de METACORTEX
# Este servidor maneja todos los canales de contacto con personas necesitadas

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

echo "🚨 METACORTEX - Sistema de Contacto de Emergencia"
echo "=================================================="
echo ""

# Verificar que exista .env
if [ ! -f .env ]; then
    echo "⚠️  Archivo .env no encontrado"
    echo ""
    echo "Copiando .env.example a .env..."
    cp .env.example .env
    echo "✅ Archivo .env creado"
    echo ""
    echo "⚠️  IMPORTANTE: Debes configurar al menos UN canal de comunicación:"
    echo "   1. Abrir .env en tu editor"
    echo "   2. Configurar TELEGRAM_BOT_TOKEN (recomendado)"
    echo "   3. O configurar TWILIO para WhatsApp"
    echo "   4. O configurar EMAIL_PASSWORD para email"
    echo ""
    read -p "¿Deseas continuar sin configurar? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "Configuración cancelada. Por favor configura .env antes de continuar."
        exit 1
    fi
fi

# Verificar puerto
PORT=${EMERGENCY_PORT:-8200}
echo "🔍 Verificando puerto $PORT..."

if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Puerto $PORT ya está en uso"
    PID=$(lsof -t -i:$PORT)
    echo "   Proceso: $PID"
    read -p "¿Deseas detener el proceso anterior? (y/N): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        kill $PID
        sleep 2
        echo "✅ Proceso anterior detenido"
    else
        echo "❌ No se puede iniciar el servidor con el puerto ocupado"
        exit 1
    fi
fi

echo ""
echo "📋 Configuración:"
echo "   Puerto: $PORT"
echo "   Host: 0.0.0.0 (accesible desde red)"
echo "   Logs: metacortex_emergency.log"
echo ""

# Crear directorio para solicitudes si no existe
mkdir -p emergency_requests

echo "🚀 Iniciando servidor de emergencias..."
echo ""
echo "📡 El servidor estará disponible en:"
echo "   • Local: http://localhost:$PORT"
echo "   • Red: http://$(hostname -I | awk '{print $1}'):$PORT"
echo "   • API: http://localhost:$PORT/docs (documentación interactiva)"
echo ""
echo "🆘 Canales de contacto activos:"
echo "   ✅ Formulario Web: POST http://localhost:$PORT/api/emergency/request"

# Verificar configuración de canales
if grep -q "TELEGRAM_BOT_TOKEN=.*[^x]" .env 2>/dev/null && ! grep -q "TELEGRAM_BOT_TOKEN=1234567890" .env 2>/dev/null; then
    echo "   ✅ Telegram Bot: Configurado"
else
    echo "   ⏸️  Telegram Bot: No configurado"
fi

if grep -q "TWILIO_ACCOUNT_SID=.*[^x]" .env 2>/dev/null && ! grep -q "TWILIO_ACCOUNT_SID=ACxxx" .env 2>/dev/null; then
    echo "   ✅ WhatsApp (Twilio): Configurado"
else
    echo "   ⏸️  WhatsApp (Twilio): No configurado"
fi

if grep -q "EMAIL_PASSWORD=.*[^r]" .env 2>/dev/null && ! grep -q "EMAIL_PASSWORD=your_app_specific_password" .env 2>/dev/null; then
    echo "   ✅ Email: Configurado"
else
    echo "   ⏸️  Email: No configurado"
fi

echo ""
echo "📝 Para probar el sistema:"
echo "   python scripts/test_emergency_contact.py"
echo ""
echo "🛑 Para detener el servidor: Ctrl+C"
echo ""
echo "=================================================="
echo ""

# Iniciar servidor con uvicorn
uvicorn metacortex_sinaptico.emergency_contact_system:app \
    --host 0.0.0.0 \
    --port $PORT \
    --reload \
    --log-level info \
    --access-log \
    2>&1 | tee metacortex_emergency.log
