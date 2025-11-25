#!/bin/bash
# ============================================================================
# 🚀 QUICK START - Inicio Rápido de METACORTEX con IA Completa
# ============================================================================

echo "============================================================================"
echo "🚀 METACORTEX QUICK START"
echo "============================================================================"
echo ""

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directorio del proyecto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ============================================================================
# 1. VERIFICAR DEPENDENCIAS
# ============================================================================

echo -e "${BLUE}[1/6] Verificando dependencias...${NC}"

# Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 no encontrado${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 encontrado: $(python3 --version)${NC}"

# Pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 no encontrado${NC}"
    exit 1
fi
echo -e "${GREEN}✅ pip3 encontrado${NC}"

# Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠️  Ollama no encontrado${NC}"
    echo "   Instalando Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install ollama 2>/dev/null || {
            echo -e "${RED}❌ No se pudo instalar Ollama con Homebrew${NC}"
            echo "   Instala manualmente desde: https://ollama.com"
            exit 1
        }
    else
        # Linux
        curl -fsSL https://ollama.com/install.sh | sh
    fi
fi
echo -e "${GREEN}✅ Ollama encontrado${NC}"

# ============================================================================
# 2. INSTALAR DEPENDENCIAS PYTHON
# ============================================================================

echo ""
echo -e "${BLUE}[2/6] Instalando/Actualizando dependencias Python...${NC}"

# Activar venv si existe
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✅ Entorno virtual activado${NC}"
else
    echo -e "${YELLOW}⚠️  Creando entorno virtual...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    echo -e "${GREEN}✅ Entorno virtual creado y activado${NC}"
fi

# Instalar dependencias
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✅ Dependencias instaladas${NC}"

# ============================================================================
# 3. VERIFICAR OLLAMA Y MODELOS
# ============================================================================

echo ""
echo -e "${BLUE}[3/6] Verificando Ollama y modelos...${NC}"

# Iniciar Ollama si no está corriendo
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Iniciando servidor Ollama...${NC}"
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Verificar modelos
MODELS_NEEDED=("mistral:latest" "mistral:instruct" "mistral-nemo:latest")
for model in "${MODELS_NEEDED[@]}"; do
    if ollama list | grep -q "$model"; then
        echo -e "${GREEN}✅ Modelo $model disponible${NC}"
    else
        echo -e "${YELLOW}⚠️  Descargando $model...${NC}"
        ollama pull "$model"
    fi
done

# ============================================================================
# 4. VERIFICAR CONFIGURACIÓN
# ============================================================================

echo ""
echo -e "${BLUE}[4/6] Verificando configuración...${NC}"

# Verificar .env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  Archivo .env no encontrado${NC}"
    echo "   Creando .env de ejemplo..."
    cat > .env << 'EOF'
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_token_here

# WhatsApp (Twilio)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# Email
EMERGENCY_EMAIL=emergency@metacortex.ai
EMAIL_PASSWORD=your_email_password

# Sistema
ENVIRONMENT=development
DEBUG=true
EOF
    echo -e "${YELLOW}   ⚠️  Por favor configura .env con tus credenciales${NC}"
    echo -e "${YELLOW}   Presiona ENTER para continuar (el sistema usará fallbacks)...${NC}"
    read
fi

if [ -f ".env" ]; then
    if grep -q "your_token_here" .env; then
        echo -e "${YELLOW}⚠️  .env necesita configuración${NC}"
    else
        echo -e "${GREEN}✅ Archivo .env configurado${NC}"
    fi
fi

# Crear directorios necesarios
mkdir -p logs emergency_requests ml_models
echo -e "${GREEN}✅ Directorios creados${NC}"

# ============================================================================
# 5. DETENER PROCESOS ANTERIORES
# ============================================================================

echo ""
echo -e "${BLUE}[5/6] Deteniendo procesos anteriores...${NC}"

# Detener Emergency Contact System
pkill -9 -f "python.*emergency_contact_system.py" 2>/dev/null && \
    echo -e "${GREEN}✅ Emergency Contact System detenido${NC}" || \
    echo -e "${YELLOW}   (no había procesos corriendo)${NC}"

# Detener Unified System
pkill -9 -f "python.*unified_startup.py" 2>/dev/null && \
    echo -e "${GREEN}✅ Unified System detenido${NC}" || \
    echo -e "${YELLOW}   (no había procesos corriendo)${NC}"

sleep 2

# ============================================================================
# 6. INICIAR SISTEMA UNIFICADO
# ============================================================================

echo ""
echo -e "${BLUE}[6/6] Iniciando Sistema Unificado...${NC}"
echo ""

# Limpiar logs antiguos
> logs/unified_system.log
> logs/emergency_contact_stdout.log

# Iniciar sistema
echo -e "${GREEN}🚀 Iniciando METACORTEX Unified System...${NC}"
echo ""

nohup .venv/bin/python3 unified_startup.py > logs/unified_system.log 2>&1 &
UNIFIED_PID=$!

# Esperar a que inicie
echo "⏳ Esperando inicio del sistema..."
sleep 5

# Verificar que está corriendo
if kill -0 $UNIFIED_PID 2>/dev/null; then
    echo ""
    echo "============================================================================"
    echo -e "${GREEN}✅ SISTEMA INICIADO EXITOSAMENTE${NC}"
    echo "============================================================================"
    echo ""
    echo "📊 INFORMACIÓN DEL SISTEMA:"
    echo "   • PID: $UNIFIED_PID"
    echo "   • Web Interface: http://localhost:8080"
    echo "   • API Status: http://localhost:8080/api/status"
    echo "   • Logs: tail -f logs/unified_system.log"
    echo ""
    echo "📞 CANALES DE CONTACTO:"
    echo "   • Telegram Bot: @metacortex_divine_bot"
    echo "   • Web Form: http://localhost:8080"
    echo "   • Email: emergency@metacortex.ai"
    echo ""
    echo "🧠 SISTEMAS ACTIVOS:"
    echo "   • AI Integration Layer (Ollama)"
    echo "   • Divine Protection System"
    echo "   • Emergency Contact System"
    echo "   • Telegram Bot (con IA)"
    echo "   • WhatsApp Bot (si está configurado)"
    echo "   • Web Interface"
    echo ""
    echo "🔍 VERIFICACIÓN:"
    
    # Esperar un poco más para que todo inicie
    sleep 3
    
    # Verificar web interface
    if curl -s http://localhost:8080/api/status > /dev/null 2>&1; then
        echo -e "   ${GREEN}✅ Web Interface: FUNCIONANDO${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Web Interface: Iniciando...${NC}"
    fi
    
    # Verificar Ollama
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "   ${GREEN}✅ Ollama: FUNCIONANDO${NC}"
    else
        echo -e "   ${RED}❌ Ollama: NO RESPONDE${NC}"
    fi
    
    echo ""
    echo "============================================================================"
    echo ""
    echo "💡 COMANDOS ÚTILES:"
    echo ""
    echo "   Ver logs en tiempo real:"
    echo "   $ tail -f logs/unified_system.log"
    echo ""
    echo "   Verificar estado:"
    echo "   $ curl http://localhost:8080/api/status | json_pp"
    echo ""
    echo "   Detener sistema:"
    echo "   $ kill $UNIFIED_PID"
    echo ""
    echo "   Ver este PID más tarde:"
    echo "   $ ps aux | grep unified_startup"
    echo ""
    echo "============================================================================"
    echo ""
    echo -e "${GREEN}🛡️  Divine Protection System ACTIVO - Listo para ayudar${NC}"
    echo ""
    
    # Ofrecer ver logs
    echo -e "${YELLOW}¿Deseas ver los logs en tiempo real? (y/n)${NC}"
    read -t 5 -n 1 response
    echo ""
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Presiona Ctrl+C para salir..."
        sleep 1
        tail -f logs/unified_system.log
    fi
    
else
    echo ""
    echo "============================================================================"
    echo -e "${RED}❌ ERROR AL INICIAR EL SISTEMA${NC}"
    echo "============================================================================"
    echo ""
    echo "Ver logs para más información:"
    echo "$ tail -50 logs/unified_system.log"
    echo ""
    exit 1
fi
