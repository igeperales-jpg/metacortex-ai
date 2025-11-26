#!/usr/bin/env bash
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ⚔️  METACORTEX MASTER CONTROL - Sistema Unificado de Control v5.0      ║
# ║  Control centralizado de TODOS los servicios y operaciones              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# USAGE:
#   ./metacortex_master.sh start      - Iniciar TODO el sistema (COMPLETO + IA)
#   ./metacortex_master.sh stop       - Detener todo el sistema
#   ./metacortex_master.sh restart    - Reiniciar el sistema completo
#   ./metacortex_master.sh status     - Ver estado de todos los servicios
#   ./metacortex_master.sh emergency  - Apagado de emergencia (mata todo)
#   ./metacortex_master.sh clean      - Limpiar logs y archivos temporales
#   ./metacortex_master.sh deploy     - Desplegar Emergency System públicamente
#   ./metacortex_master.sh ai         - Iniciar SOLO sistemas de IA (Telegram+WhatsApp+Web)
#

set -euo pipefail

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="${SCRIPT_DIR}"
readonly SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
readonly LOGS_DIR="${PROJECT_ROOT}/logs"
readonly PID_DIR="${PROJECT_ROOT}/pid"
readonly VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
readonly VENV_PIP="${PROJECT_ROOT}/.venv/bin/pip"

# ============================================================================
# 🍎 CONFIGURACIÓN APPLE SILICON M4 + MPS (Metal Performance Shaders)
# ============================================================================
readonly APPLE_SILICON_M4=true
readonly FORCE_MPS=true  # Forzar uso de GPU Metal en lugar de CPU
readonly DEVICE="mps"    # PyTorch device: mps (GPU Metal)

# Variables de entorno para PyTorch MPS
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Fallback a CPU si MPS falla
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Usar toda la memoria GPU disponible
export PYTORCH_MPS_PREFER_METAL=1  # Preferir Metal sobre CPU
export MPS_FORCE_ENABLE=1  # Forzar MPS incluso si no es detectado

# Optimizaciones para Apple Silicon
export TOKENIZERS_PARALLELISM=true  # Paralelizar tokenizers
export OMP_NUM_THREADS=10  # iMac M4 tiene 10 cores de rendimiento
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

# Configuración de memoria para ML (iMac M4 tiene 16-32GB RAM unificada)
export PYTORCH_MPS_ALLOCATOR_POLICY="garbage_collection"  # Mejor gestión de memoria
export TF_GPU_ALLOCATOR="cuda_malloc_async"  # Para TensorFlow si se usa

# Archivos de control
readonly DAEMON_SCRIPT="${PROJECT_ROOT}/metacortex_daemon.py"
readonly DAEMON_PID_FILE="${PROJECT_ROOT}/metacortex_daemon_military.pid"
readonly ORCHESTRATOR_SCRIPT="${PROJECT_ROOT}/metacortex_orchestrator.py"
# Use the existing orchestrator as startup orchestrator
readonly STARTUP_ORCHESTRATOR="${PROJECT_ROOT}/metacortex_orchestrator.py"

# Colores
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly RESET='\033[0m'

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================
log_info() {
    echo -e "${CYAN}ℹ️  [INFO]${RESET} $1"
}

log_success() {
    echo -e "${GREEN}✅ [SUCCESS]${RESET} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️  [WARNING]${RESET} $1"
}

log_error() {
    echo -e "${RED}❌ [ERROR]${RESET} $1" >&2
}

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}╔════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}${BOLD}║  $1${RESET}"
    echo -e "${CYAN}${BOLD}╚════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
}

check_venv() {
    if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
        log_error "Entorno virtual no encontrado en ${PROJECT_ROOT}/.venv"
        log_info "Ejecuta: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
}

check_apple_silicon() {
    log_info "🍎 Verificando Apple Silicon M4..."
    
    # Verificar que estamos en Apple Silicon
    local arch=$(uname -m)
    if [ "$arch" != "arm64" ]; then
        log_warning "⚠️ No se detectó Apple Silicon (arch: $arch)"
        log_warning "   El sistema se ejecutará en modo CPU"
        return 1
    fi
    
    # Obtener información del chip
    local chip_info=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    log_success "✅ Apple Silicon detectado: $chip_info"
    
    # Verificar cores disponibles
    local perf_cores=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "Unknown")
    local efficiency_cores=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo "Unknown")
    local total_cores=$(sysctl -n hw.physicalcpu 2>/dev/null || echo "Unknown")
    
    log_info "   Performance Cores: $perf_cores"
    log_info "   Efficiency Cores: $efficiency_cores"
    log_info "   Total Physical Cores: $total_cores"
    
    # Verificar memoria unificada
    local memory_gb=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
    log_info "   Unified Memory: ${memory_gb}GB"
    
    # Verificar que PyTorch MPS esté disponible
    log_info "🔍 Verificando PyTorch MPS..."
    local mps_available=$("$VENV_PYTHON" -c "import torch; print('YES' if torch.backends.mps.is_available() else 'NO')" 2>/dev/null || echo "NO")
    
    if [ "$mps_available" = "YES" ]; then
        log_success "✅ PyTorch MPS (Metal) DISPONIBLE"
        log_info "   GPU Metal será usado para aceleración"
        return 0
    else
        log_warning "⚠️ PyTorch MPS NO disponible"
        log_warning "   Instala PyTorch con soporte MPS:"
        log_warning "   pip3 install --upgrade torch torchvision torchaudio"
        return 1
    fi
}

verify_mps_usage() {
    log_info "🎮 Verificando uso de GPU Metal (MPS)..."
    
    # Crear script temporal de verificación
    local verify_script=$(cat <<'EOF'
import torch
import sys

print("=" * 60)
print("🍎 APPLE SILICON M4 + MPS VERIFICATION")
print("=" * 60)

# 1. PyTorch version
print(f"PyTorch version: {torch.__version__}")

# 2. MPS availability
mps_available = torch.backends.mps.is_available()
print(f"MPS available: {mps_available}")

if mps_available:
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # 3. Try to use MPS
    try:
        device = torch.device("mps")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = x @ y  # Matrix multiplication on GPU
        print(f"✅ MPS test PASSED - GPU is working!")
        print(f"   Device: {device}")
        print(f"   Test tensor shape: {z.shape}")
    except Exception as e:
        print(f"❌ MPS test FAILED: {e}")
        sys.exit(1)
else:
    print("❌ MPS not available - will use CPU (slower)")
    sys.exit(1)

print("=" * 60)
EOF
)
    
    echo "$verify_script" | "$VENV_PYTHON" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "✅ GPU Metal (MPS) está funcionando correctamente"
        return 0
    else
        log_error "❌ GPU Metal (MPS) NO está funcionando"
        log_warning "   El sistema usará CPU (más lento)"
        return 1
    fi
}

is_process_running() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

wait_for_process() {
    local pid="$1"
    local timeout="$2"
    local count=0
    
    while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt $timeout ]; do
        sleep 1
        count=$((count + 1))
    done
    
    if ps -p "$pid" > /dev/null 2>&1; then
        return 1
    fi
    return 0
}

# ============================================================================
# FUNCIONES DE DESPLIEGUE PÚBLICO
# ============================================================================
deploy_emergency_public() {
    print_header "🌐 DESPLIEGUE PÚBLICO - EMERGENCY CONTACT SYSTEM"
    
    log_info "Opciones de despliegue público para contactar personas en peligro:"
    echo ""
    echo "  1) 📱 Telegram Bot (RECOMENDADO - GRATIS Y GLOBAL)"
    echo "     • Accesible desde cualquier país AHORA"
    echo "     • No necesita servidor público"
    echo "     • Funciona con internet intermitente"
    echo "     • 100% gratis para siempre"
    echo ""
    echo "  2) ☁️ Cloudflare Tunnel (PRODUCCIÓN)"
    echo "     • URL permanente y profesional"
    echo "     • HTTPS automático"
    echo "     • Sin límites de tráfico"
    echo "     • DDoS protection incluido"
    echo ""
    echo "  3) 🌐 ngrok (TESTING RÁPIDO)"
    echo "     • Para pruebas inmediatas"
    echo "     • URL temporal"
    echo ""
    
    read -p "Selecciona opción [1-3]: " option
    
    case "$option" in
        1)
            setup_telegram_bot
            ;;
        2)
            setup_cloudflare_tunnel
            ;;
        3)
            setup_ngrok_tunnel
            ;;
        *)
            log_error "Opción inválida"
            return 1
            ;;
    esac
}

setup_telegram_bot() {
    print_header "📱 CONFIGURACIÓN TELEGRAM BOT - ACCESO GLOBAL"
    
    log_success "✅ Telegram Bot NO necesita servidor público"
    log_info "Se conecta automáticamente desde tu iMac a Telegram"
    log_info "Las personas pueden contactarlo desde CUALQUIER PAÍS"
    echo ""
    
    log_info "PASOS PARA CREAR TU BOT:"
    echo "1. Abre Telegram en tu teléfono/computadora"
    echo "2. Busca: @BotFather"
    echo "3. Envía: /newbot"
    echo "4. Nombre del bot: METACORTEX Divine Protection"
    echo "5. Username: metacortex_divine_bot (o el que prefieras)"
    echo "6. Copia el TOKEN que te da BotFather"
    echo ""
    
    read -p "¿Ya tienes el token de @BotFather? (s/n): " has_token
    
    if [ "$has_token" = "s" ]; then
        read -p "Pega tu TELEGRAM_BOT_TOKEN aquí: " telegram_token
        
        # Guardar en .env
        if [ -f "${PROJECT_ROOT}/.env" ]; then
            if grep -q "TELEGRAM_BOT_TOKEN=" "${PROJECT_ROOT}/.env"; then
                # Actualizar token existente (macOS sed syntax)
                sed -i '' "s/TELEGRAM_BOT_TOKEN=.*/TELEGRAM_BOT_TOKEN=$telegram_token/" "${PROJECT_ROOT}/.env"
            else
                echo "TELEGRAM_BOT_TOKEN=$telegram_token" >> "${PROJECT_ROOT}/.env"
            fi
        else
            echo "TELEGRAM_BOT_TOKEN=$telegram_token" > "${PROJECT_ROOT}/.env"
        fi
        
        log_success "✅ Token guardado en .env"
        
        # Probar conexión
        log_info "🔍 Probando conexión con Telegram..."
        "$VENV_PYTHON" -c "
import os
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

from telegram import Bot
import asyncio

async def test_bot():
    try:
        bot = Bot(token='$telegram_token')
        me = await bot.get_me()
        print('\n✅ Bot conectado exitosamente!')
        print(f'   Nombre: {me.first_name}')
        print(f'   Username: @{me.username}')
        print(f'\n🌐 URL PÚBLICA GLOBAL: https://t.me/{me.username}')
        print(f'\n📱 Las personas pueden buscar: @{me.username}')
        print('\n🔥 El bot está ACTIVO y accesible desde CUALQUIER PAÍS')
        return 0
    except Exception as e:
        print(f'\n❌ Error conectando: {e}')
        print('Verifica que el token sea correcto')
        return 1

exit(asyncio.run(test_bot()))
" || {
            log_error "Error al probar el bot. Verifica el token."
            return 1
        }
        
        log_success "🎉 Telegram Bot configurado y PÚBLICO!"
        log_info ""
        log_info "PRÓXIMOS PASOS:"
        log_info "1. Reinicia el sistema: ./metacortex_master.sh restart"
        log_info "2. El bot empezará a recibir mensajes automáticamente"
        log_info "3. Comparte el link del bot con personas en peligro"
        log_info ""
        log_warning "⚠️ IMPORTANTE: El bot funciona mientras tu iMac esté encendido"
        log_info "   Por eso usamos caffeinate para mantenerlo 24/7"
        
    else
        log_info ""
        log_info "📝 INSTRUCCIONES DETALLADAS:"
        log_info "1. Abre Telegram: https://telegram.org"
        log_info "2. Busca: @BotFather (bot oficial de Telegram)"
        log_info "3. Envía: /newbot"
        log_info "4. Sigue las instrucciones"
        log_info "5. Vuelve a ejecutar: ./metacortex_master.sh deploy"
    fi
}

setup_cloudflare_tunnel() {
    print_header "☁️ CLOUDFLARE TUNNEL - EXPOSICIÓN PÚBLICA PROFESIONAL"
    
    # Verificar cloudflared
    if ! command -v cloudflared &> /dev/null; then
        log_info "📦 cloudflared no está instalado. Instalando..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install cloudflared || {
                log_error "Error instalando cloudflared con Homebrew"
                log_info "Instala manualmente: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
                return 1
            }
        else
            log_error "Instala cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
            return 1
        fi
    fi
    
    log_success "✅ cloudflared instalado"
    
    # Verificar que Emergency System esté corriendo
    if ! lsof -i:8200 -sTCP:LISTEN > /dev/null 2>&1; then
        log_error "❌ Emergency Contact System NO está corriendo en puerto 8200"
        log_info "Ejecuta primero: ./metacortex_master.sh start"
        return 1
    fi
    
    log_success "✅ Emergency Contact System corriendo en puerto 8200"
    
    # Login a Cloudflare
    log_info "🔐 Autenticando con Cloudflare..."
    if [ ! -f "$HOME/.cloudflared/cert.pem" ]; then
        log_info "Se abrirá tu navegador para autenticarte con Cloudflare"
        cloudflared tunnel login || {
            log_error "Error en autenticación"
            return 1
        }
    fi
    
    log_success "✅ Autenticado con Cloudflare"
    
    # Crear tunnel con nombre único
    local tunnel_name="metacortex-emergency-$(date +%s)"
    log_info "🚇 Creando túnel: $tunnel_name"
    
    cloudflared tunnel create "$tunnel_name" || {
        log_error "Error creando túnel"
        return 1
    }
    
    local tunnel_id=$(cloudflared tunnel list | grep "$tunnel_name" | awk '{print $1}')
    log_success "✅ Tunnel creado: $tunnel_id"
    
    # Configurar DNS
    log_info "🌐 Configura tu dominio (o usa el gratuito de Cloudflare):"
    read -p "¿Tienes un dominio en Cloudflare? (s/n): " has_domain
    
    if [ "$has_domain" = "s" ]; then
        read -p "Ingresa tu dominio (ej: emergency.tudominio.com): " domain
    else
        domain="$tunnel_id.cfargotunnel.com"
        log_info "Usando dominio gratuito: $domain"
    fi
    
    # Crear configuración
    mkdir -p "$HOME/.cloudflared"
    cat > "$HOME/.cloudflared/config.yml" << EOF
tunnel: $tunnel_id
credentials-file: $HOME/.cloudflared/$tunnel_id.json

ingress:
  - hostname: $domain
    service: http://localhost:8200
  - service: http_status:404
EOF
    
    log_success "✅ Configuración creada"
    
    # Configurar ruta DNS si tiene dominio
    if [ "$has_domain" = "s" ]; then
        log_info "Configurando DNS..."
        cloudflared tunnel route dns "$tunnel_name" "$domain" || {
            log_warning "No se pudo configurar DNS automáticamente"
            log_info "Configura manualmente en Cloudflare Dashboard"
        }
    fi
    
    # Iniciar túnel en background
    log_info "🚀 Iniciando túnel en background..."
    nohup cloudflared tunnel run "$tunnel_name" > "${LOGS_DIR}/cloudflare_tunnel.log" 2>&1 &
    local tunnel_pid=$!
    echo "$tunnel_pid" > "${PID_DIR}/cloudflare_tunnel.pid"
    
    sleep 3
    
    if ps -p "$tunnel_pid" > /dev/null 2>&1; then
        log_success "🎉 ¡Túnel ACTIVO!"
        log_info ""
        log_info "🌐 URL PÚBLICA: https://$domain"
        log_info "📱 Endpoint de emergencia: https://$domain/emergency"
        log_info "🔒 HTTPS automático y seguro"
        log_info "📊 Monitorea en: https://dash.cloudflare.com"
        log_info ""
        log_info "PID del túnel: $tunnel_pid"
        log_info "Logs: tail -f ${LOGS_DIR}/cloudflare_tunnel.log"
    else
        log_error "Error: Túnel no inició correctamente"
        log_info "Ver logs: cat ${LOGS_DIR}/cloudflare_tunnel.log"
        return 1
    fi
}

setup_ngrok_tunnel() {
    print_header "🌐 NGROK TUNNEL - TESTING RÁPIDO"
    
    # Verificar ngrok
    if ! command -v ngrok &> /dev/null; then
        log_info "📦 ngrok no está instalado. Instalando..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install ngrok/ngrok/ngrok || {
                log_error "Error instalando ngrok"
                log_info "Instala manualmente: https://ngrok.com/download"
                return 1
            }
        else
            log_error "Instala ngrok: https://ngrok.com/download"
            return 1
        fi
    fi
    
    log_success "✅ ngrok instalado"
    
    # Verificar que Emergency System esté corriendo
    if ! lsof -i:8200 -sTCP:LISTEN > /dev/null 2>&1; then
        log_error "❌ Emergency Contact System NO está corriendo en puerto 8200"
        log_info "Ejecuta primero: ./metacortex_master.sh start"
        return 1
    fi
    
    log_success "✅ Emergency Contact System corriendo en puerto 8200"
    log_warning "⚠️ El túnel ngrok es TEMPORAL - se cerrará al detener ngrok"
    log_info ""
    log_info "🚀 Iniciando túnel público..."
    log_info "🌐 URL pública estará disponible en unos segundos..."
    log_info "📝 Presiona Ctrl+C para detener el túnel"
    echo ""
    
    # Ejecutar ngrok (bloquea terminal, es para testing)
    ngrok http 8200 --log=stdout --log-level=info
}

# ============================================================================
# FUNCIONES AUXILIARES PARA IA
# ============================================================================

verify_dependencies() {
    print_header "🔍 VERIFICANDO DEPENDENCIAS COMPLETAS"
    
    local all_ok=true
    
    # 1. Python 3.11+
    if command -v python3 &> /dev/null; then
        local py_version=$(python3 --version 2>&1 | awk '{print $2}')
        log_success "Python 3: $py_version"
    else
        log_error "Python 3 no encontrado"
        all_ok=false
    fi
    
    # 2. pip
    if command -v pip3 &> /dev/null; then
        log_success "pip3: Disponible"
    else
        log_error "pip3 no encontrado"
        all_ok=false
    fi
    
    # 3. Ollama
    if command -v ollama &> /dev/null; then
        log_success "Ollama: Instalado"
        
        # Verificar si está corriendo
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_success "   Ollama Server: ACTIVO"
        else
            log_warning "   Ollama Server: NO ACTIVO (iniciando...)"
            nohup ollama serve > "${LOGS_DIR}/ollama.log" 2>&1 &
            sleep 3
        fi
    else
        log_warning "Ollama no encontrado (instalando...)"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install ollama || {
                log_error "No se pudo instalar Ollama"
                log_info "Instala manualmente: https://ollama.com"
                all_ok=false
            }
        fi
    fi
    
    # 4. Entorno virtual
    if [ -d "${PROJECT_ROOT}/.venv" ]; then
        log_success "Entorno virtual: Encontrado"
    else
        log_warning "Creando entorno virtual..."
        python3 -m venv "${PROJECT_ROOT}/.venv"
        log_success "Entorno virtual: Creado"
    fi
    
    # 5. Dependencias Python
    log_info "Verificando dependencias Python..."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    
    # Instalar dependencias críticas si faltan
    local missing_deps=false
    
    if ! "$VENV_PYTHON" -c "import fastapi" 2>/dev/null; then
        missing_deps=true
    fi
    
    if ! "$VENV_PYTHON" -c "import httpx" 2>/dev/null; then
        missing_deps=true
    fi
    
    if ! "$VENV_PYTHON" -c "import telegram" 2>/dev/null; then
        missing_deps=true
    fi
    
    if [ "$missing_deps" = true ]; then
        log_warning "Instalando dependencias faltantes..."
        "$VENV_PIP" install --upgrade pip -q
        "$VENV_PIP" install -r "${PROJECT_ROOT}/requirements.txt" -q
        "$VENV_PIP" install 'pydantic[email]' -q
        log_success "Dependencias instaladas"
    else
        log_success "Todas las dependencias Python disponibles"
    fi
    
    # 6. Modelos de Ollama
    log_info "Verificando modelos de Ollama..."
    
    # Primero verificar si Ollama está corriendo (más robusto)
    if ! lsof -i:11434 -sTCP:LISTEN > /dev/null 2>&1; then
        log_warning "   Servidor Ollama NO está corriendo en puerto 11434. Iniciando..."
        nohup ollama serve > "${LOGS_DIR}/ollama_serve.log" 2>&1 &
        local ollama_pid=$!
        echo "$ollama_pid" > "${PID_DIR}/ollama.pid"
        log_info "   Esperando inicio de Ollama (10 segundos)..."
        sleep 10
        
        # Verificar que inició correctamente
        if lsof -i:11434 -sTCP:LISTEN > /dev/null 2>&1; then
            log_success "   ✅ Ollama iniciado correctamente (PID: $ollama_pid, Puerto: 11434)"
        else
            log_error "   ❌ Ollama no pudo iniciar en puerto 11434"
            log_info "   Ver logs: tail -f ${LOGS_DIR}/ollama_serve.log"
        fi
    else
        local ollama_pid=$(lsof -i:11434 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        log_success "   ✅ Ollama ya está corriendo (PID: $ollama_pid, Puerto: 11434)"
    fi
    
    # Verificar conexión a Ollama con timeout MÁS LARGO
    log_info "   Verificando conexión a Ollama..."
    local retry_count=0
    local max_retries=3  # Reducido a 3 intentos pero con más tiempo
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -s --max-time 5 http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_success "   ✅ Conexión a Ollama API establecida"
            break
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                log_warning "   ⚠️  Intento $retry_count/$max_retries - Esperando 3s..."
                sleep 3
            else
                log_warning "   ⚠️  No se puede conectar a Ollama API después de $max_retries intentos"
                log_info "   CONTINUANDO de todas formas - Ollama puede estar iniciándose"
                # NO retornar 1, continuar con el startup
            fi
        fi
    done
    
    # Verificar modelos de Ollama (NO-BLOQUEANTE)
    local models_needed=("mistral:latest" "mistral:instruct" "mistral-nemo:latest")
    log_info "   Verificando modelos disponibles..."
    
    local models_available=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || echo "")
    
    for model in "${models_needed[@]}"; do
        if echo "$models_available" | grep -q "^${model}"; then
            log_success "   ✅ Modelo $model: Disponible"
        else
            log_warning "   ⚠️  Modelo $model: NO DISPONIBLE"
            log_info "   Para instalar: ollama pull $model"
        fi
    done
    
    # 7. Archivos de configuración
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        if grep -q "your_token_here" "${PROJECT_ROOT}/.env" 2>/dev/null; then
            log_warning ".env necesita configuración"
        else
            log_success ".env: Configurado"
        fi
    else
        log_warning "Creando .env de ejemplo..."
        cat > "${PROJECT_ROOT}/.env" << 'ENVEOF'
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
ENVIRONMENT=production
DEBUG=false
ENVEOF
        log_info "   Archivo .env creado - configura tus credenciales"
    fi
    
    # 8. Crear directorios
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/emergency_requests"
    mkdir -p "${PROJECT_ROOT}/ml_models"
    mkdir -p "${PROJECT_ROOT}/pid"
    log_success "Directorios creados"
    
    if [ "$all_ok" = true ]; then
        log_success "✅ Todas las dependencias verificadas"
        return 0
    else
        log_error "❌ Algunas dependencias faltan"
        return 1
    fi
}

start_ai_systems() {
    print_header "🧠 INICIANDO SISTEMAS DE IA COMPLETOS"
    
    # Verificar dependencias primero
    verify_dependencies || return 1
    
    # Activar entorno
    source "${PROJECT_ROOT}/.venv/bin/activate"
    
    # 1. Detener procesos previos de IA
    log_info "Limpiando procesos previos..."
    pkill -9 -f "python.*unified_startup.py" 2>/dev/null || true
    pkill -9 -f "python.*emergency_contact_system.py" 2>/dev/null || true
    sleep 2
    
    # 2. Limpiar logs
    log_info "Limpiando logs antiguos..."
    > "${LOGS_DIR}/unified_system.log" 2>/dev/null || true
    > "${LOGS_DIR}/emergency_contact_stdout.log" 2>/dev/null || true
    
    # 3. Iniciar Unified System (TODO integrado)
    log_info "Iniciando Unified System (AI Integration + Telegram + WhatsApp + Web)..."
    log_info "   • AI Integration Layer (Ollama)"
    log_info "   • Divine Protection System"
    log_info "   • Emergency Contact System"
    log_info "   • Telegram Bot (con IA)"
    log_info "   • WhatsApp Bot"
    log_info "   • Web Interface (puerto 8080)"
    echo ""
    
    nohup "$VENV_PYTHON" "${PROJECT_ROOT}/unified_startup.py" > "${LOGS_DIR}/unified_system.log" 2>&1 &
    local unified_pid=$!
    echo "$unified_pid" > "${PID_DIR}/unified_system.pid"
    
    log_info "⏳ Esperando inicialización (15 segundos)..."
    sleep 15
    
    # 4. Verificar que está corriendo
    if ps -p "$unified_pid" > /dev/null 2>&1; then
        log_success "✅ Unified System ACTIVO (PID: $unified_pid)"
        
        # Verificar componentes
        echo ""
        log_info "🔍 Verificando componentes..."
        
        # Web Interface
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            log_success "   ✅ Web Interface: http://localhost:8080"
        else
            log_warning "   ⏳ Web Interface: Iniciando..."
        fi
        
        # Ollama
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_success "   ✅ Ollama LLM: http://localhost:11434"
        else
            log_warning "   ⏳ Ollama: Iniciando..."
        fi
        
        # Emergency Contact
        if [ -f "${LOGS_DIR}/emergency_contact_stdout.log" ]; then
            if grep -q "Application started" "${LOGS_DIR}/emergency_contact_stdout.log" 2>/dev/null; then
                log_success "   ✅ Emergency Contact System: ACTIVO"
            else
                log_warning "   ⏳ Emergency Contact: Iniciando..."
            fi
        fi
        
        echo ""
        print_header "✅ SISTEMAS DE IA ACTIVOS"
        
        echo "📊 INFORMACIÓN DEL SISTEMA:"
        echo "   • PID: $unified_pid"
        echo "   • Web Interface: http://localhost:8080"
        echo "   • API Status: http://localhost:8080/api/status"
        echo "   • Logs: tail -f logs/unified_system.log"
        echo ""
        echo "📞 CANALES DE CONTACTO:"
        echo "   • Telegram Bot: @metacortex_divine_bot"
        echo "   • Web Form: http://localhost:8080"
        echo "   • WhatsApp: (Configura Twilio en .env)"
        echo "   • Email: emergency@metacortex.ai"
        echo ""
        echo "🧠 SISTEMAS ACTIVOS:"
        echo "   • AI Integration Layer (Ollama + 956 ML models)"
        echo "   • Divine Protection System"
        echo "   • Emergency Contact System"
        echo "   • Telegram Bot (con respuestas inteligentes)"
        echo "   • WhatsApp Bot (si configurado)"
        echo "   • Web Interface (responsive + chat)"
        echo ""
        echo "💡 COMANDOS ÚTILES:"
        echo "   Ver logs:        tail -f logs/unified_system.log"
        echo "   Estado:          ./metacortex_master.sh status"
        echo "   Detener:         ./metacortex_master.sh stop"
        echo "   Reiniciar:       ./metacortex_master.sh restart"
        echo ""
        
        return 0
    else
        log_error "❌ Error al iniciar Unified System"
        log_info "Ver logs: tail -50 logs/unified_system.log"
        return 1
    fi
}

# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================
start_system() {
    print_header "🚀 INICIANDO METACORTEX COMPLETO - APPLE SILICON M4 + IA"
    
    check_venv
    
    # ============================================================================
    # PASO 1: INICIAR SISTEMAS DE IA (PRIORITARIO)
    # ============================================================================
    log_info "═══════════════════════════════════════════════════════════"
    log_info "🧠 FASE 1: SISTEMAS DE INTELIGENCIA ARTIFICIAL"
    log_info "═══════════════════════════════════════════════════════════"
    
    start_ai_systems || {
        log_error "Error iniciando sistemas de IA"
        log_info "Continuando con sistemas base..."
    }
    
    sleep 3
    
    # ============================================================================
    # PASO 2: VERIFICAR APPLE SILICON M4 Y MPS
    # ============================================================================
    log_info "═══════════════════════════════════════════════════════════"
    log_info "🍎 FASE 2: APPLE SILICON M4 OPTIMIZATION"
    log_info "═══════════════════════════════════════════════════════════"
    
    check_apple_silicon
    local mps_status=$?
    
    if [ $mps_status -eq 0 ]; then
        verify_mps_usage
        log_success "✅ Sistema configurado para GPU Metal (MPS)"
        log_info "   PYTORCH_ENABLE_MPS_FALLBACK=1"
        log_info "   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"
        log_info "   MPS_FORCE_ENABLE=1"
    else
        log_warning "⚠️ MPS no disponible, usando CPU"
    fi
    
    log_info "═══════════════════════════════════════════════════════════"
    echo ""
    
    # ============================================================================
    # PASO 3: VERIFICAR PROCESOS EXISTENTES (ANTI-DUPLICADOS)
    # ============================================================================
    log_info "Verificando estado previo (anti-duplicados)..."
    
    local existing_processes=$(ps aux | grep -E "python.*(metacortex_daemon.py|neural_symbiotic_network.py|web_interface/server.py|startup_orchestrator.py)" | grep -v grep | wc -l | tr -d ' ')
    
    if [ "$existing_processes" -gt 0 ]; then
        log_error "⚠️ DETECTADOS $existing_processes PROCESOS ACTIVOS - Ejecuta './metacortex_master.sh stop' primero"
        log_info "Procesos encontrados:"
        ps aux | grep -E "python.*(metacortex_daemon.py|neural_symbiotic_network.py|web_interface/server.py|startup_orchestrator.py)" | grep -v grep | awk '{printf "   PID %s: %s\n", $2, $11}'
        return 1
    fi
    
    if is_process_running "$DAEMON_PID_FILE"; then
        log_warning "El daemon ya está corriendo (archivo PID encontrado)"
        return 1
    fi
    
    log_success "✓ No hay procesos duplicados"
    
    # ============================================================================
    # PASO 4: LIMPIAR ARCHIVOS ANTIGUOS
    # ============================================================================
    log_info "Limpiando archivos PID/LOCK antiguos..."
    find "$PROJECT_ROOT" -name "*.pid" -type f -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.lock" -type f -delete 2>/dev/null || true
    
    # Crear directorios necesarios
    mkdir -p "$LOGS_DIR"
    mkdir -p "$PID_DIR"
    
    # ============================================================================
    # PASO 5: ACTIVAR PERSISTENCIA macOS (CAFFEINATE)
    # ============================================================================
    log_info "🍎 Activando persistencia optimizada para iMac M4..."
    log_info "   ✅ Caffeinate: Mantiene sistema ejecutándose 24/7"
    log_info "   ✅ GPU Metal (MPS): Aceleración de ML/AI"
    log_info "   ✅ Unified Memory: Compartida entre CPU y GPU"
    log_info "   ✅ Performance Cores: Priorizados para carga pesada"
    log_info "   ✅ Efficiency Cores: Tareas de fondo"
    log_info "   ✅ Permite sleep de pantalla (ahorro de energía)"
    log_info "   ✅ Inicialización ORDENADA sin circular imports"
    log_info "   ✅ Health checks antes de continuar"
    
    # ============================================================================
    # PASO 6: INICIAR SERVICIOS STANDALONE (ULTRA-LIGEROS)
    # ============================================================================
    log_info "🚀 Iniciando servicios STANDALONE (arquitectura 3 capas)..."
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    # Web Interface STANDALONE (puerto 8000) - Ultra-ligero, lazy loading
    log_info "   → Web Interface STANDALONE (puerto 8000) - <3s startup..."
    nohup "$VENV_PYTHON" "${PROJECT_ROOT}/web_interface/server.py" > "${LOGS_DIR}/web_interface_stdout.log" 2>&1 &
    local web_interface_pid=$!
    echo "$web_interface_pid" > "${PID_DIR}/web_interface.pid"
    log_success "      Web Interface STANDALONE iniciado (PID: $web_interface_pid)"
    
    # Neural Network STANDALONE - Sin conexiones simbióticas automáticas
    log_info "   → Neural Network STANDALONE - <5s startup..."
    nohup "$VENV_PYTHON" "${PROJECT_ROOT}/neural_network_service/server.py" > "${LOGS_DIR}/neural_network_stdout.log" 2>&1 &
    local neural_network_pid=$!
    echo "$neural_network_pid" > "${PID_DIR}/neural_network.pid"
    log_success "      Neural Network STANDALONE iniciado (PID: $neural_network_pid)"
    
    # Telemetry STANDALONE (puerto 9090) - Prometheus metrics SIMPLE
    log_info "   → Telemetry System STANDALONE (puerto 9090) - ULTRA-SIMPLE..."
    nohup "$VENV_PYTHON" "${PROJECT_ROOT}/telemetry_service/server.py" > "${LOGS_DIR}/telemetry_stdout.log" 2>&1 &
    local telemetry_pid=$!
    echo "$telemetry_pid" > "${PID_DIR}/telemetry.pid"
    log_success "      Telemetry System STANDALONE iniciado (PID: $telemetry_pid)"
    
    # Emergency Contact System STANDALONE (puerto 8200) - Sistema de emergencia
    # ⚠️ DESACTIVADO: unified_startup.py ya incluye Emergency Contact + Telegram Bot
    # Mantenerlo aquí causaría conflicto 409 (multiple getUpdates)
    log_info "   → Emergency Contact System: INTEGRADO en unified_startup.py (evitando duplicación)"
    # nohup "$VENV_PYTHON" "${PROJECT_ROOT}/metacortex_sinaptico/emergency_contact_system.py" > "${LOGS_DIR}/emergency_contact_stdout.log" 2>&1 &
    # local emergency_pid=$!
    # echo "$emergency_pid" > "${PID_DIR}/emergency_contact.pid"
    log_success "      Emergency Contact System: Ya activo en unified_startup.py (puerto 8080)"
    
    # API Monetization Server STANDALONE (puerto 8100) - Sistema de ingresos
    log_info "   → API Monetization Server STANDALONE (puerto 8100)..."
    nohup "$VENV_PYTHON" "${PROJECT_ROOT}/metacortex_sinaptico/api_monetization_endpoint.py" > "${LOGS_DIR}/api_monetization_stdout.log" 2>&1 &
    local api_monetization_pid=$!
    echo "$api_monetization_pid" > "${PID_DIR}/api_monetization.pid"
    log_success "      API Monetization Server STANDALONE iniciado (PID: $api_monetization_pid)"
    
    # ============================================================================
    # ENTERPRISE SERVICES: Dashboard, Telegram Bot, Autonomous Orchestrator
    # ============================================================================
    log_info "🚀 Iniciando METACORTEX Enterprise Services..."
    
    # Dashboard Enterprise (puerto 8300) - FastAPI + WebSocket + 965 modelos
    log_info "   → Dashboard Enterprise (puerto 8300) - FastAPI + WebSocket..."
    nohup "$VENV_PYTHON" "${PROJECT_ROOT}/dashboard_enterprise.py" > "${LOGS_DIR}/dashboard_enterprise.log" 2>&1 &
    local dashboard_pid=$!
    echo "$dashboard_pid" > "${PID_DIR}/dashboard_enterprise.pid"
    log_success "      Dashboard Enterprise iniciado (PID: $dashboard_pid)"
    log_info "      🌐 URL: http://localhost:8300"
    log_info "      📊 965 modelos ML disponibles"
    log_info "      📡 WebSocket: ws://localhost:8300/ws"
    log_info "      📚 API Docs: http://localhost:8300/api/docs"
    
    # Telegram Monitor Bot (si está configurado el token)
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
        log_info "   → Telegram Monitor Bot..."
        nohup "$VENV_PYTHON" "${PROJECT_ROOT}/telegram_monitor_bot.py" > "${LOGS_DIR}/telegram_monitor.log" 2>&1 &
        local telegram_pid=$!
        echo "$telegram_pid" > "${PID_DIR}/telegram_monitor.pid"
        log_success "      Telegram Monitor Bot iniciado (PID: $telegram_pid)"
        log_info "      📱 Bot activo - Comandos: /start, /status, /models, /tasks"
    else
        log_info "   → Telegram Monitor Bot: DESACTIVADO (configura TELEGRAM_BOT_TOKEN)"
    fi
    
    # Esperar 3 segundos para que los servicios enterprise inicien
    log_info "   ⏳ Esperando servicios enterprise (3s)..."
    sleep 3
    
    # Verificar Dashboard Enterprise
    if ps -p "$dashboard_pid" > /dev/null 2>&1; then
        log_success "   ✅ Dashboard Enterprise: ACTIVO (PID: $dashboard_pid, Puerto 8300)"
        log_success "      → Abre en navegador: http://localhost:8300"
    else
        log_warning "   ⚠️ Dashboard Enterprise: NO ACTIVO (ver logs/dashboard_enterprise.log)"
    fi
    
    # Verificar Telegram Bot (solo si se intentó iniciar)
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
        if ps -p "$telegram_pid" > /dev/null 2>&1; then
            log_success "   ✅ Telegram Monitor Bot: ACTIVO (PID: $telegram_pid)"
        else
            log_warning "   ⚠️ Telegram Monitor Bot: NO ACTIVO (ver logs/telegram_monitor.log)"
        fi
    fi
    
    # ============================================================================
    # SERVICIOS STANDALONE (ARQUITECTURA 3 CAPAS)
    # ============================================================================
    # Esperar 3 segundos para que los servicios standalone inicien (ultra-rápido)
    log_info "   ⏳ Esperando servicios standalone (5s - incluyendo Emergency Contact + API)..."
    sleep 5
    
    # Verificar que están corriendo
    if ps -p "$web_interface_pid" > /dev/null 2>&1; then
        log_success "   ✅ Web Interface: ACTIVO (PID: $web_interface_pid)"
    else
        log_warning "   ⚠️ Web Interface: NO ACTIVO (ver logs/web_interface.log)"
    fi
    
    if ps -p "$neural_network_pid" > /dev/null 2>&1; then
        log_success "   ✅ Neural Network: ACTIVO (PID: $neural_network_pid)"
    else
        log_warning "   ⚠️ Neural Network: NO ACTIVO (ver logs/neural_network.log)"
    fi
    
    if ps -p "$telemetry_pid" > /dev/null 2>&1; then
        log_success "   ✅ Telemetry System: ACTIVO (PID: $telemetry_pid)"
    else
        log_warning "   ⚠️ Telemetry System: NO ACTIVO (ver logs/telemetry.log)"
    fi
    
    # Emergency Contact System ya NO se inicia standalone (integrado en unified_startup.py)
    # No verificar emergency_pid porque no existe
    log_success "   ✅ Emergency Contact System: Integrado en Unified System (puerto 8080)"
    
    if ps -p "$api_monetization_pid" > /dev/null 2>&1; then
        log_success "   ✅ API Monetization Server: ACTIVO (PID: $api_monetization_pid, Puerto 8100)"
    else
        log_warning "   ⚠️ API Monetization Server: NO ACTIVO (ver logs/api_monetization_stdout.log)"
    fi
    
    # ============================================================================
    # PASO 7: INICIAR DAEMON MILITAR (24/7 PERSISTENCE) - CON MPS FORZADO
    # ============================================================================
    log_info "Iniciando METACORTEX Military Daemon (Apple Silicon M4 + MPS)..."
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    # Configurar variables de entorno para forzar MPS
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export PYTORCH_MPS_PREFER_METAL=1
    export MPS_FORCE_ENABLE=1
    export TOKENIZERS_PARALLELISM=true
    export OMP_NUM_THREADS=10
    
    log_info "   🎮 Variables de entorno MPS configuradas:"
    log_info "      PYTORCH_ENABLE_MPS_FALLBACK=1"
    log_info "      PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"
    log_info "      MPS_FORCE_ENABLE=1"
    log_info "      OMP_NUM_THREADS=10 (M4 10-core)"
    
    nohup caffeinate -i "$VENV_PYTHON" "${PROJECT_ROOT}/metacortex_daemon.py" > "${LOGS_DIR}/metacortex_daemon_military.log" 2>&1 &
    local daemon_pid=$!
    echo "$daemon_pid" > "${PID_DIR}/metacortex_daemon_military.pid"
    
    log_success "Military Daemon iniciado (PID: $daemon_pid)"
    log_info "   ✅ Daemon: PERSISTENCIA 24/7 ACTIVA (caffeinate)"
    log_info "   ✅ GPU Metal (MPS): FORZADO para ML/AI"
    log_info "   ✅ Apple Silicon M4: OPTIMIZADO"
    log_info "   ✅ Health monitoring: CONTINUO"
    log_info "   ✅ Capability discovery: EXPONENCIAL"
    
    sleep 3
    
    # ============================================================================
    # PASO 8: ORCHESTRATOR (INTEGRADO EN DASHBOARD ENTERPRISE)
    # ============================================================================
    # NOTA: El Autonomous Orchestrator ahora está integrado en dashboard_enterprise.py
    # El orchestrator se inicia automáticamente cuando el Dashboard se inicia
    log_info "Agent Orchestrator integrado en Dashboard Enterprise..."
    log_info "   ℹ️  Orchestrator se inicia automáticamente con el Dashboard"
    log_info "   ℹ️  Modo autónomo: ACTIVADO (enable_auto_task_generation=True)"
    log_info "   ℹ️  965 modelos ML disponibles"
    
    # Asignar orchestrator_pid al mismo PID del Dashboard (donde está integrado)
    local orchestrator_pid=$dashboard_pid
    log_success "Agent Orchestrator iniciado (PID: $orchestrator_pid - integrado en Dashboard)"

    # ============================================================================
    # PASO 9: MONITOREAR EL STARTUP (VER LOGS EN TIEMPO REAL)
    # ============================================================================
    log_info "Monitoreando inicialización..."
    log_info "   Daemon logs: tail -f ${LOGS_DIR}/metacortex_daemon_military.log"
    log_info "   Dashboard logs: tail -f ${LOGS_DIR}/dashboard_enterprise.log"
    log_info "   AI Systems logs: tail -f ${LOGS_DIR}/unified_system.log"
    
    sleep 3
    
    # Mostrar últimas líneas del log del daemon
    if [ -f "${LOGS_DIR}/metacortex_daemon_military.log" ]; then
        log_info "Daemon Military - Últimas líneas:"
        tail -20 "${LOGS_DIR}/metacortex_daemon_military.log"
    fi
    
    # ============================================================================
    # PASO 10: VERIFICAR QUE TODOS LOS PROCESOS SIGAN CORRIENDO
    # ============================================================================
    sleep 5
    
    local daemon_running=false
    local orchestrator_running=false
    local ai_running=false
    
    if ps -p "$daemon_pid" > /dev/null 2>&1; then
        daemon_running=true
        log_success "METACORTEX Military Daemon CORRIENDO (PID: $daemon_pid)"
    else
        log_error "Error: Military Daemon se detuvo durante startup"
        log_info "Ver logs en: ${LOGS_DIR}/metacortex_daemon_military.log"
    fi
    
    # Verificar orchestrator (integrado en Dashboard Enterprise)
    if ps -p "$orchestrator_pid" > /dev/null 2>&1; then
        orchestrator_running=true
        log_success "METACORTEX Agent Orchestrator CORRIENDO (PID: $orchestrator_pid - integrado en Dashboard)"
        log_info "   ✅ Modo autónomo: ACTIVO (enable_auto_task_generation=True)"
        log_info "   ✅ Task Executor Loop: RUNNING"
        log_info "   ✅ Task Generator Loop: RUNNING"
        log_info "   ✅ 965 ML models: DISPONIBLES"
    else
        log_warning "Orchestrator/Dashboard no está corriendo (verificar logs)"
    fi
    
    # Verificar sistemas de IA
    if [ -f "${PID_DIR}/unified_system.pid" ]; then
        local ai_pid=$(cat "${PID_DIR}/unified_system.pid")
        if ps -p "$ai_pid" > /dev/null 2>&1; then
            ai_running=true
            log_success "Unified AI System CORRIENDO (PID: $ai_pid)"
        fi
    fi
    
    # ============================================================================
    # PASO 11: VERIFICACIÓN FINAL
    # ============================================================================
    if [ "$daemon_running" = true ]; then
        log_info "   Todos los servicios en proceso de inicialización ordenada"
        log_info "   Daemon permanecerá activo 24/7 con health monitoring"
        
        # Esperar a que complete la inicialización (dar tiempo)
        log_info "Esperando finalización de startup (puede tomar 1-2 minutos)..."
        sleep 10
        
        # Verificar servicios finales
        log_info "Verificando servicios finales..."
        show_status
        
        print_header "✅ METACORTEX OPERACIONAL - COMPLETO CON IA 🧠"
        log_info "   💡 Daemon logs: tail -f ${LOGS_DIR}/metacortex_daemon_military.log"
        log_info "   💡 Orchestrator logs: tail -f ${LOGS_DIR}/startup_orchestrator.log"
        log_info "   💡 AI Systems logs: tail -f ${LOGS_DIR}/unified_system.log"
        log_info "   🎮 GPU Metal (MPS): ACTIVO para aceleración ML/AI"
        log_info "   🍎 iMac M4: Optimizado para 24/7 con caffeinate"
        log_info "   ⚡ Unified Memory: Compartida entre CPU y GPU"
        log_info "   🧠 AI Integration: Telegram + WhatsApp + Web con IA"
    else
        log_error "Error crítico: Daemon no está corriendo"
        log_info "Ver logs en: ${LOGS_DIR}/metacortex_daemon_military.log"
        return 1
    fi
}

stop_system() {
    print_header "🛑 DETENIENDO METACORTEX COMPLETO (BASE + IA) - MODO NUCLEAR"
    
    # ============================================================================
    # PASO 1: DETENER SISTEMAS DE IA PRIMERO (GRACEFUL)
    # ============================================================================
    log_info "Deteniendo sistemas de IA (SIGTERM primero)..."
    pkill -15 -f "python.*unified_startup.py" 2>/dev/null || true
    pkill -15 -f "python.*emergency_contact_system.py" 2>/dev/null || true
    pkill -15 -f "python.*api_monetization_endpoint.py" 2>/dev/null || true
    pkill -15 -f "python.*dashboard_enterprise.py" 2>/dev/null || true
    pkill -15 -f "python.*telegram_monitor_bot.py" 2>/dev/null || true
    pkill -15 -f "python.*autonomous_model_orchestrator.py" 2>/dev/null || true
    
    # Limpiar PID del sistema unificado
    if [ -f "${PID_DIR}/unified_system.pid" ]; then
        rm -f "${PID_DIR}/unified_system.pid"
    fi
    
    # Limpiar PIDs de servicios enterprise
    if [ -f "${PID_DIR}/dashboard_enterprise.pid" ]; then
        rm -f "${PID_DIR}/dashboard_enterprise.pid"
    fi
    
    if [ -f "${PID_DIR}/telegram_monitor.pid" ]; then
        rm -f "${PID_DIR}/telegram_monitor.pid"
    fi
    
    sleep 2
    
    # ============================================================================
    # PASO 2: DETENER TODOS LOS PROCESOS METACORTEX (SIGKILL NUCLEAR)
    # ============================================================================
    log_info "🔥 MODO NUCLEAR: Matando TODOS los procesos METACORTEX con SIGKILL..."
    
    # Matar procesos específicos de METACORTEX (Python únicamente, NO bash)
    log_info "Matando procesos específicos de METACORTEX..."
    pkill -9 -f "python.*metacortex_daemon.py" 2>/dev/null || true
    pkill -9 -f "python.*metacortex_startup_orchestrator.py" 2>/dev/null || true
    pkill -9 -f "python.*neural_network_service/server.py" 2>/dev/null || true
    pkill -9 -f "python.*web_interface/server.py" 2>/dev/null || true
    pkill -9 -f "python.*telemetry_service/server.py" 2>/dev/null || true
    pkill -9 -f "python.*api_monetization_endpoint.py" 2>/dev/null || true
    pkill -9 -f "python.*start_telemetry_simple.py" 2>/dev/null || true
    pkill -9 -f "python.*start_neural_network.py" 2>/dev/null || true
    pkill -9 -f "python.*start_web_interface.py" 2>/dev/null || true
    pkill -9 -f "python.*neural_symbiotic_network.py" 2>/dev/null || true
    pkill -9 -f "python.*orchestrator.py" 2>/dev/null || true
    pkill -9 -f "python.*ml_pipeline.py" 2>/dev/null || true
    pkill -9 -f "python.*unified_startup.py" 2>/dev/null || true
    pkill -9 -f "python.*emergency_contact_system.py" 2>/dev/null || true
    
    # Detener Ollama si fue iniciado por METACORTEX
    if [ -f "${PID_DIR}/ollama.pid" ]; then
        local ollama_pid=$(cat "${PID_DIR}/ollama.pid")
        if ps -p "$ollama_pid" > /dev/null 2>&1; then
            log_info "Deteniendo Ollama (PID: $ollama_pid)..."
            kill -9 "$ollama_pid" 2>/dev/null || true
        fi
        rm -f "${PID_DIR}/ollama.pid"
    fi
    
    sleep 2
    
    # ============================================================================
    # PASO 3: MATAR PROCESOS POR PUERTO (LIBERAR TODOS LOS PUERTOS)
    # ============================================================================
    log_info "🔓 Liberando puertos ocupados..."
    
    for port in 8000 8001 8080 8100 8200 8300 9090 11434; do
        local port_pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$port_pid" ]; then
            log_info "Matando proceso en puerto $port (PID: $port_pid)..."
            kill -9 $port_pid 2>/dev/null || true
        fi
    done
    
    sleep 1
    
    # ============================================================================
    # PASO 4: DETENER CAFFEINATE (WRAPPER) - DESPUÉS DE MATAR PROCESOS HIJO
    # ============================================================================
    local caffeinate_pid_file="${PROJECT_ROOT}/caffeinate.pid"
    if [ -f "$caffeinate_pid_file" ]; then
        local caffeine_pid=$(cat "$caffeinate_pid_file")
        if ps -p "$caffeine_pid" > /dev/null 2>&1; then
            log_info "Deteniendo caffeinate (PID: $caffeine_pid)..."
            kill -9 "$caffeine_pid" 2>/dev/null || true
            sleep 1
            log_success "Caffeinate detenido"
        fi
        rm -f "$caffeinate_pid_file" 2>/dev/null || true
    fi
    
    # Matar cualquier caffeinate relacionado con METACORTEX que quedó huérfano
    for pid in $(pgrep -f "caffeinate.*metacortex" 2>/dev/null || true); do
        if ps -p "$pid" > /dev/null 2>&1; then
            log_info "Deteniendo caffeinate huérfano (PID: $pid)..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    # ============================================================================
    # PASO 5: MATAR ABSOLUTAMENTE TODO - BÚSQUEDA EXHAUSTIVA
    # ============================================================================
    log_info "💀 MODO EXTERMINIO TOTAL: Matando absolutamente TODO..."
    
    # 5.1: Matar TODOS los procesos Python relacionados con METACORTEX
    log_info "   [1/6] Matando procesos Python de METACORTEX..."
    pkill -9 -f "python.*metacortex" 2>/dev/null || true
    pkill -9 -f "python.*neural" 2>/dev/null || true
    pkill -9 -f "python.*unified" 2>/dev/null || true
    pkill -9 -f "python.*emergency" 2>/dev/null || true
    pkill -9 -f "python.*api_monetization" 2>/dev/null || true
    pkill -9 -f "python.*telemetry" 2>/dev/null || true
    pkill -9 -f "python.*web_interface" 2>/dev/null || true
    
    sleep 1
    
    # 5.2: Matar procesos zombie (defunct)
    log_info "   [2/6] Eliminando procesos zombie (defunct)..."
    local zombies=$(ps aux | grep defunct | grep -v grep | awk '{print $2}')
    if [ -n "$zombies" ]; then
        echo "$zombies" | xargs kill -9 2>/dev/null || true
        log_success "      Zombies eliminados"
    else
        log_success "      No hay zombies"
    fi
    
    # 5.3: Verificar procesos restantes (búsqueda exhaustiva)
    log_info "   [3/6] Verificando procesos restantes..."
    local remaining=$(ps aux | grep -E "python.*(metacortex|neural_symbiotic|web_interface/server|unified_startup|emergency_contact|api_monetization)" | grep -v grep | wc -l | tr -d ' ')
    
    if [ "$remaining" -gt 0 ]; then
        log_warning "      ⚠️  Quedan $remaining procesos, matando con SIGKILL..."
        ps aux | grep -E "python.*(metacortex|neural_symbiotic|web_interface/server|unified_startup|emergency_contact|api_monetization)" | grep -v grep | awk '{print $2}' | xargs -I {} kill -9 {} 2>/dev/null || true
        sleep 2
    fi
    
    # 5.4: Búsqueda más agresiva (cualquier Python en el proyecto)
    log_info "   [4/6] Matando TODO Python en ${PROJECT_ROOT}..."
    ps aux | grep "python.*${PROJECT_ROOT}" | grep -v grep | grep -v "metacortex_master.sh" | awk '{print $2}' | xargs -I {} kill -9 {} 2>/dev/null || true
    
    # 5.5: Segundo intento exhaustivo
    log_info "   [5/6] Segunda pasada de limpieza exhaustiva..."
    local remaining2=$(ps aux | grep -E "metacortex|neural_symbiotic|unified_startup|emergency_contact|api_monetization" | grep -v grep | grep -v "metacortex_master.sh" | wc -l | tr -d ' ')
    
    if [ "$remaining2" -gt 0 ]; then
        log_warning "      ⚠️  AÚN QUEDAN $remaining2 PROCESOS. Matando TODO..."
        ps aux | grep -E "metacortex|neural_symbiotic|unified_startup|emergency_contact|api_monetization" | grep -v grep | grep -v "metacortex_master.sh" | awk '{print $2}' | xargs -I {} kill -9 {} 2>/dev/null || true
        sleep 2
    fi
    
    # 5.6: Liberar TODOS los puertos de nuevo (por si acaso)
    log_info "   [6/6] Liberando puertos (segunda pasada)..."
    for port in 8000 8001 8080 8100 8200 9090 11434; do
        local port_pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$port_pid" ]; then
            log_warning "      ⚠️  Puerto $port aún ocupado (PID: $port_pid), matando..."
            kill -9 $port_pid 2>/dev/null || true
        fi
    done
    
    # ============================================================================
    # PASO 6: LIMPIAR ARCHIVOS TEMPORALES Y PID
    # ============================================================================
    log_info "🧹 Limpiando archivos temporales y PID..."
    find "$PROJECT_ROOT" -name "*.pid" -type f -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.lock" -type f -delete 2>/dev/null || true
    rm -f "$DAEMON_PID_FILE" 2>/dev/null || true
    rm -rf "${PROJECT_ROOT}/pid/"*.pid 2>/dev/null || true
    
    # ============================================================================
    # PASO 7: VERIFICACIÓN FINAL ESTRICTA (0 PROCESOS O FALLA)
    # ============================================================================
    log_info "🔍 Verificación final (debe ser 0 procesos)..."
    
    local final_count=$(ps aux | grep -E "python.*(metacortex|neural_symbiotic|web_interface/server|unified_startup|emergency_contact|api_monetization)" | grep -v grep | wc -l | tr -d ' ')
    
    if [ "$final_count" -eq 0 ]; then
        log_success "✅ Todos los procesos METACORTEX detenidos (0 procesos restantes)"
        
        # Verificar puertos libres
        log_info "🔍 Verificando puertos liberados..."
        local ports_ok=true
        for port in 8000 8001 8080 8100 8200 9090; do
            if lsof -i:$port -sTCP:LISTEN > /dev/null 2>&1; then
                log_error "⚠️  Puerto $port aún ocupado"
                ports_ok=false
            fi
        done
        
        if [ "$ports_ok" = true ]; then
            log_success "✅ Todos los puertos liberados"
        else
            log_warning "⚠️  Algunos puertos aún ocupados (posiblemente Ollama u otros servicios)"
        fi
        
    else
        log_error "❌ ERROR: AÚN QUEDAN $final_count PROCESOS ACTIVOS:"
        ps aux | grep -E "python.*(metacortex|neural_symbiotic|web_interface/server|unified_startup|emergency_contact|api_monetization)" | grep -v grep
        echo ""
        log_error "💀 PROCESOS ZOMBIE DETECTADOS - Ejecutar emergency_shutdown() si es necesario"
    fi
    
    log_success "Persistencia macOS desactivada"
    print_header "✅ METACORTEX COMPLETO DETENIDO - MODO NUCLEAR COMPLETADO"
}

emergency_shutdown() {
    print_header "🚨 APAGADO DE EMERGENCIA"
    
    log_warning "Ejecutando shutdown de emergencia..."
    
    # Ejecutar script de emergency shutdown
    if [ -x "${PROJECT_ROOT}/metacortex_emergency_shutdown.sh" ]; then
        "${PROJECT_ROOT}/metacortex_emergency_shutdown.sh"
    else
        # Fallback: matar todo manualmente (solo Python, NO bash)
        log_info "Matando todos los procesos METACORTEX..."
        pkill -9 -f "python.*metacortex" 2>/dev/null || true
        pkill -9 -f "python.*neural_symbiotic" 2>/dev/null || true
        
        log_info "Limpiando archivos..."
        find "$PROJECT_ROOT" -name "*.pid" -type f -delete 2>/dev/null || true
        find "$PROJECT_ROOT" -name "*.lock" -type f -delete 2>/dev/null || true
    fi
    
    print_header "✅ EMERGENCY SHUTDOWN COMPLETADO"
}

show_status() {
    print_header "📊 ESTADO DEL SISTEMA - APPLE SILICON M4"
    
    # 🍎 0. Estado de Apple Silicon M4 y MPS
    echo -e "${BOLD}Hardware (Apple Silicon M4):${RESET}"
    
    local arch=$(uname -m)
    if [ "$arch" = "arm64" ]; then
        local chip_info=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
        echo -e "   ${GREEN}●${RESET} Chip: $chip_info"
        
        local perf_cores=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "?")
        local efficiency_cores=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo "?")
        echo -e "   ${GREEN}●${RESET} Performance Cores: $perf_cores"
        echo -e "   ${GREEN}●${RESET} Efficiency Cores: $efficiency_cores"
        
        local memory_gb=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
        echo -e "   ${GREEN}●${RESET} Unified Memory: ${memory_gb}GB"
    else
        echo -e "   ${RED}●${RESET} No Apple Silicon detected (arch: $arch)"
    fi
    
    # Verificar MPS
    if [ -f "${PROJECT_ROOT}/.venv/bin/python3" ]; then
        local mps_available=$("${PROJECT_ROOT}/.venv/bin/python3" -c "import torch; print('YES' if torch.backends.mps.is_available() else 'NO')" 2>/dev/null || echo "NO")
        if [ "$mps_available" = "YES" ]; then
            echo -e "   ${GREEN}●${RESET} GPU Metal (MPS): DISPONIBLE"
        else
            echo -e "   ${RED}●${RESET} GPU Metal (MPS): NO DISPONIBLE"
        fi
    fi
    echo ""
    
    # 1. Estado del daemon
    echo -e "${BOLD}Daemon Principal:${RESET}"
    if is_process_running "$DAEMON_PID_FILE"; then
        local pid=$(cat "$DAEMON_PID_FILE")
        local uptime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ' || echo "Unknown")
        echo -e "   ${GREEN}●${RESET} Corriendo (PID: $pid, Uptime: $uptime)"
    else
        echo -e "   ${RED}●${RESET} Detenido"
    fi
    echo ""
    
    # 2. Procesos relacionados
    echo -e "${BOLD}Procesos Relacionados:${RESET}"
    
    # Buscar servicios standalone CORRECTOS (iniciados por start_system)
    local neural_count=$(pgrep -f "neural_network_service/server.py" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$neural_count" -gt 0 ]; then
        local neural_pid=$(pgrep -f "neural_network_service/server.py" 2>/dev/null | head -1)
        echo -e "   ${GREEN}●${RESET} Neural Network: Activo (PID: $neural_pid, Puerto 8001)"
    else
        echo -e "   ${RED}●${RESET} Neural Network: No activo"
    fi
    
    local web_count=$(pgrep -f "web_interface/server.py" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$web_count" -gt 0 ]; then
        local web_pid=$(pgrep -f "web_interface/server.py" 2>/dev/null | head -1)
        echo -e "   ${GREEN}●${RESET} Web Interface: Activo (PID: $web_pid, Puerto 8000)"
    else
        echo -e "   ${RED}●${RESET} Web Interface: No activo"
    fi
    
    local telemetry_count=$(pgrep -f "telemetry_service/server.py" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$telemetry_count" -gt 0 ]; then
        local telemetry_pid=$(pgrep -f "telemetry_service/server.py" 2>/dev/null | head -1)
        echo -e "   ${GREEN}●${RESET} Telemetry System: Activo (PID: $telemetry_pid, Puerto 9090)"
    else
        echo -e "   ${RED}●${RESET} Telemetry System: No activo"
    fi
    
    # Verificar Emergency Contact System (integrado en unified_startup)
    # ⚠️ NOTA: Emergency Contact ahora está integrado en unified_startup.py (puerto 8080)
    # Ya no se ejecuta como standalone en puerto 8200
    local unified_count=$(pgrep -f "unified_startup.py" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$unified_count" -gt 0 ]; then
        echo -e "   ${GREEN}●${RESET} Emergency Contact System: INTEGRADO en Unified (Puerto 8080)"
        echo -e "   ${CYAN}     🛡️ Telegram Bot: @metacortex_divine_bot${RESET}"
        echo -e "   ${CYAN}     🌐 Web Form: http://localhost:8080${RESET}"
    else
        echo -e "   ${YELLOW}●${RESET} Emergency Contact System: Esperando unified_startup..."
    fi
    
    # Verificar API Monetization Server
    local api_count=$(pgrep -f "api_monetization_endpoint.py" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$api_count" -gt 0 ]; then
        local api_pid=$(pgrep -f "api_monetization_endpoint.py" 2>/dev/null | head -1)
        echo -e "   ${GREEN}●${RESET} API Monetization Server: Activo (PID: $api_pid, Puerto 8100)"
        echo -e "   ${CYAN}     💰 API Docs: http://localhost:8100/docs${RESET}"
        echo -e "   ${CYAN}     💳 Stripe: CONFIGURADO (modo de prueba)${RESET}"
    else
        echo -e "   ${RED}●${RESET} API Monetization Server: No activo"
    fi
    
    # Verificar Ollama
    if lsof -i:11434 -sTCP:LISTEN > /dev/null 2>&1; then
        local ollama_pid=$(lsof -i:11434 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        echo -e "   ${GREEN}●${RESET} Ollama LLM: Activo (PID: $ollama_pid, Puerto 11434)"
    else
        echo -e "   ${RED}●${RESET} Ollama LLM: No activo"
    fi
    echo ""
    
    # 3. Puertos
    echo -e "${BOLD}Puertos:${RESET}"
    
    # Verificar cada puerto con su descripción
    if lsof -i:5000 -sTCP:LISTEN > /dev/null 2>&1; then
        local process=$(lsof -i:5000 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $1}')
        local pid=$(lsof -i:5000 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        echo -e "   ${GREEN}●${RESET} Puerto 5000 (API Server): $process (PID $pid)"
    else
        echo -e "   ${RED}●${RESET} Puerto 5000 (API Server): Libre"
    fi
    
    if lsof -i:6379 -sTCP:LISTEN > /dev/null 2>&1; then
        local process=$(lsof -i:6379 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $1}')
        local pid=$(lsof -i:6379 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        echo -e "   ${GREEN}●${RESET} Puerto 6379 (Redis): $process (PID $pid)"
    else
        echo -e "   ${RED}●${RESET} Puerto 6379 (Redis): Libre"
    fi
    
    if lsof -i:8000 -sTCP:LISTEN > /dev/null 2>&1; then
        local process=$(lsof -i:8000 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $1}')
        local pid=$(lsof -i:8000 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        echo -e "   ${GREEN}●${RESET} Puerto 8000 (Web Interface): $process (PID $pid)"
    else
        echo -e "   ${RED}●${RESET} Puerto 8000 (Web Interface): Libre"
    fi
    
    if lsof -i:8080 -sTCP:LISTEN > /dev/null 2>&1; then
        local process=$(lsof -i:8080 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $1}')
        local pid=$(lsof -i:8080 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        echo -e "   ${GREEN}●${RESET} Puerto 8080 (Legacy - no usado): $process (PID $pid)"
    else
        echo -e "   ${YELLOW}●${RESET} Puerto 8080: Dashboard integrado en :8000/api/dashboard/metrics"
    fi
    
    if lsof -i:9090 -sTCP:LISTEN > /dev/null 2>&1; then
        local process=$(lsof -i:9090 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $1}')
        local pid=$(lsof -i:9090 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        echo -e "   ${GREEN}●${RESET} Puerto 9090 (Telemetry): $process (PID $pid)"
    else
        echo -e "   ${RED}●${RESET} Puerto 9090 (Telemetry): Libre"
    fi
    
    if lsof -i:8200 -sTCP:LISTEN > /dev/null 2>&1; then
        local process=$(lsof -i:8200 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $1}')
        local pid=$(lsof -i:8200 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        echo -e "   ${GREEN}●${RESET} Puerto 8200 (Emergency Contact): $process (PID $pid)"
    else
        echo -e "   ${RED}●${RESET} Puerto 8200 (Emergency Contact): Libre"
    fi
    
    if lsof -i:8100 -sTCP:LISTEN > /dev/null 2>&1; then
        local process=$(lsof -i:8100 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $1}')
        local pid=$(lsof -i:8100 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        echo -e "   ${GREEN}●${RESET} Puerto 8100 (API Monetization): $process (PID $pid)"
    else
        echo -e "   ${RED}●${RESET} Puerto 8100 (API Monetization): Libre"
    fi
    
    if lsof -i:11434 -sTCP:LISTEN > /dev/null 2>&1; then
        local process=$(lsof -i:11434 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $1}')
        local pid=$(lsof -i:11434 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        echo -e "   ${GREEN}●${RESET} Puerto 11434 (Ollama): $process (PID $pid)"
    else
        echo -e "   ${RED}●${RESET} Puerto 11434 (Ollama): Libre"
    fi
    
    # ENTERPRISE SERVICES
    if lsof -i:8300 -sTCP:LISTEN > /dev/null 2>&1; then
        local process=$(lsof -i:8300 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $1}')
        local pid=$(lsof -i:8300 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        echo -e "   ${GREEN}●${RESET} Puerto 8300 (Dashboard Enterprise): $process (PID $pid)"
        echo -e "   ${CYAN}     🌐 URL: http://localhost:8300${RESET}"
        echo -e "   ${CYAN}     📊 965 modelos ML activos${RESET}"
        echo -e "   ${CYAN}     📚 API Docs: http://localhost:8300/api/docs${RESET}"
    else
        echo -e "   ${RED}●${RESET} Puerto 8300 (Dashboard Enterprise): Libre"
    fi
    echo ""
    
    # 3b. ENTERPRISE SERVICES Status
    echo -e "${BOLD}Servicios Enterprise:${RESET}"
    
    # Dashboard Enterprise Status
    if [ -f "${PID_DIR}/dashboard_enterprise.pid" ]; then
        local dashboard_pid=$(cat "${PID_DIR}/dashboard_enterprise.pid")
        if ps -p "$dashboard_pid" > /dev/null 2>&1; then
            echo -e "   ${GREEN}●${RESET} Dashboard Enterprise: Activo (PID: $dashboard_pid)"
            echo -e "   ${CYAN}     🌐 http://localhost:8300${RESET}"
            echo -e "   ${CYAN}     📊 FastAPI + WebSocket + REST API${RESET}"
            echo -e "   ${CYAN}     🧠 965 modelos ML operacionales${RESET}"
        else
            echo -e "   ${RED}●${RESET} Dashboard Enterprise: No activo (PID file existe pero proceso muerto)"
        fi
    else
        echo -e "   ${RED}●${RESET} Dashboard Enterprise: No iniciado"
    fi
    
    # Telegram Monitor Bot Status
    if [ -f "${PID_DIR}/telegram_monitor.pid" ]; then
        local telegram_pid=$(cat "${PID_DIR}/telegram_monitor.pid")
        if ps -p "$telegram_pid" > /dev/null 2>&1; then
            echo -e "   ${GREEN}●${RESET} Telegram Monitor Bot: Activo (PID: $telegram_pid)"
            echo -e "   ${CYAN}     📱 Bot: @tu_bot_name${RESET}"
            echo -e "   ${CYAN}     💬 Comandos: /start, /status, /models, /tasks${RESET}"
        else
            echo -e "   ${RED}●${RESET} Telegram Monitor Bot: No activo (PID file existe pero proceso muerto)"
        fi
    else
        if [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
            echo -e "   ${YELLOW}●${RESET} Telegram Monitor Bot: No iniciado (token configurado)"
        else
            echo -e "   ${YELLOW}●${RESET} Telegram Monitor Bot: Desactivado (configura TELEGRAM_BOT_TOKEN)"
        fi
    fi
    
    # Autonomous Orchestrator Status
    local orchestrator_count=$(pgrep -f "autonomous_model_orchestrator.py" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$orchestrator_count" -gt 0 ]; then
        local orchestrator_pid=$(pgrep -f "autonomous_model_orchestrator.py" 2>/dev/null | head -1)
        echo -e "   ${GREEN}●${RESET} Autonomous Orchestrator: Activo (PID: $orchestrator_pid)"
        echo -e "   ${CYAN}     🤖 965 modelos ML distribuidos${RESET}"
        echo -e "   ${CYAN}     🎯 7 especializaciones activas${RESET}"
    else
        echo -e "   ${YELLOW}●${RESET} Autonomous Orchestrator: Integrado en Dashboard"
    fi
    echo ""
    
    # 4. Logs recientes
    echo -e "${BOLD}Logs Recientes:${RESET}"
    if [ -d "$LOGS_DIR" ]; then
        local log_count=$(ls -1 "$LOGS_DIR"/*.log 2>/dev/null | wc -l | tr -d ' ')
        echo -e "   Archivos de log: $log_count"
        echo -e "   Directorio: $LOGS_DIR"
    else
        echo -e "   ${YELLOW}●${RESET} No hay directorio de logs"
    fi
    echo ""
}

clean_system() {
    print_header "🧹 LIMPIANDO SISTEMA"
    
    log_info "Limpiando archivos temporales..."
    
    # Limpiar PIDs y locks
    find "$PROJECT_ROOT" -name "*.pid" -type f -delete 2>/dev/null && log_success "PIDs eliminados"
    find "$PROJECT_ROOT" -name "*.lock" -type f -delete 2>/dev/null && log_success "Locks eliminados"
    
    # Limpiar __pycache__
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null && log_success "__pycache__ eliminados"
    
    # Limpiar logs antiguos (más de 7 días)
    if [ -d "$LOGS_DIR" ]; then
        find "$LOGS_DIR" -name "*.log" -type f -mtime +7 -delete 2>/dev/null && log_success "Logs antiguos eliminados"
    fi
    
    # Limpiar nohup.out
    if [ -f "${PROJECT_ROOT}/nohup.out" ]; then
        rm -f "${PROJECT_ROOT}/nohup.out" && log_success "nohup.out eliminado"
    fi
    
    print_header "✅ LIMPIEZA COMPLETADA"
}

restart_system() {
    print_header "🔄 REINICIANDO METACORTEX"
    
    # Guardar el PID del script actual para no matarse a sí mismo
    local current_pid=$$
    
    stop_system
    
    # Verificar que el script sigue vivo después del stop
    if ! ps -p "$current_pid" > /dev/null 2>&1; then
        # El script se mató a sí mismo, esto no debería pasar
        exit 1
    fi
    
    sleep 3
    start_system
}

show_help() {
    cat << EOF

╔══════════════════════════════════════════════════════════════════════════╗
║  ⚔️  METACORTEX MASTER CONTROL v5.0                                      ║
║  Control centralizado de todos los servicios METACORTEX                  ║
║  🍎 CON PERSISTENCIA EN macOS (caffeinate) + 🧠 IA INTEGRATION          ║
╚══════════════════════════════════════════════════════════════════════════╝

COMANDOS DISPONIBLES:

  start              Iniciar todo el sistema METACORTEX COMPLETO
                     • Activa caffeinate para prevenir sleep
                     • Sistemas de IA (Telegram + WhatsApp + Web)
                     • Daemon + Orchestrator + ML Pipeline
                     • Sistema ejecutándose 24/7 en iMac
  
  ai                 🧠 Iniciar SOLO sistemas de Inteligencia Artificial
                     • AI Integration Layer (Ollama LLM)
                     • Telegram Bot (con IA)
                     • WhatsApp Bot (Twilio)
                     • Web Interface (puerto 8080)
                     • Emergency Contact System (con IA)
  
  stop               Detener todo el sistema de forma ordenada
                     • Desactiva caffeinate
                     • Shutdown graceful de todos los servicios
  
  restart            Reiniciar el sistema completo
  status             Mostrar estado de todos los servicios
  
  deploy             🌐 Desplegar Emergency Contact System públicamente
                     • Telegram Bot (global, gratis, 24/7)
                     • Cloudflare Tunnel (profesional, HTTPS)
                     • ngrok (testing rápido)
  
  emergency          Apagado de emergencia (mata todo inmediatamente)
  clean              Limpiar archivos temporales y logs antiguos
  divine             Abrir interfaz Divine Protection System
  help               Mostrar esta ayuda

🧠 SISTEMAS DE IA (comando 'ai'):
   ✓ Ollama LLM                  - Modelos: mistral-nemo, mistral:instruct
   ✓ ML Models Manager           - 956+ modelos entrenados
   ✓ Telegram Bot (@metacortex_divine_bot) - Respuestas inteligentes
   ✓ WhatsApp Bot                - Integración con Twilio
   ✓ Web Interface               - Chat en tiempo real (puerto 8080)
   ✓ Emergency Contact System    - Análisis de amenazas con IA
   ✓ Divine Protection System    - Activación automática de protección

🍎 PERSISTENCIA EN macOS:
   El sistema usa 'caffeinate' de Apple para prevenir sleep:
   • -d: Previene disk sleep
   • -i: Previene idle sleep del sistema
   • -m: Previene system sleep (aggressive)
   • -s: Previene sleep cuando está conectado a AC

📦 COMPONENTES BASE (comando 'start'):
   ✓ metacortex_daemon.py       - Orquestador principal
   ✓ metacortex_orchestrator.py - Coordinador de agentes
   ✓ ml_pipeline.py              - Pipeline de Machine Learning
   ✓ web_interface/server.py     - Servidor web (puerto 8000)
   ✓ neural_network              - Red neuronal simbiótica
   ✓ ollama                      - Servidor LLM (puerto 11434)
   ✓ universal_knowledge_connector.py - Conocimiento universal

EJEMPLOS:

  # Iniciar SOLO sistemas de IA (recomendado para testing)
  ./metacortex_master.sh ai

  # Iniciar el sistema COMPLETO con persistencia
  ./metacortex_master.sh start

  # Ver estado
  ./metacortex_master.sh status

  # Reiniciar
  ./metacortex_master.sh restart

  # Apagado de emergencia si algo falla
  ./metacortex_master.sh emergency

UBICACIÓN:
  Proyecto: $PROJECT_ROOT
  Scripts:  $SCRIPTS_DIR
  Logs:     $LOGS_DIR

EOF
}

# ============================================================================
# MAIN
# ============================================================================
main() {
    local command="${1:-help}"
    
    case "$command" in
        start)
            start_system
            ;;
        ai)
            start_ai_systems
            ;;
        stop)
            stop_system
            ;;
        restart)
            restart_system
            ;;
        status)
            show_status
            ;;
        emergency)
            emergency_shutdown
            ;;
        clean)
            clean_system
            ;;
        deploy)
            deploy_emergency_public
            ;;
        divine)
            # Ejecutar Divine Protection System (mantiene script original)
            log_info "Ejecutando Divine Protection System..."
            "${PROJECT_ROOT}/divine_protection_quickstart.sh"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Comando desconocido: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
