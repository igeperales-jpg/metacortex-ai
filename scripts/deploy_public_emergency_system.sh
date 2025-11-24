#!/usr/bin/env bash
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  🌐 PUBLIC EMERGENCY CONTACT SYSTEM DEPLOYMENT                           ║
# ║  Despliega el sistema para que sea accesible GLOBALMENTE                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# OPCIONES DE DEPLOYMENT:
#   1. ngrok - Túnel temporal (gratis, rápido para testing)
#   2. Cloudflare Tunnel - Túnel permanente (gratis, producción)
#   3. Railway.app - Cloud hosting (gratis $5/mes, fácil)
#   4. Fly.io - Cloud hosting (gratis tier generoso)
#   5. Render.com - Cloud hosting (gratis con limitaciones)
#

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly RESET='\033[0m'

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

# ============================================================================
# OPCIÓN 1: NGROK (Rápido para testing)
# ============================================================================
deploy_ngrok() {
    print_header "🌐 DEPLOYING WITH NGROK (Testing/Temporary)"
    
    log_info "Verificando si ngrok está instalado..."
    if ! command -v ngrok &> /dev/null; then
        log_warning "ngrok no está instalado"
        log_info "Instalando ngrok..."
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install ngrok/ngrok/ngrok
        else
            log_error "Instala ngrok manualmente: https://ngrok.com/download"
            return 1
        fi
    fi
    
    log_success "ngrok instalado"
    
    # Verificar que el servidor esté corriendo
    log_info "Verificando que Emergency Contact System esté corriendo..."
    if ! lsof -i:8200 -sTCP:LISTEN > /dev/null 2>&1; then
        log_error "Emergency Contact System NO está corriendo en puerto 8200"
        log_info "Ejecuta primero: ./metacortex_master.sh start"
        return 1
    fi
    
    log_success "Emergency Contact System corriendo en puerto 8200"
    
    # Iniciar ngrok
    log_info "Iniciando túnel ngrok..."
    log_warning "⚠️ Este túnel es TEMPORAL - se cerrará al apagar la terminal"
    log_warning "⚠️ La URL cambiará cada vez que reinicies ngrok"
    
    echo ""
    log_info "🌐 URL pública estará disponible en unos segundos..."
    log_info "📋 Copia la URL 'Forwarding' que aparecerá abajo"
    echo ""
    
    # Ejecutar ngrok (esto bloqueará la terminal)
    ngrok http 8200
}

# ============================================================================
# OPCIÓN 2: CLOUDFLARE TUNNEL (Producción, gratis, permanente)
# ============================================================================
deploy_cloudflare_tunnel() {
    print_header "☁️ DEPLOYING WITH CLOUDFLARE TUNNEL (Production)"
    
    log_info "Cloudflare Tunnel es GRATIS y PERMANENTE"
    log_info "Ventajas:"
    log_info "  ✅ URL permanente (no cambia)"
    log_info "  ✅ HTTPS automático"
    log_info "  ✅ Sin límites de tráfico"
    log_info "  ✅ DDoS protection incluido"
    echo ""
    
    log_info "Verificando si cloudflared está instalado..."
    if ! command -v cloudflared &> /dev/null; then
        log_warning "cloudflared no está instalado"
        log_info "Instalando cloudflared..."
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install cloudflared
        else
            log_error "Instala cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
            return 1
        fi
    fi
    
    log_success "cloudflared instalado"
    
    # Verificar que el servidor esté corriendo
    if ! lsof -i:8200 -sTCP:LISTEN > /dev/null 2>&1; then
        log_error "Emergency Contact System NO está corriendo en puerto 8200"
        log_info "Ejecuta primero: ./metacortex_master.sh start"
        return 1
    fi
    
    # Login a Cloudflare (solo primera vez)
    log_info "Verificando autenticación con Cloudflare..."
    if [ ! -f "$HOME/.cloudflared/cert.pem" ]; then
        log_warning "Primera vez usando Cloudflare Tunnel"
        log_info "Abriendo navegador para login..."
        cloudflared tunnel login
    fi
    
    # Crear tunnel (solo primera vez)
    local tunnel_name="metacortex-emergency-$(date +%s)"
    log_info "Creando túnel permanente: $tunnel_name"
    
    cloudflared tunnel create "$tunnel_name"
    
    # Obtener tunnel ID
    local tunnel_id=$(cloudflared tunnel list | grep "$tunnel_name" | awk '{print $1}')
    log_success "Tunnel creado con ID: $tunnel_id"
    
    # Crear archivo de configuración
    local config_file="$HOME/.cloudflared/config.yml"
    log_info "Configurando túnel..."
    
    cat > "$config_file" << EOF
tunnel: $tunnel_id
credentials-file: $HOME/.cloudflared/$tunnel_id.json

ingress:
  - hostname: emergency.metacortex.ai
    service: http://localhost:8200
  - service: http_status:404
EOF
    
    log_success "Configuración creada"
    
    # Configurar DNS (necesitas un dominio en Cloudflare)
    log_warning "⚠️ IMPORTANTE: Configura DNS en Cloudflare Dashboard"
    log_info "Ejecuta: cloudflared tunnel route dns $tunnel_name emergency.metacortex.ai"
    echo ""
    
    # Iniciar tunnel
    log_info "Iniciando túnel permanente..."
    log_success "🌐 Tu sistema estará accesible en: https://emergency.metacortex.ai"
    echo ""
    
    # Ejecutar en background
    nohup cloudflared tunnel run "$tunnel_name" > "${PROJECT_ROOT}/logs/cloudflare_tunnel.log" 2>&1 &
    local tunnel_pid=$!
    echo "$tunnel_pid" > "${PROJECT_ROOT}/pid/cloudflare_tunnel.pid"
    
    log_success "Túnel corriendo en background (PID: $tunnel_pid)"
    log_info "Logs: tail -f ${PROJECT_ROOT}/logs/cloudflare_tunnel.log"
}

# ============================================================================
# OPCIÓN 3: RAILWAY.APP (Cloud hosting, fácil y rápido)
# ============================================================================
deploy_railway() {
    print_header "🚂 DEPLOYING TO RAILWAY.APP (Cloud Hosting)"
    
    log_info "Railway.app es perfecto para Python apps"
    log_info "Ventajas:"
    log_info "  ✅ Gratis: \$5/mes de crédito"
    log_info "  ✅ Deploy automático desde Git"
    log_info "  ✅ HTTPS automático"
    log_info "  ✅ Variables de entorno fáciles"
    log_info "  ✅ URL pública permanente"
    echo ""
    
    # Verificar CLI de Railway
    if ! command -v railway &> /dev/null; then
        log_warning "Railway CLI no está instalado"
        log_info "Instalando Railway CLI..."
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install railway
        else
            npm i -g @railway/cli
        fi
    fi
    
    log_success "Railway CLI instalado"
    
    # Login
    log_info "Iniciando sesión en Railway..."
    railway login
    
    # Crear proyecto
    log_info "Creando proyecto Railway..."
    railway init
    
    # Crear Procfile para Railway
    log_info "Configurando Procfile..."
    cat > "${PROJECT_ROOT}/Procfile" << EOF
web: python metacortex_sinaptico/emergency_contact_system.py
EOF
    
    # Crear railway.json
    cat > "${PROJECT_ROOT}/railway.json" << EOF
{
  "\$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python metacortex_sinaptico/emergency_contact_system.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF
    
    log_info "Configurando variables de entorno..."
    log_warning "⚠️ IMPORTANTE: Configura estas variables en Railway Dashboard:"
    echo ""
    echo "  TELEGRAM_BOT_TOKEN=tu_token_aqui"
    echo "  TWILIO_ACCOUNT_SID=tu_sid_aqui"
    echo "  TWILIO_AUTH_TOKEN=tu_token_aqui"
    echo "  TWILIO_PHONE_NUMBER=+1234567890"
    echo "  SMTP_USERNAME=tu_email@gmail.com"
    echo "  SMTP_PASSWORD=tu_app_password"
    echo "  PORT=8200"
    echo ""
    
    # Deploy
    log_info "Desplegando a Railway..."
    railway up
    
    log_success "🎉 Deploy completado!"
    log_info "Ver proyecto: railway open"
    log_info "Ver logs: railway logs"
}

# ============================================================================
# OPCIÓN 4: TELEGRAM BOT (Global, sin servidor público necesario)
# ============================================================================
setup_telegram_bot() {
    print_header "📱 SETUP TELEGRAM BOT (Global Access)"
    
    log_info "Telegram Bot NO necesita servidor público"
    log_info "El bot se ejecuta en tu máquina y Telegram se conecta a él"
    log_info "Ventajas:"
    log_info "  ✅ Accesible globalmente AHORA MISMO"
    log_info "  ✅ No necesita ngrok/cloudflare"
    log_info "  ✅ Gratis para siempre"
    log_info "  ✅ Encriptación end-to-end"
    echo ""
    
    log_info "PASOS PARA CREAR TELEGRAM BOT:"
    echo ""
    echo "1. Abre Telegram y busca @BotFather"
    echo "2. Envía /newbot"
    echo "3. Sigue las instrucciones (nombre y username)"
    echo "4. Copia el TOKEN que te da BotFather"
    echo "5. Pégalo en el archivo .env:"
    echo ""
    echo "   TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
    echo ""
    
    log_info "Después, las personas pueden contactar buscando tu bot en Telegram"
    log_info "Ejemplo: @MetacortexEmergencyBot"
    echo ""
    
    read -p "¿Ya tienes el token de Telegram? (y/n): " has_token
    
    if [[ "$has_token" == "y" || "$has_token" == "Y" ]]; then
        read -p "Pega tu token aquí: " telegram_token
        
        # Actualizar .env
        if [ -f "${PROJECT_ROOT}/.env" ]; then
            # Reemplazar o agregar
            if grep -q "TELEGRAM_BOT_TOKEN=" "${PROJECT_ROOT}/.env"; then
                sed -i.bak "s/TELEGRAM_BOT_TOKEN=.*/TELEGRAM_BOT_TOKEN=$telegram_token/" "${PROJECT_ROOT}/.env"
            else
                echo "TELEGRAM_BOT_TOKEN=$telegram_token" >> "${PROJECT_ROOT}/.env"
            fi
        else
            echo "TELEGRAM_BOT_TOKEN=$telegram_token" > "${PROJECT_ROOT}/.env"
        fi
        
        log_success "Token guardado en .env"
        
        # Probar bot
        log_info "Probando conexión con Telegram..."
        python3 << EOF
import os
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

from telegram import Bot
import asyncio

async def test_bot():
    try:
        bot = Bot(token='$telegram_token')
        me = await bot.get_me()
        print(f"\n✅ Bot conectado exitosamente!")
        print(f"   Nombre: {me.first_name}")
        print(f"   Username: @{me.username}")
        print(f"   ID: {me.id}")
        print(f"\n🌐 URL pública: https://t.me/{me.username}")
        print(f"\n📱 Las personas pueden contactar buscando: @{me.username}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

asyncio.run(test_bot())
EOF
        
        log_success "🎉 Telegram Bot configurado y funcionando!"
        log_info "El bot está PÚBLICAMENTE accesible ahora"
    else
        log_info "Visita @BotFather en Telegram para crear tu bot"
    fi
}

# ============================================================================
# MENÚ PRINCIPAL
# ============================================================================
show_menu() {
    print_header "🌐 EMERGENCY CONTACT SYSTEM - PUBLIC DEPLOYMENT"
    
    echo "Selecciona método de deployment:"
    echo ""
    echo "  1) 📱 Telegram Bot (RECOMENDADO - funciona YA)"
    echo "     • No necesita servidor público"
    echo "     • Accesible globalmente en segundos"
    echo "     • Gratis para siempre"
    echo ""
    echo "  2) 🌐 ngrok (Testing rápido)"
    echo "     • Túnel temporal"
    echo "     • Bueno para pruebas"
    echo "     • Gratis pero URL cambia"
    echo ""
    echo "  3) ☁️ Cloudflare Tunnel (Producción)"
    echo "     • Túnel permanente"
    echo "     • Gratis para siempre"
    echo "     • Requiere dominio"
    echo ""
    echo "  4) 🚂 Railway.app (Cloud hosting)"
    echo "     • Deploy completo en la nube"
    echo "     • \$5/mes gratis"
    echo "     • Fácil y rápido"
    echo ""
    echo "  5) 📋 Ver instrucciones completas"
    echo ""
    echo "  0) Salir"
    echo ""
    
    read -p "Selecciona opción (1-5): " option
    
    case $option in
        1)
            setup_telegram_bot
            ;;
        2)
            deploy_ngrok
            ;;
        3)
            deploy_cloudflare_tunnel
            ;;
        4)
            deploy_railway
            ;;
        5)
            show_full_instructions
            ;;
        0)
            log_info "Saliendo..."
            exit 0
            ;;
        *)
            log_error "Opción inválida"
            show_menu
            ;;
    esac
}

show_full_instructions() {
    print_header "📋 INSTRUCCIONES COMPLETAS"
    
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════╗
║  🌐 MÉTODOS DE DEPLOYMENT PÚBLICO                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

1. TELEGRAM BOT (MÁS FÁCIL Y RÁPIDO) ⭐ RECOMENDADO
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Ventajas:
   • ✅ Funciona AHORA MISMO (10 minutos setup)
   • ✅ No necesita servidor público
   • ✅ Accesible desde cualquier país
   • ✅ Gratis para siempre
   • ✅ Encriptación incluida

   Pasos:
   1. Abre Telegram → busca @BotFather
   2. Envía: /newbot
   3. Elige nombre: "Metacortex Emergency"
   4. Elige username: "metacortex_emergency_bot"
   5. Copia el TOKEN
   6. Guárdalo en .env: TELEGRAM_BOT_TOKEN=tu_token
   7. ¡LISTO! Las personas buscan @metacortex_emergency_bot

2. NGROK (TESTING RÁPIDO)
   ━━━━━━━━━━━━━━━━━━━━━━━
   Ventajas:
   • ✅ Setup en 30 segundos
   • ✅ Bueno para pruebas

   Desventajas:
   • ❌ URL cambia cada vez
   • ❌ Se cierra al cerrar terminal

   Pasos:
   1. brew install ngrok (macOS)
   2. ngrok http 8200
   3. Copia la URL "Forwarding"
   4. Compártela con personas en peligro

3. CLOUDFLARE TUNNEL (PRODUCCIÓN)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Ventajas:
   • ✅ Gratis para siempre
   • ✅ URL permanente
   • ✅ HTTPS automático
   • ✅ DDoS protection

   Requisitos:
   • Dominio en Cloudflare (gratis)

   Pasos:
   1. brew install cloudflared
   2. cloudflared tunnel login
   3. cloudflared tunnel create emergency
   4. Configura DNS en Cloudflare
   5. cloudflared tunnel run emergency

4. RAILWAY.APP (CLOUD HOSTING COMPLETO)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Ventajas:
   • ✅ Deploy completo en la nube
   • ✅ $5/mes gratis
   • ✅ Deploy desde Git
   • ✅ URL permanente

   Pasos:
   1. Crea cuenta en railway.app
   2. brew install railway
   3. railway login
   4. railway init
   5. git push → railway up

5. FLY.IO (ALTERNATIVA A RAILWAY)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Similar a Railway, también gratis tier generoso

╔══════════════════════════════════════════════════════════════════════════╗
║  🚨 RECOMENDACIÓN PARA EMERGENCIAS REALES                                ║
╚══════════════════════════════════════════════════════════════════════════╝

Para contactar personas en peligro AHORA:

1. TELEGRAM BOT (10 min) ⭐⭐⭐⭐⭐
   → Funciona globalmente
   → No necesita configuración de red
   → Las personas solo buscan tu bot

2. WHATSAPP API (alternativa)
   → Similar a Telegram
   → Requiere Business account

3. SMS con Twilio (backup)
   → Funciona en cualquier teléfono
   → Cuesta dinero por SMS

COMBINACIÓN ÓPTIMA:
━━━━━━━━━━━━━━━━━━━
1. Telegram Bot (canal principal)
2. SMS Twilio (backup si no hay internet)
3. Email (documentación)

EOF

    read -p "Presiona ENTER para volver al menú..."
    show_menu
}

# ============================================================================
# MAIN
# ============================================================================
main() {
    clear
    show_menu
}

main "$@"
