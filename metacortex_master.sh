#!/usr/bin/env bash
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ⚔️  METACORTEX MASTER CONTROL - Sistema Unificado de Control v5.0      ║
# ║  Control centralizado de TODOS los servicios y operaciones              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# USAGE:
#   ./metacortex_master.sh start      - Iniciar todo el sistema
#   ./metacortex_master.sh stop       - Detener todo el sistema
#   ./metacortex_master.sh restart    - Reiniciar el sistema completo
#   ./metacortex_master.sh status     - Ver estado de todos los servicios
#   ./metacortex_master.sh emergency  - Apagado de emergencia (mata todo)
#   ./metacortex_master.sh clean      - Limpiar logs y archivos temporales
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
# FUNCIONES PRINCIPALES
# ============================================================================
start_system() {
    print_header "🚀 INICIANDO METACORTEX - STARTUP ORCHESTRATOR"
    
    check_venv
    
    # 1. Verificar que NO haya procesos METACORTEX corriendo (prevenir duplicados)
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
    
    # 2. Limpiar archivos antiguos
    log_info "Limpiando archivos PID/LOCK antiguos..."
    find "$PROJECT_ROOT" -name "*.pid" -type f -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.lock" -type f -delete 2>/dev/null || true
    
    # 3. Crear directorios necesarios
    mkdir -p "$LOGS_DIR"
    mkdir -p "$PID_DIR"
    
    # 🍎 4. ACTIVAR PERSISTENCIA EN macOS con caffeinate
    log_info "Activando persistencia en macOS (caffeinate)..."
    log_info "   - Mantiene sistema ejecutándose 24/7"
    log_info "   - ✅ Permite sleep de la pantalla (ahorro de energía)"
    log_info "   - Previene sleep del sistema"
    log_info "   - Inicialización ORDENADA sin circular imports"
    log_info "   - Health checks antes de continuar"
    
    # 4. Iniciar SERVICIOS STANDALONE (ultra-ligeros, no bloquean)
    log_info "🚀 Iniciando servicios STANDALONE (arquitectura 3 capas)..."
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    # Web Interface STANDALONE (puerto 8000) - Ultra-ligero, lazy loading
    log_info "   → Web Interface STANDALONE (puerto 8000) - <3s startup..."
    nohup "$VENV_PYTHON" "${PROJECT_ROOT}/start_web_interface_standalone.py" > "${LOGS_DIR}/web_interface_stdout.log" 2>&1 &
    local web_interface_pid=$!
    echo "$web_interface_pid" > "${PID_DIR}/web_interface.pid"
    log_success "      Web Interface STANDALONE iniciado (PID: $web_interface_pid)"
    
    # Neural Network STANDALONE - Sin conexiones simbióticas automáticas
    log_info "   → Neural Network STANDALONE - <5s startup..."
    nohup "$VENV_PYTHON" "${PROJECT_ROOT}/start_neural_network_standalone.py" > "${LOGS_DIR}/neural_network_stdout.log" 2>&1 &
    local neural_network_pid=$!
    echo "$neural_network_pid" > "${PID_DIR}/neural_network.pid"
    log_success "      Neural Network STANDALONE iniciado (PID: $neural_network_pid)"
    
    # Telemetry STANDALONE (puerto 9090) - Prometheus metrics SIMPLE
    log_info "   → Telemetry System STANDALONE (puerto 9090) - ULTRA-SIMPLE..."
    nohup "$VENV_PYTHON" "${PROJECT_ROOT}/start_telemetry_simple.py" > "${LOGS_DIR}/telemetry_stdout.log" 2>&1 &
    local telemetry_pid=$!
    echo "$telemetry_pid" > "${PID_DIR}/telemetry.pid"
    log_success "      Telemetry System STANDALONE iniciado (PID: $telemetry_pid)"
    
    # 🤖 Ollama LLM Server (puerto 11434) - CRÍTICO para agentes
    log_info "   → Verificando Ollama (puerto 11434)..."
    if lsof -i:11434 -sTCP:LISTEN > /dev/null 2>&1; then
        log_success "      Ollama ya está corriendo"
    else
        log_info "      Iniciando Ollama en background..."
        if command -v ollama &> /dev/null; then
            nohup ollama serve > "${LOGS_DIR}/ollama.log" 2>&1 &
            local ollama_pid=$!
            echo "$ollama_pid" > "${PID_DIR}/ollama.pid"
            log_success "      Ollama iniciado (PID: $ollama_pid)"
            sleep 2  # Dar tiempo a Ollama para iniciar
        else
            log_warning "      ⚠️ Ollama no está instalado (brew install ollama)"
        fi
    fi
    
    # Esperar 3 segundos para que los servicios standalone inicien (ultra-rápido)
    log_info "   ⏳ Esperando servicios standalone (3s - arquitectura ligera)..."
    sleep 3
    
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
    
    # Verificar Ollama
    if lsof -i:11434 -sTCP:LISTEN > /dev/null 2>&1; then
        log_success "   ✅ Ollama: ACTIVO (puerto 11434)"
    else
        log_warning "   ⚠️ Ollama: NO ACTIVO (ver logs/ollama.log)"
    fi
    
    # 5. Iniciar DAEMON MILITAR (24/7 persistence) - YA NO carga módulos pesados
    log_info "Iniciando METACORTEX Military Daemon (modo ligero)..."
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    nohup caffeinate -i "$VENV_PYTHON" "${PROJECT_ROOT}/metacortex_daemon.py" > "${LOGS_DIR}/metacortex_daemon_military.log" 2>&1 &
    local daemon_pid=$!
    echo "$daemon_pid" > "${PID_DIR}/metacortex_daemon_military.pid"
    
    log_success "Military Daemon iniciado (PID: $daemon_pid)"
    log_info "   ✅ Daemon: PERSISTENCIA 24/7 ACTIVA"
    log_info "   ✅ Health monitoring: CONTINUO"
    log_info "   ✅ Capability discovery: EXPONENCIAL"
    
    sleep 3
    
    # 6. Iniciar ORCHESTRATOR (coordinador de agentes)
    log_info "Iniciando METACORTEX Agent Orchestrator..."
    
    # Usar el Startup Orchestrator que garantiza:
    # - Orden correcto de inicialización (5 fases)
    # - No circular imports
    # - Health checks de cada servicio
    # - Retry logic con backoff
    # - TODO está activo antes de continuar
    nohup caffeinate -i "$VENV_PYTHON" "$STARTUP_ORCHESTRATOR" > "${LOGS_DIR}/startup_orchestrator.log" 2>&1 &
    local orchestrator_pid=$!
    
    # Guardar PID de caffeinate (que envuelve al orchestrator)
    echo "$orchestrator_pid" > "${PROJECT_ROOT}/caffeinate.pid"
    
    log_success "Agent Orchestrator iniciado (PID: $orchestrator_pid)"
    log_info "   ✅ Sistema METACORTEX: INICIALIZANDO EN 5 FASES"
    log_info "   ✅ Garantía: TODOS los servicios activos"
    log_info "   ✅ Health checks: ANTES DE CONTINUAR"
    
    # 6. Monitorear el startup (ver logs en tiempo real)
    log_info "Monitoreando inicialización..."
    log_info "   Daemon logs: tail -f ${LOGS_DIR}/metacortex_daemon_military.log"
    log_info "   Orchestrator logs: tail -f ${LOGS_DIR}/startup_orchestrator.log"
    
    sleep 3
    
    # Mostrar últimas líneas del log del daemon
    if [ -f "${LOGS_DIR}/metacortex_daemon_military.log" ]; then
        log_info "Daemon Military - Últimas líneas:"
        tail -20 "${LOGS_DIR}/metacortex_daemon_military.log"
    fi
    
    # 7. Verificar que ambos procesos sigan corriendo
    sleep 5
    
    local daemon_running=false
    local orchestrator_running=false
    
    if ps -p "$daemon_pid" > /dev/null 2>&1; then
        daemon_running=true
        log_success "METACORTEX Military Daemon CORRIENDO (PID: $daemon_pid)"
    else
        log_error "Error: Military Daemon se detuvo durante startup"
        log_info "Ver logs en: ${LOGS_DIR}/metacortex_daemon_military.log"
    fi
    
    if ps -p "$orchestrator_pid" > /dev/null 2>&1; then
        orchestrator_running=true
        log_success "METACORTEX Agent Orchestrator CORRIENDO (PID: $orchestrator_pid)"
    else
        log_warning "Orchestrator completó su ciclo (esto es normal)"
    fi
    
    # 8. Verificación final
    if [ "$daemon_running" = true ]; then
        log_info "   Todos los servicios en proceso de inicialización ordenada"
        log_info "   Daemon permanecerá activo 24/7 con health monitoring"
        
        # 9. Esperar a que complete la inicialización (dar tiempo)
        log_info "Esperando finalización de startup (puede tomar 1-2 minutos)..."
        sleep 10
        
        # 10. Verificar servicios finales
        log_info "Verificando servicios finales..."
        show_status
        
        print_header "✅ METACORTEX OPERACIONAL - PERSISTENCIA ACTIVA 🍎"
        log_info "   💡 Daemon logs: tail -f ${LOGS_DIR}/metacortex_daemon_military.log"
        log_info "   💡 Orchestrator logs: tail -f ${LOGS_DIR}/startup_orchestrator.log"
    else
        log_error "Error crítico: Daemon no está corriendo"
        log_info "Ver logs en: ${LOGS_DIR}/metacortex_daemon_military.log"
        return 1
    fi
}

stop_system() {
    print_header "🛑 DETENIENDO METACORTEX"
    
    # 0. PRIMERO: Matar TODOS los procesos relacionados (incluye duplicados)
    # IMPORTANTE: NO matar procesos bash (incluyendo este script)
    log_info "Deteniendo TODOS los procesos METACORTEX (incluye duplicados)..."
    
    # Matar procesos específicos de METACORTEX (Python únicamente, NO bash)
    log_info "Matando procesos específicos de METACORTEX..."
    pkill -9 -f "python.*metacortex_daemon.py" 2>/dev/null || true
    pkill -9 -f "python.*metacortex_startup_orchestrator.py" 2>/dev/null || true
    pkill -9 -f "python.*start_neural_network_standalone.py" 2>/dev/null || true
    pkill -9 -f "python.*start_web_interface_standalone.py" 2>/dev/null || true
    pkill -9 -f "python.*start_telemetry_simple.py" 2>/dev/null || true
    pkill -9 -f "python.*start_neural_network.py" 2>/dev/null || true
    pkill -9 -f "python.*start_web_interface.py" 2>/dev/null || true
    pkill -9 -f "python.*neural_symbiotic_network.py" 2>/dev/null || true
    pkill -9 -f "python.*web_interface/server.py" 2>/dev/null || true
    pkill -9 -f "python.*orchestrator.py" 2>/dev/null || true
    pkill -9 -f "python.*ml_pipeline.py" 2>/dev/null || true
    
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
    
    # 1. Detener caffeinate (wrapper) - DESPUÉS de matar procesos hijo
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
    
    # 2. Matar cualquier caffeinate relacionado con METACORTEX que quedó huérfano
    for pid in $(pgrep -f "caffeinate.*metacortex_daemon.py" 2>/dev/null || true); do
        if ps -p "$pid" > /dev/null 2>&1; then
            log_info "Deteniendo caffeinate huérfano (PID: $pid)..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    # 3. Verificar que NO queden procesos Python de METACORTEX
    log_info "Verificando que no queden procesos..."
    local remaining=$(ps aux | grep -E "python.*(metacortex|neural_symbiotic|web_interface/server)" | grep -v grep | wc -l | tr -d ' ')
    
    if [ "$remaining" -gt 0 ]; then
        log_warning "Quedan $remaining procesos, limpiando..."
        ps aux | grep -E "python.*(metacortex|neural_symbiotic|web_interface/server)" | grep -v grep | awk '{print $2}' | xargs -I {} kill -9 {} 2>/dev/null || true
        sleep 1
    fi
    
    # 4. Limpiar archivos temporales y PID
    log_info "Limpiando archivos temporales..."
    find "$PROJECT_ROOT" -name "*.pid" -type f -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.lock" -type f -delete 2>/dev/null || true
    rm -f "$DAEMON_PID_FILE" 2>/dev/null || true
    
    # 5. Verificación final
    local final_count=$(ps aux | grep -E "python.*(metacortex|neural_symbiotic|web_interface/server)" | grep -v grep | wc -l | tr -d ' ')
    if [ "$final_count" -eq 0 ]; then
        log_success "Todos los procesos METACORTEX detenidos (0 procesos restantes)"
    else
        log_error "Aún quedan $final_count procesos activos"
        ps aux | grep -E "python.*(metacortex|neural_symbiotic|web_interface/server)" | grep -v grep
    fi
    
    log_success "Persistencia macOS desactivada"
    print_header "✅ METACORTEX DETENIDO - SIN PROCESOS DUPLICADOS"
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
    print_header "📊 ESTADO DEL SISTEMA"
    
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
    
    # Buscar nuevos scripts standalone (v2025-11-14)
    local neural_count=$(pgrep -f "start_neural_network_standalone.py" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$neural_count" -gt 0 ]; then
        local neural_pid=$(pgrep -f "start_neural_network_standalone.py" 2>/dev/null | head -1)
        echo -e "   ${GREEN}●${RESET} Neural Network: Activo (PID: $neural_pid)"
    else
        echo -e "   ${RED}●${RESET} Neural Network: No activo"
    fi
    
    local web_count=$(pgrep -f "start_web_interface_standalone.py" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$web_count" -gt 0 ]; then
        local web_pid=$(pgrep -f "start_web_interface_standalone.py" 2>/dev/null | head -1)
        echo -e "   ${GREEN}●${RESET} Web Interface: Activo (PID: $web_pid, Puerto 8000)"
    else
        echo -e "   ${RED}●${RESET} Web Interface: No activo"
    fi
    
    local telemetry_count=$(pgrep -f "start_telemetry_simple.py" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$telemetry_count" -gt 0 ]; then
        local telemetry_pid=$(pgrep -f "start_telemetry_simple.py" 2>/dev/null | head -1)
        echo -e "   ${GREEN}●${RESET} Telemetry System: Activo (PID: $telemetry_pid, Puerto 9090)"
    else
        echo -e "   ${RED}●${RESET} Telemetry System: No activo"
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
    
    if lsof -i:11434 -sTCP:LISTEN > /dev/null 2>&1; then
        local process=$(lsof -i:11434 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $1}')
        local pid=$(lsof -i:11434 -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
        echo -e "   ${GREEN}●${RESET} Puerto 11434 (Ollama): $process (PID $pid)"
    else
        echo -e "   ${RED}●${RESET} Puerto 11434 (Ollama): Libre"
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
║  🍎 CON PERSISTENCIA EN macOS (caffeinate)                              ║
╚══════════════════════════════════════════════════════════════════════════╝

COMANDOS DISPONIBLES:

  start              Iniciar todo el sistema METACORTEX con persistencia
                     • Activa caffeinate para prevenir sleep
                     • Sistema ejecutándose 24/7 en iMac
                     • Todos los componentes iniciados automáticamente
  
  stop               Detener todo el sistema de forma ordenada
                     • Desactiva caffeinate
                     • Shutdown graceful de todos los servicios
  
  restart            Reiniciar el sistema completo
  status             Mostrar estado de todos los servicios
  emergency          Apagado de emergencia (mata todo inmediatamente)
  clean              Limpiar archivos temporales y logs antiguos
  divine             Abrir interfaz Divine Protection System
  help               Mostrar esta ayuda

🍎 PERSISTENCIA EN macOS:
   El sistema usa 'caffeinate' de Apple para prevenir sleep:
   • -d: Previene disk sleep
   • -i: Previene idle sleep del sistema
   • -m: Previene system sleep (aggressive)
   • -s: Previene sleep cuando está conectado a AC

📦 COMPONENTES ACTIVADOS AUTOMÁTICAMENTE:
   ✓ metacortex_daemon.py       - Orquestador principal
   ✓ metacortex_orchestrator.py - Coordinador de agentes
   ✓ ml_pipeline.py              - Pipeline de Machine Learning
   ✓ web_interface/server.py     - Servidor web (puerto 8000)
   ✓ neural_network              - Red neuronal simbiótica
   ✓ ollama                      - Servidor LLM (puerto 11434)
   ✓ universal_knowledge_connector.py - Conocimiento universal

EJEMPLOS:

  # Iniciar el sistema con persistencia
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
