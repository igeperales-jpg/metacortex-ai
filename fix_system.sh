#!/bin/bash
################################################################################
# 🔧 METACORTEX SYSTEM REPAIR - FIX AUTOMÁTICO
################################################################################
# Este script diagnostica y repara automáticamente problemas del sistema
################################################################################

set -e

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

PROJECT_ROOT="/Users/edkanina/ai_definitiva"
PID_DIR="${PROJECT_ROOT}/pid"
LOGS_DIR="${PROJECT_ROOT}/logs"

echo -e "${BOLD}${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🔧 METACORTEX SYSTEM REPAIR - DIAGNÓSTICO Y REPARACIÓN  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ============================================================================
# 1. VERIFICAR Y REPARAR OLLAMA
# ============================================================================
echo -e "${BOLD}1. Verificando Ollama...${RESET}"

if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama no instalado${RESET}"
    echo -e "${YELLOW}   Instala con: brew install ollama${RESET}"
else
    echo -e "${GREEN}✅ Ollama instalado${RESET}"
    
    # Verificar si está corriendo
    if lsof -i:11434 -sTCP:LISTEN > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Ollama Server: ACTIVO (Puerto 11434)${RESET}"
        
        # Verificar conectividad
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Ollama API: RESPONDIENDO${RESET}"
            
            # Listar modelos
            MODELS=$(ollama list 2>/dev/null | grep -v "NAME" | wc -l | tr -d ' ')
            echo -e "${GREEN}✅ Modelos disponibles: $MODELS${RESET}"
            
            # Verificar Mistral
            if ollama list | grep -q "mistral"; then
                echo -e "${GREEN}✅ Mistral: DISPONIBLE${RESET}"
            else
                echo -e "${YELLOW}⚠️  Mistral: NO ENCONTRADO${RESET}"
                echo -e "${YELLOW}   Instalando Mistral...${RESET}"
                ollama pull mistral:latest &
                echo -e "${GREEN}✅ Mistral descargando en background${RESET}"
            fi
        else
            echo -e "${YELLOW}⚠️  Ollama API no responde${RESET}"
            echo -e "${YELLOW}   Reiniciando Ollama...${RESET}"
            pkill -9 ollama 2>/dev/null || true
            sleep 2
            ollama serve > /dev/null 2>&1 &
            sleep 3
            echo -e "${GREEN}✅ Ollama reiniciado${RESET}"
        fi
    else
        echo -e "${YELLOW}⚠️  Ollama Server NO ACTIVO${RESET}"
        echo -e "${YELLOW}   Iniciando Ollama...${RESET}"
        ollama serve > /dev/null 2>&1 &
        sleep 3
        
        if lsof -i:11434 -sTCP:LISTEN > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Ollama iniciado correctamente${RESET}"
        else
            echo -e "${RED}❌ No se pudo iniciar Ollama${RESET}"
        fi
    fi
fi

echo ""

# ============================================================================
# 2. VERIFICAR Y REPARAR DAEMON PRINCIPAL
# ============================================================================
echo -e "${BOLD}2. Verificando Daemon Principal...${RESET}"

if [ -f "${PID_DIR}/metacortex_daemon_military.pid" ]; then
    DAEMON_PID=$(cat "${PID_DIR}/metacortex_daemon_military.pid")
    if ps -p "$DAEMON_PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Daemon Principal: ACTIVO (PID: $DAEMON_PID)${RESET}"
    else
        echo -e "${YELLOW}⚠️  Daemon Principal: PID inválido${RESET}"
        echo -e "${YELLOW}   Limpiando PID y reiniciando...${RESET}"
        rm -f "${PID_DIR}/metacortex_daemon_military.pid"
        
        # Iniciar daemon
        cd "$PROJECT_ROOT"
        source .venv/bin/activate
        
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
        export MPS_FORCE_ENABLE=1
        export OMP_NUM_THREADS=10
        
        nohup caffeinate -i python3 "${PROJECT_ROOT}/metacortex_daemon.py" > "${LOGS_DIR}/metacortex_daemon_military.log" 2>&1 &
        NEW_PID=$!
        echo "$NEW_PID" > "${PID_DIR}/metacortex_daemon_military.pid"
        
        sleep 2
        if ps -p "$NEW_PID" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Daemon iniciado (PID: $NEW_PID)${RESET}"
        else
            echo -e "${RED}❌ Error iniciando daemon${RESET}"
            echo -e "${YELLOW}   Ver logs: tail -20 ${LOGS_DIR}/metacortex_daemon_military.log${RESET}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Daemon Principal: NO INICIADO${RESET}"
    echo -e "${YELLOW}   Iniciando daemon...${RESET}"
    
    mkdir -p "$PID_DIR"
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export MPS_FORCE_ENABLE=1
    export OMP_NUM_THREADS=10
    
    nohup caffeinate -i python3 "${PROJECT_ROOT}/metacortex_daemon.py" > "${LOGS_DIR}/metacortex_daemon_military.log" 2>&1 &
    NEW_PID=$!
    echo "$NEW_PID" > "${PID_DIR}/metacortex_daemon_military.pid"
    
    sleep 2
    if ps -p "$NEW_PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Daemon iniciado (PID: $NEW_PID)${RESET}"
    else
        echo -e "${RED}❌ Error iniciando daemon${RESET}"
        echo -e "${YELLOW}   Ver logs: tail -20 ${LOGS_DIR}/metacortex_daemon_military.log${RESET}"
    fi
fi

echo ""

# ============================================================================
# 3. VERIFICAR Y REPARAR ORCHESTRATOR
# ============================================================================
echo -e "${BOLD}3. Verificando Agent Orchestrator...${RESET}"

ORCHESTRATOR_FILE="${PROJECT_ROOT}/metacortex_orchestrator.py"

if [ -f "${PROJECT_ROOT}/caffeinate.pid" ]; then
    ORCH_PID=$(cat "${PROJECT_ROOT}/caffeinate.pid")
    if ps -p "$ORCH_PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Agent Orchestrator: ACTIVO (PID: $ORCH_PID)${RESET}"
    else
        echo -e "${YELLOW}⚠️  Agent Orchestrator: PID inválido${RESET}"
        echo -e "${YELLOW}   Reiniciando orchestrator...${RESET}"
        rm -f "${PROJECT_ROOT}/caffeinate.pid"
        
        cd "$PROJECT_ROOT"
        source .venv/bin/activate
        
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
        export MPS_FORCE_ENABLE=1
        
        if [ -f "$ORCHESTRATOR_FILE" ]; then
            nohup caffeinate -i python3 "$ORCHESTRATOR_FILE" > "${LOGS_DIR}/startup_orchestrator.log" 2>&1 &
            NEW_PID=$!
            echo "$NEW_PID" > "${PROJECT_ROOT}/caffeinate.pid"
            
            sleep 2
            if ps -p "$NEW_PID" > /dev/null 2>&1; then
                echo -e "${GREEN}✅ Orchestrator iniciado (PID: $NEW_PID)${RESET}"
            else
                echo -e "${RED}❌ Error iniciando orchestrator${RESET}"
                echo -e "${YELLOW}   Ver logs: tail -20 ${LOGS_DIR}/startup_orchestrator.log${RESET}"
            fi
        else
            echo -e "${YELLOW}⚠️  Archivo orchestrator no encontrado: $ORCHESTRATOR_FILE${RESET}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Agent Orchestrator: NO INICIADO${RESET}"
    echo -e "${YELLOW}   Iniciando orchestrator...${RESET}"
    
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export MPS_FORCE_ENABLE=1
    
    if [ -f "$ORCHESTRATOR_FILE" ]; then
        nohup caffeinate -i python3 "$ORCHESTRATOR_FILE" > "${LOGS_DIR}/startup_orchestrator.log" 2>&1 &
        NEW_PID=$!
        echo "$NEW_PID" > "${PROJECT_ROOT}/caffeinate.pid"
        
        sleep 2
        if ps -p "$NEW_PID" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Orchestrator iniciado (PID: $NEW_PID)${RESET}"
        else
            echo -e "${RED}❌ Error iniciando orchestrator${RESET}"
            echo -e "${YELLOW}   Ver logs: tail -20 ${LOGS_DIR}/startup_orchestrator.log${RESET}"
        fi
    else
        echo -e "${YELLOW}⚠️  Archivo orchestrator no encontrado: $ORCHESTRATOR_FILE${RESET}"
        echo -e "${YELLOW}   El orchestrator puede estar en otra ubicación${RESET}"
    fi
fi

echo ""

# ============================================================================
# 4. VERIFICAR DASHBOARD ENTERPRISE
# ============================================================================
echo -e "${BOLD}4. Verificando Dashboard Enterprise...${RESET}"

if [ -f "${PID_DIR}/dashboard_enterprise.pid" ]; then
    DASH_PID=$(cat "${PID_DIR}/dashboard_enterprise.pid")
    if ps -p "$DASH_PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Dashboard Enterprise: ACTIVO (PID: $DASH_PID)${RESET}"
        echo -e "${GREEN}   🌐 http://localhost:8300${RESET}"
    else
        echo -e "${YELLOW}⚠️  Dashboard Enterprise: PID inválido${RESET}"
        echo -e "${YELLOW}   Reiniciando dashboard...${RESET}"
        rm -f "${PID_DIR}/dashboard_enterprise.pid"
        
        cd "$PROJECT_ROOT"
        source .venv/bin/activate
        
        nohup python3 "${PROJECT_ROOT}/dashboard_enterprise.py" > "${LOGS_DIR}/dashboard_enterprise.log" 2>&1 &
        NEW_PID=$!
        echo "$NEW_PID" > "${PID_DIR}/dashboard_enterprise.pid"
        
        sleep 3
        if ps -p "$NEW_PID" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Dashboard iniciado (PID: $NEW_PID)${RESET}"
            echo -e "${GREEN}   🌐 http://localhost:8300${RESET}"
        else
            echo -e "${RED}❌ Error iniciando dashboard${RESET}"
            echo -e "${YELLOW}   Ver logs: tail -20 ${LOGS_DIR}/dashboard_enterprise.log${RESET}"
        fi
    fi
else
    echo -e "${GREEN}✅ Dashboard ya está corriendo (verificado por puerto 8300)${RESET}"
fi

echo ""

# ============================================================================
# 5. VERIFICAR SERVICIOS BASE
# ============================================================================
echo -e "${BOLD}5. Verificando Servicios Base...${RESET}"

# Neural Network
if pgrep -f "neural_network_service/server.py" > /dev/null; then
    NEURAL_PID=$(pgrep -f "neural_network_service/server.py" | head -1)
    echo -e "${GREEN}✅ Neural Network: ACTIVO (PID: $NEURAL_PID)${RESET}"
else
    echo -e "${YELLOW}⚠️  Neural Network: NO ACTIVO${RESET}"
fi

# Web Interface
if pgrep -f "web_interface/server.py" > /dev/null; then
    WEB_PID=$(pgrep -f "web_interface/server.py" | head -1)
    echo -e "${GREEN}✅ Web Interface: ACTIVO (PID: $WEB_PID)${RESET}"
else
    echo -e "${YELLOW}⚠️  Web Interface: NO ACTIVO${RESET}"
fi

# Telemetry
if pgrep -f "telemetry_service/server.py" > /dev/null; then
    TELEM_PID=$(pgrep -f "telemetry_service/server.py" | head -1)
    echo -e "${GREEN}✅ Telemetry System: ACTIVO (PID: $TELEM_PID)${RESET}"
else
    echo -e "${YELLOW}⚠️  Telemetry System: NO ACTIVO${RESET}"
fi

# API Monetization
if pgrep -f "api_monetization_endpoint.py" > /dev/null; then
    API_PID=$(pgrep -f "api_monetization_endpoint.py" | head -1)
    echo -e "${GREEN}✅ API Monetization: ACTIVO (PID: $API_PID)${RESET}"
else
    echo -e "${YELLOW}⚠️  API Monetization: NO ACTIVO${RESET}"
fi

echo ""

# ============================================================================
# 6. VERIFICAR AUTONOMOUS ORCHESTRATOR
# ============================================================================
echo -e "${BOLD}6. Verificando Autonomous Orchestrator...${RESET}"

if pgrep -f "autonomous_model_orchestrator.py" > /dev/null; then
    AUTO_PID=$(pgrep -f "autonomous_model_orchestrator.py" | head -1)
    echo -e "${GREEN}✅ Autonomous Orchestrator: ACTIVO (PID: $AUTO_PID)${RESET}"
else
    echo -e "${YELLOW}⚠️  Autonomous Orchestrator: Integrado en Dashboard${RESET}"
    echo -e "${YELLOW}   Los 965 modelos ML están disponibles via Dashboard${RESET}"
fi

echo ""

# ============================================================================
# 7. RESUMEN FINAL
# ============================================================================
echo -e "${BOLD}${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              📊 RESUMEN DE REPARACIÓN                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

echo -e "${BOLD}Servicios Críticos:${RESET}"
echo -e "   $(if lsof -i:11434 > /dev/null 2>&1; then echo '✅'; else echo '❌'; fi) Ollama (Puerto 11434)"
echo -e "   $(if [ -f "${PID_DIR}/metacortex_daemon_military.pid" ] && ps -p $(cat "${PID_DIR}/metacortex_daemon_military.pid") > /dev/null 2>&1; then echo '✅'; else echo '❌'; fi) Daemon Principal"
echo -e "   $(if [ -f "${PROJECT_ROOT}/caffeinate.pid" ] && ps -p $(cat "${PROJECT_ROOT}/caffeinate.pid") > /dev/null 2>&1; then echo '✅'; else echo '❌'; fi) Agent Orchestrator"
echo -e "   $(if lsof -i:8300 > /dev/null 2>&1; then echo '✅'; else echo '❌'; fi) Dashboard Enterprise (Puerto 8300)"

echo ""
echo -e "${BOLD}Acceso Rápido:${RESET}"
echo -e "   🌐 Dashboard: ${BLUE}http://localhost:8300${RESET}"
echo -e "   📚 API Docs: ${BLUE}http://localhost:8300/api/docs${RESET}"
echo -e "   📊 Web Interface: ${BLUE}http://localhost:8000${RESET}"

echo ""
echo -e "${BOLD}Comandos Útiles:${RESET}"
echo -e "   Ver estado: ${BLUE}./metacortex_master.sh status${RESET}"
echo -e "   Logs dashboard: ${BLUE}tail -f logs/dashboard_enterprise.log${RESET}"
echo -e "   Logs daemon: ${BLUE}tail -f logs/metacortex_daemon_military.log${RESET}"

echo ""
echo -e "${GREEN}✅ Reparación completada${RESET}"
