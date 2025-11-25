#!/bin/bash

# ============================================================================
# 🍎 METACORTEX - Monitor de Salud del Sistema
# Monitoreo continuo de logs, procesos y rendimiento
# Apple Silicon M4 + MPS Optimized
# ============================================================================

WORKSPACE="/Users/edkanina/ai_definitiva"
LOG_DIR="$WORKSPACE/logs"
DAEMON_LOG="$LOG_DIR/metacortex_daemon_military.log"
MONITOR_LOG="$LOG_DIR/system_monitor.log"

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# FUNCIONES DE MONITOREO
# ============================================================================

log_monitor() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

check_daemon() {
    local PID=$(pgrep -f "metacortex_daemon.py" | head -1)
    if [ -n "$PID" ]; then
        local UPTIME=$(ps -p $PID -o etime | tail -1 | tr -d ' ')
        echo -e "${GREEN}✅${NC} Daemon: ACTIVO (PID: $PID, Uptime: $UPTIME)"
        log_monitor "✅ Daemon activo: PID=$PID, Uptime=$UPTIME"
        return 0
    else
        echo -e "${RED}❌${NC} Daemon: NO ACTIVO"
        log_monitor "❌ Daemon NO activo"
        return 1
    fi
}

check_mps() {
    local MPS_CHECK=$(python3 -c "import torch; print('OK' if torch.backends.mps.is_available() else 'FAIL')" 2>/dev/null)
    if [ "$MPS_CHECK" = "OK" ]; then
        echo -e "${GREEN}✅${NC} GPU Metal (MPS): DISPONIBLE"
        log_monitor "✅ MPS disponible"
        return 0
    else
        echo -e "${RED}❌${NC} GPU Metal (MPS): NO DISPONIBLE"
        log_monitor "❌ MPS no disponible"
        return 1
    fi
}

check_ollama() {
    local OLLAMA_PID=$(pgrep -f "ollama serve")
    if [ -n "$OLLAMA_PID" ]; then
        local RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags 2>/dev/null)
        if [ "$RESPONSE" = "200" ]; then
            echo -e "${GREEN}✅${NC} Ollama LLM: ACTIVO (PID: $OLLAMA_PID, Puerto: 11434)"
            log_monitor "✅ Ollama activo: PID=$OLLAMA_PID"
            return 0
        fi
    fi
    echo -e "${YELLOW}⚠️${NC}  Ollama LLM: NO RESPONDE"
    log_monitor "⚠️ Ollama no responde"
    return 1
}

check_redis() {
    local REDIS_PID=$(pgrep -f "redis-server" | head -1)
    if [ -n "$REDIS_PID" ]; then
        if redis-cli ping > /dev/null 2>&1; then
            echo -e "${GREEN}✅${NC} Redis Cache: ACTIVO (PID: $REDIS_PID, Puerto: 6379)"
            log_monitor "✅ Redis activo: PID=$REDIS_PID"
            return 0
        fi
    fi
    echo -e "${YELLOW}⚠️${NC}  Redis Cache: NO RESPONDE"
    log_monitor "⚠️ Redis no responde"
    return 1
}

check_caffeinate() {
    local CAFF_PID=$(pgrep -f "caffeinate.*metacortex_daemon")
    if [ -n "$CAFF_PID" ]; then
        echo -e "${GREEN}✅${NC} Persistencia 24/7: ACTIVA (Caffeinate PID: $CAFF_PID)"
        log_monitor "✅ Caffeinate activo: PID=$CAFF_PID"
        return 0
    else
        echo -e "${RED}❌${NC} Persistencia 24/7: NO ACTIVA"
        log_monitor "❌ Caffeinate no activo"
        return 1
    fi
}

check_errors() {
    local ERROR_COUNT=$(grep -i "error\|critical\|exception" "$DAEMON_LOG" 2>/dev/null | tail -100 | wc -l | tr -d ' ')
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}⚠️${NC}  Errores en log: $ERROR_COUNT (últimas 100 líneas)"
        log_monitor "⚠️ $ERROR_COUNT errores en últimas 100 líneas"
        
        # Mostrar últimos 3 errores
        echo -e "${BLUE}   Últimos errores:${NC}"
        grep -i "error\|critical\|exception" "$DAEMON_LOG" | tail -3 | while read line; do
            echo "   $line"
        done
    else
        echo -e "${GREEN}✅${NC} Sin errores críticos en últimas 100 líneas"
        log_monitor "✅ Sin errores en últimas 100 líneas"
    fi
}

check_log_size() {
    if [ -f "$DAEMON_LOG" ]; then
        local SIZE=$(du -h "$DAEMON_LOG" | awk '{print $1}')
        local LINES=$(wc -l < "$DAEMON_LOG" | tr -d ' ')
        echo -e "${BLUE}📊${NC} Log Principal: $SIZE ($LINES líneas)"
        log_monitor "📊 Log: $SIZE, $LINES líneas"
        
        # Advertir si el log es muy grande
        local SIZE_MB=$(du -m "$DAEMON_LOG" | awk '{print $1}')
        if [ "$SIZE_MB" -gt 50 ]; then
            echo -e "${YELLOW}⚠️${NC}  Log grande (>50MB), considerar rotación"
            log_monitor "⚠️ Log grande: ${SIZE_MB}MB"
        fi
    fi
}

check_autonomous_decisions() {
    local DECISION_COUNT=$(ls -1 "$LOG_DIR/autonomous_decisions/" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$DECISION_COUNT" -gt 0 ]; then
        echo -e "${GREEN}🤖${NC} Decisiones Autónomas: $DECISION_COUNT registradas"
        log_monitor "🤖 $DECISION_COUNT decisiones autónomas"
        
        # Mostrar últimas 3 decisiones
        echo -e "${BLUE}   Últimas decisiones:${NC}"
        ls -t "$LOG_DIR/autonomous_decisions/" | head -3 | while read file; do
            echo "   - $file"
        done
    else
        echo -e "${BLUE}🤖${NC} Decisiones Autónomas: 0 (modo vigilancia)"
        log_monitor "🤖 Sin decisiones (modo vigilancia)"
    fi
}

get_system_resources() {
    # CPU y Memoria del daemon
    local DAEMON_PID=$(pgrep -f "metacortex_daemon.py" | head -1)
    if [ -n "$DAEMON_PID" ]; then
        local CPU=$(ps -p $DAEMON_PID -o %cpu | tail -1 | tr -d ' ')
        local MEM=$(ps -p $DAEMON_PID -o rss | tail -1 | awk '{printf "%.1f MB", $1/1024}')
        echo -e "${BLUE}💻${NC} Recursos Daemon: CPU $CPU%, Memoria $MEM"
        log_monitor "💻 Daemon: CPU=$CPU%, MEM=$MEM"
    fi
}

calculate_health_score() {
    local score=0
    local max_score=100
    
    # Daemon (30 puntos)
    check_daemon > /dev/null 2>&1 && score=$((score + 30))
    
    # MPS (20 puntos)
    check_mps > /dev/null 2>&1 && score=$((score + 20))
    
    # Ollama (15 puntos)
    check_ollama > /dev/null 2>&1 && score=$((score + 15))
    
    # Redis (15 puntos)
    check_redis > /dev/null 2>&1 && score=$((score + 15))
    
    # Caffeinate (10 puntos)
    check_caffeinate > /dev/null 2>&1 && score=$((score + 10))
    
    # Sin errores recientes (10 puntos)
    local ERROR_COUNT=$(grep -i "error\|critical\|exception" "$DAEMON_LOG" 2>/dev/null | tail -100 | wc -l | tr -d ' ')
    [ "$ERROR_COUNT" -eq 0 ] && score=$((score + 10))
    
    local percentage=$((score * 100 / max_score))
    
    if [ $percentage -ge 90 ]; then
        echo -e "${GREEN}🏆${NC} Health Score: $percentage/100 (EXCELENTE)"
    elif [ $percentage -ge 70 ]; then
        echo -e "${YELLOW}⚠️${NC}  Health Score: $percentage/100 (ACEPTABLE)"
    else
        echo -e "${RED}❌${NC} Health Score: $percentage/100 (CRÍTICO)"
    fi
    
    log_monitor "🏆 Health Score: $percentage/100"
}

# ============================================================================
# MONITOR PRINCIPAL
# ============================================================================

monitor_once() {
    clear
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  🍎 METACORTEX - Monitor de Salud del Sistema             ║"
    echo "║     Apple Silicon M4 + MPS Optimization                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "⏰ Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  COMPONENTES PRINCIPALES"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    check_daemon
    check_mps
    check_ollama
    check_redis
    check_caffeinate
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  DIAGNÓSTICO"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    check_errors
    check_log_size
    check_autonomous_decisions
    get_system_resources
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PUNTUACIÓN FINAL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    calculate_health_score
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

monitor_continuous() {
    local INTERVAL=${1:-60}  # Default: 60 segundos
    
    echo "🔄 Iniciando monitoreo continuo cada $INTERVAL segundos..."
    echo "   Presiona Ctrl+C para detener"
    echo ""
    
    while true; do
        monitor_once
        echo "⏳ Próxima actualización en $INTERVAL segundos..."
        sleep $INTERVAL
    done
}

show_help() {
    cat << EOF
🍎 METACORTEX - Monitor de Salud del Sistema

USO:
    $0 [COMANDO] [OPCIONES]

COMANDOS:
    check           Ejecutar verificación única (por defecto)
    watch [N]       Monitoreo continuo cada N segundos (default: 60)
    errors          Mostrar últimos 20 errores del log
    logs            Ver últimas 50 líneas del log
    decisions       Listar decisiones autónomas
    help            Mostrar esta ayuda

EJEMPLOS:
    $0                    # Verificación única
    $0 watch             # Monitor continuo (60s)
    $0 watch 30          # Monitor continuo (30s)
    $0 errors            # Ver errores
    $0 logs              # Ver log reciente
    $0 decisions         # Ver decisiones autónomas

EOF
}

# ============================================================================
# COMANDOS ADICIONALES
# ============================================================================

show_errors() {
    echo "🔍 Últimos 20 errores del log:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    grep -i "error\|critical\|exception" "$DAEMON_LOG" | tail -20
}

show_logs() {
    echo "📄 Últimas 50 líneas del log:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    tail -50 "$DAEMON_LOG"
}

show_decisions() {
    echo "🤖 Decisiones autónomas registradas:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -d "$LOG_DIR/autonomous_decisions" ]; then
        local COUNT=$(ls -1 "$LOG_DIR/autonomous_decisions/" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$COUNT" -gt 0 ]; then
            ls -lh "$LOG_DIR/autonomous_decisions/" | tail -n +2
        else
            echo "No hay decisiones registradas (sistema en modo vigilancia)"
        fi
    else
        echo "Directorio de decisiones no encontrado"
    fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    cd "$WORKSPACE" || exit 1
    
    case "${1:-check}" in
        check)
            monitor_once
            ;;
        watch)
            monitor_continuous "${2:-60}"
            ;;
        errors)
            show_errors
            ;;
        logs)
            show_logs
            ;;
        decisions)
            show_decisions
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Comando desconocido: $1"
            echo "Usa '$0 help' para ver comandos disponibles"
            exit 1
            ;;
    esac
}

main "$@"
