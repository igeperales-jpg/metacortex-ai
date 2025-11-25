#!/bin/bash
# =============================================================================
# METACORTEX - Sistema de Backup Automático de Configuraciones
# =============================================================================
# Este script crea backups seguros de todas las configuraciones críticas
# para prevenir pérdida de datos durante restauraciones de git
# =============================================================================

set -e

PROJECT_ROOT="/Users/edkanina/ai_definitiva"
BACKUP_DIR="${PROJECT_ROOT}/config_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  💾 BACKUP AUTOMÁTICO DE CONFIGURACIONES                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Crear directorio de backups si no existe
mkdir -p "${BACKUP_DIR}"

# Crear subdirectorio con timestamp
CURRENT_BACKUP="${BACKUP_DIR}/backup_${TIMESTAMP}"
mkdir -p "${CURRENT_BACKUP}"

echo -e "${YELLOW}📁 Creando backup en: ${CURRENT_BACKUP}${NC}"
echo ""

# =============================================================================
# 1. BACKUP DE .ENV (CONFIGURACIÓN CRÍTICA)
# =============================================================================
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo -e "${GREEN}✅ Backing up .env (tokens, API keys, configuración)${NC}"
    cp "${PROJECT_ROOT}/.env" "${CURRENT_BACKUP}/.env"
    
    # Crear también un backup permanente con timestamp
    cp "${PROJECT_ROOT}/.env" "${BACKUP_DIR}/.env.${TIMESTAMP}"
else
    echo -e "${RED}⚠️  .env no encontrado${NC}"
fi

# =============================================================================
# 2. BACKUP DE SCRIPTS CRÍTICOS
# =============================================================================
echo -e "${GREEN}✅ Backing up scripts críticos${NC}"
mkdir -p "${CURRENT_BACKUP}/scripts"

if [ -f "${PROJECT_ROOT}/metacortex_master.sh" ]; then
    cp "${PROJECT_ROOT}/metacortex_master.sh" "${CURRENT_BACKUP}/metacortex_master.sh"
fi

if [ -f "${PROJECT_ROOT}/unified_startup.py" ]; then
    cp "${PROJECT_ROOT}/unified_startup.py" "${CURRENT_BACKUP}/unified_startup.py"
fi

if [ -f "${PROJECT_ROOT}/metacortex_daemon.py" ]; then
    cp "${PROJECT_ROOT}/metacortex_daemon.py" "${CURRENT_BACKUP}/metacortex_daemon.py"
fi

# =============================================================================
# 3. BACKUP DE CONFIGURACIONES JSON
# =============================================================================
echo -e "${GREEN}✅ Backing up configuraciones JSON${NC}"
if [ -f "${PROJECT_ROOT}/divine_protection_config.json" ]; then
    cp "${PROJECT_ROOT}/divine_protection_config.json" "${CURRENT_BACKUP}/divine_protection_config.json"
fi

# =============================================================================
# 4. BACKUP DE BASE DE DATOS SQLite
# =============================================================================
echo -e "${GREEN}✅ Backing up bases de datos${NC}"
if [ -f "${PROJECT_ROOT}/metacortex.sqlite" ]; then
    cp "${PROJECT_ROOT}/metacortex.sqlite" "${CURRENT_BACKUP}/metacortex.sqlite"
fi

if [ -f "${PROJECT_ROOT}/emergency_requests.db" ]; then
    cp "${PROJECT_ROOT}/emergency_requests.db" "${CURRENT_BACKUP}/emergency_requests.db"
fi

# =============================================================================
# 5. BACKUP DE MODELOS ML ENTRENADOS (solo metadatos)
# =============================================================================
echo -e "${GREEN}✅ Backing up metadatos de ML${NC}"
if [ -d "${PROJECT_ROOT}/ml_models" ]; then
    mkdir -p "${CURRENT_BACKUP}/ml_models"
    # Solo copiar archivos de configuración, no los modelos pesados
    find "${PROJECT_ROOT}/ml_models" -name "*.json" -exec cp {} "${CURRENT_BACKUP}/ml_models/" \;
    find "${PROJECT_ROOT}/ml_models" -name "*.yaml" -exec cp {} "${CURRENT_BACKUP}/ml_models/" \;
fi

# =============================================================================
# 6. CREAR SCRIPT DE RESTAURACIÓN AUTOMÁTICA
# =============================================================================
echo -e "${GREEN}✅ Creando script de restauración automática${NC}"

cat > "${CURRENT_BACKUP}/RESTORE.sh" << 'EOF'
#!/bin/bash
# Script de restauración automática
# Ejecuta: bash RESTORE.sh

set -e

BACKUP_DIR="$(dirname "$0")"
PROJECT_ROOT="/Users/edkanina/ai_definitiva"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🔄 RESTAURANDO CONFIGURACIONES DESDE BACKUP              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Backup actual antes de restaurar
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo "📦 Creando backup de seguridad del .env actual..."
    cp "${PROJECT_ROOT}/.env" "${PROJECT_ROOT}/.env.before_restore"
fi

# Restaurar .env
if [ -f "${BACKUP_DIR}/.env" ]; then
    echo "✅ Restaurando .env..."
    cp "${BACKUP_DIR}/.env" "${PROJECT_ROOT}/.env"
else
    echo "⚠️  No se encontró .env en el backup"
fi

# Restaurar scripts
if [ -f "${BACKUP_DIR}/metacortex_master.sh" ]; then
    echo "✅ Restaurando metacortex_master.sh..."
    cp "${BACKUP_DIR}/metacortex_master.sh" "${PROJECT_ROOT}/metacortex_master.sh"
    chmod +x "${PROJECT_ROOT}/metacortex_master.sh"
fi

if [ -f "${BACKUP_DIR}/unified_startup.py" ]; then
    echo "✅ Restaurando unified_startup.py..."
    cp "${BACKUP_DIR}/unified_startup.py" "${PROJECT_ROOT}/unified_startup.py"
fi

if [ -f "${BACKUP_DIR}/metacortex_daemon.py" ]; then
    echo "✅ Restaurando metacortex_daemon.py..."
    cp "${BACKUP_DIR}/metacortex_daemon.py" "${PROJECT_ROOT}/metacortex_daemon.py"
fi

# Restaurar configuraciones
if [ -f "${BACKUP_DIR}/divine_protection_config.json" ]; then
    echo "✅ Restaurando divine_protection_config.json..."
    cp "${BACKUP_DIR}/divine_protection_config.json" "${PROJECT_ROOT}/divine_protection_config.json"
fi

# Restaurar bases de datos
if [ -f "${BACKUP_DIR}/metacortex.sqlite" ]; then
    echo "✅ Restaurando metacortex.sqlite..."
    cp "${BACKUP_DIR}/metacortex.sqlite" "${PROJECT_ROOT}/metacortex.sqlite"
fi

if [ -f "${BACKUP_DIR}/emergency_requests.db" ]; then
    echo "✅ Restaurando emergency_requests.db..."
    cp "${BACKUP_DIR}/emergency_requests.db" "${PROJECT_ROOT}/emergency_requests.db"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ RESTAURACIÓN COMPLETADA                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "💡 Verifica que todo esté correcto:"
echo "   grep 'TELEGRAM_BOT_TOKEN' ${PROJECT_ROOT}/.env"
echo ""
echo "🚀 Para reiniciar el sistema:"
echo "   cd ${PROJECT_ROOT}"
echo "   ./metacortex_master.sh restart"
EOF

chmod +x "${CURRENT_BACKUP}/RESTORE.sh"

# =============================================================================
# 7. CREAR MANIFIESTO DEL BACKUP
# =============================================================================
cat > "${CURRENT_BACKUP}/MANIFEST.txt" << EOF
═══════════════════════════════════════════════════════════════
METACORTEX - BACKUP DE CONFIGURACIONES
═══════════════════════════════════════════════════════════════

Fecha de backup: $(date)
Timestamp: ${TIMESTAMP}
Usuario: $(whoami)
Hostname: $(hostname)

ARCHIVOS RESPALDADOS:
───────────────────────────────────────────────────────────────
EOF

# Listar archivos en el backup
find "${CURRENT_BACKUP}" -type f -not -name "MANIFEST.txt" | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  ✓ $(basename "$file") - ${size}" >> "${CURRENT_BACKUP}/MANIFEST.txt"
done

cat >> "${CURRENT_BACKUP}/MANIFEST.txt" << EOF

═══════════════════════════════════════════════════════════════
INSTRUCCIONES DE RESTAURACIÓN:
═══════════════════════════════════════════════════════════════

1. Para restaurar TODAS las configuraciones:
   bash ${CURRENT_BACKUP}/RESTORE.sh

2. Para restaurar solo el .env:
   cp ${CURRENT_BACKUP}/.env /Users/edkanina/ai_definitiva/.env

3. Para verificar el contenido del .env:
   cat ${CURRENT_BACKUP}/.env | grep -v '^#' | grep -v '^$'

═══════════════════════════════════════════════════════════════
EOF

# =============================================================================
# 8. LIMPIAR BACKUPS ANTIGUOS (mantener últimos 10)
# =============================================================================
echo -e "${YELLOW}🧹 Limpiando backups antiguos (manteniendo últimos 10)${NC}"
cd "${BACKUP_DIR}"
ls -t backup_* 2>/dev/null | tail -n +11 | xargs rm -rf 2>/dev/null || true

# Limpiar .env antiguos (mantener últimos 20)
ls -t .env.* 2>/dev/null | tail -n +21 | xargs rm -f 2>/dev/null || true

# =============================================================================
# RESUMEN
# =============================================================================
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  ✅ BACKUP COMPLETADO EXITOSAMENTE                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}📁 Backup guardado en:${NC}"
echo -e "   ${CURRENT_BACKUP}"
echo ""
echo -e "${GREEN}📋 Archivos respaldados:${NC}"
cat "${CURRENT_BACKUP}/MANIFEST.txt" | grep "✓"
echo ""
echo -e "${YELLOW}💡 Para restaurar este backup:${NC}"
echo -e "   bash ${CURRENT_BACKUP}/RESTORE.sh"
echo ""
echo -e "${YELLOW}📊 Backups totales disponibles:${NC}"
echo -e "   $(ls -d ${BACKUP_DIR}/backup_* 2>/dev/null | wc -l) backups"
echo ""
