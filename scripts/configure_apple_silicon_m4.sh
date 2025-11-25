#!/usr/bin/env bash
#
# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  🍎 APPLE SILICON M4 + MPS CONFIGURATION SCRIPT                      ║
# ║  Configura PyTorch y TensorFlow para usar GPU Metal                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝
#

set -euo pipefail

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

log_info() { echo -e "${CYAN}ℹ️  [INFO]${RESET} $1"; }
log_success() { echo -e "${GREEN}✅ [SUCCESS]${RESET} $1"; }
log_warning() { echo -e "${YELLOW}⚠️  [WARNING]${RESET} $1"; }
log_error() { echo -e "${RED}❌ [ERROR]${RESET} $1" >&2; }

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}╔════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}${BOLD}║  $1${RESET}"
    echo -e "${CYAN}${BOLD}╚════════════════════════════════════════════════════════╝${RESET}"
    echo ""
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
VENV_PIP="${PROJECT_ROOT}/.venv/bin/pip3"

print_header "🍎 CONFIGURACIÓN APPLE SILICON M4 + MPS"

# 1. Verificar arquitectura
log_info "Verificando arquitectura del sistema..."
ARCH=$(uname -m)

if [ "$ARCH" != "arm64" ]; then
    log_error "Este script es solo para Apple Silicon (arm64)"
    log_error "Arquitectura detectada: $ARCH"
    exit 1
fi

log_success "Apple Silicon detectado (arm64)"

# 2. Obtener información del chip
CHIP_INFO=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
log_info "Chip: $CHIP_INFO"

PERF_CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "Unknown")
EFF_CORES=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo "Unknown")
TOTAL_CORES=$(sysctl -n hw.physicalcpu 2>/dev/null || echo "Unknown")

log_info "Performance Cores: $PERF_CORES"
log_info "Efficiency Cores: $EFF_CORES"
log_info "Total Cores: $TOTAL_CORES"

MEMORY_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
log_info "Unified Memory: ${MEMORY_GB}GB"

# 3. Verificar entorno virtual
if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
    log_error "Entorno virtual no encontrado"
    log_info "Ejecuta: python3 -m venv .venv"
    exit 1
fi

log_success "Entorno virtual encontrado"

# 4. Verificar PyTorch con MPS
print_header "🔍 VERIFICANDO PYTORCH MPS"

log_info "Verificando instalación de PyTorch..."

PYTORCH_INSTALLED=$("$VENV_PYTHON" -c "import torch; print('YES')" 2>/dev/null || echo "NO")

if [ "$PYTORCH_INSTALLED" = "NO" ]; then
    log_warning "PyTorch no instalado"
    log_info "Instalando PyTorch con soporte MPS..."
    
    "$VENV_PIP" install --upgrade torch torchvision torchaudio
    
    log_success "PyTorch instalado"
fi

# 5. Verificar disponibilidad de MPS
log_info "Verificando disponibilidad de MPS..."

MPS_CHECK=$("$VENV_PYTHON" <<'EOF'
import torch
import sys

print("=" * 60)
print("🍎 APPLE SILICON MPS VERIFICATION")
print("=" * 60)

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    try:
        device = torch.device("mps")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = x @ y
        print(f"✅ MPS test PASSED - GPU is operational!")
        print(f"   Test completed on device: {device}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ MPS test FAILED: {e}")
        sys.exit(1)
else:
    print("❌ MPS not available")
    sys.exit(1)
EOF
)

MPS_EXIT=$?

echo "$MPS_CHECK"

if [ $MPS_EXIT -eq 0 ]; then
    log_success "GPU Metal (MPS) funcionando correctamente"
else
    log_error "GPU Metal (MPS) no disponible"
    log_warning "Posibles causas:"
    log_warning "  1. PyTorch no compilado con soporte MPS"
    log_warning "  2. macOS < 12.3 (se requiere macOS 12.3+)"
    log_warning "  3. Versión antigua de PyTorch"
    log_info "Intenta: pip3 install --upgrade torch torchvision torchaudio"
    exit 1
fi

# 6. Crear archivo de configuración MPS
print_header "📝 CREANDO CONFIGURACIÓN MPS"

MPS_CONFIG="${PROJECT_ROOT}/mps_config.py"

cat > "$MPS_CONFIG" <<'EOF'
"""
🍎 Apple Silicon M4 + MPS (Metal Performance Shaders) Configuration
Configuración automática para usar GPU Metal en lugar de CPU
"""
import os
import torch

# ============================================================================
# CONFIGURACIÓN DE VARIABLES DE ENTORNO
# ============================================================================

# Forzar uso de MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Usar toda la memoria GPU
os.environ['PYTORCH_MPS_PREFER_METAL'] = '1'
os.environ['MPS_FORCE_ENABLE'] = '1'

# Optimizaciones para Apple Silicon
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['OMP_NUM_THREADS'] = '10'  # M4 tiene 10 performance cores
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['OPENBLAS_NUM_THREADS'] = '10'

# Gestión de memoria
os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_device():
    """
    Obtiene el mejor device disponible (prioridad: MPS > CUDA > CPU)
    
    Returns:
        torch.device: Device óptimo para ejecución
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def verify_mps():
    """
    Verifica que MPS esté disponible y funcionando
    
    Returns:
        bool: True si MPS está operacional
    """
    if not torch.backends.mps.is_available():
        print("⚠️ MPS no disponible - usando CPU")
        return False
    
    try:
        device = torch.device("mps")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = x @ y
        print(f"✅ MPS verificado - GPU Metal operacional")
        return True
    except Exception as e:
        print(f"❌ Error verificando MPS: {e}")
        return False


def get_system_info():
    """
    Obtiene información del sistema Apple Silicon
    
    Returns:
        dict: Información del sistema
    """
    import platform
    import subprocess
    
    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }
    
    # Obtener información del chip Apple
    try:
        chip_info = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True
        ).strip()
        info["chip"] = chip_info
    except:
        info["chip"] = "Unknown"
    
    return info


def print_config():
    """Imprime la configuración actual de MPS"""
    print("=" * 60)
    print("🍎 APPLE SILICON M4 + MPS CONFIGURATION")
    print("=" * 60)
    
    info = get_system_info()
    for key, value in info.items():
        print(f"{key:20s}: {value}")
    
    device = get_device()
    print(f"{'device':20s}: {device}")
    
    print("=" * 60)


# ============================================================================
# AUTO-CONFIGURACIÓN AL IMPORTAR
# ============================================================================

# Detectar y configurar automáticamente
DEVICE = get_device()

if __name__ == "__main__":
    print_config()
    verify_mps()
EOF

log_success "Configuración MPS creada: $MPS_CONFIG"

# 7. Probar configuración
print_header "🧪 PROBANDO CONFIGURACIÓN"

"$VENV_PYTHON" "$MPS_CONFIG"

# 8. Crear script de activación
ACTIVATE_SCRIPT="${PROJECT_ROOT}/.venv/bin/activate_mps"

cat > "$ACTIVATE_SCRIPT" <<EOF
#!/usr/bin/env bash
# Auto-generated MPS activation script

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_PREFER_METAL=1
export MPS_FORCE_ENABLE=1
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection

echo "🍎 MPS environment variables configured"
echo "   PYTORCH_ENABLE_MPS_FALLBACK=1"
echo "   PYTORCH_MPS_PREFER_METAL=1"
echo "   OMP_NUM_THREADS=10"
EOF

chmod +x "$ACTIVATE_SCRIPT"

log_success "Script de activación MPS creado: $ACTIVATE_SCRIPT"

# 9. Resumen final
print_header "✅ CONFIGURACIÓN COMPLETADA"

echo -e "${GREEN}${BOLD}Todo listo para usar GPU Metal (MPS) en iMac M4!${RESET}"
echo ""
echo "Para activar MPS en tu sesión:"
echo "  source .venv/bin/activate_mps"
echo ""
echo "Para verificar MPS en Python:"
echo "  python3 -c 'import mps_config; mps_config.verify_mps()'"
echo ""
echo "Dispositivo configurado: ${CYAN}${BOLD}MPS (GPU Metal)${RESET}"
echo ""

log_success "Sistema optimizado para Apple Silicon M4"
