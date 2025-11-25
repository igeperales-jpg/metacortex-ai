#!/usr/bin/env python3
"""
🔑 STRIPE SETUP WIZARD - Interactive Setup for LIVE API Keys
============================================================

Este script te guía paso a paso para configurar Stripe LIVE (producción).

IMPORTANTE: Esto es para PRODUCCIÓN con dinero REAL.
Si prefieres probar sin riesgo, usa Test Mode primero.

Autor: METACORTEX Team
Fecha: 24 de Noviembre de 2025
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv, set_key

# Colores para terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}")
    print(f"{text}")
    print(f"{'='*80}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def ask_confirmation(question):
    """Pedir confirmación al usuario"""
    while True:
        response = input(f"{Colors.WARNING}{question} (sí/no): {Colors.ENDC}").lower()
        if response in ['sí', 'si', 'yes', 'y', 's']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Por favor responde 'sí' o 'no'")

def validate_stripe_key(key, key_type):
    """Valida formato de API key de Stripe"""
    if key_type == "secret":
        if not key.startswith("sk_live_"):
            print_error(f"Secret key debe empezar con 'sk_live_' (tienes: {key[:10]}...)")
            return False
    elif key_type == "publishable":
        if not key.startswith("pk_live_"):
            print_error(f"Publishable key debe empezar con 'pk_live_' (tienes: {key[:10]}...)")
            return False
    elif key_type == "webhook":
        if not key.startswith("whsec_"):
            print_error(f"Webhook secret debe empezar con 'whsec_' (tienes: {key[:10]}...)")
            return False
    
    # Verificar longitud mínima
    if len(key) < 30:
        print_error(f"API key parece muy corta (len={len(key)})")
        return False
    
    return True

def test_stripe_connection(secret_key):
    """Test de conexión con Stripe"""
    print_info("Probando conexión con Stripe...")
    
    try:
        import stripe
        
        stripe.api_key = secret_key
        
        # Retrieve balance
        balance = stripe.Balance.retrieve()
        
        if balance.livemode:
            print_success("Conexión LIVE exitosa!")
            print_info(f"   Balance disponible: ${balance.available[0].amount / 100:.2f} {balance.available[0].currency.upper()}")
            print_info(f"   Balance pendiente: ${balance.pending[0].amount / 100:.2f}")
            return True
        else:
            print_warning("Conexión exitosa pero aún en TEST mode")
            print_warning("Asegúrate de usar keys que empiecen con 'sk_live_'")
            return False
            
    except stripe.error.AuthenticationError:
        print_error("API key inválida")
        print_info("   → Verifica que copiaste la key completa")
        print_info("   → Verifica que empiece con 'sk_live_'")
        return False
        
    except Exception as e:
        print_error(f"Error de conexión: {e}")
        return False

def main():
    """Main wizard"""
    print_header("🔑 STRIPE SETUP WIZARD - LIVE MODE (PRODUCCIÓN)")
    
    print(f"{Colors.WARNING}")
    print("⚠️  IMPORTANTE:")
    print("   Este setup es para PRODUCCIÓN con dinero REAL")
    print("   - Procesarás pagos REALES de clientes")
    print("   - Stripe cobrará fees reales (2.9% + $0.30)")
    print("   - Necesitas cuenta verificada y banco vinculado")
    print(f"{Colors.ENDC}")
    
    if not ask_confirmation("¿Deseas continuar con setup de PRODUCCIÓN?"):
        print_info("Setup cancelado. Para test mode, usa keys que empiecen con 'sk_test_'")
        sys.exit(0)
    
    # Verificar si stripe está instalado
    try:
        import stripe
        print_success("Librería 'stripe' encontrada")
    except ImportError:
        print_error("Librería 'stripe' no instalada")
        if ask_confirmation("¿Deseas instalarla ahora?"):
            subprocess.run([sys.executable, "-m", "pip", "install", "stripe"])
        else:
            print_error("Necesitas instalar: pip install stripe")
            sys.exit(1)
    
    # Encontrar archivo .env
    env_path = Path.cwd() / ".env"
    
    if not env_path.exists():
        print_error(f".env no encontrado en: {env_path}")
        if ask_confirmation("¿Deseas crearlo desde .env.example?"):
            example_path = Path.cwd() / ".env.example"
            if example_path.exists():
                subprocess.run(["cp", str(example_path), str(env_path)])
                print_success(".env creado")
            else:
                print_error(".env.example tampoco existe")
                sys.exit(1)
        else:
            sys.exit(1)
    
    print_success(f".env encontrado: {env_path}")
    
    # Cargar .env actual
    load_dotenv(env_path)
    
    # Backup de .env
    import shutil
    from datetime import datetime
    backup_path = env_path.with_suffix(f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(env_path, backup_path)
    print_success(f"Backup creado: {backup_path}")
    
    print_header("PASO 1: OBTENER API KEYS DE STRIPE")
    
    print_info("Necesitas obtener tus API keys LIVE de Stripe:")
    print_info("1. Ve a: https://dashboard.stripe.com/")
    print_info("2. Cambia a 'Live mode' (toggle arriba-derecha)")
    print_info("3. Navega a: Developers → API keys")
    print_info("4. Copia 'Secret key' (sk_live_...)")
    print()
    
    # Obtener Secret Key
    while True:
        secret_key = input(f"{Colors.BOLD}Pega tu STRIPE_SECRET_KEY (sk_live_...): {Colors.ENDC}").strip()
        
        if not secret_key:
            print_error("Key vacía")
            continue
        
        if validate_stripe_key(secret_key, "secret"):
            break
    
    # Obtener Publishable Key
    while True:
        publishable_key = input(f"{Colors.BOLD}Pega tu STRIPE_PUBLISHABLE_KEY (pk_live_...): {Colors.ENDC}").strip()
        
        if not publishable_key:
            print_error("Key vacía")
            continue
        
        if validate_stripe_key(publishable_key, "publishable"):
            break
    
    # Webhook Secret (opcional)
    print()
    print_info("Webhook Secret es opcional pero recomendado para pagos automáticos")
    print_info("Para obtenerlo:")
    print_info("1. Dashboard → Developers → Webhooks")
    print_info("2. Add endpoint → URL de tu API")
    print_info("3. Copiar 'Signing secret' (whsec_...)")
    print()
    
    if ask_confirmation("¿Deseas configurar Webhook Secret ahora?"):
        while True:
            webhook_secret = input(f"{Colors.BOLD}Pega tu STRIPE_WEBHOOK_SECRET (whsec_...): {Colors.ENDC}").strip()
            
            if not webhook_secret:
                print_warning("Webhook secret vacío, se puede configurar después")
                webhook_secret = "whsec_YOUR_WEBHOOK_SECRET_HERE"
                break
            
            if validate_stripe_key(webhook_secret, "webhook"):
                break
    else:
        webhook_secret = "whsec_YOUR_WEBHOOK_SECRET_HERE"
    
    print_header("PASO 2: TEST DE CONEXIÓN")
    
    if test_stripe_connection(secret_key):
        print_success("¡API keys LIVE verificadas!")
    else:
        print_error("No se pudo verificar conexión")
        if not ask_confirmation("¿Deseas continuar de todas formas?"):
            print_info("Setup cancelado. Verifica tus API keys.")
            sys.exit(1)
    
    print_header("PASO 3: ACTUALIZAR .env")
    
    # Actualizar .env
    set_key(env_path, "STRIPE_SECRET_KEY", secret_key)
    set_key(env_path, "STRIPE_PUBLISHABLE_KEY", publishable_key)
    set_key(env_path, "STRIPE_WEBHOOK_SECRET", webhook_secret)
    set_key(env_path, "ENVIRONMENT", "production")
    set_key(env_path, "DEBUG", "false")
    
    print_success(".env actualizado con keys LIVE")
    
    # Verificar permisos
    os.chmod(env_path, 0o600)
    print_success(f"Permisos de .env configurados a 600 (privado)")
    
    print_header("PASO 4: VERIFICACIÓN FINAL")
    
    # Verificar que .env está en .gitignore
    gitignore_path = Path.cwd() / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            if '.env' in f.read():
                print_success(".env está en .gitignore ✓")
            else:
                print_warning(".env NO está en .gitignore")
                if ask_confirmation("¿Deseas añadirlo?"):
                    with open(gitignore_path, 'a') as f:
                        f.write("\n# Environment variables\n.env\n.env.local\n.env.*.local\n")
                    print_success(".env añadido a .gitignore")
    
    print_header("✅ SETUP COMPLETADO")
    
    print_success("Stripe LIVE configurado exitosamente!")
    print()
    print_info("Próximos pasos:")
    print_info("1. Iniciar API server:")
    print_info("   python metacortex_sinaptico/api_monetization_endpoint.py")
    print()
    print_info("2. Test del server:")
    print_info("   curl http://localhost:8100/health")
    print()
    print_info("3. Ver documentación completa:")
    print_info("   cat docs/STRIPE_SETUP_REAL.md")
    print()
    print_warning("⚠️ RECUERDA:")
    print_warning("   - Esto procesa pagos REALES")
    print_warning("   - Stripe cobrará fees reales")
    print_warning("   - Habilita 2FA en tu cuenta Stripe")
    print_warning("   - Nunca compartas tu SECRET_KEY")
    print()
    print(f"{Colors.OKGREEN}{Colors.BOLD}¡Listo para generar ingresos REALES! 💰{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup cancelado por usuario{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
