# 🔑 STRIPE SETUP - API Keys REALES (PRODUCCIÓN)

**Fecha:** 24 de Noviembre de 2025  
**Modo:** PRODUCCIÓN (Real Money) ⚠️

---

## ⚠️ **ADVERTENCIA IMPORTANTE**

Este es el setup para **PRODUCCIÓN CON DINERO REAL**.

- ✅ Usarás API keys **LIVE** (sk_live_...)
- ✅ Procesarás pagos REALES de clientes
- ✅ Stripe cobrará fees reales (2.9% + $0.30 por transacción)
- ⚠️ Necesitas **activar tu cuenta** completamente
- ⚠️ Requiere **verificación de identidad** (KYC)
- ⚠️ Requiere **cuenta bancaria verificada** para recibir fondos

Si prefieres empezar **sin riesgo**, usa primero [Test Mode](#modo-test-alternativa-segura).

---

## 📋 **PASO 1: CREAR CUENTA STRIPE**

### **1.1 Registro Inicial**

1. **Ir a:** https://dashboard.stripe.com/register

2. **Llenar formulario:**
   ```
   Email: tu-email@example.com
   Nombre completo: Tu Nombre
   País: España (o tu país)
   Contraseña: (segura, >12 caracteres)
   ```

3. **Verificar email:**
   - Revisa tu bandeja de entrada
   - Click en el link de verificación
   - Vuelve al dashboard

### **1.2 Información de Negocio**

Stripe te pedirá información sobre tu negocio:

```
Tipo de negocio: Individual / Company
Nombre del negocio: METACORTEX AI Services
Descripción: AI-powered APIs and automation services
Website: https://tu-dominio.com (opcional al inicio)
Categoría: Software / Technology
```

**Importante:** Sé honesto y preciso. Stripe revisa esta información.

---

## 🏦 **PASO 2: VERIFICACIÓN DE CUENTA**

### **2.1 Verificación de Identidad (KYC)**

Stripe requiere verificar tu identidad para prevenir fraude:

**Documentos necesarios:**
- ✅ DNI/Pasaporte (foto)
- ✅ Comprobante de domicilio (factura luz/agua < 3 meses)
- ✅ Número de teléfono

**Proceso:**
1. Dashboard → Settings → Business Details
2. Click en "Complete verification"
3. Sube documentos
4. Espera 1-3 días hábiles

### **2.2 Cuenta Bancaria**

Para recibir pagos, necesitas vincular una cuenta bancaria:

**España (SEPA):**
```
IBAN: ES12 3456 7890 1234 5678 9012
Titular: Tu Nombre (debe coincidir con cuenta Stripe)
```

**Otros países:**
- USA: Routing + Account Number
- UK: Sort Code + Account Number
- MX: CLABE

**Configurar:**
1. Dashboard → Settings → Bank accounts and scheduling
2. Add bank account
3. Stripe hará 2 micro-depósitos (€0.01, €0.02)
4. Verificar montos en 1-2 días

---

## 🔑 **PASO 3: OBTENER API KEYS REALES**

### **3.1 Activar Modo Live**

**Dashboard Stripe:**
```
1. Top-right corner: Ver toggle "Test mode" / "Live mode"
2. Cambiar a "Live mode" (color naranja)
3. Si no puedes, es porque falta completar verificación
```

### **3.2 Obtener Secret Key (CRÍTICA - NO COMPARTIR)**

1. **Navegar:**
   ```
   Dashboard → Developers → API keys
   ```

2. **Verificar modo:**
   ```
   Arriba debe decir: "Live mode API keys"
   ```

3. **Copiar Secret Key:**
   ```
   Standard keys
   ├── Publishable key: pk_live_51abc...  (OK compartir)
   └── Secret key: sk_live_51def...       (⚠️ NUNCA compartir)
   
   Click en "Reveal live key token"
   Copiar TODA la key (empieza con sk_live_)
   ```

**Ejemplo de keys LIVE:**
```bash
# ESTAS SON REALES (no test)
STRIPE_SECRET_KEY=sk_live_XXXXXXXXXXXXXXXXXXXXX_REPLACE_WITH_YOUR_REAL_KEY
STRIPE_PUBLISHABLE_KEY=pk_live_XXXXXXXXXXXXXXXXXXXXX_REPLACE_WITH_YOUR_REAL_KEY
```

### **3.3 Webhook Secret (para pagos automáticos)**

1. **Navegar:**
   ```
   Dashboard → Developers → Webhooks
   ```

2. **Add endpoint:**
   ```
   Endpoint URL: https://tu-dominio.com/api/v1/webhook/stripe
   
   ⚠️ IMPORTANTE: Debe ser HTTPS (no HTTP)
   ⚠️ Si aún no tienes dominio, usar ngrok temporalmente
   ```

3. **Seleccionar eventos:**
   ```
   ✅ payment_intent.succeeded
   ✅ payment_intent.payment_failed
   ✅ customer.subscription.created
   ✅ customer.subscription.updated
   ✅ customer.subscription.deleted
   ✅ invoice.payment_succeeded
   ✅ invoice.payment_failed
   ```

4. **Copiar Webhook Secret:**
   ```
   whsec_abc123def456...
   ```

---

## 🔧 **PASO 4: CONFIGURAR EN .env**

### **4.1 Backup del .env actual**

```bash
cd /Users/edkanina/ai_definitiva
cp .env .env.backup.$(date +%Y%m%d)
ls -la .env*
```

### **4.2 Editar .env**

```bash
# Abrir editor
nano .env

# O con VSCode
code .env
```

### **4.3 Actualizar Keys LIVE**

**ANTES (test mode):**
```bash
STRIPE_SECRET_KEY=sk_test_YOUR_KEY_HERE
STRIPE_PUBLISHABLE_KEY=pk_test_YOUR_KEY_HERE
STRIPE_WEBHOOK_SECRET=whsec_YOUR_WEBHOOK_SECRET_HERE
```

**DESPUÉS (LIVE mode):**
```bash
# =============================================================================
# STRIPE PRODUCTION (LIVE MODE) ⚠️ REAL MONEY
# =============================================================================
STRIPE_SECRET_KEY=sk_live_XXXXXXXXXXXXXXXXXXXXX_REPLACE_WITH_YOUR_REAL_KEY
STRIPE_PUBLISHABLE_KEY=pk_live_XXXXXXXXXXXXXXXXXXXXX_REPLACE_WITH_YOUR_REAL_KEY
STRIPE_WEBHOOK_SECRET=whsec_XXXXXXXXXXXXXXXXXXX_REPLACE_WITH_YOUR_WEBHOOK_SECRET

# Modo de entorno
ENVIRONMENT=production
DEBUG=false
```

**Guardar:**
```bash
# Nano: Ctrl+X, Y, Enter
# VSCode: Cmd+S
```

### **4.4 Verificar Permisos**

```bash
# .env debe ser privado (600)
chmod 600 .env
ls -la .env

# Debe mostrar: -rw------- (solo tú puedes leer/escribir)
```

---

## ✅ **PASO 5: TESTING DE CONEXIÓN**

### **5.1 Test de API Key**

```bash
cd /Users/edkanina/ai_definitiva
source .venv/bin/activate

# Test connection
python3 << 'PYTHON_EOF'
import stripe
import os
from dotenv import load_dotenv

load_dotenv()

# Cargar key LIVE
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

print("🔍 Testing Stripe LIVE connection...")
print(f"API Key: {stripe.api_key[:20]}... (masked)")

try:
    # Retrieve balance (debe funcionar con key LIVE válida)
    balance = stripe.Balance.retrieve()
    
    print("\n✅ STRIPE LIVE CONNECTION SUCCESS!")
    print(f"   Available balance: ${balance.available[0].amount / 100:.2f} {balance.available[0].currency.upper()}")
    print(f"   Pending balance: ${balance.pending[0].amount / 100:.2f} {balance.pending[0].currency.upper()}")
    print(f"   Livemode: {balance.livemode}")
    
    if not balance.livemode:
        print("\n⚠️ WARNING: Still in test mode!")
    
except stripe.error.AuthenticationError as e:
    print(f"\n❌ AUTHENTICATION ERROR: {e}")
    print("   → Check your STRIPE_SECRET_KEY in .env")
    print("   → Make sure it starts with 'sk_live_'")
    
except stripe.error.StripeError as e:
    print(f"\n❌ STRIPE ERROR: {e}")
    
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")

PYTHON_EOF
```

**Output esperado:**
```
🔍 Testing Stripe LIVE connection...
API Key: sk_live_51MNoPqRsT...

✅ STRIPE LIVE CONNECTION SUCCESS!
   Available balance: $0.00 USD
   Pending balance: $0.00 USD
   Livemode: True
```

### **5.2 Test de Payment Intent (REAL)**

**⚠️ CUIDADO: Esto creará un payment intent REAL**

```bash
python3 << 'PYTHON_EOF'
import stripe
import os
from dotenv import load_dotenv

load_dotenv()
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

print("💳 Creating REAL payment intent...")

try:
    # Crear payment intent de $1.00 USD
    intent = stripe.PaymentIntent.create(
        amount=100,  # $1.00 (en centavos)
        currency='usd',
        description='METACORTEX API Test Payment (REAL)',
        metadata={
            'test': 'true',
            'environment': 'production'
        }
    )
    
    print(f"\n✅ Payment Intent Created (REAL):")
    print(f"   ID: {intent.id}")
    print(f"   Amount: ${intent.amount / 100:.2f} {intent.currency.upper()}")
    print(f"   Status: {intent.status}")
    print(f"   Client Secret: {intent.client_secret[:30]}...")
    print(f"   Livemode: {intent.livemode}")
    
    if intent.livemode:
        print("\n⚠️ This is a REAL payment intent in production!")
        print("   You can complete it with a real card at:")
        print(f"   https://dashboard.stripe.com/test/payments/{intent.id}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")

PYTHON_EOF
```

---

## 🚀 **PASO 6: INICIAR API MONETIZATION SERVER**

### **6.1 Test del Server**

```bash
cd /Users/edkanina/ai_definitiva
source .venv/bin/activate

# Iniciar server en background
python metacortex_sinaptico/api_monetization_endpoint.py &

# Guardar PID
API_SERVER_PID=$!
echo "API Server PID: $API_SERVER_PID"

# Esperar 5 segundos
sleep 5

# Test health endpoint
curl http://localhost:8100/health | python -m json.tool

# Test root endpoint
curl http://localhost:8100/ | python -m json.tool
```

**Output esperado:**
```json
{
  "service": "METACORTEX API Monetization",
  "version": "1.0.0",
  "status": "operational",
  "payment_processor": "active"
}
```

### **6.2 Test de Registro (sin pago)**

```bash
# Registrar usuario de prueba
curl -X POST http://localhost:8100/api/v1/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "name": "Test User",
    "company": "Test Company"
  }' | python -m json.tool
```

**Output esperado:**
```json
{
  "success": true,
  "user_id": "user_abc123...",
  "api_key": "mctx_free_def456...",
  "jwt_token": "eyJhbGc...",
  "plan": "free",
  "requests_limit": 100,
  "message": "Account created successfully"
}
```

### **6.3 Test de Suscripción (PAGO REAL)**

**⚠️ CUIDADO: Esto intentará crear una suscripción REAL en Stripe**

```bash
# Primero, necesitas un payment method en Stripe
# Usa este comando SOLO si estás seguro:

curl -X POST http://localhost:8100/api/v1/subscribe \
  -H "Content-Type: application/json" \
  -d '{
    "plan_id": "basic",
    "email": "tu-email@example.com",
    "payment_method": "stripe"
  }' | python -m json.tool
```

---

## 🌐 **PASO 7: CONFIGURAR DOMINIO PÚBLICO (HTTPS)**

Para webhooks de Stripe, necesitas HTTPS público.

### **Opción A: Ngrok (Temporal - para testing)**

```bash
# Instalar ngrok
brew install ngrok

# Autenticar (obtener token en ngrok.com)
ngrok config add-authtoken TU_TOKEN_NGROK

# Crear túnel público a puerto 8100
ngrok http 8100

# Copiar URL pública (https://abc123.ngrok.io)
```

**Configurar webhook en Stripe:**
```
URL: https://abc123.ngrok.io/api/v1/webhook/stripe
```

### **Opción B: Dominio Real + Let's Encrypt (PRODUCCIÓN)**

**Requisitos:**
- Dominio propio (e.g., api.metacortex.ai)
- VPS/servidor con IP pública

**Setup:**
```bash
# 1. Instalar certbot
sudo apt install certbot python3-certbot-nginx

# 2. Obtener certificado SSL (GRATIS)
sudo certbot --nginx -d api.tu-dominio.com

# 3. Configurar nginx como proxy
sudo nano /etc/nginx/sites-available/api

# Contenido:
server {
    listen 443 ssl;
    server_name api.tu-dominio.com;
    
    ssl_certificate /etc/letsencrypt/live/api.tu-dominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.tu-dominio.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8100;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# 4. Habilitar y recargar
sudo ln -s /etc/nginx/sites-available/api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

**Webhook URL:**
```
https://api.tu-dominio.com/api/v1/webhook/stripe
```

---

## 💰 **PASO 8: PRIMERA TRANSACCIÓN REAL**

### **8.1 Crear Producto en Stripe**

1. **Dashboard → Products**
2. **Add product:**
   ```
   Name: METACORTEX API - Basic Plan
   Description: 1,000 API requests per month
   Price: $29.00 USD / month (recurring)
   ```

3. **Copiar Price ID:**
   ```
   price_1abc123def456...
   ```

### **8.2 Test con Tarjeta Real**

**⚠️ ADVERTENCIA: Esto cobrará dinero REAL**

```bash
# Usar TU tarjeta real para probar
# Stripe cobrará $29 USD

# O usar tarjeta de prueba si aún estás en test mode:
# Número: 4242 4242 4242 4242
# Fecha: 12/34
# CVC: 123
```

### **8.3 Verificar Pago en Dashboard**

1. **Dashboard → Payments**
2. Debe aparecer el pago de $29.00
3. Status: "Succeeded"
4. Fee: ~$1.14 (2.9% + $0.30)
5. Net: ~$27.86

---

## 📊 **PASO 9: MONITOREO Y REPORTING**

### **9.1 Dashboard de Ingresos**

```bash
# Ver reporte de ingresos REAL
python3 << 'PYTHON_EOF'
import stripe
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

print("💰 STRIPE REVENUE REPORT (REAL)")
print("="*60)

# Balance actual
balance = stripe.Balance.retrieve()
available = balance.available[0].amount / 100
pending = balance.pending[0].amount / 100

print(f"\n💵 Balance:")
print(f"   Available: ${available:.2f}")
print(f"   Pending: ${pending:.2f}")
print(f"   Total: ${available + pending:.2f}")

# Pagos de los últimos 30 días
payments = stripe.PaymentIntent.list(
    limit=100,
    created={'gte': int((datetime.now() - timedelta(days=30)).timestamp())}
)

total_revenue = sum(p.amount / 100 for p in payments.data if p.status == 'succeeded')
total_fees = total_revenue * 0.029 + (len([p for p in payments.data if p.status == 'succeeded']) * 0.30)
net_revenue = total_revenue - total_fees

print(f"\n📈 Last 30 days:")
print(f"   Transactions: {len([p for p in payments.data if p.status == 'succeeded'])}")
print(f"   Gross Revenue: ${total_revenue:.2f}")
print(f"   Stripe Fees: ${total_fees:.2f}")
print(f"   Net Revenue: ${net_revenue:.2f}")

# Suscripciones activas
subscriptions = stripe.Subscription.list(status='active', limit=100)

print(f"\n🔄 Active Subscriptions: {len(subscriptions.data)}")
for sub in subscriptions.data[:5]:
    print(f"   - {sub.id}: ${sub.plan.amount / 100:.2f}/{sub.plan.interval}")

PYTHON_EOF
```

### **9.2 Alertas Automáticas**

Añadir a tu código:

```python
# webhook handler en api_monetization_endpoint.py

@app.post("/api/v1/webhook/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)
    
    if event.type == "payment_intent.succeeded":
        amount = event.data.object.amount / 100
        
        # ALERTA: Nuevo pago recibido
        send_notification(
            title="💰 Nuevo Pago Recibido",
            message=f"${amount:.2f} USD de {event.data.object.customer}",
            channel="telegram"  # o Discord, email, etc.
        )
    
    return {"received": True}
```

---

## 🔒 **SEGURIDAD CRÍTICA**

### **Checklist de Seguridad LIVE:**

```bash
✅ 1. .env está en .gitignore
   grep ".env" .gitignore

✅ 2. .env tiene permisos 600
   chmod 600 .env && ls -la .env

✅ 3. Secret keys NUNCA en código
   grep -r "sk_live" . --exclude-dir=node_modules --exclude=.env

✅ 4. HTTPS obligatorio en producción
   # Verificar en nginx o servidor

✅ 5. Webhook signature verification
   # Ya implementado en api_monetization_endpoint.py

✅ 6. Rate limiting activo
   # Ya implementado con slowapi

✅ 7. Logging de transacciones
   tail -f logs/stripe_transactions.log

✅ 8. 2FA en cuenta Stripe
   Dashboard → Settings → Security → Enable 2FA

✅ 9. Alertas de fraude configuradas
   Dashboard → Radar → Rules → Add rule

✅ 10. Backup diario de datos
   crontab -e
   # 0 2 * * * /path/to/backup_script.sh
```

---

## 📞 **SOPORTE**

### **Stripe Support:**
- Docs: https://stripe.com/docs
- Support: https://support.stripe.com/
- Chat: Dashboard → bottom-right (? icon)

### **Errores Comunes:**

**Error: "Invalid API Key"**
```
→ Verificar que key empiece con sk_live_
→ Verificar que no tiene espacios extra
→ Verificar que está en .env correctamente
```

**Error: "Account not activated"**
```
→ Completar verificación de identidad
→ Añadir cuenta bancaria
→ Esperar aprobación de Stripe (1-3 días)
```

**Error: "Webhook signature invalid"**
```
→ Verificar STRIPE_WEBHOOK_SECRET en .env
→ Verificar que endpoint URL es correcta
→ Verificar que es HTTPS
```

---

## ✅ **CHECKLIST FINAL**

```
✅ Cuenta Stripe creada
✅ Verificación de identidad completada
✅ Cuenta bancaria vinculada
✅ API keys LIVE obtenidas
✅ .env actualizado con keys LIVE
✅ Test de conexión exitoso
✅ API server corriendo
✅ Webhooks configurados
✅ HTTPS configurado (ngrok o dominio real)
✅ Primera transacción de prueba
✅ Monitoreo y alertas activas
✅ Security checklist completado
```

---

## 🎯 **PRÓXIMOS PASOS**

1. **Marketing:**
   - Landing page para API
   - Twitter, Reddit, Hacker News
   - Blog posts sobre casos de uso

2. **Optimización:**
   - A/B testing de precios
   - Analítica de conversión
   - Feedback de usuarios

3. **Escalado:**
   - Añadir más planes (Enterprise, Custom)
   - Referral program
   - Partnerships con empresas

---

**¡YA ESTÁS LISTO PARA PROCESAR PAGOS REALES Y GENERAR INGRESOS! 💰**

**Primera Meta: $100 USD en 30 días**

---

**Última actualización:** 24 de Noviembre de 2025  
**Autor:** METACORTEX Team
