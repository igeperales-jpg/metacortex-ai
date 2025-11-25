# 🚀 METACORTEX Deployment Guide

## Guía Completa de Despliegue en Hostinger y Otros Servidores

---

## 📋 Tabla de Contenidos

1. [Requisitos Previos](#requisitos-previos)
2. [Configuración Local](#configuración-local)
3. [Deployment en Hostinger](#deployment-en-hostinger)
4. [Deployment en VPS (DigitalOcean, AWS, etc.)](#deployment-en-vps)
5. [Configuración de Dominios](#configuración-de-dominios)
6. [Configuración de Canales](#configuración-de-canales)
7. [Monitoreo y Mantenimiento](#monitoreo-y-mantenimiento)

---

## 🔧 Requisitos Previos

### Software Necesario
- **Python 3.11+**
- **pip** (gestor de paquetes)
- **Git** (opcional, para deployment)
- **Ollama** (para modelos de lenguaje)

### Dependencias Python
```bash
pip install -r requirements.txt
```

### Variables de Entorno Requeridas

Crear archivo `.env` en la raíz del proyecto:

```env
# Telegram Bot
TELEGRAM_BOT_TOKEN=tu_token_aqui

# WhatsApp (Twilio)
TWILIO_ACCOUNT_SID=tu_account_sid
TWILIO_AUTH_TOKEN=tu_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# Email (opcional)
EMERGENCY_EMAIL=emergency@tudominio.com
EMAIL_PASSWORD=tu_password_smtp

# Sistema
ENVIRONMENT=production
DEBUG=false
```

---

## 💻 Configuración Local

### 1. Instalar Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Iniciar servidor
ollama serve
```

### 2. Descargar Modelos

```bash
ollama pull mistral:latest
ollama pull mistral:instruct
ollama pull mistral-nemo:latest
```

### 3. Iniciar Sistema Local

```bash
# Opción 1: Sistema unificado (RECOMENDADO)
python unified_startup.py

# Opción 2: Script maestro
./metacortex_master.sh start

# Verificar que funciona
curl http://localhost:8080/api/status
```

---

## 🌐 Deployment en Hostinger

### Opción 1: Python App (Recomendado)

1. **Acceder al Panel de Hostinger**
   - Ir a "Advanced" → "Python App"

2. **Crear Nueva Aplicación Python**
   ```
   Python version: 3.11 o superior
   Application root: /home/usuario/metacortex
   Application URL: tudominio.com
   Application startup file: unified_startup.py
   Application Entry point: main
   ```

3. **Subir Código**
   ```bash
   # Desde tu máquina local
   rsync -avz --exclude='.git' --exclude='__pycache__' \
         --exclude='ml_models' --exclude='logs' \
         ./ usuario@tudominio.com:/home/usuario/metacortex/
   ```

4. **Instalar Dependencias**
   ```bash
   # SSH a tu servidor
   ssh usuario@tudominio.com
   
   cd /home/usuario/metacortex
   pip install -r requirements.txt
   ```

5. **Configurar Variables de Entorno**
   - En panel de Hostinger → Python App → Environment Variables
   - Añadir todas las variables del archivo `.env`

6. **Iniciar Aplicación**
   - Click en "Start Application"
   - Verificar en: https://tudominio.com/api/status

### Opción 2: VPS Manual

Si Hostinger no soporta Python Apps, usar VPS:

1. **Crear VPS en Hostinger**
   - Panel → VPS → Create New

2. **Conectar por SSH**
   ```bash
   ssh root@tu-ip-server
   ```

3. **Instalar Dependencias del Sistema**
   ```bash
   apt update && apt upgrade -y
   apt install python3.11 python3-pip nginx git -y
   ```

4. **Clonar o Subir Código**
   ```bash
   cd /var/www
   git clone https://github.com/tu-usuario/metacortex.git
   cd metacortex
   ```

5. **Crear Entorno Virtual**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

6. **Configurar Systemd Service**
   ```bash
   nano /etc/systemd/system/metacortex.service
   ```
   
   Contenido:
   ```ini
   [Unit]
   Description=METACORTEX Unified System
   After=network.target
   
   [Service]
   Type=simple
   User=www-data
   WorkingDirectory=/var/www/metacortex
   Environment="PATH=/var/www/metacortex/.venv/bin"
   ExecStart=/var/www/metacortex/.venv/bin/python unified_startup.py
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   Activar:
   ```bash
   systemctl enable metacortex
   systemctl start metacortex
   systemctl status metacortex
   ```

7. **Configurar Nginx como Reverse Proxy**
   ```bash
   nano /etc/nginx/sites-available/metacortex
   ```
   
   Contenido:
   ```nginx
   server {
       listen 80;
       server_name tudominio.com www.tudominio.com;
   
       location / {
           proxy_pass http://127.0.0.1:8080;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```
   
   Activar:
   ```bash
   ln -s /etc/nginx/sites-available/metacortex /etc/nginx/sites-enabled/
   nginx -t
   systemctl restart nginx
   ```

8. **Configurar SSL con Let's Encrypt**
   ```bash
   apt install certbot python3-certbot-nginx -y
   certbot --nginx -d tudominio.com -d www.tudominio.com
   ```

---

## 🌍 Deployment en VPS (DigitalOcean, AWS, etc.)

### DigitalOcean Droplet

1. **Crear Droplet**
   - Ubuntu 22.04 LTS
   - 2 GB RAM mínimo (4 GB recomendado para Ollama)
   - Añadir tu SSH key

2. **Seguir pasos de "Opción 2: VPS Manual"** arriba

### AWS EC2

1. **Crear Instancia EC2**
   - AMI: Ubuntu Server 22.04 LTS
   - Tipo: t3.medium (2 vCPU, 4 GB RAM)
   - Security Group: Abrir puertos 80, 443, 22

2. **Conectar y configurar**
   ```bash
   ssh -i tu-key.pem ubuntu@tu-ec2-ip
   # Seguir pasos de "Opción 2: VPS Manual"
   ```

---

## 🔗 Configuración de Dominios

### Hostinger DNS

1. **Acceder a DNS Zone Editor**
   - Panel → Domains → Manage → DNS Zone Editor

2. **Añadir Registros**
   ```
   Tipo: A
   Nombre: @
   Apunta a: IP_DE_TU_SERVIDOR
   TTL: 14400
   
   Tipo: A
   Nombre: www
   Apunta a: IP_DE_TU_SERVIDOR
   TTL: 14400
   ```

3. **Para Subdominio de API (opcional)**
   ```
   Tipo: A
   Nombre: api
   Apunta a: IP_DE_TU_SERVIDOR
   TTL: 14400
   ```

---

## 📱 Configuración de Canales

### Telegram Bot

1. **Crear Bot**
   - Hablar con @BotFather en Telegram
   - Comando: `/newbot`
   - Seguir instrucciones
   - Copiar token

2. **Configurar Webhook (opcional)**
   ```bash
   curl https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://tudominio.com/telegram/webhook
   ```

3. **Añadir Token a .env**
   ```env
   TELEGRAM_BOT_TOKEN=tu_token_aqui
   ```

### WhatsApp (Twilio)

1. **Crear Cuenta en Twilio**
   - https://www.twilio.com/try-twilio

2. **Activar WhatsApp Sandbox**
   - Console → Messaging → Try it Out → Try WhatsApp

3. **Obtener Credenciales**
   - Account SID
   - Auth Token
   - WhatsApp Number

4. **Configurar Webhook**
   ```
   URL: https://tudominio.com/whatsapp/webhook
   Método: POST
   ```

5. **Añadir a .env**
   ```env
   TWILIO_ACCOUNT_SID=tu_account_sid
   TWILIO_AUTH_TOKEN=tu_auth_token
   TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
   ```

### Email (SMTP)

1. **Usar Gmail**
   - Activar "App Password" en Google Account
   - Settings → Security → 2-Step Verification → App passwords

2. **Configurar en .env**
   ```env
   EMERGENCY_EMAIL=tu_email@gmail.com
   EMAIL_PASSWORD=tu_app_password
   ```

---

## 📊 Monitoreo y Mantenimiento

### Logs

```bash
# Ver logs en tiempo real
tail -f logs/unified_system.log

# Ver logs de Telegram
tail -f logs/emergency_contact_stdout.log

# Ver logs de sistema (si usas systemd)
journalctl -u metacortex -f
```

### Status Endpoints

```bash
# Estado general
curl https://tudominio.com/api/status

# Health check
curl https://tudominio.com/health

# Estadísticas
curl https://tudominio.com/api/emergency/stats
```

### Backup

```bash
# Crear backup de base de datos
cp metacortex.sqlite backups/metacortex_$(date +%Y%m%d).sqlite

# Backup de configuración
tar -czf backups/config_$(date +%Y%m%d).tar.gz .env divine_protection_config.json

# Backup de modelos ML (opcional, ocupa mucho espacio)
tar -czf backups/ml_models_$(date +%Y%m%d).tar.gz ml_models/
```

### Actualización

```bash
# Método 1: Con Git
cd /var/www/metacortex
git pull origin main
systemctl restart metacortex

# Método 2: Manual
# Subir nuevos archivos con rsync
rsync -avz ./ usuario@servidor:/var/www/metacortex/
ssh usuario@servidor 'cd /var/www/metacortex && systemctl restart metacortex'
```

---

## 🆘 Troubleshooting

### Bot de Telegram no responde

```bash
# Verificar logs
tail -f logs/emergency_contact_stdout.log

# Verificar token
curl https://api.telegram.org/bot<TOKEN>/getMe

# Reiniciar servicio
systemctl restart metacortex
```

### Ollama no funciona

```bash
# Verificar que está corriendo
curl http://localhost:11434/api/tags

# Iniciar Ollama
ollama serve

# Ver logs
journalctl -u ollama -f
```

### Error de permisos

```bash
# Dar permisos correctos
chown -R www-data:www-data /var/www/metacortex
chmod +x unified_startup.py
```

---

## 📞 Soporte

Si tienes problemas:

1. **Revisar logs**: `tail -f logs/*.log`
2. **Verificar status**: `curl https://tudominio.com/api/status`
3. **Contactar soporte**: emergency@metacortex.ai

---

## ✅ Checklist de Deployment

- [ ] Python 3.11+ instalado
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Ollama instalado y corriendo
- [ ] Modelos descargados
- [ ] Archivo `.env` configurado
- [ ] Telegram bot creado y token añadido
- [ ] WhatsApp/Twilio configurado (opcional)
- [ ] Dominio configurado y apuntando a servidor
- [ ] SSL certificado instalado
- [ ] Nginx configurado como reverse proxy
- [ ] Systemd service configurado y activo
- [ ] Firewall configurado (puertos 80, 443, 22)
- [ ] Logs funcionando correctamente
- [ ] Backup configurado
- [ ] Monitoreo activo

---

**¡Sistema listo para salvar vidas! 🛡️**
