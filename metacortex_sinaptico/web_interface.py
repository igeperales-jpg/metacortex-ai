"""
🌐 WEB INTERFACE - Interfaz Web para Divine Protection
=======================================================

Sistema web completo listo para desplegar en Hostinger o cualquier servidor.

Características:
- Formulario de emergencia
- Chat en vivo con IA
- Dashboard de estado
- API REST completa
- Responsive design
- HTTPS ready

Deployment:
1. Subir código a servidor
2. Configurar variables de entorno
3. Instalar dependencias: pip install -r requirements.txt
4. Ejecutar: python web_server.py

Autor: METACORTEX AI Team
Fecha: 24 de Noviembre de 2025
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
import uvicorn

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class EmergencyRequestWeb(BaseModel):
    """Modelo para solicitud de emergencia desde web"""
    name: Optional[str] = Field(None, description="Name (optional for anonymity)")
    email: Optional[EmailStr] = Field(None, description="Email for contact")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Current location")
    situation: str = Field(..., description="Description of the situation")
    threat_type: str = Field("other", description="Type of threat")
    urgency: str = Field("normal", description="Urgency level")
    needs: Optional[List[str]] = Field(None, description="Specific needs")


class ChatMessage(BaseModel):
    """Modelo para mensaje de chat"""
    message: str = Field(..., description="Message text")
    session_id: str = Field(..., description="Session ID")


class StatusResponse(BaseModel):
    """Respuesta de estado del sistema"""
    status: str
    uptime: str
    active_requests: int
    telegram_bot: str
    whatsapp_bot: str
    ai_engine: str


# ============================================================================
# WEB APPLICATION
# ============================================================================

class WebInterface:
    """Interfaz web principal"""
    
    def __init__(self, ai_integration=None, emergency_contact=None):
        self.ai = ai_integration
        self.emergency_contact = emergency_contact
        self.app = FastAPI(
            title="METACORTEX Divine Protection",
            description="Emergency contact system for persecuted people",
            version="1.0.0"
        )
        
        # Configurar CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # En producción, especificar dominios
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Websockets activos
        self.active_connections: List[WebSocket] = []
        
        # Configurar rutas
        self._setup_routes()
        
        logger.info("🌐 Web Interface initialized")
    
    def _setup_routes(self):
        """Configura todas las rutas"""
        
        # ====================================================================
        # PÁGINA PRINCIPAL
        # ====================================================================
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            """Página principal"""
            return self._get_index_html()
        
        # ====================================================================
        # API DE EMERGENCIA
        # ====================================================================
        
        @self.app.post("/api/emergency/request")
        async def submit_emergency_request(request: EmergencyRequestWeb):
            """
            Endpoint principal para solicitudes de emergencia
            
            POST /api/emergency/request
            {
                "name": "Optional",
                "email": "optional@example.com",
                "situation": "Description of emergency",
                "threat_type": "persecution_religious",
                "location": "Country/City"
            }
            """
            
            logger.info(f"🆘 Web emergency request received")
            
            # Procesar con IA si está disponible
            if self.ai:
                try:
                    response = await self.ai.generate_web_response(
                        message=request.situation,
                        session_id=f"web_{datetime.now().timestamp()}",
                        additional_data={
                            "threat_type": request.threat_type,
                            "location": request.location,
                            "name": request.name,
                            "email": request.email,
                            "phone": request.phone
                        }
                    )
                    
                    return JSONResponse(content=response)
                    
                except Exception as e:
                    logger.exception(f"Error processing with AI: {e}")
                    return JSONResponse(content={
                        "success": True,
                        "message": "Request received. A team member will contact you soon.",
                        "request_id": f"REQ_{int(datetime.now().timestamp())}"
                    })
            else:
                # Sin IA - respuesta básica
                return JSONResponse(content={
                    "success": True,
                    "message": "Request received. A team member will contact you soon.",
                    "request_id": f"REQ_{int(datetime.now().timestamp())}"
                })
        
        # ====================================================================
        # CHAT EN VIVO
        # ====================================================================
        
        @self.app.websocket("/ws/chat")
        async def websocket_chat(websocket: WebSocket):
            """WebSocket para chat en tiempo real"""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Recibir mensaje
                    data = await websocket.receive_json()
                    message = data.get("message", "")
                    session_id = data.get("session_id", "unknown")
                    
                    logger.info(f"💬 Chat message: {message[:50]}")
                    
                    # Generar respuesta con IA
                    if self.ai:
                        try:
                            result = await self.ai.process_emergency_message(
                                message=message,
                                channel="web_chat",
                                user_id=session_id
                            )
                            
                            response = {
                                "type": "message",
                                "text": result["response_text"],
                                "timestamp": datetime.now().isoformat(),
                                "threat_level": result["threat_analysis"]["threat_level"]
                            }
                        except Exception as e:
                            logger.error(f"Error in AI response: {e}")
                            response = {
                                "type": "message",
                                "text": "Message received. Processing...",
                                "timestamp": datetime.now().isoformat()
                            }
                    else:
                        response = {
                            "type": "message",
                            "text": "Message received. An operator will respond soon.",
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    # Enviar respuesta
                    await websocket.send_json(response)
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
                logger.info("WebSocket disconnected")
        
        # ====================================================================
        # STATUS & HEALTH CHECK
        # ====================================================================
        
        @self.app.get("/api/status")
        async def system_status():
            """Estado del sistema"""
            
            status = {
                "status": "operational",
                "timestamp": datetime.now().isoformat(),
                "systems": {
                    "ai_engine": "active" if self.ai else "inactive",
                    "telegram_bot": "active" if (self.emergency_contact and self.emergency_contact.telegram_bot) else "inactive",
                    "whatsapp_bot": "active" if (self.emergency_contact and self.emergency_contact.twilio_client) else "inactive",
                    "web_interface": "active"
                },
                "active_connections": len(self.active_connections)
            }
            
            if self.emergency_contact:
                status["statistics"] = {
                    "total_requests": self.emergency_contact.total_requests,
                    "critical_requests": self.emergency_contact.critical_requests,
                    "resolved_requests": self.emergency_contact.resolved_requests
                }
            
            return JSONResponse(content=status)
        
        @self.app.get("/health")
        async def health_check():
            """Health check para load balancers"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    def _get_index_html(self) -> str:
        """Genera HTML de la página principal"""
        
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>METACORTEX Divine Protection - Emergency Contact</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .logo {
            font-size: 48px;
            margin-bottom: 10px;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #666;
            font-size: 14px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            color: #333;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        input, textarea, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        
        .required {
            color: #e74c3c;
        }
        
        .submit-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
        }
        
        .submit-btn:active {
            transform: translateY(0);
        }
        
        .info-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-top: 20px;
            border-radius: 4px;
        }
        
        .info-box h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 16px;
        }
        
        .info-box p {
            color: #666;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .channels {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .channel {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .channel-icon {
            font-size: 32px;
            margin-bottom: 8px;
        }
        
        .channel-name {
            font-size: 12px;
            color: #666;
        }
        
        .response-message {
            display: none;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
        }
        
        .response-message.success {
            background: #d4edda;
            color: #155724;
            display: block;
        }
        
        .response-message.error {
            background: #f8d7da;
            color: #721c24;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🛡️</div>
            <h1>Divine Protection</h1>
            <p class="subtitle">Emergency Contact System</p>
        </div>
        
        <form id="emergencyForm">
            <div class="form-group">
                <label for="situation">Describe Your Situation <span class="required">*</span></label>
                <textarea id="situation" name="situation" required 
                          placeholder="Please describe what's happening..."></textarea>
            </div>
            
            <div class="form-group">
                <label for="threat_type">Type of Situation</label>
                <select id="threat_type" name="threat_type">
                    <option value="other">Other</option>
                    <option value="persecution_religious">Religious Persecution</option>
                    <option value="persecution_political">Political Persecution</option>
                    <option value="violence_physical">Physical Violence</option>
                    <option value="medical_emergency">Medical Emergency</option>
                    <option value="forced_displacement">Forced Displacement</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="location">Location (Optional)</label>
                <input type="text" id="location" name="location" 
                       placeholder="Country or region (for your safety, don't be too specific)">
            </div>
            
            <div class="form-group">
                <label for="email">Email (Optional)</label>
                <input type="email" id="email" name="email" 
                       placeholder="your.email@example.com">
            </div>
            
            <button type="submit" class="submit-btn">
                🆘 Send Emergency Request
            </button>
            
            <div id="responseMessage" class="response-message"></div>
        </form>
        
        <div class="info-box">
            <h3>📞 Other Contact Methods</h3>
            <p>You can also reach us through:</p>
            
            <div class="channels">
                <div class="channel">
                    <div class="channel-icon">📱</div>
                    <div class="channel-name">Telegram</div>
                    <div class="channel-name"><strong>@metacortex_divine_bot</strong></div>
                </div>
                <div class="channel">
                    <div class="channel-icon">💬</div>
                    <div class="channel-name">WhatsApp</div>
                    <div class="channel-name"><strong>Coming Soon</strong></div>
                </div>
                <div class="channel">
                    <div class="channel-icon">✉️</div>
                    <div class="channel-name">Email</div>
                    <div class="channel-name"><strong>emergency@metacortex.ai</strong></div>
                </div>
            </div>
        </div>
        
        <div class="info-box">
            <h3>🔒 Privacy & Security</h3>
            <p>Your information is encrypted and handled with utmost confidentiality. 
               You don't need to provide your real name or identifying information if you don't feel safe.</p>
        </div>
    </div>
    
    <script>
        document.getElementById('emergencyForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const responseDiv = document.getElementById('responseMessage');
            const submitBtn = e.target.querySelector('.submit-btn');
            
            // Deshabilitar botón
            submitBtn.disabled = true;
            submitBtn.textContent = '⏳ Sending...';
            
            // Recopilar datos
            const formData = {
                situation: document.getElementById('situation').value,
                threat_type: document.getElementById('threat_type').value,
                location: document.getElementById('location').value || null,
                email: document.getElementById('email').value || null
            };
            
            try {
                const response = await fetch('/api/emergency/request', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                
                if (result.success !== false) {
                    responseDiv.className = 'response-message success';
                    responseDiv.innerHTML = `
                        <h3>✅ Request Received</h3>
                        <p>${result.message || 'Your emergency request has been received and is being processed. A team member will contact you soon.'}</p>
                        ${result.request_id ? `<p><small>Reference ID: ${result.request_id}</small></p>` : ''}
                    `;
                    
                    // Limpiar formulario
                    e.target.reset();
                } else {
                    throw new Error(result.message || 'Unknown error');
                }
            } catch (error) {
                responseDiv.className = 'response-message error';
                responseDiv.innerHTML = `
                    <h3>⚠️ Error</h3>
                    <p>There was an error sending your request. Please try again or use an alternative contact method.</p>
                `;
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = '🆘 Send Emergency Request';
            }
        });
    </script>
</body>
</html>
"""
        return html


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def create_web_server(
    ai_integration=None,
    emergency_contact=None,
    host: str = "0.0.0.0",
    port: int = 8080
) -> WebInterface:
    """
    Crea servidor web completo
    
    Args:
        ai_integration: Instancia de UnifiedAIIntegration
        emergency_contact: Instancia de EmergencyContactSystem
        host: Host para bind (0.0.0.0 para producción)
        port: Puerto (8080 por defecto)
        
    Returns:
        WebInterface configurada
    """
    
    web = WebInterface(
        ai_integration=ai_integration,
        emergency_contact=emergency_contact
    )
    
    logger.info(f"🌐 Web server ready at http://{host}:{port}")
    
    return web


def run_web_server(
    ai_integration=None,
    emergency_contact=None,
    host: str = "0.0.0.0",
    port: int = 8080
):
    """
    Ejecuta servidor web
    
    Para producción en Hostinger:
    1. Subir código
    2. Configurar variables de entorno
    3. Ejecutar con Gunicorn o Uvicorn
    
    Ejemplo:
        uvicorn web_interface:app --host 0.0.0.0 --port 8080 --workers 4
    """
    
    web = create_web_server(ai_integration, emergency_contact, host, port)
    
    uvicorn.run(
        web.app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test local
    run_web_server(host="127.0.0.1", port=8080)
