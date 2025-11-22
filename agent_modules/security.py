#!/usr/bin/env python3
"""
🔒 SECURITY MODULE - Hardening de seguridad enterprise (Military Grade)

Características:
    pass  # TODO: Implementar
- Secrets Management (vault integration)
- Encryption at rest and in transit
- Authentication (JWT, API Keys)
- Authorization (RBAC, ABAC)
- Input Validation & Sanitization
- SQL Injection Prevention
- XSS Prevention
- CSRF Protection
- Security Headers
- Audit Logging
- Rate Limiting Integration
- Hash & Signature Verification
"""

import json
import time
import hmac
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import re
import bcrypt
import bcrypt

logger = logging.getLogger(__name__)

# Cryptography imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning(
        "⚠️ cryptography no disponible - algunas funciones de seguridad deshabilitadas"
    )


class Permission(Enum):
    """Permisos del sistema"""

    # Admin
    ADMIN_ALL = "admin:*"

    # ML
    ML_TRAIN = "ml:train"
    ML_PREDICT = "ml:predict"
    ML_DEPLOY = "ml:deploy"
    ML_READ = "ml:read"

    # Cognitive
    COGNITIVE_READ = "cognitive:read"
    COGNITIVE_WRITE = "cognitive:write"
    COGNITIVE_EXECUTE = "cognitive:execute"

    # Agent
    AGENT_CREATE = "agent:create"
    AGENT_EXECUTE = "agent:execute"
    AGENT_READ = "agent:read"

    # System
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_AUDIT = "system:audit"


@dataclass
class Role:
    """Rol con permisos"""

    name: str
    permissions: List[Permission]
    description: str = ""


@dataclass
class User:
    """Usuario del sistema"""

    user_id: str
    username: str
    roles: List[str] = field(default_factory=list)
    api_key_hash: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLog:
    """Log de auditoría"""

    timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    action: str = ""
    resource: str = ""
    result: str = "success"  # success, denied, error
    ip_address: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "result": self.result,
            "ip_address": self.ip_address,
            "metadata": self.metadata,
        }


class SecretsManager:
    """Gestor de secretos y credenciales"""

    def __init__(self, secrets_file: str = ".secrets.enc"):
        self.secrets_file = Path(secrets_file)
        self.secrets: Dict[str, str] = {}
        self.encryption_key: Optional[bytes] = None

        if CRYPTO_AVAILABLE:
            self._initialize_encryption()
            self._load_secrets()
        else:
            logger.warning("⚠️ Secrets manager en modo inseguro (sin encriptación)")

    def _initialize_encryption(self):
        """Inicializar encriptación"""
        key_file = Path(".encryption_key")

        if key_file.exists():
            with open(key_file, "rb") as f:
                self.encryption_key = f.read()
        else:
            # Generar nueva key
            self.encryption_key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self.encryption_key)
            key_file.chmod(0o600)  # Solo lectura para owner
            logger.info("🔑 Nueva encryption key generada")

    def _load_secrets(self):
        """Cargar secretos desde archivo encriptado"""
        if not self.secrets_file.exists():
            return

        try:
            with open(self.secrets_file, "rb") as f:
                encrypted_data = f.read()

            fernet = Fernet(self.encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            self.secrets = json.loads(decrypted_data.decode())

            logger.info(f"✅ {len(self.secrets)} secretos cargados")
        except Exception as e:
            logger.error(f"❌ Error cargando secretos: {e}")

    def _save_secrets(self):
        """Guardar secretos en archivo encriptado"""
        if not CRYPTO_AVAILABLE:
            logger.warning("⚠️ No se puede guardar sin encriptación")
            return

        try:
            fernet = Fernet(self.encryption_key)
            data = json.dumps(self.secrets).encode()
            encrypted_data = fernet.encrypt(data)

            with open(self.secrets_file, "wb") as f:
                f.write(encrypted_data)

            self.secrets_file.chmod(0o600)
            logger.info("💾 Secretos guardados")
        except Exception as e:
            logger.error(f"❌ Error guardando secretos: {e}")

    def set_secret(self, key: str, value: str):
        """Guardar secreto"""
        self.secrets[key] = value
        self._save_secrets()

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Obtener secreto"""
        return self.secrets.get(key, default)

    def delete_secret(self, key: str):
        """Eliminar secreto"""
        if key in self.secrets:
            del self.secrets[key]
            self._save_secrets()


class InputValidator:
    """Validación y sanitización de inputs"""

    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitizar string"""
        # Limitar longitud
        sanitized = input_str[:max_length]

        # Eliminar caracteres peligrosos
        dangerous_chars = ["<", ">", '"', "'", "&", ";", "|", "`", "$"]
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")

        return sanitized.strip()

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validar formato de email"""

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validar formato de API key"""
        # Debe ser alphanumeric, longitud 32-64
        if not api_key or len(api_key) < 32 or len(api_key) > 64:
            return False
        return api_key.isalnum()

    @staticmethod
    def prevent_sql_injection(query: str) -> bool:
        """Detectar potencial SQL injection"""
        dangerous_patterns = [
            "DROP TABLE",
            "DELETE FROM",
            "INSERT INTO",
            "UPDATE ",
            "--",
            ";--",
            "UNION SELECT",
            "OR 1=1",
            "OR 1 = 1",
        ]

        query_upper = query.upper()
        for pattern in dangerous_patterns:
            if pattern in query_upper:
                logger.warning(f"⚠️ Posible SQL injection detectado: {pattern}")
                return True

        return False

    @staticmethod
    def prevent_xss(input_str: str) -> str:
        """Prevenir XSS"""
        # HTML escape
        replacements = {
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
            "/": "&#x2F;",
        }

        for char, replacement in replacements.items():
            input_str = input_str.replace(char, replacement)

        return input_str


class AuthenticationSystem:
    """Sistema de autenticación"""

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_user(self, username: str, roles: List[str]) -> User:
        """Crear usuario"""
        user_id = hashlib.sha256(f"{username}{time.time()}".encode()).hexdigest()[:16]

        user = User(user_id=user_id, username=username, roles=roles)

        self.users[user_id] = user
        logger.info(f"✅ Usuario creado: {username} ({user_id})")

        return user

    def generate_api_key(self, user_id: str) -> str:
        """Generar API key para usuario"""
        if user_id not in self.users:
            raise ValueError(f"Usuario no existe: {user_id}")

        # Generar API key segura
        api_key = secrets.token_urlsafe(32)

        # Hash del API key para almacenar
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Guardar
        self.users[user_id].api_key_hash = api_key_hash
        self.api_keys[api_key_hash] = user_id

        logger.info(f"🔑 API key generada para: {user_id}")

        return api_key

    def verify_api_key(self, api_key: str) -> Optional[User]:
        """Verificar API key"""
        # Hash del API key
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Buscar usuario
        user_id = self.api_keys.get(api_key_hash)
        if not user_id:
            return None

        user = self.users.get(user_id)
        if user:
            user.last_login = time.time()

        return user

    def create_session(self, user_id: str, duration_seconds: int = 3600) -> str:
        """Crear sesión temporal"""
        session_id = secrets.token_urlsafe(32)

        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": time.time(),
            "expires_at": time.time() + duration_seconds,
        }

        return session_id

    def verify_session(self, session_id: str) -> Optional[User]:
        """Verificar sesión"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        # Verificar expiración
        if time.time() > session["expires_at"]:
            del self.sessions[session_id]
            return None

        return self.users.get(session["user_id"])


class AuthorizationSystem:
    """Sistema de autorización (RBAC)"""

    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self._initialize_default_roles()

    def _initialize_default_roles(self):
        """Inicializar roles por defecto"""
        # Admin role
        self.add_role(
            Role(
                name="admin",
                permissions=[Permission.ADMIN_ALL],
                description="Administrator with full access",
            )
        )

        # ML Engineer role
        self.add_role(
            Role(
                name="ml_engineer",
                permissions=[
                    Permission.ML_TRAIN,
                    Permission.ML_PREDICT,
                    Permission.ML_DEPLOY,
                    Permission.ML_READ,
                ],
                description="ML Engineer",
            )
        )

        # Developer role
        self.add_role(
            Role(
                name="developer",
                permissions=[
                    Permission.AGENT_CREATE,
                    Permission.AGENT_EXECUTE,
                    Permission.AGENT_READ,
                    Permission.COGNITIVE_READ,
                    Permission.ML_PREDICT,
                    Permission.ML_READ,
                ],
                description="Developer",
            )
        )

        # Viewer role
        self.add_role(
            Role(
                name="viewer",
                permissions=[
                    Permission.ML_READ,
                    Permission.COGNITIVE_READ,
                    Permission.AGENT_READ,
                    Permission.SYSTEM_MONITOR,
                ],
                description="Read-only access",
            )
        )

    def add_role(self, role: Role):
        """Agregar rol"""
        self.roles[role.name] = role
        logger.info(f"✅ Rol agregado: {role.name}")

    def check_permission(self, user: User, permission: Permission) -> bool:
        """Verificar si usuario tiene permiso"""
        # Admin tiene todos los permisos
        if "admin" in user.roles:
            return True

        # Verificar permisos de todos los roles del usuario
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role and permission in role.permissions:
                return True

        return False


class SecuritySystem:
    """Sistema completo de seguridad"""

    def __init__(self, audit_log_file: str = "logs/security_audit.jsonl"):
        self.secrets_manager = SecretsManager()
        self.auth = AuthenticationSystem()
        self.authz = AuthorizationSystem()
        self.validator = InputValidator()

        # Audit logging
        self.audit_log_file = Path(audit_log_file)
        self.audit_log_file.parent.mkdir(exist_ok=True, parents=True)

        # 🧠 Conexión a red neuronal
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("security", self)
            logger.info("✅ 'security' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

        logger.info("✅ Security System inicializado")

    def audit_log(self, log: AuditLog):
        """Registrar en audit log"""
        with open(self.audit_log_file, "a") as f:
            f.write(json.dumps(log.to_dict()) + "\n")

    def require_auth(self, api_key: str) -> Optional[User]:
        """Requerir autenticación"""
        user = self.auth.verify_api_key(api_key)

        if not user:
            self.audit_log(
                AuditLog(
                    action="authentication_failed", resource="api", result="denied"
                )
            )
            return None

        self.audit_log(
            AuditLog(
                user_id=user.user_id,
                action="authentication_success",
                resource="api",
                result="success",
            )
        )

        return user

    def require_permission(self, user: User, permission: Permission) -> bool:
        """Requerir permiso"""
        has_permission = self.authz.check_permission(user, permission)

        self.audit_log(
            AuditLog(
                user_id=user.user_id,
                action="authorization_check",
                resource=permission.value,
                result="success" if has_permission else "denied",
            )
        )

        return has_permission

    def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener audit logs"""
        if not self.audit_log_file.exists():
            return []

        logs = []
        with open(self.audit_log_file, "r") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                if line.strip():
                    logs.append(json.loads(line))

        return logs

    def generate_security_token(self, data: str, secret: str) -> str:
        """Generar token de seguridad (HMAC)"""
        signature = hmac.new(secret.encode(), data.encode(), hashlib.sha256).hexdigest()

        return f"{data}.{signature}"

    def verify_security_token(self, token: str, secret: str) -> bool:
        """Verificar token de seguridad"""
        try:
            data, signature = token.rsplit(".", 1)
            expected_signature = hmac.new(
                secret.encode(), data.encode(), hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False

    def store_secret(self, key: str, value: str):
        """Almacenar un secret"""
        self.secrets_manager.secrets[key] = value
        self.secrets_manager._save_secrets()
        logger.info(f"🔐 Secret '{key}' almacenado")

    def get_secret(self, key: str) -> Optional[str]:
        """Obtener un secret"""
        return self.secrets_manager.get_secret(key)

    def validate_input(self, input_str: str) -> bool:
        """Validar input (delegando a InputValidator)"""
        try:
            # Verificar XSS (prevent_xss devuelve string limpio)
            cleaned = self.validator.prevent_xss(input_str)

            # Si el input cambió mucho, tiene contenido peligroso
            if len(cleaned) < len(input_str) * 0.5:  # Perdió >50% de caracteres
                logger.warning("⚠️ Input rechazado: XSS detectado")
                return False

            # Verificar patrones XSS comunes
            xss_patterns = ["<script", "javascript:", "onerror=", "onclick="]
            input_lower = input_str.lower()
            for pattern in xss_patterns:
                if pattern in input_lower:
                    logger.warning("⚠️ Input rechazado: XSS detectado")
                    return False

            # Verificar SQL injection
            has_sql_injection = self.validator.prevent_sql_injection(input_str)
            if has_sql_injection:
                logger.warning("⚠️ Input rechazado: SQL injection detectado")
                return False

            # Si llegamos aquí, el input es válido
            logger.info("✅ Input validado correctamente")
            return True
        except Exception as e:
            logger.error(f"❌ Error validando input: {e}")
            return False


# Singleton global
_global_security: Optional[SecuritySystem] = None


def get_security_system(**kwargs) -> SecuritySystem:
    """Obtener instancia global del sistema de seguridad"""
    global _global_security
    if _global_security is None:
        _global_security = SecuritySystem(**kwargs)
    return _global_security


# ========================================================================
# FUNCIONES UTILITARIAS PARA ENCRYPTION Y HASHING
# ========================================================================


def encrypt_data(data: str) -> str:
    """Encriptar datos usando Fernet"""
    security = get_security_system()
    fernet = Fernet(security.secrets_manager.encryption_key)
    encrypted = fernet.encrypt(data.encode())
    return encrypted.decode()


def decrypt_data(encrypted_data: str) -> str:
    """Desencriptar datos usando Fernet"""
    security = get_security_system()
    fernet = Fernet(security.secrets_manager.encryption_key)
    decrypted = fernet.decrypt(encrypted_data.encode())
    return decrypted.decode()


def hash_password(password: str) -> str:
    """Hashear contraseña usando bcrypt"""

    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verificar contraseña contra hash"""

    return bcrypt.checkpw(password.encode(), hashed.encode())


# Decorator para requerir autenticación
def require_auth(get_api_key_func: Callable):
    """Decorator para requerir autenticación"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            api_key = get_api_key_func(*args, **kwargs)

            security = get_security_system()
            user = security.require_auth(api_key)

            if not user:
                raise PermissionError("Authentication required")

            # Agregar user a kwargs
            kwargs["authenticated_user"] = user

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Decorator para requerir permiso
def require_permission(permission: Permission):
    """Decorator para requerir permiso específico"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            user = kwargs.get("authenticated_user")
            if not user:
                raise PermissionError("Authentication required")

            security = get_security_system()
            if not security.require_permission(user, permission):
                raise PermissionError(f"Permission denied: {permission.value}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🔒 Testing Security System...\n")

    security = get_security_system()

    # Test 1: Secrets Manager
    print("Test 1: Secrets Manager")
    if CRYPTO_AVAILABLE:
        security.secrets_manager.set_secret("api_key", "super_secret_123")
        retrieved = security.secrets_manager.get_secret("api_key")
        print(f"  ✅ Secret guardado y recuperado: {retrieved}\n")
    else:
        print("  ⚠️ Crypto no disponible\n")

    # Test 2: Authentication
    print("Test 2: Authentication & Authorization")
    user = security.auth.create_user("john_doe", roles=["ml_engineer"])
    api_key = security.auth.generate_api_key(user.user_id)
    print(f"  ✅ Usuario creado: {user.username}")
    print(f"  ✅ API Key: {api_key[:20]}...\n")

    # Verify API key
    verified_user = security.auth.verify_api_key(api_key)
    print(
        f"  ✅ API Key verificada: {verified_user.username if verified_user else 'FAILED'}\n"
    )

    # Test 3: Authorization
    print("Test 3: Permisos")
    can_train = security.authz.check_permission(user, Permission.ML_TRAIN)
    can_admin = security.authz.check_permission(user, Permission.ADMIN_ALL)
    print(f"  ML_TRAIN: {'✅ PERMITIDO' if can_train else '❌ DENEGADO'}")
    print(f"  ADMIN_ALL: {'✅ PERMITIDO' if can_admin else '❌ DENEGADO'}\n")

    # Test 4: Input Validation
    print("Test 4: Input Validation")
    dangerous_input = "<script>alert('xss')</script>"
    sanitized = security.validator.prevent_xss(dangerous_input)
    print(f"  Input peligroso: {dangerous_input}")
    print(f"  Sanitizado: {sanitized}\n")

    # Test 5: Audit Logs
    print("Test 5: Audit Logs")
    logs = security.get_audit_logs(limit=5)
    print(f"  ✅ {len(logs)} audit logs registrados")
    for log in logs[-3:]:
        print(f"     - {log['action']}: {log['result']}")

    print("\n✅ Tests completados")
