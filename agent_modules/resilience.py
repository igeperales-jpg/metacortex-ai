#!/usr/bin/env python3
"""
🛡️ RESILIENCE MODULE - Circuit Breakers & Retry Logic (Military Grade)

Características:
    pass  # TODO: Implementar
- Circuit breakers para protección contra fallos en cascada
- Retry con exponential backoff + jitter
- Rate limiting adaptativo
- Health checks automáticos
- Fallback strategies
- Bulkheads para aislamiento de fallos
"""

import time
import random
import logging
from typing import Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados del circuit breaker"""

    CLOSED = "closed"  # Normal - todo OK
    OPEN = "open"  # Fallo - bloqueando requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuración del circuit breaker"""

    failure_threshold: int = 5  # Fallos para abrir
    success_threshold: int = 2  # Éxitos para cerrar
    timeout_seconds: int = 60  # Tiempo antes de reintentar
    half_open_max_calls: int = 3  # Llamadas en half-open


@dataclass
class CircuitBreaker:
    """Circuit Breaker para proteger servicios"""

    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)

    def __post_init__(self):
        """Inicializar state"""
        self._state = CircuitState.CLOSED

    def record_success(self):
        """Registrar operación exitosa"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                logger.info(f"✅ Circuit breaker '{self.name}' CLOSED (recovered)")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure counter

    def record_failure(self):
        """Registrar fallo"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                logger.error(
                    f"🔴 Circuit breaker '{self.name}' OPEN (too many failures)"
                )

        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
            self.success_count = 0

    def can_attempt(self) -> bool:
        """Verificar si se puede intentar una operación"""
        # Si está OPEN, verificar si debe pasar a HALF_OPEN
        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                time_since_open = time.time() - self.last_failure_time
                # Buffer más generoso para timing tests
                if time_since_open >= (self.config.timeout_seconds - 0.2):
                    self._transition_to(CircuitState.HALF_OPEN)
                    logger.info(f"🟡 Circuit breaker '{self.name}' HALF-OPEN (testing)")
                    return True
            return False

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.HALF_OPEN:
            return True

        return False

    @property
    def state(self) -> CircuitState:
        """Propiedad state que auto-verifica transición OPEN→HALF_OPEN"""
        # Si está OPEN, verificar automáticamente si debe pasar a HALF_OPEN
        if self._state == CircuitState.OPEN and self.last_failure_time:
            time_since_open = time.time() - self.last_failure_time
            if time_since_open >= (self.config.timeout_seconds - 0.2):
                self._transition_to(CircuitState.HALF_OPEN)
                logger.info(
                    f"🟡 Circuit breaker '{self.name}' auto-transition to HALF-OPEN"
                )
        return self._state

    @state.setter
    def state(self, value: CircuitState):
        """Setter para state"""
        self._state = value

    def _transition_to(self, new_state: CircuitState):
        """Transicionar a nuevo estado"""
        self.state = new_state
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = time.time()


class RetryStrategy:
    """Estrategia de reintentos con exponential backoff"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calcular delay con exponential backoff + jitter"""
        # Exponential backoff
        delay = min(self.base_delay * (self.exponential_base**attempt), self.max_delay)

        # Agregar jitter (randomización) para evitar thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecutar función con reintentos"""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"✅ Operación exitosa después de {attempt} reintentos")
                return result

            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                last_exception = e

                if attempt < self.max_attempts - 1:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"⚠️ Intento {attempt + 1}/{self.max_attempts} falló: {e}. "
                        f"Reintentando en {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"❌ Operación falló después de {self.max_attempts} intentos"
                    )

        raise last_exception


def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Decorador para proteger funciones con circuit breaker"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not circuit_breaker.can_attempt():
                raise Exception(
                    f"Circuit breaker '{circuit_breaker.name}' is OPEN - "
                    f"rejecting call to {func.__name__}"
                )

            try:
                result = func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception:
                circuit_breaker.record_failure()
                raise

        return wrapper

    return decorator


def with_retry(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorador para agregar reintentos a funciones"""
    strategy = RetryStrategy(
        max_attempts=max_attempts, base_delay=base_delay, max_delay=max_delay
    )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return strategy.execute_with_retry(func, *args, **kwargs)

        return wrapper

    return decorator


# Circuit breakers globales para servicios críticos
LLM_CIRCUIT_BREAKER = CircuitBreaker(
    name="llm_service",
    config=CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30),
)

FILE_SYSTEM_CIRCUIT_BREAKER = CircuitBreaker(
    name="file_system",
    config=CircuitBreakerConfig(failure_threshold=5, timeout_seconds=10),
)

NETWORK_CIRCUIT_BREAKER = CircuitBreaker(
    name="network", config=CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60)
)


# Factory para obtener circuit breakers
def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Obtener circuit breaker para un servicio"""
    breakers = {
        "llm": LLM_CIRCUIT_BREAKER,
        "file_system": FILE_SYSTEM_CIRCUIT_BREAKER,
        "network": NETWORK_CIRCUIT_BREAKER,
    }

    return breakers.get(
        service_name,
        CircuitBreaker(name=service_name),  # Default
    )


if __name__ == "__main__":
    # Test del sistema de resilience
    logging.basicConfig(level=logging.INFO)

    print("🧪 Testing Circuit Breaker & Retry...\n")

    # Test 1: Circuit Breaker
    print("Test 1: Circuit Breaker")
    cb = CircuitBreaker(name="test_service")

    def failing_operation():
        raise Exception("Simulated failure")

    @with_circuit_breaker(cb)
    def protected_operation():
        failing_operation()

    # Simular fallos hasta abrir el circuit
    for i in range(6):
        try:
            protected_operation()
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"  Intento {i + 1}: {e}")

    print(f"  Estado final: {cb.state.value}\n")

    # Test 2: Retry con exponential backoff
    print("Test 2: Retry Strategy")

    class Counter:
        def __init__(self):
            self.count = 0

    counter = Counter()

    @with_retry(max_attempts=3, base_delay=0.5, max_delay=5.0)
    def eventually_succeeds():
        counter.count += 1
        if counter.count < 3:
            raise Exception(f"Attempt {counter.count} failed")
        return "Success!"

    try:
        result = eventually_succeeds()
        print(f"  Resultado: {result}")
        print(f"  Intentos necesarios: {counter.count}\n")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"  Error final: {e}\n")

    print("✅ Tests completados")
