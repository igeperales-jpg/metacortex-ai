#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis Connection Pool
=====================

Módulo para gestionar un pool de conexiones a Redis.
"""
from __future__ import annotations

import logging
import redis
from typing import Optional

logger = logging.getLogger(__name__)

_connection_pool: Optional[redis.BlockingConnectionPool] = None

def get_redis_client(
    host: str = "localhost", 
    port: int = 6379, 
    db: int = 0,
    max_connections: int = 20,
    decode_responses: bool = True
) -> redis.Redis:
    """
    Retorna una instancia de cliente Redis usando un pool de conexiones singleton.
    
    Los parámetros de conexión solo se usan la primera vez que se crea el pool.
    """
    global _connection_pool
    if _connection_pool is None:
        logger.info(f"Creando nuevo pool de conexiones Redis para {host}:{port}")
        try:
            _connection_pool = redis.BlockingConnectionPool(
                host=host,
                port=port,
                db=db,
                max_connections=max_connections,
                timeout=10,
                decode_responses=decode_responses,
            )
        except Exception as e:
            logger.error(f"No se pudo crear el pool de conexiones de Redis: {e}")
            raise

    return redis.Redis(connection_pool=_connection_pool)
