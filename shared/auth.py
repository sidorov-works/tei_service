# shared/auth.py

"""
JWT-based аутентификация для межсервисного взаимодействия.
Использует HS256 для подписи токенов с коротким сроком жизни.
"""

from jose import jwt, JWTError
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
from typing import Optional, Dict, Any
import logging

from shared.config import config

logger = logging.getLogger(__name__)

# Настройки JWT
SECRET_KEY = config.INTERNAL_API_SECRET
ALGORITHM = "HS256"
# Токен живет 30 секунд - достаточно для межсервисного запроса
# Это защищает от replay-атак (повторного использования перехваченного токена)
TOKEN_EXPIRE_SECONDS = 30

# Схема аутентификации - ожидаем токен в заголовке Authorization: Bearer <token>
security = HTTPBearer(auto_error=False)


def create_service_token(
    service_name: str = "unknown",
    extra_payload: Optional[Dict[str, Any]] = None
) -> str:
    """
    Создает JWT токен для межсервисной аутентификации.
    
    Args:
        service_name: Идентификатор сервиса-отправителя
        extra_payload: Дополнительные данные для включения в токен
    
    Returns:
        str: JWT токен
    
    Пример payload:
    ```
    {
        "iss": "encoder-client",             # отправитель
        "iat": 1700000000,                   # время выпуска
        "exp": 1700000030,                   # истекает через 30 сек
        "service": "encoder-client",         # сервис
        "request_id": "550e8400-e29b-41d4"   # можно добавить для trace
    }
    ```
    """
    current_time = int(time.time())
    
    # Базовый payload
    payload = {
        "iss": service_name,                          # кто выпустил токен
        "iat": current_time,                          # когда выпущен
        "exp": current_time + TOKEN_EXPIRE_SECONDS,   # когда истекает
        "service": service_name,                      # имя сервиса
    }
    
    # Добавляем дополнительные данные, если есть
    if extra_payload:
        payload.update(extra_payload)
    
    # Создаем подписанный токен
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    
    logger.debug(f"Created JWT token for {service_name}, expires in {TOKEN_EXPIRE_SECONDS}s")
    return token


async def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Зависимость FastAPI для проверки JWT токена.
    
    Использование в эндпоинтах:
    @app.post("/encode")
    async def encode(..., token_data: dict = Depends(verify_jwt_token)):
        service_name = token_data.get("service")
    
    Returns:
        Dict[str, Any]: Декодированный payload токена
    
    Raises:
        HTTPException 403: Если токен отсутствует, недействителен или истек
    """
    # Проверяем, что токен вообще предоставлен
    if not credentials:
        logger.warning("No authorization credentials provided")
        raise HTTPException(
            status_code=403,
            detail="Missing authentication token"
        )
    
    token = credentials.credentials
    
    try:
        # Декодируем и проверяем подпись
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        
        # Дополнительная проверка: токен должен быть выпущен для внутренних сервисов
        # Можно проверять конкретные значения, если нужно
        issuer = payload.get("iss")
        if not issuer:
            logger.warning("Token missing issuer")
            raise HTTPException(
                status_code=403,
                detail="Invalid token: missing issuer"
            )
        
        # Логируем успешную аутентификацию (для отладки)
        logger.debug(f"JWT token verified for service: {issuer}")
        
        return payload
        
    except JWTError as e:
        # Ошибка декодирования или недействительная подпись
        logger.warning(f"JWT verification failed: {str(e)}")
        raise HTTPException(
            status_code=403,
            detail=f"Invalid token: {str(e)}"
        )


# Упрощенная версия для эндпоинтов, 
# которым не требуется конкретная информация из payload, 
# а только нужен сам факт аутентификации
async def require_auth(
    _: Dict[str, Any] = Depends(verify_jwt_token)
) -> None:
    """
    Упрощенная зависимость, когда нужно только проверить наличие токена,
    но payload не требуется.
    
    Использование:
    @app.post("/encode")
    def encode(..., _: None = Depends(require_auth)):
        ...
    """
    pass