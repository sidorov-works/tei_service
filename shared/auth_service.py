# shared/auth-service.py

"""
JWT-based аутентификация для межсервисного взаимодействия.
Использует HS256 для подписи токенов с коротким сроком жизни.

В данном модуле - функции, необходимые только серверу
"""

from jose import jwt, JWTError
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List
import logging

from shared.config import config

logger = logging.getLogger(__name__)

# Схема аутентификации - ожидаем токен в заголовке Authorization: Bearer <token>
security = HTTPBearer(auto_error=False)

async def verify_jwt_token(
    secret_key: str = config.INTERNAL_API_SECRET,
    allowed_algorithms: Optional[List[str]] = config.ALLOWED_JWT_ALGORITHMS,
    # Список разрешенных JWT алгоритмов
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
            secret_key,
            # В аргумент algorithms передается список РАЗРЕШЕННЫХ алгоритмов, 
            # а не конкретное указание на алгоритм, 
            # котрый был использован для создания подписи (токена).
            # А конкретный алгоритм, который будет использован для декодирования, 
            # якобы, уже прописан в заголовке токена (
            # token header - первая из трех частей строки токена)
            algorithms=allowed_algorithms
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