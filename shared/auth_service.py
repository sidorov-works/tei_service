# shared/auth_service.py

"""
JWT-based аутентификация для межсервисного взаимодействия.
Использует HS256 для подписи токенов с коротким сроком жизни.

В данном модуле - функции, необходимые только серверу
"""

from jose import jwt, JWTError
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List
from shared.utils.logger import logger
from shared.config import config

# Схема аутентификации - ожидаем токен в заголовке Authorization: Bearer <token>
security = HTTPBearer(auto_error=False)

async def verify_jwt_token(
    # Явное добавление параметра request: Request в зависимость - решение проблемы 
    # "конфликта за тело запроса": FastAPI видит, что request уже используется 
    # в зависимости. Это говорит FastAPI: "не блокируй парсинг тела для основного эндпоинта, 
    # потому что request уже обрабатывается на уровне зависимости"
    request: Request,
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
            config.INTERNAL_API_SECRET,
            # В аргумент algorithms передается список РАЗРЕШЕННЫХ алгоритмов, 
            # а не конкретное указание на алгоритм, 
            # котрый был использован для создания подписи (токена).
            # А конкретный алгоритм, который будет использован для декодирования, 
            # якобы, уже прописан в заголовке токена (
            # token header - первая из трех частей строки токена)
            algorithms=config.ALLOWED_JWT_ALGORITHMS
        )
        
        # Дополнительная проверка: токен должен быть выпущен для внутренних сервисов
        # Можно проверять конкретные значения, если нужно
        issuer = payload.get("iss")
        if not issuer:
            raise HTTPException(
                status_code=403,
                detail="Invalid token: missing issuer"
            )
        
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
async def require_jwt_auth(
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

async def require_header_secret(
    # Явное добавление параметра request: Request в зависимость - решение проблемы 
    # "конфликта за тело запроса": FastAPI видит, что request уже используется 
    # в зависимости. Это говорит FastAPI: "не блокируй парсинг тела для основного эндпоинта, 
    # потому что request уже обрабатывается на уровне зависимости"
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Зависимость FastAPI для проверки секретного заголовка.
    
    Использование в эндпоинтах:
    @app.post("/encode")
    async def encode(..., _ = Depends(require_header_secret)):
    
    Raises:
        HTTPException 403: Если секретный заголовок отсутствует или не совпадает
    """
    # Проверяем, что токен вообще предоставлен
    if not credentials:
        logger.warning("No authorization credentials provided")
        raise HTTPException(
            status_code=403,
            detail="Missing API secret key"
        )
    
    API_SECRET = credentials.credentials
    
    if API_SECRET != config.INTERNAL_API_SECRET:
        logger.warning(f"Invalid API secret")
        raise HTTPException(status_code=403, detail="Forbidden")