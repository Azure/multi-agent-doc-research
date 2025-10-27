FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir uv

RUN uv pip install --system -e .

EXPOSE 7860

# entrypoint.sh 스크립트 생성 및 실행 권한 부여
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# 환경변수 확인 및 로깅\n\
echo "Starting Chainlit application..."\n\
echo "PORT: 7860"\n\
echo "HOST: 0.0.0.0"\n\
\n\
# CHAINLIT_AUTH_SECRET이 설정되지 않은 경우 자동 생성\n\
if [ -z "$CHAINLIT_AUTH_SECRET" ]; then\n\
    export CHAINLIT_AUTH_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")\n\
    echo "Generated CHAINLIT_AUTH_SECRET for this session"\n\
else\n\
    echo "Using provided CHAINLIT_AUTH_SECRET"\n\
fi\n\
\n\
# 백엔드 연결 확인\n\
if [ ! -z "$API_URL" ]; then\n\
    echo "Backend API URL: $API_URL"\n\
fi\n\
\n\
# Socket.IO 세션 복구를 위한 환경변수 설정\n\
export CHAINLIT_SOCKET_IO_TRANSPORTS="polling,websocket"\n\
export CHAINLIT_SOCKET_IO_UPGRADE="true"\n\
export CHAINLIT_SOCKET_IO_REMEMBER_TRANSPORT="false"\n\
\n\
# Chainlit 애플리케이션 실행\n\
echo "Executing: chainlit run src/app_chainlit.py --port 7860 --host 0.0.0.0"\n\
exec chainlit run src/app_chainlit.py --port 7860 --host 0.0.0.0 --headless' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]