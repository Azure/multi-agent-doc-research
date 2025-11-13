FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NODE_PATH=/usr/local/lib/node_modules \
    PATH=/usr/local/bin:$PATH

# 프로젝트 루트의 필수 파일들 복사
COPY pyproject.toml README.md ./
# backend 코드 복사
COPY app/backend/ .

RUN pip install --no-cache-dir uv

# ⭐ agent-framework 관련 패키지 완전 제거 후 재설치
RUN uv pip uninstall --system agent-framework agent-framework-core agent-framework-azure-ai agent-framework-a2a agent-framework-copilotstudio agent-framework-devui agent-framework-mem0 agent-framework-redis || true


RUN uv pip install --system -e .


# PATH 환경변수에 npm global bin 추가
RUN echo "export PATH=/usr/local/bin:\$PATH" >> /etc/bash.bashrc \
    && echo "export NODE_PATH=/usr/local/lib/node_modules" >> /etc/bash.bashrc

EXPOSE 8000

# entrypoint 스크립트 생성
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# PATH 환경변수 설정\n\
export PATH=/usr/local/bin:$PATH\n\
export NODE_PATH=/usr/local/lib/node_modules\n\
\n\
echo "Starting Backend application..."\n\
echo "PATH: $PATH"\n\
echo "NODE_PATH: $NODE_PATH"\n\
\n\

\n\
# Backend 애플리케이션 실행\n\
echo "Executing: uv run run.py"\n\
exec uv run run.py' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]