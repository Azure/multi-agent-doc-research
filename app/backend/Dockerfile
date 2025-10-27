FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NODE_PATH=/usr/local/lib/node_modules \
    PATH=/usr/local/bin:$PATH

# Node.js와 npm 설치 (YouTube MCP 서버 설치를 위해)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir uv

RUN uv pip install --system -e .

# YouTube MCP 서버 설치 및 확인
RUN npm install -g youtube-data-mcp-server \
    && npm list -g youtube-data-mcp-server \
    && which youtube-data-mcp-server \
    && youtube-data-mcp-server --help || echo "MCP server installed but help command failed"

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
# YouTube MCP 서버 확인\n\
if command -v youtube-data-mcp-server >/dev/null 2>&1; then\n\
    echo "✅ YouTube MCP Server found: $(which youtube-data-mcp-server)"\n\
else\n\
    echo "❌ YouTube MCP Server not found"\n\
    echo "Available executables in /usr/local/bin:"\n\
    ls -la /usr/local/bin/ | grep -i youtube || echo "No youtube executables found"\n\
    \n\
    # 수동 심볼릭 링크 생성 시도\n\
    if [ -f "/usr/local/lib/node_modules/youtube-data-mcp-server/bin/youtube-data-mcp-server" ]; then\n\
        ln -sf /usr/local/lib/node_modules/youtube-data-mcp-server/bin/youtube-data-mcp-server /usr/local/bin/youtube-data-mcp-server\n\
        chmod +x /usr/local/bin/youtube-data-mcp-server\n\
        echo "✅ Manual symlink created"\n\
    else\n\
        echo "❌ MCP server binary not found"\n\
    fi\n\
fi\n\
\n\
# Backend 애플리케이션 실행\n\
echo "Executing: uv run run.py"\n\
exec uv run run.py' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]