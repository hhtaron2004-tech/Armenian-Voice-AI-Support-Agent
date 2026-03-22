#!/bin/bash

echo "🚀 Starting Armenian Voice AI..."

# 1. Start LiveKit server
sudo docker run --rm -d \
  -p 7880:7880 \
  -p 7881:7881 \
  -p 7882:7882/udp \
  livekit/livekit-server:latest \
  --dev --bind 0.0.0.0
echo "✅ LiveKit server started"

sleep 2

# 2. Start HTTP server for browser client
pkill -f "http.server 8080" 2>/dev/null
python -m http.server 8080 --directory frontend &>/dev/null &
echo "✅ HTTP server started"

# 3. Start agent
PYTHONPATH=. python src/main.py dev &
AGENT_PID=$!
echo "✅ Agent started"

sleep 8

# 4. Generate token
echo ""
echo "🔑 Token:"
python -c "
import jwt, time
payload = {
    'iss': 'devkey',
    'sub': 'user1',
    'iat': int(time.time()),
    'exp': int(time.time()) + 3600,
    'video': {'room': 'test', 'roomJoin': True, 'canPublish': True, 'canSubscribe': True}
}
token = jwt.encode(payload, 'secret', algorithm='HS256')
print(token)
"

echo ""
echo "🌐 Open browser: http://localhost:8080/index.html"
echo "   Connect with the token above"
echo ""
read -p "Press ENTER after you connect in browser..."

# 5. Dispatch agent
python -c "
import asyncio
from livekit.api import LiveKitAPI
from livekit.api.agent_dispatch_service import CreateAgentDispatchRequest

async def main():
    api = LiveKitAPI('http://localhost:7880', 'devkey', 'secret')
    await api.agent_dispatch.create_dispatch(
        CreateAgentDispatchRequest(room='test', agent_name='')
    )
    print('✅ Agent dispatched! Start speaking.')
    await api.aclose()

asyncio.run(main())
"

wait $AGENT_PID