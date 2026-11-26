#!/usr/bin/env bash
set -e

echo "Starting FastAPI Backend on http://127.0.0.1:8000 ..."
python -m uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

echo "Starting React Vite Frontend on http://localhost:5173 ..."
(cd frontend && npm run dev) &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" EXIT
wait
