@echo off
echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "FastAPI Backend" cmd /k "python -m uvicorn backend.main:app --reload --port 8000"

echo Starting React Vite Frontend on http://localhost:5173 ...
start "React Frontend" cmd /k "cd frontend && npm run dev"

echo Both servers launched. Open http://localhost:5173 in your browser.
