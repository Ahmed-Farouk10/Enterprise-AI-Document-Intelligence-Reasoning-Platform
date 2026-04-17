@echo off
REM ========================================
REM DocuCentric Local Development Startup
REM Windows Batch Script
REM ========================================

echo.
echo ========================================
echo DocuCentric - Local Development Setup
echo ========================================
echo.

REM Check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Docker Compose is not installed.
    pause
    exit /b 1
)

echo [OK] Docker found
echo [OK] Docker Compose found
echo.

REM Check if .env files exist
if not exist backend\.env (
    echo WARNING: backend\.env not found. Creating from template...
    if exist backend\.env.example (
        copy backend\.env.example backend\.env
        echo.
        echo Please edit backend\.env with your configuration
        echo Especially set:
        echo   - LLM_PROVIDER=groq
        echo   - GROQ_API_KEY=your-key-here
        echo.
        pause
    ) else (
        echo ERROR: backend\.env.example not found
        pause
        exit /b 1
    )
)

if not exist frontend\.env.local (
    echo WARNING: frontend\.env.local not found. Creating from template...
    if exist frontend\.env.example (
        copy frontend\.env.example frontend\.env.local
        echo [OK] frontend\.env.local created
    )
)

echo.
echo ========================================
echo Starting DocuCentric Services...
echo ========================================
echo.

REM Start services
echo Building and starting services...
docker-compose up -d --build

echo.
echo ========================================
echo Services Started!
echo ========================================
echo.
echo Service Status:
docker-compose ps
echo.
echo Access Points:
echo   - Frontend:    http://localhost:3000
echo   - Backend API: http://localhost:8000
echo   - API Docs:    http://localhost:8000/docs
echo   - Health Check: http://localhost:8000/health
echo.
echo View Logs:
echo   docker-compose logs -f backend
echo   docker-compose logs -f frontend
echo   docker-compose logs -f celery-worker
echo.
echo Stop Services:
echo   docker-compose down
echo.
echo Tip: Run 'docker-compose logs -f' to view all logs
echo.
pause
