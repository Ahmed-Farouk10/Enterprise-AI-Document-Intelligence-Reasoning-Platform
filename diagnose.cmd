@echo off
REM DocuCentric Diagnostic Script
REM Tests all critical endpoints

echo ========================================
echo DocuCentric System Diagnostic
echo ========================================
echo.

echo 1. Testing Backend Health...
curl -s http://localhost:8000/health | findstr "healthy"
if %errorlevel% equ 0 (
    echo [OK] Backend is healthy
) else (
    echo [FAIL] Backend not responding
)
echo.

echo 2. Testing Chat Sessions...
curl -s http://localhost:8000/api/chat/sessions | findstr "id"
if %errorlevel% equ 0 (
    echo [OK] Chat sessions endpoint working
) else (
    echo [FAIL] Chat sessions endpoint failed
)
echo.

echo 3. Testing Documents...
curl -s http://localhost:8000/api/documents?page=1^&page_size=20 | findstr "items"
if %errorlevel% equ 0 (
    echo [OK] Documents endpoint working
) else (
    echo [FAIL] Documents endpoint failed
)
echo.

echo 4. Testing Knowledge Graph...
curl -s http://localhost:8000/api/graph/stats | findstr "total"
if %errorlevel% equ 0 (
    echo [OK] Knowledge graph endpoint working
) else (
    echo [FAIL] Knowledge graph endpoint failed
)
echo.

echo 5. Testing Frontend...
curl -s http://localhost:3000 | findstr "DOCTYPE"
if %errorlevel% equ 0 (
    echo [OK] Frontend is running
) else (
    echo [FAIL] Frontend not responding
)
echo.

echo ========================================
echo Diagnostic Complete
echo ========================================
echo.
echo If all tests passed:
echo   - Backend API is working
echo   - Frontend should work
echo   - Refresh browser (Ctrl+F5)
echo.
pause
