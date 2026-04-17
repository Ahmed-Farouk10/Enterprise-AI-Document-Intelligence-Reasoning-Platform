# Troubleshooting Guide

## Common Issues and Solutions

### 1. Backend Won't Start / ERR_EMPTY_RESPONSE

**Symptom:** Frontend shows `net::ERR_EMPTY_RESPONSE` errors

**Cause:** Windows environment variable `LLM_PROVIDER` is set to an invalid value

**Solution:**
```cmd
# Check current value
echo %LLM_PROVIDER%

# If it says "huggingface" or anything other than groq/openai/gemini/ollama/openrouter:
setx LLM_PROVIDER "groq"
setx GROQ_API_KEY "your-api-key"

# Restart Docker
docker-compose down
docker-compose up -d
```

### 2. Frontend Can't Connect to Backend

**Symptom:** `GET http://localhost:8000/api/... net::ERR_EMPTY_RESPONSE`

**Solutions:**

1. **Check backend is running:**
   ```bash
   docker-compose ps backend
   # Should show "Up" and "(healthy)"
   ```

2. **Check backend logs:**
   ```bash
   docker-compose logs --tail=50 backend
   ```

3. **Verify backend responds:**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Restart backend:**
   ```bash
   docker-compose restart backend
   ```

### 3. Node.js Version Error

**Symptom:** `You are using Node.js 18.x. For Next.js, Node.js version ">=20.9.0" is required.`

**Solution:** Already fixed in Dockerfile - uses `node:20-alpine`. Rebuild:
```bash
docker-compose up -d --build frontend
```

### 4. Knowledge Graph Not Working

**Symptom:** "Failed to fetch graph stats"

**Solution:** The `/api/graph/stats` and `/api/graph/data` endpoints have been added. Restart backend:
```bash
docker-compose restart backend
```

### 5. Chat Not Working

**Symptom:** Messages don't send or no response

**Checklist:**
1. ✅ Backend is running: `docker-compose ps`
2. ✅ Redis is running: `docker-compose ps redis`
3. ✅ LLM provider configured: `echo %LLM_PROVIDER%`
4. ✅ API key set (if using Groq/OpenAI/Gemini)

**Test backend directly:**
```bash
# Health check
curl http://localhost:8000/health

# Create session
curl -X POST http://localhost:8000/api/chat/sessions -H "Content-Type: application/json" -d "{\"title\":\"Test\"}"

# Send message
curl -X POST http://localhost:8000/api/chat/sessions/{session-id}/messages -H "Content-Type: application/json" -d "{\"content\":\"Hello\"}"
```

### 6. Docker Build Takes Forever

**Cause:** Downloading large packages (torch, transformers, etc.)

**Solution:** Use the simplified requirements.txt (already updated). If still slow:
```bash
# Clean Docker cache
docker system prune -a

# Rebuild
docker-compose up -d --build --no-cache
```

### 7. Database Errors

**Symptom:** Database connection failed

**Solution:**
```bash
# Reset database
docker-compose down -v
docker-compose up -d
```

### 8. Redis Connection Refused

**Symptom:** Redis connection errors in backend logs

**Solution:**
```bash
# Restart Redis
docker-compose restart redis

# Check Redis is healthy
docker-compose ps redis
```

---

## Quick Diagnostic Commands

```bash
# Check all services
docker-compose ps

# Check backend health
curl http://localhost:8000/health

# Check system info
curl http://localhost:8000/system/info

# View backend logs
docker-compose logs -f backend

# View frontend logs
docker-compose logs -f frontend

# Test API endpoint
curl http://localhost:8000/api/chat/sessions

# Check environment variables
docker-compose exec backend env | findstr LLM_PROVIDER
```

---

## Environment Variables

### Windows System Variables

Check if these are set (they override .env files):
```cmd
echo %LLM_PROVIDER%
echo %GROQ_API_KEY%
echo %OPENAI_API_KEY%
```

To change:
```cmd
setx LLM_PROVIDER "groq"
setx GROQ_API_KEY "your-key"
# Restart terminal/Docker after changes
```

### Docker Environment

Check what the container sees:
```bash
docker-compose exec backend env | findstr LLM
docker-compose exec frontend env | findstr API
```

---

## Clean Slate

If nothing works, start completely fresh:

```bash
# Stop everything
docker-compose down -v

# Clean Docker
docker system prune -af --volumes

# Set environment variables
setx LLM_PROVIDER "groq"
setx GROQ_API_KEY "your-key"

# Restart
docker-compose up -d --build

# Monitor
docker-compose logs -f
```

---

## Need More Help?

1. Check logs: `docker-compose logs -f`
2. Test API directly: `curl http://localhost:8000/health`
3. Review this guide
4. Open issue on GitHub with:
   - `docker-compose ps` output
   - `docker-compose logs --tail=100 backend` output
   - Browser console errors
