# Local Development Guide

## Quick Start for Local Development

This guide will help you run DocuCentric locally using Docker Compose.

---

## Prerequisites

Before starting, ensure you have:

- **Docker Desktop** installed and running
  - Windows: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Mac: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Linux: `sudo apt install docker.io docker-compose-v2`
- **Git** installed
- At least **8GB RAM** available for Docker

---

## Option 1: Docker Compose (Recommended)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd "Enterprise AI Document Intelligence  Reasoning Platform"
```

### Step 2: Configure Environment

**Backend Configuration:**

```bash
# Copy environment template
cp backend/.env.example backend/.env

# Edit .env and set your LLM provider
# Example using Groq (free tier):
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your-key-here

# Or OpenAI:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

**Frontend Configuration:**

```bash
# Copy environment template
cp frontend/.env.example frontend/.env.local

# Default is already set to http://localhost:8000
# No changes needed unless backend runs on different port
```

### Step 3: Start Services

**Windows:**
```bash
# Double-click start-dev.cmd
# Or run in terminal:
start-dev.cmd
```

**Linux/Mac:**
```bash
# Make script executable
chmod +x start-dev.sh

# Run startup script
./start-dev.sh
```

**Or manually:**
```bash
docker-compose up -d --build
```

### Step 4: Access Application

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### Step 5: View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f redis
docker-compose logs -f celery-worker
```

### Step 6: Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

---

## Option 2: Manual Setup (No Docker)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your LLM provider and API key

# Start Redis (required)
# Windows: Download and run Redis from https://github.com/tporadowski/redis/releases
# Linux: sudo systemctl start redis-server
# Mac: brew services start redis

# Start backend
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local

# Start development server
npm run dev
```

### Access

- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

---

## Docker Compose Profiles

The docker-compose.yml includes different profiles for various use cases:

### Development (Default)

Runs frontend in development mode with hot reload:

```bash
docker-compose up -d
```

### Full Stack

Includes Celery worker for background processing:

```bash
docker-compose --profile full up -d
```

### Production

Runs production-optimized setup with PostgreSQL:

```bash
docker-compose --profile production up -d
```

---

## Troubleshooting

### Issue: Docker Build Fails

**Solution:**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose up -d --build --no-cache
```

### Issue: Port Already in Use

**Solution:**
```bash
# Find process using port 8000
# Windows:
netstat -ano | findstr :8000
# Linux/Mac:
lsof -i :8000

# Kill process
# Windows:
taskkill /PID <PID> /F
# Linux/Mac:
kill -9 <PID>
```

### Issue: Redis Connection Refused

**Solution:**
```bash
# Check Redis is running
docker-compose ps redis

# Restart Redis
docker-compose restart redis

# Check logs
docker-compose logs redis
```

### Issue: Backend Can't Connect to Redis

**Solution:**

The backend uses `redis://redis:6379/0` inside Docker. If running backend manually outside Docker, change to:

```env
# In backend/.env
REDIS_URL=redis://localhost:6379/0
```

### Issue: Frontend Shows API Errors

**Solutions:**

1. **Check backend is running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify frontend .env.local:**
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Check CORS in backend:**
   Backend has CORS enabled with `allow_origins=["*"]` for development.

4. **Restart frontend container:**
   ```bash
   docker-compose restart frontend
   ```

### Issue: LLM API Errors

**Solution:**

1. **Verify API key is set:**
   ```bash
   docker-compose exec backend env | grep GROQ_API_KEY
   ```

2. **Test API connection:**
   ```bash
   # Groq
   curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models
   
   # OpenAI
   curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
   ```

3. **Update .env with correct key**

4. **Restart backend:**
   ```bash
   docker-compose restart backend
   ```

---

## Development Workflow

### Making Changes to Backend

1. Edit files in `backend/` directory
2. Uvicorn auto-reloads on file changes
3. If using Docker, changes reflect immediately:
   ```bash
   docker-compose logs -f backend
   ```

### Making Changes to Frontend

1. Edit files in `frontend/` directory
2. Next.js hot reload updates automatically
3. If using Docker, changes reflect immediately:
   ```bash
   docker-compose logs -f frontend
   ```

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests (if added)
cd frontend
npm test
```

---

## Environment Variables Reference

### Backend (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai/groq/gemini/ollama/openrouter) | `groq` |
| `GROQ_API_KEY` | Groq API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `GEMINI_API_KEY` | Gemini API key | - |
| `DATABASE_URL` | Database connection string | `sqlite:///./data/app.db` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `TAVILY_API_KEY` | Tavily search API key (optional) | - |
| `DEBUG` | Enable debug mode | `false` |
| `LOG_LEVEL` | Log level (DEBUG/INFO/WARNING/ERROR) | `INFO` |

### Frontend (.env.local)

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |
| `NODE_ENV` | Node environment | `development` |

---

## Performance Tips

### 1. Increase Docker Memory

**Docker Desktop:**
- Settings → Resources → Memory: Increase to 8GB+

### 2. Use Docker Volumes

Persistent volumes speed up rebuilds:

```bash
# Don't remove volumes on down
docker-compose down  # NOT docker-compose down -v
```

### 3. Build Cache

Speed up subsequent builds:

```bash
# First build (slow)
docker-compose up -d --build

# Subsequent builds (fast, uses cache)
docker-compose up -d
```

### 4. Selective Service Start

Only start what you need:

```bash
# Backend only
docker-compose up -d backend redis

# Frontend only (backend must be running)
docker-compose up -d frontend
```

---

## Cleaning Up

### Remove All Docker Data

```bash
# Stop services
docker-compose down

# Remove all containers, networks, volumes
docker-compose down -v

# Clean Docker system
docker system prune -a --volumes
```

### Reset Database

```bash
# Stop services
docker-compose down

# Remove database volume
docker volume rm enterprise-ai-document-intelligence-reasoning-platform_postgres_data

# Restart
docker-compose up -d
```

### Reset Vector Store

```bash
# Via API
curl -X POST http://localhost:8000/system/reset/vector-store

# Or manually
docker-compose down -v
docker-compose up -d
```

---

## Next Steps

1. **Upload a document:** http://localhost:3000/dashboard
2. **Create a chat session**
3. **Ask questions about your document**
4. **Check API docs:** http://localhost:8000/docs

---

**Need help?** Check [docs/DEPLOYMENT.md](DEPLOYMENT.md) or open an issue on GitHub.
