# Quick Start Reference Card

## 🚀 Get Started in 2 Minutes

### 1. Install Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) - Required
- 8GB+ RAM available

### 2. Clone & Configure

```bash
# Clone repository
git clone <repository-url>
cd "Enterprise AI Document Intelligence  Reasoning Platform"

# Copy environment files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local

# Edit backend/.env - Set your LLM provider:
LLM_PROVIDER=groq
GROQ_API_KEY=your-key-here
```

### 3. Start

**Windows:**
```bash
start-dev.cmd
```

**Linux/Mac:**
```bash
chmod +x start-dev.sh
./start-dev.sh
```

### 4. Access

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### 5. Stop

```bash
docker-compose down
```

---

## 📋 Essential Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f redis
```

### Restart Services

```bash
# Restart single service
docker-compose restart backend

# Restart all
docker-compose restart
```

### Clean Start

```bash
# Stop and remove everything
docker-compose down -v

# Fresh build
docker-compose up -d --build --no-cache
```

---

## 🔧 Common Issues

### Port Already in Use

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

### Build Fails

```bash
# Clean and rebuild
docker system prune -a
docker-compose up -d --build --no-cache
```

### Can't Connect to Backend

1. Check backend running: `curl http://localhost:8000/health`
2. Verify `NEXT_PUBLIC_API_URL=http://localhost:8000` in `frontend/.env.local`
3. Restart: `docker-compose restart frontend backend`

---

## 📚 Documentation

| Document | Location |
|----------|----------|
| Main README | `README.md` |
| Architecture | `docs/ARCHITECTURE.md` |
| Deployment | `docs/DEPLOYMENT.md` |
| Local Development | `docs/LOCAL_DEVELOPMENT.md` |
| Frontend | `docs/FRONTEND.md` |

---

## 🎯 Quick Tests

1. **Health Check:** http://localhost:8000/health
2. **API Docs:** http://localhost:8000/docs
3. **Upload Document:** http://localhost:3000/dashboard
4. **Ask Question:** Create chat session → Upload doc → Ask question

---

## 💡 Tips

- **First build takes 2-3 minutes** (downloading images)
- **Subsequent builds are fast** (Docker cache)
- **View logs** to see what's happening: `docker-compose logs -f`
- **Edit code** while running - hot reload works automatically
- **Don't delete volumes** unless you want to reset data: `docker-compose down` (without `-v`)

---

**Need help?** Check `docs/LOCAL_DEVELOPMENT.md` or open an issue on GitHub.
