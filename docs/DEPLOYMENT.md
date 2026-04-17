# Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Prerequisites

### System Requirements

- **CPU:** 4+ cores (8+ recommended for production)
- **RAM:** 8GB minimum (16GB+ recommended)
- **Storage:** 20GB+ (depends on document volume)
- **Network:** Stable internet connection for LLM API calls

### Software Requirements

- Python 3.11+
- Redis 7+
- PostgreSQL 16+ (optional for production)
- Docker & Docker Compose (for containerized deployment)

---

## Local Development

### 1. Clone Repository

```bash
git clone <repository-url>
cd "Enterprise AI Document Intelligence  Reasoning Platform"
```

### 2. Setup Backend

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

# Create environment file
cp .env.example .env

# Edit .env with your configuration
# At minimum, set:
# - LLM_PROVIDER=groq
# - GROQ_API_KEY=your-key
```

### 3. Start Redis

**Windows:**
```bash
# Download Redis for Windows
# Or use WSL2:
wsl redis-server
```

**Linux/Mac:**
```bash
redis-server
```

### 4. Run Backend

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Access API

- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health
- **System Info:** http://localhost:8000/system/info

---

## Docker Deployment

### Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

### Production

1. **Configure environment:**

```bash
# Create production .env
cat > backend/.env << EOF
LLM_PROVIDER=groq
GROQ_API_KEY=your-production-key
DATABASE_URL=postgresql://user:password@postgres:5432/docucentric
REDIS_URL=redis://redis:6379/0
ENVIRONMENT=production
DEBUG=false
EOF
```

2. **Deploy:**

```bash
# Build and start
docker-compose up -d --build

# Verify services
docker-compose ps

# Check logs
docker-compose logs -f backend
```

3. **Scale workers:**

```bash
# Scale Celery workers for background processing
docker-compose up -d --scale celery-worker=4
```

---

## Production Deployment

### Architecture

```
                    ┌──────────┐
                    │  Nginx   │  ← Reverse proxy, SSL termination
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
        ┌─────▼────┐ ┌──▼──┐ ┌────▼────┐
        │ Backend  │ │Backend│ │ Backend │  ← Multiple instances
        │ Instance │ │ ...  │ │ Instance│
        └─────┬────┘ └──┬──┘ └────┬────┘
              │          │          │
              └──────────┼──────────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
        ┌─────▼────┐ ┌──▼──┐ ┌────▼────┐
        │PostgreSQL│ │Redis│ │LanceDB  │
        │          │ │     │ │(local)  │
        └──────────┘ └─────┘ └─────────┘
```

### 1. Server Setup

**Ubuntu 22.04 Example:**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y docker.io docker-compose-v2 nginx certbot python3-certbot-nginx

# Start Docker
sudo systemctl enable docker
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER
```

### 2. Configure PostgreSQL

```bash
# Create docker network
docker network create docucentric-network

# Start PostgreSQL
docker run -d \
  --name docucentric-postgres \
  --network docucentric-network \
  -e POSTGRES_DB=docucentric \
  -e POSTGRES_USER=docucentric \
  -e POSTGRES_PASSWORD=your-secure-password \
  -v /data/postgres:/var/lib/postgresql/data \
  postgres:16-alpine

# Verify
docker exec docucentric-postgres psql -U docucentric -c "SELECT version();"
```

### 3. Configure Redis

```bash
# Start Redis with persistence
docker run -d \
  --name docucentric-redis \
  --network docucentric-network \
  -v /data/redis:/data \
  redis:7-alpine \
  redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru

# Verify
docker exec docucentric-redis redis-cli ping
```

### 4. Deploy Backend

```bash
# Create .env file
cat > backend/.env << EOF
LLM_PROVIDER=groq
GROQ_API_KEY=your-production-key
DATABASE_URL=postgresql://docucentric:your-secure-password@postgres:5432/docucentric
REDIS_URL=redis://redis:6379/0
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
EOF

# Deploy with Docker Compose
docker-compose -f docker-compose.yml up -d --build

# Verify deployment
curl http://localhost:8000/health
```

### 5. Configure Nginx

```nginx
# /etc/nginx/sites-available/docucentric
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;  # Frontend
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://localhost:8000;  # Backend
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # SSE support for streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    location /docs {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/docucentric /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 6. SSL Certificate (Let's Encrypt)

```bash
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 0 * * * certbot renew --quiet
```

---

## Cloud Deployment

### AWS Deployment

**Using ECS:**

```yaml
# docker-compose.ecs.yml
version: '3.8'
services:
  backend:
    image: your-ecr-repo/docucentric-backend:latest
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
```

**Deploy:**

```bash
# Build and push to ECR
docker build -t docucentric-backend backend/
docker tag docucentric-backend:latest your-ecr-repo/docucentric-backend:latest
docker push your-ecr-repo/docucentric-backend:latest

# Deploy to ECS
aws ecs update-service --cluster docucentric --service backend --force-new-deployment
```

### Azure Deployment

**Using Azure Container Apps:**

```bash
# Create resource group
az group create --name docucentric --location eastus

# Create Container App environment
az containerapp env create --name docucentric-env --resource-group docucentric

# Deploy backend
az containerapp create \
  --name docucentric-backend \
  --resource-group docucentric \
  --environment docucentric-env \
  --image your-acr.azurecr.io/docucentric-backend:latest \
  --target-port 8000 \
  --ingress external \
  --env-vars \
    LLM_PROVIDER=groq \
    GROQ_API_KEY=secretref:grok-key \
    DATABASE_URL=secretref:db-url \
    REDIS_URL=secretref:redis-url
```

### GCP Deployment

**Using Cloud Run:**

```bash
# Build container
docker build -t gcr.io/your-project/docucentric-backend backend/

# Push to Google Container Registry
docker push gcr.io/your-project/docucentric-backend

# Deploy to Cloud Run
gcloud run deploy docucentric-backend \
  --image gcr.io/your-project/docucentric-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars LLM_PROVIDER=groq \
  --set-secrets GROQ_API_KEY=groq-key:latest
```

---

## Monitoring & Maintenance

### Health Checks

```bash
# API health
curl http://your-domain.com/health

# Component health
curl http://your-domain.com/system/info

# Cache stats
curl http://your-domain.com/api/cache/stats
```

### Log Management

**View logs:**

```bash
# Docker Compose
docker-compose logs -f backend

# Specific service
docker-compose logs -f celery-worker

# Last 100 lines
docker-compose logs --tail=100 backend
```

**Log rotation:**

```bash
# Configure Docker log rotation
cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

sudo systemctl restart docker
```

### Backup Strategy

**Database backup:**

```bash
#!/bin/bash
# backup.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# PostgreSQL backup
docker exec docucentric-postgres pg_dump -U docucentric docucentric > \
  $BACKUP_DIR/postgres_$TIMESTAMP.sql

# Compress
gzip $BACKUP_DIR/postgres_$TIMESTAMP.sql

# Keep only last 7 days
find $BACKUP_DIR -name "postgres_*.sql.gz" -mtime +7 -delete
```

**Cron job:**

```bash
# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh
```

### Performance Tuning

**Redis optimization:**

```bash
# In docker-compose.yml
command: redis-server \
  --appendonly yes \
  --maxmemory 512mb \
  --maxmemory-policy allkeys-lru \
  --save 900 1 \
  --save 300 10 \
  --save 60 10000
```

**PostgreSQL optimization:**

```sql
-- Increase connection pool
ALTER SYSTEM SET max_connections = 200;

-- Optimize for your workload
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '768MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';

-- Reload configuration
SELECT pg_reload_conf();
```

### Updating

**Zero-downtime update:**

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose up -d --build backend

# Verify
curl http://localhost:8000/health

# Rollback if needed
docker-compose up -d --scale backend=0
docker-compose up -d backend  # Uses previous image
```

---

## Troubleshooting

### Common Issues

**1. Redis connection refused:**

```bash
# Check Redis is running
docker ps | grep redis

# Check connectivity
docker exec docucentric-backend redis-cli -h redis ping
```

**2. Database migration errors:**

```bash
# Check migration status
docker exec docucentric-backend alembic current

# Upgrade to latest
docker exec docucentric-backend alembic upgrade head
```

**3. LLM API errors:**

```bash
# Check API key is set
docker exec docucentric-backend env | grep GROQ_API_KEY

# Test LLM connection
curl -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models
```

### Debug Mode

```bash
# Enable debug logging
echo "LOG_LEVEL=DEBUG" >> backend/.env
echo "DEBUG=true" >> backend/.env

# Restart backend
docker-compose restart backend

# View detailed logs
docker-compose logs -f backend | grep ERROR
```

---

## Security Checklist

- [ ] Change default database passwords
- [ ] Enable SSL/TLS
- [ ] Configure CORS for specific origins
- [ ] Set up rate limiting
- [ ] Enable firewall (only open ports 80, 443)
- [ ] Regular security updates
- [ ] Monitor logs for suspicious activity
- [ ] Backup database regularly
- [ ] Use secrets manager for API keys
- [ ] Enable network policies (if using Kubernetes)

---

**Need help?** Check [GitHub Issues](https://github.com/your-repo/issues) or contact support.
