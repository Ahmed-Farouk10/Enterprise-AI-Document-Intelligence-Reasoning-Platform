#!/bin/bash
# ========================================
# DocuCentric Local Development Startup
# ========================================

set -e

echo "🚀 DocuCentric - Local Development Setup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker found"
echo -e "${GREEN}✓${NC} Docker Compose found"
echo ""

# Check if .env files exist
if [ ! -f backend/.env ]; then
    echo -e "${YELLOW}⚠️  backend/.env not found. Creating from template...${NC}"
    if [ -f backend/.env.example ]; then
        cp backend/.env.example backend/.env
        echo -e "${YELLOW}📝 Please edit backend/.env with your configuration${NC}"
        echo ""
        read -p "Press enter to continue after editing..."
    else
        echo -e "${RED}❌ backend/.env.example not found${NC}"
        exit 1
    fi
fi

if [ ! -f frontend/.env.local ]; then
    echo -e "${YELLOW}⚠️  frontend/.env.local not found. Creating from template...${NC}"
    if [ -f frontend/.env.example ]; then
        cp frontend/.env.example frontend/.env.local
        echo -e "${GREEN}✓${NC} frontend/.env.local created"
    fi
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Starting DocuCentric Services...${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

# Start services
echo -e "${YELLOW}📦 Building and starting services...${NC}"
docker-compose up -d --build

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Services Started!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "📊 Service Status:"
docker-compose ps
echo ""
echo "🌐 Access Points:"
echo "   - Frontend:    http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs:    http://localhost:8000/docs"
echo "   - Health Check: http://localhost:8000/health"
echo ""
echo "📝 View Logs:"
echo "   docker-compose logs -f backend"
echo "   docker-compose logs -f frontend"
echo "   docker-compose logs -f celery-worker"
echo ""
echo "🛑 Stop Services:"
echo "   docker-compose down"
echo ""
echo -e "${YELLOW}💡 Tip: Run 'docker-compose logs -f' to view all logs${NC}"
