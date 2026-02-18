# Docker Setup Guide

This guide explains how to run the IntelliMeet application using Docker Compose for production-ready deployment.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Environment variables configured (see `.env` setup below)

## Quick Start

1. **Create environment file** (copy from `backend/.env.example`):
   ```bash
   cp backend/.env.example .env
   ```

2. **Edit `.env`** and set your Azure OpenAI credentials and MongoDB URI:
   ```env
   AZURE_OPENAI_KEY=your_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-mini
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
   JWT_SECRET_KEY=your-secure-random-string-min-32-chars
   ```

3. **Build and start all services**:
   ```bash
   docker-compose up --build
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Services

### MongoDB
- **Port:** 27017
- **Default credentials:** `admin` / `changeme` (change in production!)
- **Data persistence:** Volume `mongodb_data`
- **Health check:** Enabled

### Backend (FastAPI)
- **Port:** 8000
- **Health check:** `/health` endpoint
- **Hot reload:** Enabled in development (volume mount)
- **Dependencies:** Auto-installed from `requirements.txt`

### Frontend (React + Nginx)
- **Port:** 3000 (mapped to nginx port 80)
- **Build:** Multi-stage build for optimized production bundle
- **Static files:** Served via nginx with gzip compression
- **SPA routing:** Configured to serve `index.html` for all routes

## Environment Variables

### Required for Docker Compose

Create a `.env` file in the project root with:

```env
# Azure OpenAI
AZURE_OPENAI_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

# Optional API versions (defaults shown)
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_EMBEDDING_API_VERSION=2024-08-01-preview
AZURE_OPENAI_AGENT_API_VERSION=2024-08-01-preview

# JWT Secret (generate a secure random string for production)
JWT_SECRET_KEY=your-secure-random-string-min-32-chars

# Frontend API URL (for React build)
REACT_APP_API_URL=http://localhost:8000
```

**Note:** MongoDB connection is automatically configured in `docker-compose.yml` to use the MongoDB service.

## Production Deployment

### Security Checklist

1. **Change MongoDB credentials** in `docker-compose.yml`:
   ```yaml
   environment:
     MONGO_INITDB_ROOT_USERNAME: your_secure_username
     MONGO_INITDB_ROOT_PASSWORD: your_secure_password
   ```

2. **Update MongoDB URI** in backend environment:
   ```yaml
   environment:
     - MONGO_URI=mongodb://your_secure_username:your_secure_password@mongodb:27017/?authSource=admin
   ```

3. **Generate secure JWT secret**:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

4. **Disable hot reload** in `docker-compose.yml` (remove volume mount and use production command):
   ```yaml
   backend:
     # Remove volumes section
     command: uvicorn main:app --host 0.0.0.0 --port 8000
   ```

5. **Use HTTPS** in production (configure nginx with SSL certificates).

### Building for Production

```bash
# Build images
docker-compose build

# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes (⚠️ deletes data)
docker-compose down -v
```

## Development Mode

The `docker-compose.yml` includes volume mounts for hot reload during development:

- Backend: `./backend:/app` (code changes reflect immediately)
- Frontend: Built once, changes require rebuild

To rebuild frontend after changes:
```bash
docker-compose build frontend
docker-compose up -d frontend
```

## Troubleshooting

### MongoDB connection issues
- Check MongoDB health: `docker-compose ps mongodb`
- View MongoDB logs: `docker-compose logs mongodb`
- Verify credentials match in `docker-compose.yml`

### Backend not starting
- Check environment variables: `docker-compose config`
- View backend logs: `docker-compose logs backend`
- Verify Azure OpenAI credentials are correct

### Frontend not loading
- Check nginx logs: `docker-compose logs frontend`
- Verify `REACT_APP_API_URL` is set correctly
- Rebuild frontend: `docker-compose build frontend`

### Port conflicts
- Change ports in `docker-compose.yml` if 3000, 8000, or 27017 are in use
- Update `REACT_APP_API_URL` if backend port changes

## Health Checks

All services include health checks:
- **MongoDB:** Ping test via mongosh
- **Backend:** HTTP GET to `/health`
- **Frontend:** HTTP GET to `/health` (nginx)

View health status:
```bash
docker-compose ps
```

## Data Persistence

MongoDB data is persisted in a Docker volume `mongodb_data`. To backup:

```bash
# Create backup
docker run --rm -v intellimeet_mongodb_data:/data -v $(pwd):/backup mongo:7.0 tar czf /backup/mongodb-backup.tar.gz /data

# Restore backup
docker run --rm -v intellimeet_mongodb_data:/data -v $(pwd):/backup mongo:7.0 tar xzf /backup/mongodb-backup.tar.gz -C /
```

## Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove containers, networks, and volumes
docker-compose down -v

# Remove images
docker-compose down --rmi all
```
