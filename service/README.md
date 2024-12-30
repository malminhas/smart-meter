# Smart Meter Advisor Service

This service provides energy usage analysis for smart meter data. It consists of a frontend web application and a backend API, deployed using Docker containers and managed with Terraform.

## 1. Architecture

mermaid
graph TD
User[User Browser] -->|HTTPS| Caddy[Caddy Reverse Proxy]
Caddy -->|Port 8081| Frontend[Frontend Container<br/>Nginx]
Caddy -->|Port 8080| WordPress[WordPress Container]
Frontend -->|Port 8000| Backend[Backend Container<br/>FastAPI]
Backend -->|Analysis| Backend

## 2. Components

### Frontend
- Served by Nginx on port 8081
- Static HTML/JavaScript application
- Located at `/energy-assistant/`
- Handles user input and displays analysis results
- Communicates with backend via REST API

### Backend
- FastAPI application on port 8000
- Provides REST API for smart meter data analysis
- Endpoints prefixed with `/energy-assistant/api/`
- [API Documentation](https://smartmeteradvisor.uk/energy-assistant/api/docs)

### Reverse Proxy (Caddy)
- Handles SSL/TLS termination
- Routes traffic to appropriate containers
- Manages paths for both WordPress and Smart Meter application
- Provides security headers and HTTPS redirection

## 3. Deployment

The service is deployed using Infrastructure as Code (IaC) principles:

### Docker Containers
- `smart-meter-frontend`: Nginx serving static content
- `smart-meter-backend`: Python FastAPI application
- Containers are networked together using Docker networking

### Terraform Configuration
- Manages container lifecycle
- Sets up Docker networking
- Configures container environment variables
- Handles port mappings and dependencies

### Directory Structure
```
service/
├── docker/
│ ├── Dockerfile.frontend
│ ├── Dockerfile.backend
│ └── nginx.conf
├── terraform/
│ ├── main.tf
│ ├── variables.tf
│ └── outputs.tf
├── smart-meter-advisor-frontend.html
└── smart-meter-advisor-backend.py
```

## 4. Deployment Process

### Build and deploy containers:

```bash
cd service/terraform
terraform init
terraform apply
```

### Verify deployment locally:

```bash
docker ps
curl -I http://localhost:8081/energy-assistant/
```

### Caddyfile changes:
On Production server:
```bash
cat /etc/caddy/Caddyfile
```

### Access the application in Production:
On Production server:
```bash
open https://smartmeteradvisor.uk/energy-assistant/
```

## 5. Testing

### Test the API is working using pytest:
```bash
pytest test_backend.py -v -s
```

Test the frontend is working using curl:
```bash
curl -X GET "https://smartmeteradvisor.uk/energy-assistant/api/v1/analysis"
```

## 6. References
### URLs
- Frontend: https://smartmeteradvisor.uk/energy-assistant/
- API Documentation: https://smartmeteradvisor.uk/energy-assistant/api/docs
- API Base URL: https://smartmeteradvisor.uk/energy-assistant/api/

### Port Mappings
- 80/443: Caddy (HTTPS)
- 8080: WordPress
- 8081: Smart Meter Frontend
- 8000: Smart Meter Backend API

### Security
- All traffic is encrypted using TLS 1.2/1.3
- Security headers are enforced via Caddy
- CORS is configured for the domain
- HTTP automatically redirects to HTTPS

### Monitoring
- Container logs available via Docker
- Caddy provides detailed access and error logs
- Health checks configured for all services

### Maintenance
To update the service:
1. Make code changes
2. Rebuild containers: `terraform apply`
3. Check logs: `docker logs smart-meter-frontend`
4. Monitor Caddy: `journalctl -u caddy -f`
