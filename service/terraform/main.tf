terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0.0"
    }
  }
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# Build Docker images first
resource "null_resource" "docker_build" {
  provisioner "local-exec" {
    command = <<-EOT
      docker build -t smart-meter-backend:latest -f ../docker/Dockerfile.backend ..
      docker build -t smart-meter-frontend:latest -f ../docker/Dockerfile.frontend ..
    EOT
  }

  # Force rebuild every time
  triggers = {
    always_run = "${timestamp()}"
  }
}

# Use pre-built backend image
resource "docker_image" "backend" {
  name = "smart-meter-backend:latest"
  keep_locally = true
  depends_on = [null_resource.docker_build]
}

# Use pre-built frontend image
resource "docker_image" "frontend" {
  name = "smart-meter-frontend:latest"
  keep_locally = true
  depends_on = [null_resource.docker_build]
}

resource "docker_network" "smart_meter_network" {
  name = "smart_meter_network"
}

resource "docker_container" "backend" {
  name  = "smart-meter-backend"
  image = docker_image.backend.name
  hostname = "smart-meter-backend"
  
  networks_advanced {
    name = docker_network.smart_meter_network.name
    aliases = ["smart-meter-backend"]
  }

  ports {
    internal = 8000
    external = 8000
  }

  env = [
    "CORS_ORIGINS=http://localhost:8080",
    "BASE_PATH=/energy-assistant"
  ]

  restart = "unless-stopped"
}

resource "docker_container" "frontend" {
  name  = "smart-meter-frontend"
  image = docker_image.frontend.name
  
  networks_advanced {
    name = docker_network.smart_meter_network.name
  }

  ports {
    internal = 80
    external = 8080
  }

  restart = "unless-stopped"
  
  # Add logging
  logs = true
  must_run = true

  depends_on = [
    docker_container.backend
  ]
}