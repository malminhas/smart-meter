# main.tf
# -------
# This code automates building, networking, and deploying Dockerized backend and frontend
# applications in a controlled environment. It uses Terraform to define and manage the
# infrastructure, Docker to containerize the frontend and backend, and Docker Compose to
# orchestrate the two containers.  The code ensures that the backend and frontend are built,
# networked, and deployed in a controlled manner, with the frontend depending on the backend.

# Define the Terraform block with required provider information
terraform {
  required_providers {
    # Specify the Docker provider from "kreuzwerker" with a version constraint
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0.0"
    }
  }
}

# Configures the Docker provider to interact with Docker via the Unix socket.
provider "docker" {
  # Specify the Docker host using a Unix socket
  host = "unix:///var/run/docker.sock"
}

# Build Docker images first using a local-exec provisioner
resource "null_resource" "docker_build" {
  # Use a local-exec provisioner to execute shell commands for building Docker images
  provisioner "local-exec" {
    command = <<-EOT
      docker build -t smart-meter-backend:latest -f ../docker/Dockerfile.backend ..
      docker build -t smart-meter-frontend:latest -f ../docker/Dockerfile.frontend ..
    EOT
  }

  # Force the build to run every time by using a trigger based on the current timestamp
  triggers = {
    always_run = "${timestamp()}"
  }
}

# Define a Docker image resource for the backend.
# Specify the name of the Docker image. Keep the image locally after it's pulled.
# Depend on the Docker build resource to ensure the image is built first.
resource "docker_image" "backend" {
  name = "smart-meter-backend:latest"
  keep_locally = true
  depends_on = [null_resource.docker_build]
}

# Define a Docker image resource for the frontend.
# Specify the name of the Docker image. Keep the image locally after it's pulled.
# Depend on the Docker build resource to ensure the image is built first.
resource "docker_image" "frontend" {
  name = "smart-meter-frontend:latest"
  keep_locally = true
  depends_on = [null_resource.docker_build]
}

# Create a Docker network for container communication
resource "docker_network" "smart_meter_network" {
  name = "smart_meter_network"
}

# Define a Docker container for the backend.
# Specify the container name and image.
# Set the hostname to "smart-meter-backend"
resource "docker_container" "backend" {
  name  = "smart-meter-backend"
  image = docker_image.backend.name
  hostname = "smart-meter-backend"
  
  # Connect the container to the custom Docker network with an alias
  networks_advanced {
    name = docker_network.smart_meter_network.name
    aliases = ["smart-meter-backend"]
  }

  # Map the internal port 8000 to the external port 8000
  ports {
    internal = 8000
    external = 8000
  }

  # Set environment variables for the backend container
  env = [
    "CORS_ORIGINS=http://localhost:8081",
    "BASE_PATH=/energy-assistant"
  ]

  # Configure the container to restart unless stopped manually
  restart = "unless-stopped"
}

# Define a Docker container for the frontend.
# Specify the container name and image.
# Set the hostname to "smart-meter-frontend"
resource "docker_container" "frontend" {
  name  = "smart-meter-frontend"
  image = docker_image.frontend.name
  
  # Connect the container to the custom Docker network
  networks_advanced {
    name = docker_network.smart_meter_network.name
  }

  # Map the internal port 80 to the external port 8081
  ports {
    internal = 80
    external = 8081
  }

  # Configure the container to restart unless stopped manually
  restart = "unless-stopped"
  
  # Add logging
  logs = true
  must_run = true

  # Depend on the backend container to ensure it's running before starting the frontend
  depends_on = [
    docker_container.backend
  ]
}