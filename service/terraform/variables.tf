variable "backend_port" {
  description = "Port for the backend service"
  type        = number
  default     = 8000
}

variable "frontend_port" {
  description = "Port for the frontend service"
  type        = number
  default     = 80
} 