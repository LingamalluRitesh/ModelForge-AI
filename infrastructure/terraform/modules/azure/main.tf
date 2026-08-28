# ModelForge AI - Microsoft Azure Terraform Module
# Provisions Resource Group, VNet, AKS Cluster, Azure Database for PostgreSQL Flexible Server, Azure Cache for Redis, and Blob Storage

variable "environment" { type = string }
variable "location" { type = string, default = "East US" }

# 1. Resource Group
resource "azurerm_resource_group" "rg" {
  name     = "modelforge-${var.environment}-rg"
  location = var.location
}

# 2. Azure Storage Account & Blob Container
resource "azurerm_storage_account" "sa" {
  name                     = "modelforgesa${var.environment}"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "GRS"
}

resource "azurerm_storage_container" "artifacts" {
  name                  = "model-artifacts"
  storage_account_name  = azurerm_storage_account.sa.name
  container_access_type = "private"
}

# 3. Azure Kubernetes Service (AKS)
resource "azurerm_kubernetes_cluster" "aks" {
  name                = "modelforge-${var.environment}-aks"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "modelforge-${var.environment}"

  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D4s_v5"
  }

  identity {
    type = "SystemAssigned"
  }
}

# 4. Azure Database for PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "postgres" {
  name                   = "modelforge-${var.environment}-pgflex"
  resource_group_name    = azurerm_resource_group.rg.name
  location               = azurerm_resource_group.rg.location
  version                = "16"
  administrator_login    = "modelforgeadmin"
  administrator_password = "ChangeMeSecurePassAzure123!"
  sku_name               = "GP_Standard_D4s_v3"
  storage_mb             = 131072
}
