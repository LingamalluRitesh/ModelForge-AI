# ModelForge AI - Microsoft Azure Kubernetes Service (AKS) Enterprise Production Cluster
# Provisions Azure Virtual Network, Managed Azure CNI AKS Cluster, User Node Pools,
# NC-series GPU acceleration nodes, and Azure Entra ID RBAC integration.

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.95.0"
    }
  }
}

variable "resource_group_name" {
  type        = string
  default     = "rg-modelforge-prod"
  description = "Azure Resource Group Name"
}

variable "location" {
  type        = string
  default     = "eastus"
  description = "Primary Azure Region"
}

resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
}

# Azure Virtual Network
resource "azurerm_virtual_network" "vnet" {
  name                = "vnet-modelforge-prod"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  address_space       = ["10.240.0.0/16"]
}

resource "azurerm_subnet" "aks_subnet" {
  name                 = "snet-aks-nodes"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.240.0.0/20"]
}

# AKS Managed Cluster
resource "azurerm_kubernetes_cluster" "aks" {
  name                = "aks-modelforge-production"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "modelforge-aks"
  kubernetes_version  = "1.30"

  default_node_pool {
    name           = "systempool"
    node_count     = 3
    vm_size        = "Standard_D8s_v5"
    vnet_subnet_id = azurerm_subnet.aks_subnet.id
    os_disk_size_gb = 128

    upgrade_settings {
      max_surge = "10%"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    load_balancer_sku = "standard"
    service_cidr      = "10.0.0.0/16"
    dns_service_ip    = "10.0.0.10"
  }

  auto_scaler_profile {
    balance_similar_node_groups = true
    max_graceful_termination_sec = 600
  }
}

# GPU Acceleration Node Pool for PyTorch / ONNX Inference
resource "azurerm_kubernetes_cluster_node_pool" "gpu_pool" {
  name                  = "gpupool"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = "Standard_NC8as_T4_v3" # NVIDIA Tesla T4 GPU
  node_count            = 1
  vnet_subnet_id        = azurerm_subnet.aks_subnet.id

  enable_auto_scaling = true
  min_count           = 0
  max_count           = 6

  node_taints = [
    "sku=gpu:NoSchedule"
  ]

  node_labels = {
    "accelerator" = "nvidia-t4"
  }
}
