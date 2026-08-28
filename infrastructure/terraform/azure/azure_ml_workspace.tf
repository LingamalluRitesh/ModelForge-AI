# ModelForge AI - Azure Machine Learning Workspace Infrastructure

resource "azurerm_machine_learning_workspace" "modelforge_ml_workspace" {
  name                    = "modelforge-azure-workspace"
  location                = var.azure_location
  resource_group_name     = var.azure_resource_group
  application_insights_id = azurerm_application_insights.modelforge_app_insights.id
  key_vault_id            = azurerm_key_vault.modelforge_keyvault.id
  storage_account_id      = azurerm_storage_account.modelforge_storage.id

  identity {
    type = "SystemAssigned"
  }

  tags = {
    Environment = "production"
    Product     = "ModelForge-AI"
  }
}
