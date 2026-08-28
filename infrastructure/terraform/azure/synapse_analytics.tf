# ModelForge AI - Azure Synapse Analytics Datalake

resource "azurerm_synapse_workspace" "modelforge_synapse" {
  name                                 = "modelforge-synapse-dw"
  resource_group_name                  = var.azure_resource_group
  location                             = var.azure_location
  storage_data_lake_gen2_filesystem_id = azurerm_storage_data_lake_gen2_filesystem.modelforge_adls.id
  sql_administrator_login              = "sqladminuser"
  sql_administrator_login_password     = var.synapse_admin_password

  identity {
    type = "SystemAssigned"
  }

  tags = {
    Environment = "production"
  }
}
