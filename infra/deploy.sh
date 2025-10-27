
#!/bin/bash

# Set variables
RESOURCE_GROUP="doc-research-rg"
LOCATION="eastus"
DEPLOYMENT_NAME="doc-research-deployment-$(date +%Y%m%d%H%M%S)"

# Create resource group if it doesn't exist
az group create --name $RESOURCE_GROUP --location $LOCATION

# Deploy the bicep template
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --name $DEPLOYMENT_NAME \
  --template-file main.bicep \
  --parameters @main.parameters.json

# Extract the backend and frontend URLs from the deployment output
BACKEND_URL=$(az deployment group show --resource-group $RESOURCE_GROUP --name $DEPLOYMENT_NAME --query "properties.outputs.backendUrl.value" -o tsv)
FRONTEND_URL=$(az deployment group show --resource-group $RESOURCE_GROUP --name $DEPLOYMENT_NAME --query "properties.outputs.frontendUrl.value" -o tsv)

echo "Deployment complete!"
echo "Backend URL: https://$BACKEND_URL"
echo "Frontend URL: https://$FRONTEND_URL"