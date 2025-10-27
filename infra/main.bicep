@description('Specifies the location for resources.')
param location string = resourceGroup().location

@description('Specifies the environment name prefix used for generating uniqueness for resources.')
param environmentName string = 'plan-search'

@description('Specifies the container image for the backend application.')
param backendContainerImage string

@description('Specifies the container image for the frontend application.')
param frontendContainerImage string

@description('The Azure Container Registry name')
param acrName string 

resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'aca-identity'
  location: location
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, managedIdentity.id, 'acrpull')
  scope: acr
  properties: {
    principalId: managedIdentity.properties.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalType: 'ServicePrincipal'
  }
}

var containerAppEnvName = '${environmentName}-env'
var backendAppName = '${environmentName}-backend'
var frontendAppName = '${environmentName}-frontend'
var logAnalyticsWorkspaceName = '${environmentName}-logs'
var backendEnvVars = json(loadTextContent('./backend-env.json', 'utf-8'))

var backendEnvArray = [for key in items(backendEnvVars): {
  name: key.key
  secretRef: replace(toLower(key.key), '_', '-')
}]

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsWorkspaceName
  location: location
  properties: {
    retentionInDays: 30
    sku: {
      name: 'PerGB2018'
    }
  }
}

resource containerAppEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
  }
}

resource backendApp 'Microsoft.App/containerApps@2025-02-02-preview' = {
  name: backendAppName
  location: location
  dependsOn: [
    roleAssignment
  ]
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppEnvironment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: [
        {
          server: '${acrName}.azurecr.io'
          identity: managedIdentity.id
        }
      ]
      secrets: [for key in items(backendEnvVars): {
        name: replace(toLower(key.key), '_', '-')
        value: string(key.value)
      }]
    }
    template: {
      revisionSuffix: 'be-${uniqueString(backendContainerImage)}'
      containers: [{ 
        name: 'backend'
        image: backendContainerImage
        resources: {
          cpu: json('2')
          memory: '4Gi'
        }
        env: concat(
          backendEnvArray,
          [              
            {
              name: 'APP_USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID'
              value: managedIdentity.properties.clientId
            }
          ]
        )      
        }
      ]
      scale: {
        minReplicas: 2
        maxReplicas: 3
      }
    }
  }
}

resource frontendApp 'Microsoft.App/containerApps@2025-02-02-preview' = {
  name: frontendAppName
  location: location
  dependsOn: [
    roleAssignment
  ]
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppEnvironment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 7860
        allowInsecure: false
        transport: 'http'
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: [
        {
          server: '${acrName}.azurecr.io'
          identity: managedIdentity.id
        }
      ]
      secrets: [
        {
          name: 'auth-username'
          value: string(backendEnvVars.AUTH_USERNAME)
        }
        {
          name: 'auth-password'
          value: string(backendEnvVars.AUTH_PASSWORD)
        }
        {
          name: 'chainlit-auth-secret'
          value: string(backendEnvVars.CHAINLIT_AUTH_SECRET)
        }
        {
          name: 'youtube-api-key'
          value: string(backendEnvVars.YOUTUBE_API_KEY)
        }
      ]      
    }
    template: {
      revisionSuffix: 'fe-${uniqueString(frontendContainerImage)}'
      containers: [
        {
          name: 'frontend'
          image: frontendContainerImage
          resources: {
            cpu: json('2')
            memory: '4Gi'
          }
          env: [
            {
              name: 'SK_API_URL'
              value: 'https://${backendApp.properties.configuration.ingress.fqdn}/plan_search'
            }
            {
              name: 'API_URL'
              value: 'https://${backendApp.properties.configuration.ingress.fqdn}/chat'
            }
            {
              name: 'DEEP_SEARCH_API_URL'
              value: 'https://${backendApp.properties.configuration.ingress.fqdn}/deep_search'
            }
            {
              name: 'AUTH_USERNAME'
              secretRef: 'auth-username'
            }
            {
              name: 'AUTH_PASSWORD'
              secretRef: 'auth-password'
            }
            {
              name: 'CHAINLIT_AUTH_SECRET'
              secretRef: 'chainlit-auth-secret'
            }
            {
              name: 'YOUTUBE_API_KEY'
              secretRef: 'youtube-api-key'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

output backendUrl string = backendApp.properties.configuration.ingress.fqdn
output frontendUrl string = frontendApp.properties.configuration.ingress.fqdn
