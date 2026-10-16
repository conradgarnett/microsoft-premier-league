// Azure Container Apps deployment for 3D Tactical Replay.
// Provisions: Log Analytics, Container Registry, Container Apps environment, the app
// (API + viewer) with a user-assigned managed identity, and optionally Azure OpenAI
// (model deployment for the agents) and Cosmos DB (NoSQL, serverless) for match storage.
// The identity gets AcrPull, Cognitive Services OpenAI User and Cosmos data contributor,
// so no keys are stored anywhere.
targetScope = 'resourceGroup'

@description('azd environment name, used to derive resource names')
param environmentName string
param location string = resourceGroup().location
@description('Container image; azd sets this on deploy')
param appImage string = 'mcr.microsoft.com/k8se/quickstart:latest'
param deployOpenAI bool = true
param deployCosmos bool = true
param openAIModel string = 'gpt-4o-mini'
param openAIModelVersion string = '2024-07-18'

var token = toLower(uniqueString(subscription().id, resourceGroup().id, environmentName))
var tags = { 'azd-env-name': environmentName }

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${token}'
  location: location
  tags: tags
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'cr${token}'
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${token}'
  location: location
  tags: tags
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, 'acrpull')
  scope: registry
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  }
}

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (deployOpenAI) {
  name: 'oai-${token}'
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: { customSubDomainName: 'oai-${token}', publicNetworkAccess: 'Enabled', disableLocalAuth: true }
}

resource deployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployOpenAI) {
  parent: openai
  name: openAIModel
  sku: { name: 'GlobalStandard', capacity: 30 }
  properties: { model: { format: 'OpenAI', name: openAIModel, version: openAIModelVersion } }
}

resource openaiUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployOpenAI) {
  name: guid(resourceGroup().id, identity.id, 'openai-user')
  scope: openai
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
  }
}

resource cosmos 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' = if (deployCosmos) {
  name: 'cosmos-${token}'
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    locations: [ { locationName: location, failoverPriority: 0 } ]
    capabilities: [ { name: 'EnableServerless' } ]
    disableLocalAuth: true
  }
}

resource cosmosDb 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-05-15' = if (deployCosmos) {
  parent: cosmos
  name: 'tactical'
  properties: { resource: { id: 'tactical' } }
}

resource cosmosContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = if (deployCosmos) {
  parent: cosmosDb
  name: 'matches'
  properties: { resource: { id: 'matches', partitionKey: { paths: [ '/match_id' ], kind: 'Hash' } } }
}

// Cosmos DB Built-in Data Contributor (data-plane RBAC)
resource cosmosData 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = if (deployCosmos) {
  parent: cosmos
  name: guid(cosmos.id, identity.id, 'data-contributor')
  properties: {
    principalId: identity.properties.principalId
    roleDefinitionId: '${cosmos.id}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000002'
    scope: cosmos.id
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${token}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: { customerId: logs.properties.customerId, sharedKey: logs.listKeys().primarySharedKey }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-${token}'
  location: location
  tags: union(tags, { 'azd-service-name': 'app' })
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identity.id}': {} } }
  dependsOn: [ acrPull ]
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: { external: true, targetPort: 8000, transport: 'auto' }
      registries: [ { server: registry.properties.loginServer, identity: identity.id } ]
    }
    template: {
      containers: [
        {
          name: 'app'
          image: appImage
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
            { name: 'AZURE_OPENAI_ENDPOINT', value: deployOpenAI ? openai!.properties.endpoint : '' }
            { name: 'AZURE_OPENAI_DEPLOYMENT', value: deployOpenAI ? openAIModel : '' }
            { name: 'AZURE_OPENAI_USE_ENTRA', value: '1' }
            { name: 'STORE', value: deployCosmos ? 'cosmos' : 'local' }
            { name: 'COSMOS_ENDPOINT', value: deployCosmos ? cosmos!.properties.documentEndpoint : '' }
          ]
          probes: [ { type: 'Startup', httpGet: { path: '/api/health', port: 8000 }, periodSeconds: 10, failureThreshold: 30 } ]
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 3 }
    }
  }
}

output AZURE_CONTAINER_REGISTRY_ENDPOINT string = registry.properties.loginServer
output SERVICE_APP_URI string = 'https://${app.properties.configuration.ingress.fqdn}'
