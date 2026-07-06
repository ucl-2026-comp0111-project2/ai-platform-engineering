#!/usr/bin/env bash
# Strategic patch script for caipe-supervisor-agent to hot-reload local agent.py modifications.
#
# Usage:
#   ./patch-agent.sh               – patch agent.py + inject RAG service token (recommended)
#   ./patch-agent.sh --temp-only   – only disable MCP auth (no token, fast workaround)
#   ./patch-agent.sh --token-only  – only inject/refresh the RAG service token
#
set -euo pipefail

NAMESPACE="caipe"
DEPLOYMENT="caipe-supervisor-agent"
LOCAL_PATH="ai_platform_engineering/multi_agents/platform_engineer/protocol_bindings/a2a/agent.py"
CONTAINER_PATH="/app/ai_platform_engineering/multi_agents/platform_engineer/protocol_bindings/a2a/agent.py"
LOCAL_EXECUTOR_PATH="ai_platform_engineering/multi_agents/platform_engineer/protocol_bindings/a2a/agent_executor.py"
CONTAINER_EXECUTOR_PATH="/app/ai_platform_engineering/multi_agents/platform_engineer/protocol_bindings/a2a/agent_executor.py"

MODE="${1:-}"

# ---------------------------------------------------------------------------
# TEMP FIX – Disable MCP auth on the RAG server
# Run with --temp-only for a fast workaround that does NOT survive pod restarts
# on the RAG server side.  The permanent fix (token injection below) is preferred.
# ---------------------------------------------------------------------------
patch_rag_mcp_auth_disabled() {
  echo "🔧 [TEMP FIX] Setting MCP_AUTH_ENABLED=false on rag-server..."
  kubectl set env -n "$NAMESPACE" deployment/rag-server MCP_AUTH_ENABLED=false
  echo "⏳ Waiting for rag-server rollout..."
  kubectl rollout status -n "$NAMESPACE" deployment/rag-server --timeout=120s
  echo "✅ rag-server MCP auth disabled."
}

# ---------------------------------------------------------------------------
# PERMANENT FIX – Fetch a Keycloak client_credentials token and inject it into
# the supervisor as RAG_MCP_SERVICE_TOKEN.
#
# The caipe-platform Keycloak client (service account) has client_credentials
# enabled and its token is accepted by the RAG MCP server (client_credentials
# tokens are granted the RBAC_CLIENT_CREDENTIALS_ROLE, default: ingestonly).
#
# The token is short-lived (~5 min for Keycloak defaults) so the supervisor
# must be restarted when it expires.  For long-running clusters, set up a
# CronJob or re-run this script periodically.
# ---------------------------------------------------------------------------
patch_rag_service_token() {
  echo "🔑 [PERMANENT FIX] Fetching RAG service token via Keycloak client_credentials..."

  # Resolve Keycloak client secret from the existing k8s secret
  local client_secret
  client_secret=$(kubectl get secret caipe-platform-secret -n "$NAMESPACE" \
    -o jsonpath='{.data.OIDC_CLIENT_SECRET}' 2>/dev/null | base64 --decode || true)

  if [[ -z "$client_secret" ]]; then
    echo "⚠️  caipe-platform-secret not found or missing OIDC_CLIENT_SECRET."
    echo "   Trying fallback: KEYCLOAK_CLIENT_SECRET env var or default dev secret..."
    client_secret="${KEYCLOAK_CLIENT_SECRET:-caipe-platform-dev-secret}"
  fi

  # Resolve OIDC issuer from running rag-server env (most reliable source)
  local oidc_issuer
  oidc_issuer=$(kubectl exec -n "$NAMESPACE" deployment/rag-server -- \
    printenv OIDC_ISSUER 2>/dev/null || echo "")

  if [[ -z "$oidc_issuer" ]]; then
    # Fallback: construct from KEYCLOAK_ISSUER_URL env or well-known value
    oidc_issuer="${KEYCLOAK_ISSUER_URL:-https://caipe.local.me/realms/caipe}"
    echo "   OIDC_ISSUER not found in rag-server; using fallback: $oidc_issuer"
  fi

  local token_url="${oidc_issuer}/protocol/openid-connect/token"
  echo "   Token URL: $token_url"

  # Port-forward Keycloak for the token request (in-cluster DNS may not be reachable locally)
  local pf_port=17080
  echo "   Starting port-forward to caipe-keycloak:8080 → localhost:${pf_port}..."
  kubectl port-forward svc/caipe-keycloak -n "$NAMESPACE" "${pf_port}:8080" >/dev/null 2>&1 &
  local pf_pid=$!
  sleep 4

  # Replace cluster hostname with localhost for the port-forwarded call
  local local_token_url
  local_token_url=$(echo "$token_url" | sed "s|https://caipe.local.me|http://localhost:${pf_port}|")

  local service_token
  service_token=$(curl -s -X POST "$local_token_url" \
    -d "grant_type=client_credentials" \
    -d "client_id=caipe-platform" \
    -d "client_secret=${client_secret}" \
    | sed -n 's/.*"access_token":"\([^"]*\)".*/\1/p' || true)

  kill "$pf_pid" 2>/dev/null || true

  if [[ -z "$service_token" ]]; then
    echo "❌ Failed to fetch service token from Keycloak."
    echo "   Check that caipe-keycloak is running and caipe-platform client has service_accounts_enabled=true."
    exit 1
  fi

  echo "✅ Service token obtained (${#service_token} chars)."
  echo "⚙️  Injecting RAG_MCP_SERVICE_TOKEN into ${DEPLOYMENT}..."
  kubectl set env -n "$NAMESPACE" "deployment/${DEPLOYMENT}" \
    "RAG_MCP_SERVICE_TOKEN=${service_token}"
  echo "✅ RAG_MCP_SERVICE_TOKEN injected."
}

# ---------------------------------------------------------------------------
# Patch agent.py via ConfigMap volume mount
# ---------------------------------------------------------------------------
patch_agent_py() {
  echo ""
  echo "⚙️  Creating supervisor-custom-patches ConfigMap from local files..."
  kubectl create configmap supervisor-custom-patches -n "$NAMESPACE" \
    --from-file=agent.py="${LOCAL_PATH}" \
    --from-file=agent_executor.py="${LOCAL_EXECUTOR_PATH}" \
    --dry-run=client -o yaml | kubectl apply -f -

  echo "⚙️  Patching ${DEPLOYMENT} deployment to mount patched files..."
  kubectl patch deploy "$DEPLOYMENT" -n "$NAMESPACE" --type=strategic -p "
{
  \"spec\": {
    \"template\": {
      \"spec\": {
        \"volumes\": [
          {
            \"name\": \"supervisor-custom-patches-volume\",
            \"configMap\": {
              \"name\": \"supervisor-custom-patches\"
            }
          }
        ],
        \"containers\": [
          {
            \"name\": \"supervisor-agent\",
            \"volumeMounts\": [
              {
                \"name\": \"supervisor-custom-patches-volume\",
                \"mountPath\": \"${CONTAINER_PATH}\",
                \"subPath\": \"agent.py\"
              },
              {
                \"name\": \"supervisor-custom-patches-volume\",
                \"mountPath\": \"${CONTAINER_EXECUTOR_PATH}\",
                \"subPath\": \"agent_executor.py\"
              }
            ]
          }
        ]
      }
    }
  }
}
"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ "$MODE" == "--temp-only" ]]; then
  echo "=== Running TEMP FIX only (MCP_AUTH_ENABLED=false) ==="
  patch_rag_mcp_auth_disabled
  echo ""
  echo "🔄 Restarting ${DEPLOYMENT} to pick up auth change..."
  kubectl rollout restart -n "$NAMESPACE" "deployment/${DEPLOYMENT}"
  kubectl rollout status -n "$NAMESPACE" "deployment/${DEPLOYMENT}" --timeout=120s
  echo "🎉 Done. Note: this is a temporary fix — re-run with no args for the permanent token."

elif [[ "$MODE" == "--token-only" ]]; then
  echo "=== Refreshing RAG service token only ==="
  patch_rag_service_token
  echo ""
  echo "🔄 Restarting ${DEPLOYMENT} to reload token..."
  kubectl rollout restart -n "$NAMESPACE" "deployment/${DEPLOYMENT}"
  kubectl rollout status -n "$NAMESPACE" "deployment/${DEPLOYMENT}" --timeout=120s
  echo "🎉 RAG service token refreshed."

else
  echo "=== Patching agent.py + injecting RAG service token ==="
  patch_rag_service_token
  patch_agent_py
  echo ""
  echo "🔄 Waiting for ${DEPLOYMENT} rollout to complete..."
  kubectl rollout status "deploy/${DEPLOYMENT}" -n "$NAMESPACE" --timeout=120s
  echo "🎉 Success! Patched agent.py and RAG service token deployed to ${DEPLOYMENT}."
fi
