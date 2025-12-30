"""Script to discover available models in your LiteLLM proxy."""

import os
import sys
import ssl
import httpx
from dotenv import load_dotenv

# Disable SSL verification
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()

litellm_key = os.getenv("LITELLM_API_KEY")
litellm_url = os.getenv("LITELLM_BASE_URL")

print("="*70)
print("LITELLM AVAILABLE MODELS DISCOVERY")
print("="*70)

if not litellm_key or not litellm_url:
    print("\n❌ ERROR: LiteLLM configuration not found")
    print("\nMake sure your .env file contains:")
    print("  LITELLM_API_KEY=sk-your-token")
    print("  LITELLM_BASE_URL=https://your-proxy.com/anthropic")
    sys.exit(1)

print(f"\n✓ LiteLLM API Key: {litellm_key[:8]}...{litellm_key[-4:]}")
print(f"✓ Base URL: {litellm_url}")

# Extract the base URL without /anthropic suffix
base_url = litellm_url.replace("/anthropic", "").replace("/v1", "").rstrip("/")

# Try different possible endpoints for listing models
endpoints = [
    f"{base_url}/v1/models",
    f"{base_url}/models",
    f"{base_url}/model/info",
    f"{base_url}/v1/model/info",
]

print("\n" + "="*70)
print("SEARCHING FOR MODELS ENDPOINT...")
print("="*70)

# Create HTTP client with SSL disabled
client = httpx.Client(verify=False, timeout=30.0)

models_data = None
working_endpoint = None

for endpoint in endpoints:
    try:
        print(f"\nTrying: {endpoint}")

        response = client.get(
            endpoint,
            headers={
                "Authorization": f"Bearer {litellm_key}",
                "Content-Type": "application/json"
            }
        )

        if response.status_code == 200:
            models_data = response.json()
            working_endpoint = endpoint
            print(f"✓ SUCCESS! Found models endpoint")
            break
        else:
            print(f"  Status {response.status_code}: {response.text[:100]}")

    except Exception as e:
        print(f"  Failed: {str(e)[:100]}")

if not models_data:
    print("\n" + "="*70)
    print("⚠️  Could not find models endpoint automatically")
    print("="*70)
    print("\nPlease ask your IT/DevOps team for:")
    print("1. List of available model names in your LiteLLM config")
    print("2. The specific model name for Claude 3.5 Sonnet")
    print("\nCommon model names in LiteLLM:")
    print("  - claude-3-5-sonnet")
    print("  - anthropic/claude-3-5-sonnet")
    print("  - claude-sonnet-3.5")
    print("  - anthropic.claude-3-5-sonnet")
    print("\nOnce you have the correct name, update it in your .env:")
    print("  CLAUDE_MODEL_NAME=the-correct-model-name")
    sys.exit(1)

# Parse and display models
print("\n" + "="*70)
print("AVAILABLE MODELS")
print("="*70)

# Handle different response formats
if isinstance(models_data, dict):
    if "data" in models_data:
        # OpenAI-compatible format
        models_list = models_data["data"]
    elif "models" in models_data:
        models_list = models_data["models"]
    elif "model_list" in models_data:
        models_list = models_data["model_list"]
    else:
        models_list = [models_data]
elif isinstance(models_data, list):
    models_list = models_data
else:
    print(f"Unknown response format: {type(models_data)}")
    print(f"Raw response: {models_data}")
    sys.exit(1)

# Find Claude/Anthropic models
claude_models = []
other_models = []

for model in models_list:
    if isinstance(model, dict):
        model_id = model.get("id") or model.get("model_name") or model.get("name") or str(model)
    else:
        model_id = str(model)

    model_id_lower = model_id.lower()
    if "claude" in model_id_lower or "anthropic" in model_id_lower:
        claude_models.append(model)
    else:
        other_models.append(model)

# Display Claude models prominently
if claude_models:
    print("\n🎯 CLAUDE/ANTHROPIC MODELS (Use these!):")
    print("-" * 70)
    for i, model in enumerate(claude_models, 1):
        if isinstance(model, dict):
            model_id = model.get("id") or model.get("model_name") or model.get("name")
            print(f"\n  {i}. {model_id}")

            # Show additional info if available
            if "created" in model:
                print(f"     Created: {model.get('created')}")
            if "owned_by" in model:
                print(f"     Owner: {model.get('owned_by')}")
            if "description" in model:
                print(f"     Description: {model.get('description')}")
        else:
            print(f"  {i}. {model}")

if other_models:
    print(f"\n\n📋 OTHER AVAILABLE MODELS ({len(other_models)} total):")
    print("-" * 70)
    for i, model in enumerate(other_models[:10], 1):  # Show first 10
        if isinstance(model, dict):
            model_id = model.get("id") or model.get("model_name") or model.get("name")
            print(f"  {i}. {model_id}")
        else:
            print(f"  {i}. {model}")

    if len(other_models) > 10:
        print(f"  ... and {len(other_models) - 10} more")

# Provide recommendations
print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

if claude_models:
    # Get the first Claude model as recommendation
    if isinstance(claude_models[0], dict):
        recommended = claude_models[0].get("id") or claude_models[0].get("model_name") or claude_models[0].get("name")
    else:
        recommended = str(claude_models[0])

    print(f"\n✓ Found {len(claude_models)} Claude model(s)")
    print(f"\n📝 RECOMMENDED: Use this model name:")
    print(f"\n   {recommended}")
    print(f"\n1. Update your .env file:")
    print(f"   CLAUDE_MODEL_NAME={recommended}")
    print(f"\n2. Or update app.py line 104 to use:")
    print(f"   claude_model=\"{recommended}\"")
    print(f"\n3. Then run: uv run python test_claude.py")
else:
    print("\n⚠️  No Claude models found in the list")
    print("\nThis could mean:")
    print("1. Your LiteLLM uses custom model aliases")
    print("2. Models are configured differently")
    print("3. You need special permissions")
    print("\n👉 Contact your IT/DevOps team and ask:")
    print("   'What model name should I use for Claude 3.5 Sonnet?'")

print("\n" + "="*70)
