"""Test script to validate Claude API connection."""

import os
import sys
import ssl
from dotenv import load_dotenv
from anthropic import Anthropic
import httpx

# Disable SSL verification
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
print("Loading .env file...")
load_dotenv()

# Check for LiteLLM or Anthropic configuration
litellm_key = os.getenv("LITELLM_API_KEY")
litellm_url = os.getenv("LITELLM_BASE_URL")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

print("\n" + "=" * 60)
print("CLAUDE API CONNECTION TEST")
print("=" * 60)

# Determine which configuration to use
if litellm_key and litellm_url:
    print("\n✓ Using LiteLLM Proxy configuration")
    api_key = litellm_key
    base_url = litellm_url
    mode = "litellm"
elif anthropic_key:
    print("\n✓ Using Direct Anthropic API configuration")
    api_key = anthropic_key
    base_url = None
    mode = "anthropic"
else:
    print("\n❌ ERROR: No API configuration found")
    print("\nYou need EITHER:")
    print("  Option 1 (LiteLLM): LITELLM_API_KEY + LITELLM_BASE_URL")
    print("  Option 2 (Direct): ANTHROPIC_API_KEY")
    print("\nTroubleshooting:")
    print("1. Check if .env file exists in project root")
    print("2. See .env.example for configuration examples")
    print("3. Make sure there are no quotes around values")
    print("4. Check for extra spaces before/after values")
    sys.exit(1)

# Display API key info (masked)
key_length = len(api_key)
masked_key = (
    api_key[:8] + "*" * (key_length - 12) + api_key[-4:] if key_length > 12 else "***"
)
print(f"\n✓ API key found: {masked_key}")
print(f"  Length: {key_length} characters")

if mode == "litellm":
    print(f"\n✓ LiteLLM proxy URL: {base_url}")

# Check for common issues
issues = []
if api_key.startswith('"') or api_key.startswith("'"):
    issues.append("Key starts with a quote character")
if api_key.endswith('"') or api_key.endswith("'"):
    issues.append("Key ends with a quote character")
if api_key.startswith(" ") or api_key.endswith(" "):
    issues.append("Key has leading or trailing spaces")
if "\n" in api_key or "\r" in api_key:
    issues.append("Key contains newline characters")

if issues:
    print("\n⚠️  WARNING: Potential issues detected:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nCleaning API key...")
    api_key = api_key.strip().strip('"').strip("'")
    print(f"  Cleaned key length: {len(api_key)}")

# Test Claude API connection
print("\nTesting Claude API connection...")
print("(This will make a test API call - minimal cost)")

try:
    # Create httpx client with SSL disabled
    http_client = httpx.Client(verify=False, timeout=30.0)

    # Create Anthropic client
    if base_url:
        # Using LiteLLM proxy
        client = Anthropic(api_key=api_key, base_url=base_url, http_client=http_client)
        print(f"Connecting via LiteLLM proxy: {base_url}")
    else:
        # Direct Anthropic API
        client = Anthropic(api_key=api_key, http_client=http_client)
        print("Connecting directly to Anthropic API")

    # Make a simple test call
    print("Sending test message to Claude...")
    response = client.messages.create(
        model=os.getenv("CLAUDE_MODEL_NAME"),
        max_tokens=50,
        messages=[
            {
                "role": "user",
                "content": "Say 'Hello, I am working!' if you can read this.",
            }
        ],
    )

    # Display success
    answer = response.content[0].text
    print("\n" + "=" * 60)
    if mode == "litellm":
        print("✅ SUCCESS! LiteLLM proxy connection works!")
    else:
        print("✅ SUCCESS! Direct Anthropic API connection works!")
    print("=" * 60)
    print(f"\nClaude's response: {answer}")
    print(f"\nToken usage:")
    print(f"  - Input tokens: {response.usage.input_tokens}")
    print(f"  - Output tokens: {response.usage.output_tokens}")

    if mode == "litellm":
        print("\n✓ Your LiteLLM proxy is configured correctly")
    else:
        print("\n✓ Your Anthropic API key is valid")

except Exception as e:
    print("\n" + "=" * 60)
    print("❌ ERROR: Claude API connection failed")
    print("=" * 60)
    print(f"\nError message: {e}")
    print("\nTroubleshooting:")
    print("1. Verify your API key at https://console.anthropic.com/")
    print("2. Check if the key has been revoked or expired")
    print("3. Try creating a new API key")
    print("4. Make sure you copied the entire key (no truncation)")
    sys.exit(1)

print("\n" + "=" * 60)
print("All checks passed! Your RAG system should work.")
print("=" * 60)
