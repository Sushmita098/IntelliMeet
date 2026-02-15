r"""
Test script to verify Azure OpenAI connectivity.
Run from backend folder: python test_scripts/test_azure_openai.py
Or: cd backend && .venv\Scripts\python.exe test_scripts\test_azure_openai.py
"""
import os
import sys

# Load .env from backend folder (parent of test_scripts)
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(backend_dir, ".env")
sys.path.insert(0, backend_dir)
os.chdir(backend_dir)

from dotenv import load_dotenv

load_dotenv(env_path)


def main():
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    if not api_key:
        print("ERROR: AZURE_OPENAI_KEY not set in .env")
        sys.exit(1)
    if not endpoint:
        print("ERROR: AZURE_OPENAI_ENDPOINT not set in .env")
        sys.exit(1)

    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment}")
    print(f"API Version: {api_version}")
    print("Calling Azure OpenAI...")

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint.rstrip("/"),
        )

        # Use Responses API for 2025.x or when explicitly requested
        use_responses_api = api_version.startswith("2025") or os.getenv(
            "AZURE_OPENAI_USE_RESPONSES_API", ""
        ).lower() in ("1", "true", "yes")

        if use_responses_api:
            response = client.responses.create(
                model=deployment,
                input="Say hello in one short sentence.",
            )
            answer = response.output_text
        else:
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            )
            answer = response.choices[0].message.content

        print(f"OK - Response: {answer}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
