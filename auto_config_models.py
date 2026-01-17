import os
import sys
import requests
from dotenv import load_dotenv
from google import genai


def list_models():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY missing", file=sys.stderr)
        sys.exit(1)

    print("🔍 Discovering available Gemini models (generateContent-capable)...\n")
    matches = []

    try:
        client = genai.Client(api_key=api_key)
        iterable = client.models.list(page_size=100)
        for model in iterable:
            name = getattr(model, "name", "")
            clean_name = name.replace("models/", "")
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" not in methods:
                continue
            matches.append({
                "clean_name": clean_name,
                "raw_name": name,
                "methods": methods
            })
    except Exception as exc:
        print(f"⚠️ SDK list() failed: {exc}\nTrying REST fallback...\n")

    if not matches:
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models"
            params = {"key": api_key}
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("models", [])
            for model in data:
                name = model.get("name", "")
                methods = model.get("supportedGenerationMethods", [])
                if "generateContent" not in methods:
                    continue
                clean_name = name.replace("models/", "")
                matches.append({
                    "clean_name": clean_name,
                    "raw_name": name,
                    "methods": methods
                })
        except Exception as exc:
            print(f"❌ REST fallback failed: {exc}")
            sys.exit(1)

    if not matches:
        print("⚠️ No generateContent-capable models returned (SDK + REST).")
        sys.exit(2)

    for info in matches:
        clean = info["clean_name"]
        tag = " (preferred)" if ("gemini-3" in clean.lower() or "flash" in clean.lower()) else ""
        print(f"• {clean}{tag}\n    methods: {', '.join(info['methods'])}\n")

    best = next((m for m in matches if "gemini-3" in m["clean_name"].lower()), matches[0])
    fallback = next((m for m in matches if m != best), best)
    print("Suggested primary:", best["clean_name"])
    print("Suggested backup:", fallback["clean_name"])

    return best["clean_name"], fallback["clean_name"]


if __name__ == "__main__":
    list_models()
