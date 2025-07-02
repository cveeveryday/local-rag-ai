#!/usr/bin/env python3
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        print("✅ Health check passed")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_query():
    try:
        payload = {"question": "What are the main topics in the documents?"}
        response = requests.post(f"{BASE_URL}/query", json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Query successful: {result['answer'][:100000]}...")
        return True
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG Application...")
    tests_passed = test_health() and test_query()
    sys.exit(0 if tests_passed else 1)
