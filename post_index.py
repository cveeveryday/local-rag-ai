#!/usr/bin/env python3
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def index_documents():
    try:
        payload = {}
        response = requests.post(f"{BASE_URL}/index", json=payload)
        response.raise_for_status()
        result = response.json()
        if not result.get("success"):
            raise Exception("Indexing failed") 
        else:
            print("✅ Indexing documents successful")
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Indexing of Documents...")
    docs_indexed = index_documents()
    sys.exit(0 if docs_indexed else 1)
