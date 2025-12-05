#!/usr/bin/env python3
"""
API Diagnostic Tool - Tests failing LLM API endpoints
"""

import os
import json
import requests
from typing import Dict, Optional

class APITester:
    def __init__(self):
        self.results = {}
        
    def test_anthropic(self, api_key: str) -> Dict:
        """Test Anthropic Claude API"""
        print("\n🔍 Testing Anthropic API...")
        
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}]
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            return {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text[:200]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def test_google_gemini(self, api_key: str) -> Dict:
        """Test Google Gemini API"""
        print("\n🔍 Testing Google Gemini API...")
        
        # Try multiple endpoint variations
        endpoints = [
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
            f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}",
        ]
        
        payload = {
            "contents": [{
                "parts": [{"text": "Hi"}]
            }]
        }
        
        for i, url in enumerate(endpoints, 1):
            try:
                print(f"  Trying endpoint {i}/{len(endpoints)}...")
                response = requests.post(url, json=payload, timeout=30)
                if response.status_code == 200:
                    return {
                        "status": "success",
                        "endpoint": url.split("?")[0],
                        "status_code": response.status_code
                    }
                else:
                    print(f"    ❌ Status {response.status_code}")
            except Exception as e:
                print(f"    ❌ Error: {str(e)[:50]}")
                
        return {"status": "error", "message": "All endpoints failed"}
    
    def test_cohere(self, api_key: str) -> Dict:
        """Test Cohere API"""
        print("\n🔍 Testing Cohere API...")
        
        # Try both .com and .ai domains, v1 and v2
        endpoints = [
            "https://api.cohere.com/v2/chat",
            "https://api.cohere.ai/v2/chat",
            "https://api.cohere.com/v1/chat",
            "https://api.cohere.ai/v1/chat",
        ]
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        for i, url in enumerate(endpoints, 1):
            payload = {
                "model": "command-r-plus",
                "message": "Hi" if "v2" in url else None,
                "messages": [{"role": "USER", "content": "Hi"}] if "v1" in url else None
            }
            payload = {k: v for k, v in payload.items() if v is not None}
            
            try:
                print(f"  Trying endpoint {i}/{len(endpoints)}...")
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                if response.status_code == 200:
                    return {
                        "status": "success",
                        "endpoint": url,
                        "status_code": response.status_code
                    }
                else:
                    print(f"    ❌ Status {response.status_code}: {response.text[:100]}")
            except Exception as e:
                print(f"    ❌ Error: {str(e)[:50]}")
                
        return {"status": "error", "message": "All endpoints failed"}
    
    def test_fireworks(self, api_key: str) -> Dict:
        """Test Fireworks AI API"""
        print("\n🔍 Testing Fireworks AI API...")
        
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            return {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "response": response.text[:200]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def test_cerebras(self, api_key: str) -> Dict:
        """Test Cerebras API"""
        print("\n🔍 Testing Cerebras API...")
        
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3.1-70b",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            return {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "response": response.text[:200]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def test_endpoint_reachability(self):
        """Test if endpoints are reachable"""
        print("\n🌐 Testing endpoint reachability...")
        
        endpoints = [
            "https://api.anthropic.com",
            "https://generativelanguage.googleapis.com",
            "https://api.cohere.com",
            "https://api.cohere.ai",
            "https://api.fireworks.ai",
            "https://api.cerebras.ai"
        ]
        
        results = {}
        for endpoint in endpoints:
            try:
                response = requests.head(endpoint, timeout=5)
                results[endpoint] = {
                    "reachable": True,
                    "status_code": response.status_code
                }
                print(f"  ✅ {endpoint}: {response.status_code}")
            except Exception as e:
                results[endpoint] = {
                    "reachable": False,
                    "error": str(e)[:50]
                }
                print(f"  ❌ {endpoint}: {str(e)[:50]}")
        
        return results
    
    def run_diagnostics(self):
        """Run all diagnostic tests"""
        print("=" * 60)
        print("🧪 API DIAGNOSTIC TOOL")
        print("=" * 60)
        
        # Check for environment variables
        api_keys = {
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY"),
            "fireworks": os.getenv("FIREWORKS_API_KEY"),
            "cerebras": os.getenv("CEREBRAS_API_KEY")
        }
        
        print("\n📋 API Key Status:")
        for provider, key in api_keys.items():
            status = "✅ Found" if key else "❌ Missing"
            masked_key = f"{key[:8]}...{key[-4:]}" if key and len(key) > 12 else "Not set"
            print(f"  {provider.capitalize()}: {status} ({masked_key})")
        
        # Test endpoint reachability
        self.results["reachability"] = self.test_endpoint_reachability()
        
        # Test each API if key is available
        if api_keys["anthropic"]:
            self.results["anthropic"] = self.test_anthropic(api_keys["anthropic"])
        else:
            print("\n⚠️  Skipping Anthropic test (no API key)")
            
        if api_keys["google"]:
            self.results["google"] = self.test_google_gemini(api_keys["google"])
        else:
            print("\n⚠️  Skipping Google test (no API key)")
            
        if api_keys["cohere"]:
            self.results["cohere"] = self.test_cohere(api_keys["cohere"])
        else:
            print("\n⚠️  Skipping Cohere test (no API key)")
            
        if api_keys["fireworks"]:
            self.results["fireworks"] = self.test_fireworks(api_keys["fireworks"])
        else:
            print("\n⚠️  Skipping Fireworks test (no API key)")
            
        if api_keys["cerebras"]:
            self.results["cerebras"] = self.test_cerebras(api_keys["cerebras"])
        else:
            print("\n⚠️  Skipping Cerebras test (no API key)")
        
        # Print summary
        self.print_summary()
        
        # Save results
        with open("api_diagnostic_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("\n💾 Results saved to api_diagnostic_results.json")
    
    def print_summary(self):
        """Print diagnostic summary"""
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        for provider in ["anthropic", "google", "cohere", "fireworks", "cerebras"]:
            if provider in self.results:
                result = self.results[provider]
                status = "✅ Working" if result.get("status") == "success" else "❌ Failed"
                print(f"\n{provider.upper()}: {status}")
                
                if result.get("status") == "error":
                    if "status_code" in result:
                        print(f"  Status Code: {result['status_code']}")
                    if "error" in result:
                        print(f"  Error: {result['error']}")
                    if "message" in result:
                        print(f"  Message: {result['message']}")
                elif result.get("status") == "success":
                    if "endpoint" in result:
                        print(f"  Working Endpoint: {result['endpoint']}")

if __name__ == "__main__":
    tester = APITester()
    tester.run_diagnostics()
