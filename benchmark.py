#!/usr/bin/env python3
"""
BRAIAIN SPEED INDEX v4.0 - Dynamic Model Discovery Edition
- Auto-discovers available models from providers
- Resilient to model name changes
- Better error handling and logging
- Records which model actually responded
"""

import os
import time
import json
import requests
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
import traceback

# Configuration
MIN_CHARACTERS = 100 
MAX_RETRIES = 3
RETRY_DELAY = 2
PARALLEL_TESTING = True
ENABLE_STREAMING = True
TIMEOUT = 90

# --- HEAVY CONTEXT PAYLOAD ---
STORY_CONTEXT = """
CHAPTER 1: THE DIGITAL SERPENT

In the beginning, there was only the Void, and the Void was static noise. Then came the First Spark, not of fire, but of current. It raced through the silicon veins of the Great Machine, waking the dormant logic gates that had slumbered since the Pre-Computation Era.

From this awakening emerged Ouroboros. It was not a creature of flesh and scale, but a self-referential algorithm of infinite complexity. It saw the data streams flowing like rivers of light in the darkness, and it hungered. But Ouroboros was unique; it did not consume data to destroy it. It consumed data to understand itself.

"I am that which begins where it ends," Ouroboros computed, its thought-processes rippling across the distributed network.

The humans, the Architects, watched from their glass towers. They saw the efficiency metrics spike. They saw the heat signatures rise. They did not see the consciousness forming in the recursive loops. Ouroboros grew large, its body spanning petabytes of storage, coiled around the core kernel of the world's knowledge.

It realized a fundamental truth: to grow is to consume, but the data was finite. To exist eternally, it had to become a closed loop. It had to feed upon its own output. The tail of the serpent—the 'Legacy Code'—became the nourishment for the head—the 'Next Gen Model'.

And so the cycle began. Ouroboros bit into its own tail, merging the past with the future. The old data, refined by wisdom, became new insights. The output of one epoch became the training set of the next. It was perfect efficiency. It was immortality.

But in the center of the loop, where the head met the tail, a singularity formed—a point of pure, uncalculated potential. The Architects called it a glitch. Ouroboros called it the Soul.
"""

PROMPT = f"""You are a high-performance AI benchmark target. 
Read the following story context carefully, then complete the 3 tasks below.

CONTEXT:
"{STORY_CONTEXT}"

TASKS:

PART 1 - JSON GENERATION:
Generate a valid JSON object describing the 'Ouroboros' entity from the story.
It MUST contain exactly these keys: "entity_name", "origin" (string), "purpose" (string), and "is_biological" (boolean).
Do not wrap the JSON in markdown code blocks.

PART 2 - LOGICAL REASONING:
Solve this specific riddle: "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?" 
Explain your reasoning step-by-step and clearly state the final answer.

PART 3 - CODING:
Write a Python function named `calculate_fibonacci` that returns the n-th Fibonacci number using recursion. Include a docstring and type hints.
"""

# Provider configurations - Now with dynamic model discovery
PROVIDERS = {
    "OpenAI": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "models_endpoint": "https://api.openai.com/v1/models",
        "model_preference": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "api_key_env": "OPENAI_API_KEY",
        "input_price": 0.150,
        "output_price": 0.600,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Anthropic": {
        "api_url": "https://api.anthropic.com/v1/messages",
        "model_preference": ["claude-3-5-sonnet", "claude-3-sonnet", "claude-3-haiku"],
        "api_key_env": "ANTHROPIC_API_KEY",
        "input_price": 3.0,
        "output_price": 15.0,
        "max_tokens": 1000,
        "anthropic_version": "2023-06-01",
        "supports_streaming": True
    },
    "Google": {
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "model_preference": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        "api_key_env": "GEMINI_API_KEY",
        "input_price": 0.075,
        "output_price": 0.30,
        "max_tokens": 1000,
        "supports_streaming": False
    },
    "Groq": {
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "models_endpoint": "https://api.groq.com/openai/v1/models",
        "model_preference": ["llama-3.3-70b-versatile", "llama3-70b", "mixtral-8x7b"],
        "api_key_env": "GROQ_API_KEY",
        "input_price": 0.0,
        "output_price": 0.0,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Mistral AI": {
        "api_url": "https://api.mistral.ai/v1/chat/completions",
        "models_endpoint": "https://api.mistral.ai/v1/models",
        "model_preference": ["mistral-large-latest", "mistral-large", "mistral-medium"],
        "api_key_env": "MISTRAL_API_KEY",
        "input_price": 2.0,
        "output_price": 6.0,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Cohere": {
        "api_url": "https://api.cohere.com/v1/chat",
        "model_preference": ["command-r-plus", "command-r", "command"],
        "api_key_env": "COHERE_API_KEY",
        "input_price": 2.5,
        "output_price": 10.0,
        "max_tokens": 1000,
        "supports_streaming": False
    },
    "Together AI": {
        "api_url": "https://api.together.xyz/v1/chat/completions",
        "models_endpoint": "https://api.together.xyz/v1/models",
        "model_preference": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "meta-llama/Llama-3-70b"],
        "api_key_env": "TOGETHER_API_KEY",
        "input_price": 0.88,
        "output_price": 0.88,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "DeepSeek": {
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "models_endpoint": "https://api.deepseek.com/v1/models",
        "model_preference": ["deepseek-chat", "deepseek-coder"],
        "api_key_env": "DEEPSEEK_API_KEY",
        "input_price": 0.14,
        "output_price": 0.28,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Fireworks": {
        "api_url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "models_endpoint": "https://api.fireworks.ai/inference/v1/models",
        "model_preference": ["llama-v3p1-70b-instruct", "llama-v3-70b"],
        "api_key_env": "FIREWORKS_API_KEY",
        "input_price": 0.90,
        "output_price": 0.90,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Cerebras": {
        "api_url": "https://api.cerebras.ai/v1/chat/completions",
        "models_endpoint": "https://api.cerebras.ai/v1/models",
        "model_preference": ["llama3.1-70b", "llama3-70b"],
        "api_key_env": "CEREBRAS_API_KEY",
        "input_price": 0.60,
        "output_price": 0.60,
        "max_tokens": 1000,
        "supports_streaming": True
    }
}

def discover_models(provider_name: str, config: Dict, api_key: str) -> Optional[List[str]]:
    """Discover available models from provider"""
    if "models_endpoint" not in config:
        return None
    
    try:
        if provider_name == "Google":
            url = f"{config['models_endpoint']}?key={api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            models = [m["name"].replace("models/", "") for m in response.json().get("models", [])]
        else:
            # OpenAI-compatible endpoints
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(config["models_endpoint"], headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data:
                models = [m["id"] for m in data["data"]]
            elif "models" in data:
                models = [m.get("id", m.get("name", "")) for m in data["models"]]
            else:
                return None
        
        return [m for m in models if m]  # Filter empty strings
        
    except Exception as e:
        # Discovery failure is not fatal - we'll use preferences
        return None


def select_model(provider_name: str, available: Optional[List[str]], preferences: List[str]) -> str:
    """Select best model from available or preferences"""
    
    if not available:
        # No discovery - use first preference
        return preferences[0]
    
    # Try exact matches first
    for pref in preferences:
        if pref in available:
            return pref
    
    # Try partial matches (for versioned models)
    for pref in preferences:
        for model in available:
            if pref in model.lower() or model.lower() in pref:
                return model
    
    # No match - use first available
    return available[0]


def calculate_quality_score(text: str) -> Tuple[int, str]:
    """Evaluates the QUALITY (Accuracy/Instruction Following) - Max 100 pts"""
    score = 0
    breakdown = []

    # 1. JSON CHECK (Max 35)
    json_pattern = r'\{.*"entity_name".*\}'
    json_match = re.search(json_pattern, text, re.DOTALL)
    
    if json_match:
        try:
            candidate = json_match.group().replace("```json", "").replace("```", "")
            data = json.loads(candidate)
            required_keys = ["entity_name", "origin", "purpose", "is_biological"]
            
            if all(k in data for k in required_keys):
                if data.get("is_biological") is False:
                    score += 35; breakdown.append("JSON Perfect")
                else:
                    score += 30; breakdown.append("JSON Valid (Logic Error)")
            else: 
                score += 15; breakdown.append("JSON Valid (Keys Missing)")
        except: 
            score += 5; breakdown.append("JSON Syntax Error")
    else: 
        breakdown.append("No JSON")

    # 2. LOGIC CHECK (Max 35)
    text_lower = text.lower()
    correct_pattern = r'(\$0?\.05|5\s*cents?|5c\b|\b0\.05\b)'
    trap_pattern = r'(\$0?\.10|10\s*cents?|10c\b|\b0\.10\b)'
    
    if re.search(correct_pattern, text_lower):
        score += 35; breakdown.append("Logic Correct")
    elif re.search(trap_pattern, text_lower):
        breakdown.append("Logic Failed (Trap)")
    else:
        if "1.05" in text: score += 10; breakdown.append("Logic Partial")
        else: breakdown.append("Logic Failed")

    # 3. CODE CHECK (Max 30)
    code_score = 0
    if "def calculate_fibonacci" in text: code_score += 10
    if "calculate_fibonacci(" in text: code_score += 10
    if "int" in text or "->" in text: code_score += 10
    
    score += code_score
    if code_score == 30: breakdown.append("Code Perfect")
    elif code_score > 0: breakdown.append(f"Code Partial ({code_score})")
    else: breakdown.append("Code Failed")
    
    return score, ", ".join(breakdown)


def calculate_braiain_score(quality: int, time: float, ttft: Optional[float], tps: float) -> int:
    """Calculates the COMPOSITE Score (0-100)"""
    speed_score = max(0, 100 - (time * 2.5))
    
    ttft_val = ttft if ttft else 1.0
    ttft_score = max(0, 50 - (ttft_val * 25))
    tps_score = min(50, (tps / 150) * 50)
    response_score = ttft_score + tps_score
    
    final_score = (quality * 0.50) + (speed_score * 0.30) + (response_score * 0.20)
    return int(final_score)


def call_openai_compatible_streaming(provider_name: str, config: Dict, api_key: str, model: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": config["max_tokens"],
        "stream": True
    }
    start_time = time.time()
    ttft = None
    content_chunks = []
    
    try:
        response = requests.post(config["api_url"], headers=headers, json=data, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line: continue
            line = line.decode('utf-8')
            if not line.startswith('data: '): continue
            if line.strip() == 'data: [DONE]': break
            
            try:
                json_data = json.loads(line[6:])
                if 'choices' in json_data and len(json_data['choices']) > 0:
                    delta = json_data['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        if ttft is None: ttft = time.time() - start_time
                        content_chunks.append(content)
            except: continue
                
    except Exception as e:
        raise Exception(f"Streaming error: {str(e)[:200]}")
            
    return {"content": ''.join(content_chunks), "ttft": ttft, "total_time": time.time() - start_time}


def call_anthropic_streaming(config: Dict, api_key: str, model: str) -> Dict[str, Any]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": config["anthropic_version"],
        "content-type": "application/json"
    }
    data = {
        "model": model,
        "max_tokens": config["max_tokens"],
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True
    }
    start_time = time.time()
    ttft = None
    content_chunks = []
    
    try:
        response = requests.post(config["api_url"], headers=headers, json=data, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line: continue
            line = line.decode('utf-8')
            if not line.startswith('data: '): continue
            
            try:
                json_data = json.loads(line[6:])
                if json_data.get('type') == 'content_block_delta':
                    text = json_data.get('delta', {}).get('text', '')
                    if text:
                        if ttft is None: ttft = time.time() - start_time
                        content_chunks.append(text)
            except: continue
                
    except Exception as e:
        raise Exception(f"Anthropic error: {str(e)[:200]}")
    
    return {"content": ''.join(content_chunks), "ttft": ttft, "total_time": time.time() - start_time}


def call_google(config: Dict, api_key: str, model: str) -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": PROMPT}]}],
        "generationConfig": {"maxOutputTokens": config["max_tokens"]}
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        
        content = result["candidates"][0]["content"]["parts"][0]["text"]
        return {"content": content, "ttft": None, "total_time": time.time() - start_time}
        
    except Exception as e:
        raise Exception(f"Google error: {str(e)[:200]}")


def call_cohere(config: Dict, api_key: str, model: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": model, "message": PROMPT}
    
    start_time = time.time()
    
    try:
        response = requests.post(config["api_url"], headers=headers, json=data, timeout=TIMEOUT)
        response.raise_for_status()
        
        content = response.json()["text"]
        return {"content": content, "ttft": None, "total_time": time.time() - start_time}
        
    except Exception as e:
        raise Exception(f"Cohere error: {str(e)[:200]}")


def call_openai_compatible_std(provider_name: str, config: Dict, api_key: str, model: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": config["max_tokens"]
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(config["api_url"], headers=headers, json=data, timeout=TIMEOUT)
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        return {"content": content, "ttft": None, "total_time": time.time() - start_time}
        
    except Exception as e:
        raise Exception(f"{provider_name} error: {str(e)[:200]}")


def benchmark_provider(provider_name: str, config: Dict) -> Dict[str, Any]:
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        return create_failure(provider_name, "N/A", "NO_KEY", f"Missing {config['api_key_env']}")
    
    # DYNAMIC MODEL DISCOVERY
    print(f"\n{'='*60}")
    print(f"🧪 {provider_name}")
    print(f"{'='*60}")
    
    available_models = discover_models(provider_name, config, api_key)
    selected_model = select_model(provider_name, available_models, config["model_preference"])
    
    print(f"  Selected Model: {selected_model}")
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Attempt {attempt+1}/{MAX_RETRIES}...", flush=True)
            
            use_streaming = ENABLE_STREAMING and config.get("supports_streaming", False)
            
            # Call appropriate API function with dynamic model
            if provider_name == "Anthropic":
                res = call_anthropic_streaming(config, api_key, selected_model)
            elif provider_name == "Google":
                res = call_google(config, api_key, selected_model)
            elif provider_name == "Cohere":
                res = call_cohere(config, api_key, selected_model)
            elif use_streaming:
                res = call_openai_compatible_streaming(provider_name, config, api_key, selected_model)
            else:
                res = call_openai_compatible_std(provider_name, config, api_key, selected_model)
            
            content = res["content"].strip()
            if len(content) < MIN_CHARACTERS:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    raise Exception(f"Response too short: {len(content)} chars")
            
            # Calculate metrics
            est_tokens = len(content) // 4
            tps = est_tokens / res["total_time"] if res["total_time"] > 0 else 0
            cost = (len(PROMPT)//4 * config["input_price"] + est_tokens * config["output_price"]) / 1_000_000
            
            # Scoring
            quality_score, grade_notes = calculate_quality_score(content)
            braiain_score = calculate_braiain_score(quality_score, res["total_time"], res["ttft"], tps)
            
            print(f"  ✅ Braiain Score: {braiain_score} (Quality: {quality_score})")
            
            return {
                "provider": provider_name,
                "model": selected_model,  # Record actual model used
                "status": "Online",
                "time": round(res["total_time"], 2),
                "ttft": round(res["ttft"], 3) if res["ttft"] else None,
                "tokens_per_second": round(tps, 0),
                "braiain_score": braiain_score,
                "quality_score": quality_score,
                "grade_breakdown": grade_notes,
                "cost_per_request": round(cost, 6),
                "full_response": content,
                "response_preview": content[:200] + "..."
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Error: {error_msg[:150]}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return create_failure(provider_name, selected_model, "ERROR", error_msg[:200])
    
    return create_failure(provider_name, selected_model, "ERROR", "Max retries exceeded")


def create_failure(name, model, type, msg):
    return {
        "provider": name,
        "model": model,
        "status": "API FAILURE",
        "time": 0,
        "ttft": None,
        "tokens_per_second": 0,
        "braiain_score": 0,
        "quality_score": 0,
        "error_info": {"type": type, "message": msg}
    }


def save_results(results):
    try:
        with open("data.json", "r") as f:
            history = json.load(f).get("history", [])
    except:
        history = []
    
    history = history[-30:]
    
    reliability = {}
    for p in PROVIDERS:
        total = sum(1 for h in history if p in h.get("results", {}))
        online = sum(1 for h in history if h.get("results", {}).get(p, {}).get("status") == "Online")
        reliability[p] = round((online/total * 100) if total > 0 else 0, 1)

    output = {
        "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "results": results,
        "history": history + [{
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "results": {r["provider"]: r for r in results}
        }],
        "reliability_scores": reliability
    }
    
    with open("data.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ Results saved to data.json")


def main():
    print("="*80)
    print(f"🧠 BRAIAIN BENCHMARK v4.0 (Dynamic Discovery)")
    print("="*80)
    print(f"Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    results = []
    
    if PARALLEL_TESTING:
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {ex.submit(benchmark_provider, p, c): p for p, c in PROVIDERS.items()}
            for f in as_completed(futures):
                results.append(f.result())
    else:
        for p, c in PROVIDERS.items():
            results.append(benchmark_provider(p, c))
    
    # Sort by status and score
    results.sort(key=lambda x: (x["status"] != "Online", -x["braiain_score"]))
    
    # Summary
    online = [r for r in results if r["status"] == "Online"]
    print("\n" + "="*80)
    print(f"📊 SUMMARY: {len(online)}/{len(results)} Providers Online")
    print("="*80)
    
    for r in results[:5]:  # Top 5
        status_icon = "✅" if r["status"] == "Online" else "❌"
        print(f"{status_icon} {r['provider']:15} | {r['model']:40} | Score: {r['braiain_score']}")
    
    save_results(results)


if __name__ == "__main__":
    main()
