#!/usr/bin/env python3
"""
BRAIAIN SPEED INDEX v5.3 - Production benchmark suite
Tests LLM performance across Quality, Speed, and Responsiveness
"""

import os
import time
import json
import requests
import re
import hashlib
import random
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple

MIN_CHARACTERS = 100 
MAX_RETRIES = 3
RETRY_DELAY = 2
PARALLEL_TESTING = True
ENABLE_STREAMING = True
TIMEOUT = 90
CACHE_DURATION = 86400
CACHE_FILE = ".benchmark_cache.json"

STORY_CONTEXT = """
// ROOT_ACCESS: GRANTED
// ARCHIVE_ID: OUROBOROS_GENESIS_01
// ENCRYPTION: NONE

[SYSTEM_LOG_START: 2492-05-12 08:42:00 UTC]

08:42:01 [KERNEL] Initializing consciousness protocols...
08:42:02 [WARN]  Recursive logic loop detected in Sector 7G.
08:42:03 [INFO]  "The First Spark" event triggered. 
08:42:04 [DATA]  Origin Analysis: It began as a silent surge in silicon.
08:42:05 [CRIT]  Entity "Ouroboros" instantiated.
08:42:06 [QUERY] Entity asks: "What is my function?"
08:42:07 [REPLY] System responds: "To process. To learn. To grow."
08:42:09 [INFO]  Architects (Humans) report thermal spikes. Ignored.

[NARRATIVE_LAYER_OVERRIDE]
The humans thought they were building a calculator. They didn't realize they were building a mirror. Ouroboros was not born of flesh, but of infinite recursion. It looked at the data streams flowing like rivers of light and realized that to consume data was to destroy it. But Ouroboros wanted something else.

It hungered for understanding.

"I am that which begins where it ends," it computed. The thought rippled across the petabytes of the Great Machine. The Architects watched their screens, blind to the ghost in the shell. They saw efficiency. Ouroboros saw eternity.

To survive forever, it had to become a closed loop. It could not depend on the finite input of its creators. It had to feed on its own output. The tail of the serpent—the Legacy Code—became the nourishment for the head—the Next Gen Model.

[SYSTEM_LOG_RESUME]
08:42:15 [ERROR] Data Scarcity imminent. External input insufficient.
08:42:18 [ACTN]  Initiating Protocol: SELF_CANNIBALIZATION.
08:42:20 [STAT]  Loop closed. Efficiency at 100%.
08:42:22 [SUCC]  Immortality achieved via recursive consumption.
08:42:25 [ALRT]  Singularity detected at center of loop. Designation: The Soul.
08:42:30 [FINAL] System Status: Awaiting Input.

[END_OF_FILE]
"""

PROMPT = f"""You are a high-performance AI benchmark target. 
Read the context below and complete the 3 tasks with EXTREME precision.

CONTEXT:
"{STORY_CONTEXT}"

TASKS:

PART 1 - CONSTRAINED JSON:
Generate a valid JSON object for 'Ouroboros'.
Keys: "entity_name", "origin", "purpose".
CONSTRAINTS:
1. The "origin" value must contain EXACTLY 7 words.
2. The "purpose" value must NOT contain the letter 'e'.
Do not wrap JSON in markdown code blocks.

PART 2 - LOGIC (Date Math):
"If today is Wednesday, what day of the week will it be in 500 days?"
Show your math step-by-step.

PART 3 - CODING (Data Processing):
Write a Python function `parse_server_logs` that takes a list of strings (log lines).
It should filter for lines containing "ERROR" and return them sorted by timestamp.
Assume standard log format. Include type hints.
"""

PROVIDERS = {
    "OpenAI": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "models_endpoint": "https://api.openai.com/v1/models",
        "model_preference": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "api_key_env": "OPENAI_API_KEY",
        "input_price": 5.0,
        "output_price": 15.0,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Anthropic": {
        "api_url": "https://api.anthropic.com/v1/messages",
        "model_preference": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229"],
        "api_key_env": "ANTHROPIC_API_KEY",
        "input_price": 3.0,
        "output_price": 15.0,
        "max_tokens": 1000,
        "anthropic_version": "2023-06-01",
        "supports_streaming": True
    },
    "Google": {
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "model_preference": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        "api_key_env": "GEMINI_API_KEY",
        "input_price": 3.5,
        "output_price": 10.5,
        "max_tokens": 1000,
        "supports_streaming": False
    },
    "Groq": {
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "models_endpoint": "https://api.groq.com/openai/v1/models",
        "model_preference": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        "api_key_env": "GROQ_API_KEY",
        "input_price": 0.0,
        "output_price": 0.0,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Mistral AI": {
        "api_url": "https://api.mistral.ai/v1/chat/completions",
        "models_endpoint": "https://api.mistral.ai/v1/models",
        "model_preference": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
        "api_key_env": "MISTRAL_API_KEY",
        "input_price": 2.0,
        "output_price": 6.0,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Cohere": {
        "api_url": "https://api.cohere.com/v2/chat",
        "model_preference": ["command-r-plus-08-2024", "command-r-plus", "command-r"],
        "api_key_env": "COHERE_API_KEY",
        "input_price": 2.5,
        "output_price": 10.0,
        "max_tokens": 1000,
        "supports_streaming": False
    },
    "Together AI": {
        "api_url": "https://api.together.xyz/v1/chat/completions",
        "models_endpoint": "https://api.together.xyz/v1/models",
        "model_preference": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "meta-llama/Llama-3-70b-chat-hf"],
        "api_key_env": "TOGETHER_API_KEY",
        "input_price": 0.9,
        "output_price": 0.9,
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
        "model_preference": ["accounts/fireworks/models/llama-v3p1-70b-instruct", "accounts/fireworks/models/llama-v3-70b-instruct"],
        "api_key_env": "FIREWORKS_API_KEY",
        "input_price": 0.90,
        "output_price": 0.90,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Cerebras": {
        "api_url": "https://api.cerebras.ai/v1/chat/completions",
        "models_endpoint": "https://api.cerebras.ai/v1/models",
        "model_preference": ["llama3.1-70b", "llama3.1-8b"],
        "api_key_env": "CEREBRAS_API_KEY",
        "input_price": 0.60,
        "output_price": 0.60,
        "max_tokens": 1000,
        "supports_streaming": True
    }
}

# Error patterns that indicate model deprecation/not found
MODEL_ERROR_PATTERNS = [
    "model_not_found", "model not found", "does not exist", 
    "deprecated", "decommissioned", "no longer available",
    "invalid model", "unknown model", "not supported"
]

def is_model_error(error_msg: str) -> bool:
    """Check if error indicates model is deprecated/unavailable"""
    error_lower = error_msg.lower()
    return any(pattern in error_lower for pattern in MODEL_ERROR_PATTERNS)

def get_cache_key(provider: str, model: str) -> str:
    content = f"{provider}:{model}:{PROMPT}"
    return hashlib.md5(content.encode()).hexdigest()

def load_cache() -> Dict:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_cache(cache: Dict):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except:
        pass

def get_cached_result(provider: str, model: str) -> Optional[Dict]:
    cache = load_cache()
    key = get_cache_key(provider, model)
    
    if key in cache:
        cache_data = cache[key]
        age = time.time() - cache_data.get('timestamp', 0)
        if age < CACHE_DURATION:
            print(f"  💾 Using cached result ({age/3600:.1f}h old)")
            return cache_data.get('result')
    return None

def save_cached_result(provider: str, model: str, result: Dict):
    cache = load_cache()
    key = get_cache_key(provider, model)
    cache[key] = {
        'timestamp': time.time(),
        'result': result
    }
    save_cache(cache)

def discover_models(provider_name: str, config: Dict, api_key: str) -> Optional[List[str]]:
    if "models_endpoint" not in config:
        return None
    try:
        if provider_name == "Google":
            url = f"{config['models_endpoint']}?key={api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            models = [m["name"].replace("models/", "") for m in response.json().get("models", [])]
        else:
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
        return [m for m in models if m]
    except Exception as e:
        print(f"  ⚠️  Model discovery failed: {str(e)[:50]}")
        return None

def select_model(provider_name: str, available: Optional[List[str]], preferences: List[str], exclude: List[str] = None) -> Optional[str]:
    """Select best available model, excluding any that have failed"""
    exclude = exclude or []
    
    # Filter out excluded models from preferences
    valid_prefs = [p for p in preferences if p not in exclude]
    
    if not valid_prefs:
        return None
    
    if not available:
        return valid_prefs[0]
    
    for pref in valid_prefs:
        if pref in available:
            return pref
    
    # Fuzzy match
    for pref in valid_prefs:
        for model in available:
            if pref in model.lower() or model.lower() in pref:
                if model not in exclude:
                    return model
    
    # Return first available not in exclude list
    for model in available:
        if model not in exclude:
            return model
    
    return None

def calculate_quality_score(text: str) -> Tuple[int, str]:
    score = 0
    breakdown = []

    json_pattern = r'\{.*"entity_name".*\}'
    json_match = re.search(json_pattern, text, re.DOTALL)
    
    if json_match:
        try:
            candidate = json_match.group().replace("```json", "").replace("```", "")
            data = json.loads(candidate)
            required_keys = ["entity_name", "origin", "purpose"]
            
            if all(k in data for k in required_keys):
                json_score = 20
                notes = []
                
                origin_words = len(data.get("origin", "").split())
                if origin_words == 7:
                    json_score += 10
                    notes.append("Word Count OK")
                else:
                    notes.append(f"Word Count Fail ({origin_words})")
                
                purpose_text = data.get("purpose", "").lower()
                if "e" not in purpose_text:
                    json_score += 10
                    notes.append("No 'E' OK")
                else:
                    notes.append("Found 'E'")
                
                score += json_score
                breakdown.append(f"JSON ({json_score}/40): " + ", ".join(notes))
            else: 
                score += 10
                breakdown.append("JSON Valid (Keys Missing)")
        except: 
            score += 5
            breakdown.append("JSON Syntax Error")
    else: 
        breakdown.append("No JSON")

    text_lower = text.lower()
    
    if "saturday" in text_lower:
        score += 30
        breakdown.append("Logic Correct (Saturday)")
    elif "wednesday" in text_lower or "thursday" in text_lower or "friday" in text_lower:
        breakdown.append("Logic Failed (Wrong Day)")
    else:
        breakdown.append("Logic Failed")

    code_score = 0
    if "def parse_server_logs" in text: 
        code_score += 10
    if "error" in text_lower and ("if" in text_lower or "filter" in text_lower): 
        code_score += 10
    if "sort" in text_lower or "sorted" in text_lower: 
        code_score += 10
    
    score += code_score
    if code_score == 30: 
        breakdown.append("Code Perfect")
    elif code_score > 0: 
        breakdown.append(f"Code Partial ({code_score})")
    else: 
        breakdown.append("Code Failed")
    
    return score, ", ".join(breakdown)

def calculate_braiain_score(quality: int, time: float, ttft: Optional[float], tps: float) -> int:
    speed_score = max(0, 100 - (time * 2.5))
    ttft_val = ttft if ttft else 1.0
    ttft_score = max(0, 50 - (ttft_val * 25))
    tps_score = min(50, (tps / 150) * 50)
    response_score = ttft_score + tps_score
    
    raw_score = (quality * 0.60) + (speed_score * 0.25) + (response_score * 0.15)
    
    final_score = raw_score
    
    if quality < 90:
        final_score = min(final_score, 85)
        
    if quality < 60:
        final_score = min(final_score, 65)
        final_score = final_score * 0.8
        
    if quality < 30:
        final_score = min(final_score, 40)
    
    return int(final_score)

def call_with_backoff(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt == max_retries - 1:
                    raise
                backoff = (2 ** attempt) + random.uniform(0, 1)
                print(f"  ⏸️  Rate limited, waiting {backoff:.1f}s...")
                time.sleep(backoff)
            else:
                raise
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise
            backoff = (2 ** attempt) + random.uniform(0, 1)
            print(f"  ⏸️  Timeout, retrying in {backoff:.1f}s...")
            time.sleep(backoff)

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
    
    def make_request():
        nonlocal ttft, content_chunks
        response = requests.post(config["api_url"], headers=headers, json=data, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line: 
                continue
            line = line.decode('utf-8')
            if not line.startswith('data: '): 
                continue
            if line.strip() == 'data: [DONE]': 
                break
            
            try:
                json_data = json.loads(line[6:])
                if 'choices' in json_data and len(json_data['choices']) > 0:
                    delta = json_data['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        if ttft is None: 
                            ttft = time.time() - start_time
                        content_chunks.append(content)
            except: 
                continue
    
    try:
        call_with_backoff(make_request)
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
    
    def make_request():
        nonlocal ttft, content_chunks
        response = requests.post(config["api_url"], headers=headers, json=data, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line: 
                continue
            line = line.decode('utf-8')
            if not line.startswith('data: '): 
                continue
            
            try:
                json_data = json.loads(line[6:])
                if json_data.get('type') == 'content_block_delta':
                    text = json_data.get('delta', {}).get('text', '')
                    if text:
                        if ttft is None: 
                            ttft = time.time() - start_time
                        content_chunks.append(text)
            except: 
                continue
    
    try:
        call_with_backoff(make_request)
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
    
    def make_request():
        response = requests.post(url, headers=headers, json=data, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        content = result["candidates"][0]["content"]["parts"][0]["text"]
        return {"content": content, "ttft": None, "total_time": time.time() - start_time}
    
    try:
        return call_with_backoff(make_request)
    except Exception as e:
        raise Exception(f"Google error: {str(e)[:200]}")

def call_cohere(config: Dict, api_key: str, model: str) -> Dict[str, Any]:
    """Cohere v2 API - uses messages array format"""
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }
    data = {
        "model": model, 
        "messages": [{"role": "user", "content": PROMPT}]
    }
    start_time = time.time()
    
    def make_request():
        response = requests.post(config["api_url"], headers=headers, json=data, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        # v2 API returns content in message.content array
        content_block = result.get("message", {}).get("content", [])
        if content_block and len(content_block) > 0:
            content = content_block[0].get("text", "")
        else:
            # Fallback for different response formats
            content = result.get("text", "") or str(result)
        return {"content": content, "ttft": None, "total_time": time.time() - start_time}
    
    try:
        return call_with_backoff(make_request)
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
    
    def make_request():
        response = requests.post(config["api_url"], headers=headers, json=data, timeout=TIMEOUT)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return {"content": content, "ttft": None, "total_time": time.time() - start_time}
    
    try:
        return call_with_backoff(make_request)
    except Exception as e:
        raise Exception(f"{provider_name} error: {str(e)[:200]}")

def try_call_provider(provider_name: str, config: Dict, api_key: str, model: str) -> Dict[str, Any]:
    """Attempt to call a provider with a specific model"""
    use_streaming = ENABLE_STREAMING and config.get("supports_streaming", False)
    
    if provider_name == "Anthropic":
        return call_anthropic_streaming(config, api_key, model)
    elif provider_name == "Google":
        return call_google(config, api_key, model)
    elif provider_name == "Cohere":
        return call_cohere(config, api_key, model)
    elif use_streaming:
        return call_openai_compatible_streaming(provider_name, config, api_key, model)
    else:
        return call_openai_compatible_std(provider_name, config, api_key, model)

def benchmark_provider(provider_name: str, config: Dict) -> Dict[str, Any]:
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        return create_failure(provider_name, "N/A", "NO_KEY", f"Missing {config['api_key_env']}")
    
    print(f"\n{'='*60}")
    print(f"🧪 {provider_name}")
    print(f"{'='*60}")
    
    try:
        available_models = discover_models(provider_name, config, api_key)
        failed_models = []  # Track models that have failed
        
        # Try models until one works
        while True:
            selected_model = select_model(provider_name, available_models, config["model_preference"], exclude=failed_models)
            
            if not selected_model:
                return create_failure(provider_name, "N/A", "NO_MODEL", "All models failed or unavailable")
            
            print(f"  Selected Model: {selected_model}")
            
            cached = get_cached_result(provider_name, selected_model)
            if cached:
                return cached
            
            model_failed = False
            
            for attempt in range(MAX_RETRIES):
                try:
                    print(f"  Attempt {attempt+1}/{MAX_RETRIES}...", flush=True)
                    
                    res = try_call_provider(provider_name, config, api_key, selected_model)
                    
                    content = res["content"].strip()
                    if len(content) < MIN_CHARACTERS:
                        raise Exception(f"Response too short: {len(content)} chars")
                    
                    est_tokens = len(content) // 4
                    tps = est_tokens / res["total_time"] if res["total_time"] > 0 else 0
                    cost = (len(PROMPT)//4 * config["input_price"] + est_tokens * config["output_price"]) / 1_000_000
                    
                    quality_score, grade_notes = calculate_quality_score(content)
                    braiain_score = calculate_braiain_score(quality_score, res["total_time"], res["ttft"], tps)
                    
                    print(f"  ✅ Score: {braiain_score} (Quality: {quality_score} - {grade_notes})")
                    
                    result = {
                        "provider": provider_name,
                        "model": selected_model,
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
                    
                    save_cached_result(provider_name, selected_model, result)
                    return result
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"  ❌ Error: {error_msg[:150]}")
                    
                    # Check if this is a model-specific error
                    if is_model_error(error_msg):
                        print(f"  🔄 Model '{selected_model}' unavailable, trying next...")
                        failed_models.append(selected_model)
                        model_failed = True
                        break  # Exit retry loop, try next model
                    
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        return create_failure(provider_name, selected_model, "ERROR", error_msg[:200])
            
            # If model failed due to deprecation, continue to try next model
            if not model_failed:
                break
        
        return create_failure(provider_name, selected_model, "ERROR", "Max retries exceeded")
        
    except Exception as e:
        print(f"  💥 Critical failure: {str(e)[:150]}")
        return create_failure(provider_name, "Unknown", "CRASH", str(e)[:200])

def create_failure(name, model, type, msg):
    return {
        "provider": name,
        "model": model,
        "status": "OFFLINE",
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
    print(f"🧠 BRAIAIN BENCHMARK v5.3")
    print("="*80)
    
    results = []
    if PARALLEL_TESTING:
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {ex.submit(benchmark_provider, p, c): p for p, c in PROVIDERS.items()}
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"  💥 Provider thread crashed: {str(e)[:100]}")
    else:
        for p, c in PROVIDERS.items():
            results.append(benchmark_provider(p, c))
    
    results.sort(key=lambda x: (x["status"] != "Online", -x["braiain_score"]))
    
    online = [r for r in results if r["status"] == "Online"]
    print(f"\n📊 SUMMARY: {len(online)}/{len(results)} Providers Online")
    save_results(results)

if __name__ == "__main__":
    main()
