# backend/app/services/mistral_service.py
#
# SMART FOOD ANALYSIS FLOW:
#
#  User uploads image
#       ↓
#  AI identifies food NAME only  (cheap — minimal tokens)
#       ↓
#  Check food_database table in Supabase
#       ↓
#  FOUND? → return stored nutrients instantly  ⚡ (zero AI cost)
#  NOT FOUND? → full AI nutrition analysis → save to food_database 💾
#
# Result: common foods (dal, roti, biryani etc.) never hit AI again after first time.

import os
import base64
import httpx
import json
import logging
import re

logger = logging.getLogger(__name__)

AI_PROVIDER     = os.getenv("AI_PROVIDER", "mistral").lower()
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY    = os.getenv("SUPABASE_SERVICE_KEY", "")


# ── Prompts ───────────────────────────────────────────────────────────────────

# Step 1 prompt: CHEAP — just identify name, no nutrition needed
NAME_ONLY_PROMPT = """Look at this food image. Identify what food it is.
Return ONLY this JSON, nothing else:
{
  "food_name": "exact food name",
  "cuisine_type": "indian/chinese/italian/american/etc",
  "is_indian_food": true/false,
  "portion_size": "e.g. 1 bowl, 1 plate, 2 pieces",
  "serving_grams": 200
}"""

# Step 2 prompt: FULL — only called when food NOT in database
FULL_NUTRITION_PROMPT = """You are a nutrition expert. Analyze this food image carefully.
Return ONLY this JSON, nothing else:
{
  "food_name": "exact food name",
  "cuisine_type": "indian/chinese/italian/american/etc",
  "is_indian_food": true/false,
  "calories": 000,
  "protein_g": 00.0,
  "carbs_g": 00.0,
  "fat_g": 00.0,
  "fiber_g": 0.0,
  "sugar_g": 0.0,
  "sodium_mg": 000.0,
  "portion_size": "e.g. 1 bowl (200g)",
  "serving_grams": 200,
  "description": "brief 1-sentence description",
  "ingredients_detected": ["ingredient1", "ingredient2"],
  "cooking_method": "fried/grilled/steamed/baked/etc",
  "confidence": 0.95
}

INDIAN FOOD REFERENCE (per standard serving):
Dal Tadka 200g=180cal P9 C28 F4 | Dal Makhani 200g=220cal P10 C30 F7
Palak Paneer 200g=280cal P14 C12 F20 | Butter Chicken 200g=320cal P26 C10 F20
Chicken Biryani 300g=450cal P28 C55 F12 | Veg Biryani 300g=380cal P10 C65 F8
Roti 30g=80cal P3 C15 F1 | Paratha 60g=150cal P4 C22 F5 | Rice 150g=200cal P4 C44 F0
Idli 2pc=130cal P4 C26 F0 | Masala Dosa 150g=220cal P5 C35 F7
Chole 200g=270cal P14 C40 F6 | Rajma 200g=250cal P13 C38 F4
Paneer Butter Masala 200g=350cal P16 C15 F25
Samosa 60g=150cal P3 C18 F7 | Pav Bhaji 250g=380cal P9 C58 F12"""

TEXT_PROMPT_TEMPLATE = """You are a nutrition expert specializing in Indian and global foods.
Analyze this food: "{food_text}"
Cuisine hint: {cuisine_hint}

Return ONLY this JSON:
{{
  "food_name": "exact food name",
  "cuisine_type": "indian/chinese/etc",
  "is_indian_food": true/false,
  "calories": 000,
  "protein_g": 00.0,
  "carbs_g": 00.0,
  "fat_g": 00.0,
  "fiber_g": 0.0,
  "sugar_g": 0.0,
  "sodium_mg": 000.0,
  "portion_size": "e.g. 1 bowl (200g)",
  "serving_grams": 200,
  "description": "brief description",
  "ingredients_detected": ["ingredient1"],
  "cooking_method": "cooking method",
  "confidence": 0.90
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — called by main.py
# ═══════════════════════════════════════════════════════════════════════════════

async def analyze_food_image(image_bytes: bytes, mime_type: str) -> dict:
    """
    Smart 2-step analysis:
    Step 1 → AI identifies food name only (cheap)
    Step 2 → Check Supabase food_database
      Hit  → return instantly, no more AI cost
      Miss → full AI analysis → save to database for next time
    """
    # ── Step 1: Identify name cheaply ────────────────────────────────────────
    logger.info(f"[Cache] Step 1: Identifying food name via {AI_PROVIDER}...")
    name_data = await _call_ai_vision(image_bytes, mime_type, NAME_ONLY_PROMPT)
    food_name = name_data.get("food_name", "")

    # ── Step 2: Check database ────────────────────────────────────────────────
    if food_name:
        logger.info(f"[Cache] Step 2: Looking up '{food_name}' in food_database...")
        cached = await _db_lookup(food_name)

        if cached:
            logger.info(f"[Cache] ✅ HIT — '{food_name}' found in database, skipping AI")
            await _db_increment_hit(cached["id"])
            return _build_result_from_cache(cached, name_data)

    # ── Step 3: Cache miss — full AI analysis ─────────────────────────────────
    logger.info(f"[Cache] ❌ MISS — '{food_name}' not in database, running full analysis")
    full_data = await _call_ai_vision(image_bytes, mime_type, FULL_NUTRITION_PROMPT)

    # ── Step 4: Save to database for next time ────────────────────────────────
    if full_data.get("food_name"):
        logger.info(f"[Cache] 💾 Saving '{full_data['food_name']}' to food_database")
        await _db_save(full_data)

    return full_data


async def analyze_food_text(text: str, cuisine_hint: str = "indian") -> dict:
    """
    Text analysis with same cache logic:
    Check DB first → hit = instant return, miss = AI + save
    """
    # Check cache
    cached = await _db_lookup(text)
    if cached:
        logger.info(f"[Cache] ✅ Text HIT for '{text}'")
        await _db_increment_hit(cached["id"])
        return _build_result_from_cache(cached, {})

    # AI analysis
    prompt = TEXT_PROMPT_TEMPLATE.format(food_text=text, cuisine_hint=cuisine_hint)
    result = await _call_ai_text(prompt)

    # Save to DB
    if result.get("food_name"):
        await _db_save(result)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  SUPABASE FOOD DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

async def _db_lookup(food_name: str) -> dict | None:
    """
    Look up food in Supabase — tries 3 strategies:
    1. Exact normalized name match
    2. Alias array contains match
    3. Partial name match (ilike)
    """
    if not food_name or not SUPABASE_URL or not SUPABASE_KEY:
        return None

    normalized = food_name.lower().strip()
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }

    try:
        async with httpx.AsyncClient(timeout=5) as c:

            # 1. Exact match
            r = await c.get(
                f"{SUPABASE_URL}/rest/v1/food_database",
                headers=headers,
                params={"food_name_normalized": f"eq.{normalized}", "select": "*", "limit": "1"},
            )
            if r.status_code == 200 and r.json():
                return r.json()[0]

            # 2. Alias match
            r = await c.get(
                f"{SUPABASE_URL}/rest/v1/food_database",
                headers=headers,
                params={"aliases": f'cs.{{"{normalized}"}}', "select": "*", "limit": "1"},
            )
            if r.status_code == 200 and r.json():
                return r.json()[0]

            # 3. Partial match — ordered by popularity
            r = await c.get(
                f"{SUPABASE_URL}/rest/v1/food_database",
                headers=headers,
                params={
                    "food_name_normalized": f"ilike.*{normalized}*",
                    "select": "*",
                    "order":  "hit_count.desc",
                    "limit":  "1",
                },
            )
            if r.status_code == 200 and r.json():
                return r.json()[0]

    except Exception as e:
        logger.warning(f"[Cache] DB lookup error: {e}")

    return None


async def _db_save(result: dict) -> None:
    """Save new food to database so future users get instant results"""
    if not SUPABASE_URL or not SUPABASE_KEY or not result.get("food_name"):
        return

    food_name = result.get("food_name", "Unknown")
    payload = {
        "food_name":            food_name,
        "food_name_normalized": food_name.lower().strip(),
        "cuisine_type":         result.get("cuisine_type", "unknown"),
        "is_indian_food":       result.get("is_indian_food", False),
        "calories":             result.get("calories", 0),
        "protein_g":            result.get("protein_g", 0),
        "carbs_g":              result.get("carbs_g", 0),
        "fat_g":                result.get("fat_g", 0),
        "fiber_g":              result.get("fiber_g", 0),
        "sugar_g":              result.get("sugar_g", 0),
        "sodium_mg":            result.get("sodium_mg", 0),
        "portion_size":         result.get("portion_size", "1 serving"),
        "serving_grams":        result.get("serving_grams", 100),
        "description":          result.get("description", ""),
        "ingredients":          result.get("ingredients_detected", []),
        "cooking_method":       result.get("cooking_method", ""),
        "ai_confidence":        result.get("confidence", 0.9),
        "hit_count":            1,
        "verified":             False,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(
                f"{SUPABASE_URL}/rest/v1/food_database",
                headers={
                    "apikey":        SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type":  "application/json",
                    "Prefer":        "resolution=ignore-duplicates",
                },
                json=payload,
            )
        logger.info(f"[Cache] 💾 Saved '{food_name}' to food_database")
    except Exception as e:
        logger.warning(f"[Cache] DB save error: {e}")


async def _db_increment_hit(food_id: str) -> None:
    """Track popularity — most-hit foods load fastest"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            await c.patch(
                f"{SUPABASE_URL}/rest/v1/food_database",
                headers={
                    "apikey":        SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type":  "application/json",
                },
                params={"id": f"eq.{food_id}"},
                json={"hit_count": "hit_count + 1"},
            )
    except Exception:
        pass  # non-critical


def _build_result_from_cache(cached: dict, detected: dict) -> dict:
    """Build standard result dict from cached database row"""
    return {
        "food_name":            cached.get("food_name"),
        "cuisine_type":         cached.get("cuisine_type", "unknown"),
        "is_indian_food":       cached.get("is_indian_food", False),
        "calories":             cached.get("calories", 0),
        "protein_g":            cached.get("protein_g", 0),
        "carbs_g":              cached.get("carbs_g", 0),
        "fat_g":                cached.get("fat_g", 0),
        "fiber_g":              cached.get("fiber_g", 0),
        "sugar_g":              cached.get("sugar_g", 0),
        "sodium_mg":            cached.get("sodium_mg", 0),
        # Use AI-detected portion if available, else fall back to DB value
        "portion_size":         detected.get("portion_size") or cached.get("portion_size", "1 serving"),
        "description":          cached.get("description", ""),
        "ingredients_detected": cached.get("ingredients", []),
        "cooking_method":       cached.get("cooking_method", ""),
        "confidence":           cached.get("ai_confidence", 0.9),
        "from_cache":           True,   # Flutter can show ⚡ "Instant" badge
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  AI PROVIDER CALLS
# ═══════════════════════════════════════════════════════════════════════════════

async def _call_ai_vision(image_bytes: bytes, mime_type: str, prompt: str) -> dict:
    """Route vision call to configured provider"""
    try:
        if AI_PROVIDER == "groq":
            return await _groq_vision(image_bytes, mime_type, prompt)
        elif AI_PROVIDER == "gemini":
            return await _gemini_vision(image_bytes, mime_type, prompt)
        else:
            return await _mistral_vision(image_bytes, mime_type, prompt)
    except Exception as e:
        logger.error(f"Vision AI call failed: {e}")
        return _fallback_result()


async def _call_ai_text(prompt: str) -> dict:
    """Route text call to configured provider"""
    try:
        if AI_PROVIDER == "groq":
            return await _groq_text(prompt)
        elif AI_PROVIDER == "gemini":
            return await _gemini_text(prompt)
        else:
            return await _mistral_text(prompt)
    except Exception as e:
        logger.error(f"Text AI call failed: {e}")
        return _fallback_result()


# ── Mistral ───────────────────────────────────────────────────────────────────
async def _mistral_vision(image_bytes: bytes, mime_type: str, prompt: str) -> dict:
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY not set in .env")
    b64 = base64.b64encode(image_bytes).decode()
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "pixtral-12b-2409",
                "messages": [{"role": "user", "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64}"}
                ]}],
                "max_tokens": 512, "temperature": 0.1,
            },
        )
        r.raise_for_status()
    return _parse_json(r.json()["choices"][0]["message"]["content"])


async def _mistral_text(prompt: str) -> dict:
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY not set in .env")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512, "temperature": 0.1,
            },
        )
        r.raise_for_status()
    return _parse_json(r.json()["choices"][0]["message"]["content"])


# ── Groq ──────────────────────────────────────────────────────────────────────
async def _groq_vision(image_bytes: bytes, mime_type: str, prompt: str) -> dict:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env")
    b64 = base64.b64encode(image_bytes).decode()
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
                ]}],
                "max_tokens": 512, "temperature": 0.1,
            },
        )
        r.raise_for_status()
    return _parse_json(r.json()["choices"][0]["message"]["content"])


async def _groq_text(prompt: str) -> dict:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512, "temperature": 0.1,
            },
        )
        r.raise_for_status()
    return _parse_json(r.json()["choices"][0]["message"]["content"])


# ── Gemini ────────────────────────────────────────────────────────────────────
async def _gemini_vision(image_bytes: bytes, mime_type: str, prompt: str) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env")
    b64 = base64.b64encode(image_bytes).decode()
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": b64}}
                ]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512}
            },
        )
        r.raise_for_status()
    return _parse_json(r.json()["candidates"][0]["content"]["parts"][0]["text"])


async def _gemini_text(prompt: str) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512}
            },
        )
        r.raise_for_status()
    return _parse_json(r.json()["candidates"][0]["content"]["parts"][0]["text"])


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_json(text: str) -> dict:
    """Robustly extract JSON from AI response text"""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    logger.warning(f"Could not parse AI response: {text[:200]}")
    return _fallback_result()


def _fallback_result() -> dict:
    return {
        "food_name": "Unknown Food", "cuisine_type": "unknown",
        "is_indian_food": False, "calories": 200,
        "protein_g": 5.0, "carbs_g": 30.0, "fat_g": 5.0,
        "fiber_g": 2.0, "sugar_g": 2.0, "sodium_mg": 200.0,
        "portion_size": "1 serving", "serving_grams": 100,
        "description": "Could not analyze food",
        "ingredients_detected": [], "cooking_method": "unknown",
        "confidence": 0.3,
    }