# backend/main.py
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import logging
import os

from app.services.supabase_service import SupabaseService
from app.services.mistral_service import analyze_food_image, analyze_food_text
from app.models.schemas import (
    FoodLogCreate, FoodLogResponse, AnalysisResult,
    DailyNutrition, WeeklyStats, UpdateProfileRequest,
    AnalyzeTextRequest, UserProfile, WaterLogCreate, WeightLogCreate
)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Cal AI Backend starting...")
    yield
    logger.info("👋 Cal AI Backend shutting down")


app = FastAPI(
    title="Cal AI Backend",
    description="Food tracking API — Supabase Auth + Mistral AI",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = SupabaseService()

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES    = int(os.getenv("MAX_IMAGE_SIZE_MB", "10")) * 1024 * 1024


# ── Auth dependency ───────────────────────────────────────────────────────────
async def get_current_user(authorization: str = Header(...)) -> dict:
    """
    Verify Supabase JWT token from Flutter app.
    Flutter sends: Authorization: Bearer <supabase_access_token>
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header")
    token = authorization.replace("Bearer ", "")
    user = await supabase.verify_token(token)
    if not user:
        raise HTTPException(401, "Invalid or expired token")
    return user


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0"}


# ═══════════════════════════════════════════════════════════════════════════════
#  USER PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/users/me", response_model=UserProfile)
async def get_profile(user: dict = Depends(get_current_user)):
    profile = await supabase.get_user_profile(user["id"])
    if not profile:
        raise HTTPException(404, "Profile not found")
    return UserProfile(**profile)


@app.patch("/users/me", response_model=UserProfile)
async def update_profile(
    body: UpdateProfileRequest,
    user: dict = Depends(get_current_user)
):
    updated = await supabase.update_user_profile(
        user["id"],
        body.model_dump(exclude_none=True)
    )
    return UserProfile(**updated)


# ═══════════════════════════════════════════════════════════════════════════════
#  FOOD — ANALYZE (no save, just AI analysis)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/food/analyze/image", response_model=AnalysisResult)
async def analyze_image(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Analyze food image with Mistral AI — returns nutrition, does NOT log it"""
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(400, f"Invalid file type: {file.content_type}")

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(400, "Image too large")

    try:
        result = await analyze_food_image(data, file.content_type)
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(502, f"AI analysis failed: {str(e)}")

    return AnalysisResult(**result)


@app.post("/food/analyze/text", response_model=AnalysisResult)
async def analyze_text_food(
    body: AnalyzeTextRequest,
    user: dict = Depends(get_current_user),
):
    """Analyze food by text description"""
    profile = await supabase.get_user_profile(user["id"])
    hint = body.cuisine_hint or (profile or {}).get("cuisine_preference", "indian")

    try:
        result = await analyze_food_text(body.text, hint)
    except Exception as e:
        raise HTTPException(502, f"AI analysis failed: {str(e)}")

    return AnalysisResult(**result)


# ═══════════════════════════════════════════════════════════════════════════════
#  FOOD — LOG (analyze + save image + save to Supabase)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/food/log/image", response_model=FoodLogResponse, status_code=201)
async def log_with_image(
    file: UploadFile = File(...),
    meal_type: str   = Form(...),
    logged_at: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
):
    """
    Full flow: upload image → Mistral AI analyzes → 
    saves image to Supabase Storage → saves log to Supabase DB
    """
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(400, "Invalid image type")

    img_bytes = await file.read()
    if len(img_bytes) > MAX_BYTES:
        raise HTTPException(400, "Image too large")

    # 1. AI analysis
    try:
        nutrition = await analyze_food_image(img_bytes, file.content_type)
    except Exception as e:
        raise HTTPException(502, f"AI analysis failed: {str(e)}")

    # 2. Upload image to Supabase Storage
    image_url, image_path = await supabase.upload_food_image(
        img_bytes, user["id"], file.content_type
    )

    # 3. Save food log to Supabase
    log_data = {
        "user_id":       user["id"],
        "food_name":     nutrition.get("food_name", "Unknown"),
        "meal_type":     meal_type,
        "cuisine_type":  nutrition.get("cuisine_type", "unknown"),
        "calories":      nutrition.get("calories", 0),
        "protein_g":     nutrition.get("protein_g", 0),
        "carbs_g":       nutrition.get("carbs_g", 0),
        "fat_g":         nutrition.get("fat_g", 0),
        "fiber_g":       nutrition.get("fiber_g", 0),
        "sugar_g":       nutrition.get("sugar_g", 0),
        "sodium_mg":     nutrition.get("sodium_mg", 0),
        "portion_size":  nutrition.get("portion_size"),
        "image_url":     image_url,
        "image_path":    image_path,
        "ai_confidence": nutrition.get("confidence"),
        "is_indian_food": nutrition.get("is_indian_food", False),
        "ingredients":   nutrition.get("ingredients_detected", []),
        "cooking_method": nutrition.get("cooking_method"),
        "logged_at":     logged_at,
    }

    saved = await supabase.insert_food_log(log_data)
    return FoodLogResponse(**saved)


@app.post("/food/log", response_model=FoodLogResponse, status_code=201)
async def log_manual(
    body: FoodLogCreate,
    user: dict = Depends(get_current_user),
):
    """Manual food entry — no image"""
    log_data = {
        "user_id": user["id"],
        **body.model_dump(exclude_none=True),
        "is_manual_entry": True,
    }
    saved = await supabase.insert_food_log(log_data)
    return FoodLogResponse(**saved)


# ═══════════════════════════════════════════════════════════════════════════════
#  FOOD — RETRIEVE
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/food/daily")
async def get_daily(
    date: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    return await supabase.get_daily_nutrition(user["id"], date)


@app.get("/food/weekly")
async def get_weekly(user: dict = Depends(get_current_user)):
    return await supabase.get_weekly_stats(user["id"])


@app.get("/food/history")
async def get_history(
    page: int = 1,
    limit: int = 20,
    user: dict = Depends(get_current_user),
):
    return await supabase.get_food_history(user["id"], page, limit)


@app.delete("/food/log/{log_id}")
async def delete_log(
    log_id: str,
    user: dict = Depends(get_current_user),
):
    await supabase.soft_delete_food_log(log_id, user["id"])
    return {"message": "Deleted"}


# ═══════════════════════════════════════════════════════════════════════════════
#  WATER & WEIGHT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/users/me/water", status_code=201)
async def log_water(
    body: WaterLogCreate,
    user: dict = Depends(get_current_user),
):
    return await supabase.insert_water_log(user["id"], body.amount_ml)


@app.get("/users/me/water/today")
async def get_today_water(user: dict = Depends(get_current_user)):
    return await supabase.get_today_water(user["id"])


@app.post("/users/me/weight", status_code=201)
async def log_weight(
    body: WeightLogCreate,
    user: dict = Depends(get_current_user),
):
    return await supabase.insert_weight_log(user["id"], body.weight_kg, body.note)