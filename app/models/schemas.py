# backend/app/models/schemas.py
from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


class AnalysisResult(BaseModel):
    food_name: str
    cuisine_type: str = "unknown"
    is_indian_food: bool = False
    calories: int
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0
    fiber_g: float = 0
    sugar_g: float = 0
    sodium_mg: float = 0
    portion_size: str = ""
    description: str = ""
    ingredients_detected: List[str] = []
    cooking_method: Optional[str] = None
    confidence: float = 0.9
    from_cache: bool = False        # ✅ added — cache hit flag from mistral_service
    serving_grams: Optional[int] = None  # ✅ added — returned by mistral_service


class AnalyzeTextRequest(BaseModel):
    text: str
    cuisine_hint: Optional[str] = None


class FoodLogCreate(BaseModel):
    food_name: str
    meal_type: str
    calories: int
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0
    fiber_g: float = 0
    portion_size: Optional[str] = None
    logged_at: Optional[datetime] = None


class FoodLogResponse(BaseModel):
    id: str
    user_id: str
    food_name: str
    meal_type: str
    cuisine_type: str = "unknown"
    calories: int
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0
    fiber_g: float = 0
    image_url: Optional[str] = None
    is_indian_food: bool = False
    ai_confidence: Optional[float] = None
    logged_at: Any
    created_at: Any

    class Config:
        from_attributes = True


class UserProfile(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    activity_level: Optional[str] = "moderate"
    goal: Optional[str] = "maintain"
    goal_calories: int = 2000
    goal_protein: float = 150
    goal_carbs: float = 250
    goal_fat: float = 65
    goal_water_ml: int = 2500
    cuisine_preference: str = "both"
    profile_image_url: Optional[str] = None

    class Config:
        from_attributes = True


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    activity_level: Optional[str] = None
    goal: Optional[str] = None
    goal_calories: Optional[int] = None
    goal_protein: Optional[float] = None
    goal_carbs: Optional[float] = None
    goal_fat: Optional[float] = None
    goal_water_ml: Optional[int] = None
    cuisine_preference: Optional[str] = None


class WaterLogCreate(BaseModel):
    amount_ml: int


class WeightLogCreate(BaseModel):
    weight_kg: float
    note: Optional[str] = None


class DailyNutrition(BaseModel):
    date: str
    total_calories: int
    total_protein: float
    total_carbs: float
    total_fat: float
    goal_calories: int
    goal_protein: float
    goal_carbs: float
    goal_fat: float
    entries: List[Any] = []


class WeeklyStats(BaseModel):
    days: List[Any]
    avg_calories: float
    total_calories: int
    streak: int