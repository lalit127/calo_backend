# backend/app/services/supabase_service.py
import os
import uuid
import httpx
import logging
from datetime import datetime, date, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

SUPABASE_URL      = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY      = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")


def _headers(token: str = None) -> dict:
    key = token or SUPABASE_KEY
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


class SupabaseService:

    # ── Auth: Verify Flutter's Supabase JWT ──────────────────────────────────

    async def verify_token(self, token: str) -> Optional[dict]:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(
                    f"{SUPABASE_URL}/auth/v1/user",
                    headers={
                        "apikey":        SUPABASE_ANON_KEY,
                        "Authorization": f"Bearer {token}",
                    }
                )
            if r.status_code == 200:
                return r.json()
            logger.warning(f"Token verify failed: {r.status_code} {r.text}")
            return None
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None

    # ── User Profile ──────────────────────────────────────────────────────────

    async def get_user_profile(self, user_id: str) -> Optional[dict]:
        """
        Gets user profile. If row doesn't exist yet, creates it first.
        This fixes the 404 error for brand-new users after OTP login.
        """
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"{SUPABASE_URL}/rest/v1/users",
                headers=_headers(),
                params={"id": f"eq.{user_id}", "select": "*"},
            )

        if r.status_code == 200:
            data = r.json()
            if data:
                return data[0]

        # ✅ Row doesn't exist — create a minimal profile so GET /users/me
        # never returns 404 for a freshly signed-up user
        logger.info(f"Profile not found for {user_id}, creating default row")
        try:
            created = await self._create_default_profile(user_id)
            return created
        except Exception as e:
            logger.error(f"Failed to create default profile: {e}")
            return None

    async def _create_default_profile(self, user_id: str) -> dict:
        """
        Creates a minimal user row using the auth user's email from Supabase.
        Called automatically when profile is missing.
        """
        # Fetch email from Supabase auth
        email = ""
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(
                    f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                    headers={
                        "apikey":        SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                    }
                )
            if r.status_code == 200:
                email = r.json().get("email", "")
        except Exception:
            pass

        payload = {
            "id":             user_id,
            "email":          email,
            "name":           email.split("@")[0] if email else "",
            "goal_calories":  2000,
            "goal_protein":   150.0,
            "goal_carbs":     250.0,
            "goal_fat":       65.0,
            "goal_water_ml":  2500,
        }

        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                f"{SUPABASE_URL}/rest/v1/users",
                headers={
                    **_headers(),
                    "Prefer": "resolution=merge-duplicates,return=representation",
                },
                json=payload,
            )

        if r.status_code in (200, 201):
            data = r.json()
            return data[0] if data else payload

        logger.error(f"Create profile failed: {r.status_code} {r.text}")
        return payload

    async def update_user_profile(self, user_id: str, updates: dict) -> dict:
        """
        Updates user profile. Uses upsert so it works even if the row
        doesn't exist yet (e.g. called right after onboarding completes).
        """
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        # ✅ Use upsert (POST with merge-duplicates) instead of PATCH
        # so this never fails with an empty response for new users
        payload = {"id": user_id, **updates}

        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                f"{SUPABASE_URL}/rest/v1/users",
                headers={
                    **_headers(),
                    "Prefer": "resolution=merge-duplicates,return=representation",
                },
                json=payload,
            )

        if r.status_code in (200, 201):
            data = r.json()
            if data:
                return data[0]

        # Fallback: try regular PATCH
        logger.warning(f"Upsert returned {r.status_code}, trying PATCH")
        async with httpx.AsyncClient(timeout=10) as c:
            r2 = await c.patch(
                f"{SUPABASE_URL}/rest/v1/users",
                headers=_headers(),
                params={"id": f"eq.{user_id}"},
                json=updates,
            )

        if r2.status_code in (200, 204):
            data = r2.json()
            if data:
                return data[0]
            # 204 = success but no body — fetch the row
            return await self.get_user_profile(user_id) or updates

        r2.raise_for_status()
        return updates

    async def upsert_user_profile(self, user_id: str, email: str,
                                   name: str = None) -> dict:
        payload = {
            "id":    user_id,
            "email": email,
            "name":  name or email.split("@")[0],
        }
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                f"{SUPABASE_URL}/rest/v1/users",
                headers={
                    **_headers(),
                    "Prefer": "resolution=merge-duplicates,return=representation",
                },
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        return data[0] if data else payload

    # ── Food Image Storage ────────────────────────────────────────────────────

    async def upload_food_image(self, image_bytes: bytes,
                                 user_id: str,
                                 mime_type: str) -> tuple[str, str]:
        ext  = mime_type.split("/")[-1].replace("jpeg", "jpg")
        path = f"{user_id}/{uuid.uuid4()}.{ext}"

        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{SUPABASE_URL}/storage/v1/object/food-images/{path}",
                headers={
                    "apikey":        SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type":  mime_type,
                    "x-upsert":      "true",
                },
                content=image_bytes,
            )

        if r.status_code in (200, 201):
            public_url = (
                f"{SUPABASE_URL}/storage/v1/object/public/food-images/{path}"
            )
            return public_url, path

        logger.warning(f"Image upload failed {r.status_code}: {r.text}")
        return None, None

    # ── Food Logs ─────────────────────────────────────────────────────────────

    async def insert_food_log(self, data: dict) -> dict:
        if not data.get("logged_at"):
            data["logged_at"] = datetime.now(timezone.utc).isoformat()

        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                f"{SUPABASE_URL}/rest/v1/food_logs",
                headers=_headers(),
                json=data,
            )
        r.raise_for_status()
        result = r.json()
        return result[0] if result else data

    async def get_daily_nutrition(self, user_id: str,
                                   date_str: Optional[str] = None) -> dict:
        target = date.fromisoformat(date_str) if date_str else date.today()
        start  = datetime(target.year, target.month, target.day,
                          tzinfo=timezone.utc).isoformat()
        end    = (datetime(target.year, target.month, target.day,
                           tzinfo=timezone.utc) + timedelta(days=1)).isoformat()

        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"{SUPABASE_URL}/rest/v1/food_logs",
                headers=_headers(),
                params={
                    "user_id":    f"eq.{user_id}",
                    "logged_at":  f"gte.{start}",
                    "and":        f"(logged_at.lt.{end})",
                    "deleted_at": "is.null",
                    "select":     "*",
                    "order":      "logged_at.asc",
                },
            )

        entries  = r.json() if r.status_code == 200 else []
        profile  = await self.get_user_profile(user_id) or {}
        water    = await self.get_today_water(user_id)

        meals: dict = {"breakfast": [], "lunch": [], "dinner": [], "snack": []}
        for e in entries:
            mt = e.get("meal_type", "snack")
            if mt in meals:
                meals[mt].append(e)

        return {
            "date":           target.isoformat(),
            "total_calories": sum(e.get("calories", 0) for e in entries),
            "total_protein":  round(sum(e.get("protein_g", 0) for e in entries), 1),
            "total_carbs":    round(sum(e.get("carbs_g", 0) for e in entries), 1),
            "total_fat":      round(sum(e.get("fat_g", 0) for e in entries), 1),
            "total_fiber":    round(sum(e.get("fiber_g", 0) for e in entries), 1),
            "total_water_ml": water.get("total_ml", 0),
            "goal_calories":  profile.get("goal_calories", 2000),
            "goal_protein":   profile.get("goal_protein", 150),
            "goal_carbs":     profile.get("goal_carbs", 250),
            "goal_fat":       profile.get("goal_fat", 65),
            "goal_water_ml":  profile.get("goal_water_ml", 2500),
            "meals":          meals,
            "entries":        entries,
        }

    async def get_weekly_stats(self, user_id: str) -> dict:
        today = date.today()
        start = datetime(today.year, today.month, today.day,
                         tzinfo=timezone.utc) - timedelta(days=6)

        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"{SUPABASE_URL}/rest/v1/food_logs",
                headers=_headers(),
                params={
                    "user_id":    f"eq.{user_id}",
                    "logged_at":  f"gte.{start.isoformat()}",
                    "deleted_at": "is.null",
                    "select":     "calories,protein_g,logged_at",
                },
            )

        logs    = r.json() if r.status_code == 200 else []
        profile = await self.get_user_profile(user_id) or {}
        goal_cal = profile.get("goal_calories", 2000)

        days_data = []
        for i in range(6, -1, -1):
            d = today - timedelta(days=i)
            day_logs = [
                l for l in logs
                if l.get("logged_at", "").startswith(d.isoformat())
            ]
            cal = sum(l.get("calories", 0) for l in day_logs)
            days_data.append({
                "date":     d.isoformat(),
                "calories": cal,
                "protein":  round(sum(l.get("protein_g", 0) for l in day_logs), 1),
                "logged":   cal > 0,
                "goal_met": cal >= goal_cal * 0.8,
            })

        calories_list = [d["calories"] for d in days_data if d["calories"] > 0]

        return {
            "days":           days_data,
            "avg_calories":   round(sum(calories_list) / len(calories_list), 1) if calories_list else 0,
            "avg_protein":    round(sum(d["protein"] for d in days_data) / 7, 1),
            "total_calories": sum(d["calories"] for d in days_data),
            "streak":         self._calc_streak(days_data),
        }

    async def get_food_history(self, user_id: str,
                                page: int, limit: int) -> list:
        offset = (page - 1) * limit
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"{SUPABASE_URL}/rest/v1/food_logs",
                headers=_headers(),
                params={
                    "user_id":    f"eq.{user_id}",
                    "deleted_at": "is.null",
                    "select":     "*",
                    "order":      "logged_at.desc",
                    "limit":      limit,
                    "offset":     offset,
                },
            )
        return r.json() if r.status_code == 200 else []

    async def soft_delete_food_log(self, log_id: str, user_id: str):
        async with httpx.AsyncClient(timeout=10) as c:
            await c.patch(
                f"{SUPABASE_URL}/rest/v1/food_logs",
                headers=_headers(),
                params={"id": f"eq.{log_id}", "user_id": f"eq.{user_id}"},
                json={"deleted_at": datetime.now(timezone.utc).isoformat()},
            )

    # ── Water ─────────────────────────────────────────────────────────────────

    async def insert_water_log(self, user_id: str, amount_ml: int) -> dict:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                f"{SUPABASE_URL}/rest/v1/water_logs",
                headers=_headers(),
                json={
                    "user_id":   user_id,
                    "amount_ml": amount_ml,
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        r.raise_for_status()
        return r.json()[0] if r.json() else {}

    async def get_today_water(self, user_id: str) -> dict:
        today = date.today()
        start = datetime(today.year, today.month, today.day,
                         tzinfo=timezone.utc).isoformat()

        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"{SUPABASE_URL}/rest/v1/water_logs",
                headers=_headers(),
                params={
                    "user_id":   f"eq.{user_id}",
                    "logged_at": f"gte.{start}",
                    "select":    "amount_ml",
                },
            )

        logs  = r.json() if r.status_code == 200 else []
        total = sum(l.get("amount_ml", 0) for l in logs)

        profile = await self.get_user_profile(user_id) or {}
        goal    = profile.get("goal_water_ml", 2500)

        return {
            "total_ml":   total,
            "goal_ml":    goal,
            "percentage": round(total / goal * 100, 1) if goal else 0,
        }

    # ── Weight ────────────────────────────────────────────────────────────────

    async def insert_weight_log(self, user_id: str,
                                 weight_kg: float,
                                 note: str = None) -> dict:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                f"{SUPABASE_URL}/rest/v1/weight_logs",
                headers=_headers(),
                json={
                    "user_id":    user_id,
                    "weight_kg":  weight_kg,
                    "note":       note,
                    "logged_at":  datetime.now(timezone.utc).isoformat(),
                },
            )
        await self.update_user_profile(user_id, {"weight_kg": weight_kg})
        r.raise_for_status()
        return r.json()[0] if r.json() else {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _calc_streak(self, days: list) -> int:
        streak = 0
        for d in reversed(days):
            if d.get("logged"):
                streak += 1
            else:
                break
        return streak