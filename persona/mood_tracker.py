# persona/mood_tracker.py
"""
Enhanced mood tracking system using the neutral hormone API.
NOW with proper mood calculation logic moved into mood_tracker.py!
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Import from neutral API module to avoid circular imports
from .hormone_api import (
    load_hormone_levels,
    save_hormone_levels, 
    load_mood_weights,
    get_mood_context
)

MOOD_HISTORY_FILE = Path("persona/mood_history.json")

def load_mood_history() -> list:
    """Load mood history and create file if not present."""
    if not MOOD_HISTORY_FILE.exists():
        try:
            MOOD_HISTORY_FILE.parent.mkdir(exist_ok=True)
            with open(MOOD_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            print("[Mood History]: Created default mood history file")
        except Exception as e:
            print(f"[Mood history init error]: {e}")
        return []
    try:
        with open(MOOD_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Mood history load error]: {e}")
        return []

def save_mood_history(history: list):
    """Save mood history."""
    try:
        MOOD_HISTORY_FILE.parent.mkdir(exist_ok=True)
        with open(MOOD_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[Mood history save error]: {e}")

def calculate_mood_from_hormones(hormone_levels: Dict[str, float]) -> tuple:
    """
    FIXED: Calculate mood from hormone levels using deviation-based logic.
    This is the CORRECTED mood calculation that was broken in hormone_api.py
    """
    print(f"[Mood Calculator]: Processing hormones {hormone_levels}")
    
    # Define mood patterns based on hormone deviations from baseline (0.5)
    dopamine = hormone_levels.get("dopamine", 0.5)
    serotonin = hormone_levels.get("serotonin", 0.5)
    cortisol = hormone_levels.get("cortisol", 0.5) 
    oxytocin = hormone_levels.get("oxytocin", 0.5)
    
    # Calculate deviations from baseline
    dopa_dev = dopamine - 0.5
    sero_dev = serotonin - 0.5
    cort_dev = cortisol - 0.5
    oxy_dev = oxytocin - 0.5
    
    print(f"[Mood Calculator]: Deviations - dopa:{dopa_dev:+.3f}, sero:{sero_dev:+.3f}, cort:{cort_dev:+.3f}, oxy:{oxy_dev:+.3f}")
    
    # Mood determination based on dominant patterns
    mood_scores = {}
    
    # High cortisol patterns (stress/anxiety)  
    if cort_dev > 0.1:
        if sero_dev < -0.05:  # High cortisol + Low serotonin
            mood_scores["anxious"] = abs(cort_dev) + abs(sero_dev) * 0.7
        if dopa_dev < -0.05:  # High cortisol + Low dopamine  
            mood_scores["restless"] = abs(cort_dev) + abs(dopa_dev) * 0.6
        if cort_dev > 0.15:  # Very high cortisol
            mood_scores["stressed"] = abs(cort_dev) * 1.2
    
    # Low serotonin patterns (sadness/depression)
    if sero_dev < -0.08:
        if dopa_dev < -0.05:  # Low serotonin + Low dopamine
            mood_scores["depressed"] = abs(sero_dev) + abs(dopa_dev) * 0.8
        elif cort_dev > 0.05:  # Low serotonin + High cortisol  
            mood_scores["melancholic"] = abs(sero_dev) + abs(cort_dev) * 0.6
        else:
            mood_scores["sad"] = abs(sero_dev) * 1.1
    
    # High dopamine patterns (joy/excitement)
    if dopa_dev > 0.08:
        if oxy_dev > 0.05:  # High dopamine + High oxytocin
            mood_scores["euphoric"] = dopa_dev + oxy_dev * 0.8
        elif sero_dev > 0.05:  # High dopamine + High serotonin
            mood_scores["cheerful"] = dopa_dev + sero_dev * 0.7
        else:
            mood_scores["energetic"] = dopa_dev * 1.2
    
    # High oxytocin patterns (love/affection)
    if oxy_dev > 0.1:
        if dopa_dev > 0.05:  # High oxytocin + High dopamine
            mood_scores["loving"] = oxy_dev + dopa_dev * 0.6
        elif sero_dev > 0.03:  # High oxytocin + High serotonin
            mood_scores["affectionate"] = oxy_dev + sero_dev * 0.8
        else:
            mood_scores["caring"] = oxy_dev * 1.1
    
    # Balanced positive states
    if dopa_dev > 0.03 and sero_dev > 0.03 and cort_dev < 0.1:
        mood_scores["content"] = (dopa_dev + sero_dev) * 0.8
    
    # Mixed/complex states
    if abs(dopa_dev) > 0.05 and abs(sero_dev) > 0.05 and abs(cort_dev) > 0.05:
        mood_scores["conflicted"] = (abs(dopa_dev) + abs(sero_dev) + abs(cort_dev)) * 0.4
    
    print(f"[Mood Calculator]: Mood scores calculated - {mood_scores}")
    
    # Find the highest scoring mood
    if not mood_scores:
        print("[Mood Calculator]: No strong patterns detected, defaulting to neutral")
        return "neutral", 0.5
    
    # Get the mood with highest score
    top_mood = max(mood_scores.items(), key=lambda x: x[1])
    mood_name = top_mood[0]
    raw_intensity = top_mood[1]
    
    # Scale intensity to 0.0-1.0 range (cap at 1.0)
    intensity = min(1.0, max(0.1, raw_intensity))
    
    print(f"[Mood Calculator]: Selected mood '{mood_name}' with intensity {intensity:.3f}")
    
    return mood_name, intensity

def update_mood(new_mood: str, intensity: float, reason: str = "", hormone_context: Dict = None):
    """Update current mood and log to history with enhanced context."""
    # Get mood context if not provided
    if hormone_context is None:
        hormone_context = get_mood_context(new_mood, intensity)
    
    # Update mood_adjustments.json with enhanced data
    mood_data = {
        "current_mood": new_mood,
        "intensity": intensity,  
        "context": hormone_context,
        "last_updated": datetime.utcnow().isoformat()
    }

    try:
        with open("persona/mood_adjustments.json", "w", encoding="utf-8") as f:
            json.dump(mood_data, f, indent=2)
    except Exception as e:
        print(f"[Mood update error]: {e}")
        return

    # Log to history with enhanced information
    history = load_mood_history()
    history_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "mood": new_mood,
        "intensity": intensity,
        "reason": reason,
        "is_hybrid": hormone_context.get("is_hybrid", False),
        "is_emergent": hormone_context.get("is_emergent", False),
        "stability": hormone_context.get("stability", "medium")
    }
    
    # Add hormone levels snapshot for debugging contextual changes
    if reason and ("contextual" in reason or "hormone_event" in reason):
        history_entry["hormone_snapshot"] = load_hormone_levels()
    
    history.append(history_entry)

    # Keep only last 100 mood changes
    if len(history) > 100:
        history = history[-100:]

    save_mood_history(history)
    
    # Print mood change for debugging
    mood_type = ""
    if hormone_context.get("is_hybrid"):
        mood_type = " [HYBRID]"
    elif hormone_context.get("is_emergent"):
        mood_type = " [EMERGENT]"
    
    print(f"[Mood Update]: {new_mood}{mood_type} (intensity: {intensity:.2f}) - {reason}")

def get_current_mood() -> Dict[str, Any]:
    """Get current mood settings with enhanced context."""
    try:
        with open("persona/mood_adjustments.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Current mood load error]: {e}")
        return {
            "current_mood": "neutral", 
            "intensity": 0.5,
            "context": {"is_hybrid": False, "is_emergent": False, "stability": "medium"},
            "last_updated": datetime.utcnow().isoformat()
        }

def update_mood_from_hormones(reason="hormonal_shift"):
    """
    FIXED: Calculate mood from current hormone levels and update mood accordingly.
    Now uses our own mood calculation instead of the broken hormone_api function.
    """
    print(f"[Mood From Hormones]: Starting mood update for reason '{reason}'")
    
    # Load current hormone levels
    hormone_levels = load_hormone_levels()
    print(f"[Mood From Hormones]: Current hormones - {hormone_levels}")
    
    # Use OUR fixed mood calculation (not the broken one from hormone_api)
    mood, intensity = calculate_mood_from_hormones(hormone_levels)
    print(f"[Mood From Hormones]: Calculated mood '{mood}' with intensity {intensity:.3f}")
    
    # Get enhanced context
    context = get_mood_context(mood, intensity)
    
    # Update mood with context
    update_mood(mood, intensity, reason, context)
    
    return mood, intensity, context

def handle_event_and_update_mood(event: str):
    """
    Adjust hormones based on an event, then update mood from new hormone state.
    """
    print(f"[Event Trigger]: Processing event '{event}'")
    
    # Import locally to avoid circular import
    from .hormone_adjuster import adjust_hormones
    
    # First adjust hormones based on the event
    new_hormones = adjust_hormones(event)
    print(f"[Hormone Update]: New levels - {new_hormones}")
    
    # Then update mood based on new hormone levels
    mood, intensity, context = update_mood_from_hormones(reason=f"hormone_event:{event}")
    
    print(f"[Event Processed]: {event} -> Mood: {mood} ({intensity:.2f})")
    
    return mood, intensity, context, new_hormones

def apply_sentiment_to_mood(conversation_text: str):
    """
    Enhanced sentiment analysis using contextual hormone adjustments.
    """
    print(f"[Enhanced Sentiment]: Processing input '{conversation_text}'")
    
    # Import locally to avoid circular import
    from .hormone_adjuster import apply_contextual_hormone_adjustments
    
    # Use the enhanced contextual hormone adjustment system
    new_hormones = apply_contextual_hormone_adjustments(conversation_text)
    print(f"[Enhanced Sentiment]: New hormone levels - {new_hormones}")
    
    # Update mood based on the new hormone levels using OUR fixed calculation
    mood, intensity, context = update_mood_from_hormones(reason="contextual_analysis")
    
    print(f"[Enhanced Sentiment Complete]: {mood} ({intensity:.2f}) from '{conversation_text}'")
    
    return mood, intensity, context

def get_mood_summary() -> Dict[str, Any]:
    """
    Get comprehensive summary of current mood state including hormone levels.
    """
    current_mood_data = get_current_mood()
    hormone_levels = load_hormone_levels()
    history = load_mood_history()
    recent_moods = [entry["mood"] for entry in history[-10:]] if history else []
    hybrid_count = sum(1 for entry in history[-20:] if entry.get("is_hybrid", False)) if history else 0 
    emergent_count = sum(1 for entry in history[-20:] if entry.get("is_emergent", False)) if history else 0
    
    summary = {
        "current_state": current_mood_data,
        "hormone_levels": hormone_levels,
        "recent_patterns": {
            "recent_moods": recent_moods,
            "hybrid_states_count": hybrid_count,
            "emergent_states_count": emergent_count,
            "total_mood_changes": len(history)
        },
        "complexity_indicators": {
            "has_undefined_states": hybrid_count > 0 or emergent_count > 0,
            "mood_volatility": "high" if len(set(recent_moods)) > 6 else "medium" if len(set(recent_moods)) > 3 else "low"
        }
    }
    return summary

def force_mood_recalculation():
    """
    Force a complete recalculation of mood from current hormone levels.
    """
    print("[Debug]: Forcing mood recalculation from hormones...")
    return update_mood_from_hormones(reason="manual_recalculation")

def simulate_hormone_fluctuation():
    """
    Simulate natural hormone fluctuation over time.
    """
    import random
    hormones = load_hormone_levels()
    print(f"[Hormone Fluctuation]: Before - {hormones}")
    for hormone in hormones:
        drift = random.uniform(-0.02, 0.02)
        baseline_pull = (0.5 - hormones[hormone]) * 0.01
        hormones[hormone] += drift + baseline_pull
        hormones[hormone] = max(0.0, min(1.0, hormones[hormone]))
    print(f"[Hormone Fluctuation]: After - {hormones}")
    save_hormone_levels(hormones)
    return update_mood_from_hormones(reason="natural_fluctuation")

# Debug function for testing mood calculation
def test_mood_calculation():
    """Test the mood calculation with different hormone scenarios."""
    test_cases = [
        {"name": "High Stress", "hormones": {"dopamine": 0.3, "serotonin": 0.2, "cortisol": 0.8, "oxytocin": 0.4}},
        {"name": "Happy Love", "hormones": {"dopamine": 0.8, "serotonin": 0.7, "cortisol": 0.3, "oxytocin": 0.9}},
        {"name": "Sadness", "hormones": {"dopamine": 0.2, "serotonin": 0.1, "cortisol": 0.6, "oxytocin": 0.3}},
        {"name": "Neutral", "hormones": {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5}},
        {"name": "Anger", "hormones": {"dopamine": 0.4, "serotonin": 0.3, "cortisol": 0.9, "oxytocin": 0.2}}
    ]
    
    print("\n🧪 TESTING MOOD CALCULATION")
    print("=" * 50)
    
    for case in test_cases:
        print(f"\n📝 Test: {case['name']}")
        print(f"   Hormones: {case['hormones']}")
        mood, intensity = calculate_mood_from_hormones(case['hormones'])
        print(f"   Result: {mood} (intensity: {intensity:.2f})")
    
    print("\n" + "=" * 50)
    print("✅ Mood calculation test complete!")

if __name__ == "__main__":
    test_mood_calculation()