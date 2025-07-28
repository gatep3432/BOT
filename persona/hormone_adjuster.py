# persona/hormone_adjuster.py
"""
Enhanced hormone adjustment system using ML-based sentiment analysis.
Uses emotion_nsfw_checker pipeline for emotion detection instead of hardcoded patterns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Import from neutral API module to avoid circular imports
from .hormone_api import (
    load_hormone_levels,
    save_hormone_levels,
    load_mood_weights,
    infer_mood_from_hormones,
    get_mood_context
)

# Import emotion detection pipeline
from .emotion_nsfw_checker import detect_emotion, detect_toxicity, analyze_sentiment_confidence

# ---------------------- Emotion-to-Hormone Mapping --------------------- #
_EMOTION_HORMONE_MAP = {
    # Positive emotions
    "joy": {"dopamine": 0.12, "serotonin": 0.08},
    "love": {"oxytocin": 0.15, "dopamine": 0.08, "serotonin": 0.05},
    "admiration": {"dopamine": 0.06, "oxytocin": 0.04},
    "excitement": {"dopamine": 0.10, "cortisol": 0.02},
    "gratitude": {"serotonin": 0.08, "oxytocin": 0.06},
    "relief": {"cortisol": -0.08, "serotonin": 0.05},
    "pride": {"dopamine": 0.08, "serotonin": 0.04},
    "optimism": {"dopamine": 0.06, "serotonin": 0.06},
    "caring": {"oxytocin": 0.10, "serotonin": 0.04},
    "approval": {"dopamine": 0.05, "oxytocin": 0.03},
    
    # Negative emotions
    "anger": {"cortisol": 0.12, "dopamine": -0.05, "serotonin": -0.04},
    "sadness": {"serotonin": -0.10, "cortisol": 0.06, "dopamine": -0.04},
    "fear": {"cortisol": 0.15, "serotonin": -0.06},
    "disgust": {"cortisol": 0.08, "serotonin": -0.05},
    "annoyance": {"cortisol": 0.06, "serotonin": -0.03},
    "disappointment": {"dopamine": -0.08, "serotonin": -0.05},
    "embarrassment": {"cortisol": 0.08, "oxytocin": -0.04},
    "grief": {"serotonin": -0.12, "cortisol": 0.10},
    "nervousness": {"cortisol": 0.10, "dopamine": -0.03},
    "remorse": {"serotonin": -0.06, "cortisol": 0.05},
    
    # Neutral/Complex emotions
    "surprise": {"dopamine": 0.04, "cortisol": 0.02},
    "curiosity": {"dopamine": 0.05},
    "confusion": {"cortisol": 0.03},
    "neutral": {}  # No hormone changes
}

# ---------------------- Toxicity-to-Hormone Mapping ------------------- #
_TOXICITY_HORMONE_MAP = {
    "TOXIC": {"cortisol": 0.15, "serotonin": -0.08},
    "toxic": {"cortisol": 0.15, "serotonin": -0.08},
    "severe_toxic": {"cortisol": 0.20, "serotonin": -0.12, "dopamine": -0.06},
    "obscene": {"cortisol": 0.10, "serotonin": -0.06},
    "threat": {"cortisol": 0.18, "dopamine": -0.08},
    "insult": {"cortisol": 0.12, "serotonin": -0.08},
    "identity_hate": {"cortisol": 0.15, "serotonin": -0.10}
}

# ------------------------- Processing Parameters ------------------- #
_RATE_LIMIT = 0.08  # Increased from 0.05 for more responsive changes
_DECAY_STEP = 0.01  # Decay toward baseline when neutral
_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to apply changes
_TOXICITY_THRESHOLD = 0.5  # Minimum toxicity score to trigger

def _apply_resistance(current: float, delta: float) -> float:
    """Smaller effective change near extremes (0/1)."""
    distance = 1 - abs(current - 0.5) * 2  # 1 at mid, 0 at extremes
    return delta * (0.4 + 0.6 * distance)  # min 40% efficacy at extremes

def analyze_contextual_sentiment(text: str) -> dict:
    """
    ML-based contextual sentiment analysis using emotion detection pipeline.
    Returns emotion analysis, toxicity analysis, and hormone adjustment recommendations.
    """
    print(f"[ML Sentiment]: Analyzing text: '{text}'")
    
    # Get emotion detection results
    emotions = detect_emotion(text)
    toxicity_result = detect_toxicity(text)  # This returns a single dict, not a list
    confidence = analyze_sentiment_confidence(text, emotions, toxicity_result)
    
    # Determine primary emotion
    primary_emotion = None
    emotion_intensity = 0.0
    if emotions and confidence["overall_confidence"] > _CONFIDENCE_THRESHOLD:
        primary_emotion = emotions[0]["label"].lower()
        emotion_intensity = emotions[0]["score"]
        print(f"[ML Sentiment]: Primary emotion: {primary_emotion} (confidence: {emotion_intensity:.3f})")
    else:
        primary_emotion = "neutral"
        print(f"[ML Sentiment]: Low confidence or no strong emotions detected")
    
    # FIXED: Check for toxicity - toxicity_result is a single dict, not a list
    toxicity_detected = []
    max_toxicity_score = 0.0
    
    if toxicity_result and toxicity_result.get("is_toxic", False):
        toxic_score = toxicity_result.get("score", 0.0)
        toxic_label = toxicity_result.get("label", "").lower()
        
        if toxic_score > _TOXICITY_THRESHOLD:
            toxicity_detected.append(toxic_label)
            max_toxicity_score = toxic_score
            print(f"[ML Sentiment]: Toxicity detected: {toxic_label} (score: {toxic_score:.3f})")
    
    # Calculate hormone adjustments
    hormone_deltas = {}
    adjustment_reasons = []
    
    # Apply emotion-based adjustments
    if primary_emotion in _EMOTION_HORMONE_MAP:
        emotion_deltas = _EMOTION_HORMONE_MAP[primary_emotion]
        for hormone, base_delta in emotion_deltas.items():
            # Scale by emotion intensity and confidence
            scaled_delta = base_delta * emotion_intensity * confidence["overall_confidence"]
            hormone_deltas[hormone] = hormone_deltas.get(hormone, 0) + scaled_delta
        adjustment_reasons.append(f"emotion_{primary_emotion}")
    
    # Apply toxicity-based adjustments (overrides emotion if stronger)
    if toxicity_detected:
        for tox_label in toxicity_detected:
            if tox_label in _TOXICITY_HORMONE_MAP:
                tox_deltas = _TOXICITY_HORMONE_MAP[tox_label]
                for hormone, base_delta in tox_deltas.items():
                    # Scale by toxicity score
                    scaled_delta = base_delta * max_toxicity_score
                    # Toxicity adjustments are additive (can intensify emotion adjustments)
                    hormone_deltas[hormone] = hormone_deltas.get(hormone, 0) + scaled_delta
                adjustment_reasons.append(f"toxicity_{tox_label}")
    
    return {
        "primary_emotion": primary_emotion,
        "emotion_intensity": emotion_intensity,
        "emotions_detected": emotions[:3] if emotions else [],  # Top 3 emotions
        "toxicity_detected": toxicity_detected,
        "toxicity_score": max_toxicity_score,
        "confidence_metrics": confidence,
        "hormone_deltas": hormone_deltas,
        "adjustment_reasons": adjustment_reasons,
        "detected_text": text
    }

def apply_contextual_hormone_adjustments(text: str) -> Dict[str, float]:
    """
    Primary function for ML-based contextual hormone adjustment.
    Uses emotion detection pipeline instead of hardcoded patterns.
    """
    analysis = analyze_contextual_sentiment(text)
    hormones = load_hormone_levels()
    updated = hormones.copy()
    
    print(f"[ML Hormone Adjust]: {analysis['primary_emotion']} (intensity: {analysis['emotion_intensity']:.2f})")
    
    if analysis["adjustment_reasons"]:
        print(f"[ML Hormone Adjust]: Reasons: {', '.join(analysis['adjustment_reasons'])}")
    
    if analysis["hormone_deltas"]:
        for hormone, base_delta in analysis["hormone_deltas"].items():
            # Apply rate limiting
            delta = max(-_RATE_LIMIT, min(_RATE_LIMIT, base_delta))
            # Apply resistance curve
            delta = _apply_resistance(hormones[hormone], delta)
            # Update hormone level
            updated[hormone] = max(0.0, min(1.0, hormones[hormone] + delta))
            print(f"[ML Hormone Adjust]: {hormone} {hormones[hormone]:.3f} -> {updated[hormone]:.3f} (delta: {delta:+.4f})")
    else:
        # Neutral message → gradual decay toward baseline (0.5)
        print("[ML Hormone Adjust]: Neutral input - applying baseline decay")
        for hormone in updated:
            if abs(updated[hormone] - 0.5) < _DECAY_STEP:
                updated[hormone] = 0.5
            elif updated[hormone] > 0.5:
                updated[hormone] -= _DECAY_STEP
            else:
                updated[hormone] += _DECAY_STEP
    
    save_hormone_levels(updated)
    return updated

# Legacy function mappings for compatibility
def adjust_hormones(event: str) -> Dict[str, float]:
    """Legacy function for event-based hormone adjustment."""
    # Map old events to representative text for ML analysis
    event_to_text = {
        "stress": "I feel terrible and anxious about this situation",
        "positive_feedback": "This is amazing and wonderful, I love it",
        "social_connection": "I feel love and deep connection with you",
        "neutral_interaction": "This is a normal conversation"
    }
    
    text = event_to_text.get(event, event)
    print(f"[Legacy Hormone Adjust]: Converting event '{event}' to text analysis")
    return apply_contextual_hormone_adjustments(text)

# Re-export functions with consistent naming for backward compatibility
load_hormones = load_hormone_levels
save_hormones = save_hormone_levels

def get_emotion_mapping_info() -> Dict:
    """
    Debugging function to get information about emotion-hormone mappings.
    """
    return {
        "emotion_count": len(_EMOTION_HORMONE_MAP),
        "toxicity_count": len(_TOXICITY_HORMONE_MAP),
        "emotions_mapped": list(_EMOTION_HORMONE_MAP.keys()),
        "toxicity_types": list(_TOXICITY_HORMONE_MAP.keys()),
        "rate_limit": _RATE_LIMIT,
        "confidence_threshold": _CONFIDENCE_THRESHOLD,
        "toxicity_threshold": _TOXICITY_THRESHOLD
    }

# CLI for testing
if __name__ == "__main__":
    print("🧪 ML-Enhanced Hormone Adjuster Demo — Type messages, 'quit' to exit")
    print("🤖 Using GoEmotions + Toxicity Detection Pipeline\n")
    
    # Show mapping info
    info = get_emotion_mapping_info()
    print(f"📊 Loaded {info['emotion_count']} emotion mappings and {info['toxicity_count']} toxicity types")
    print(f"⚙️ Rate limit: {info['rate_limit']}, Confidence threshold: {info['confidence_threshold']}\n")
    
    while True:
        msg = input("You: ")
        if msg.lower().startswith("quit"):
            break
            
        print("\n" + "="*50)
        new_levels = apply_contextual_hormone_adjustments(msg)
        
        # Show resulting mood
        mood_weights = load_mood_weights()
        mood, intensity = infer_mood_from_hormones(new_levels, mood_weights)
        context = get_mood_context(mood, intensity)
        
        mood_type = ""
        if context["is_hybrid"]:
            mood_type = " [HYBRID]"
        elif context["is_emergent"]:
            mood_type = " [EMERGENT]"
            
        print(f"\n🧠 Final State:")
        print(f" Hormones: {new_levels}")
        print(f" Mood: {mood}{mood_type} (intensity: {intensity:.2f})")
        print("="*50 + "\n")
