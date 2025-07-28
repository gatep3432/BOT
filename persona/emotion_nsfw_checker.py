# persona/emotion_nsfw_checker.py
"""
Enhanced emotion and toxicity detection pipeline for hormone system integration.
Uses transformers models for ML-based sentiment analysis.
"""

from transformers import pipeline
import re
from typing import List, Dict, Optional
from pathlib import Path
import json

# Initialize models (lazy loading to avoid startup delays)
_goemotions_pipeline = None
_toxicity_pipeline = None

def _get_emotion_pipeline():
    """Lazy load emotion detection pipeline."""
    global _goemotions_pipeline
    if _goemotions_pipeline is None:
        print("🔄 Loading GoEmotions model...")
        _goemotions_pipeline = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=None  # Fixed: Use top_k=None instead of return_all_scores=True
        )
        print("✅ GoEmotions model loaded")
    return _goemotions_pipeline

def _get_toxicity_pipeline():
    """Lazy load toxicity detection pipeline."""
    global _toxicity_pipeline
    if _toxicity_pipeline is None:
        print("🔄 Loading toxicity detection model...")
        _toxicity_pipeline = pipeline(
            "text-classification",
            model="unitary/unbiased-toxic-roberta",
            top_k=None  # Fixed: Use top_k=None instead of return_all_scores=True
        )
        print("✅ Toxicity model loaded")
    return _toxicity_pipeline

def preprocess_text(text: str) -> str:
    """Clean and normalize text for better analysis."""
    if not text or not text.strip():
        return ""
    
    # Remove excessive punctuation while preserving emotional indicators
    text = re.sub(r'[!]{2,}', '!', text)  # Multiple ! to single !
    text = re.sub(r'[?]{2,}', '?', text)  # Multiple ? to single ?
    text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
    
    # Normalize common contractions for better context understanding
    contractions = {
        "you're": "you are", "don't": "do not", "can't": "cannot",
        "won't": "will not", "i'm": "i am", "it's": "it is",
        "that's": "that is", "we're": "we are", "they're": "they are"
    }
    
    # Preserve original case for better model performance, but use normalized contractions
    for contraction, expansion in contractions.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
    
    return text.strip()

def detect_emotion(text: str, min_confidence: float = 0.1) -> List[Dict]:
    """
    Enhanced emotion detection with preprocessing and filtering.
    
    Args:
        text: Input text to analyze
        min_confidence: Minimum confidence threshold for including emotions
    
    Returns:
        List of detected emotions with labels and scores, sorted by confidence
    """
    if not text or not text.strip():
        return []
    
    processed_text = preprocess_text(text)
    if not processed_text:
        return []
    
    try:
        pipeline = _get_emotion_pipeline()
        results = pipeline(processed_text)
        
        # FIXED: Handle the correct pipeline output format
        # When top_k=None, results is a list of dictionaries for a single input
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            emotion_scores = results[0]  # Get scores for our single input
        elif isinstance(results, list):
            emotion_scores = results  # Already in the right format
        else:
            emotion_scores = [results]  # Single result
        
        # Filter and process results
        emotions = []
        for result in emotion_scores:
            if result["score"] > min_confidence:
                emotions.append({
                    "label": result["label"],
                    "score": result["score"]
                })
        
        # Sort by confidence (highest first)
        emotions.sort(key=lambda x: x["score"], reverse=True)
        return emotions
        
    except Exception as e:
        print(f"[Emotion Detection Error]: {e}")
        return []

def detect_toxicity(text: str, min_confidence: float = 0.3) -> List[Dict]:
    """
    Enhanced toxicity detection with context awareness.
    
    Args:
        text: Input text to analyze
        min_confidence: Minimum confidence threshold for toxicity detection
    
    Returns:
        Dictionary with toxicity information
    """
    if not text or not text.strip():
        return {"is_toxic": False, "score": 0.0, "label": "NOT_TOXIC"}
    
    processed_text = preprocess_text(text)
    if not processed_text:
        return {"is_toxic": False, "score": 0.0, "label": "NOT_TOXIC"}
    
    try:
        pipeline = _get_toxicity_pipeline()
        results = pipeline(processed_text)
        
        # FIXED: Handle the correct pipeline output format for toxicity model
        # The unitary/unbiased-toxic-roberta model returns TOXIC/NOT_TOXIC labels
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            toxicity_scores = results[0]  # Get scores for our single input
        elif isinstance(results, list):
            toxicity_scores = results  # Already in the right format
        else:
            toxicity_scores = [results]  # Single result
        
        # Find the TOXIC label result
        toxic_score = 0.0
        is_toxic = False
        toxic_label = "NOT_TOXIC"
        
        for result in toxicity_scores:
            if result["label"] == "TOXIC":
                toxic_score = result["score"]
                if toxic_score > min_confidence:
                    is_toxic = True
                    toxic_label = "TOXIC"
                break
        
        return {
            "is_toxic": is_toxic,
            "score": toxic_score,
            "label": toxic_label
        }
        
    except Exception as e:
        print(f"[Toxicity Detection Error]: {e}")
        return {"is_toxic": False, "score": 0.0, "label": "NOT_TOXIC"}

def analyze_sentiment_confidence(text: str, emotions: List[Dict], toxicity: Dict) -> Dict:
    """
    Provide confidence analysis for the sentiment detection.
    
    Args:
        text: Original text
        emotions: Detected emotions from detect_emotion()
        toxicity: Detected toxicity from detect_toxicity()
    
    Returns:
        Dictionary with confidence metrics
    """
    confidence_metrics = {
        "overall_confidence": 0.0,
        "emotion_clarity": 0.0,
        "context_ambiguity": 0.0,
        "text_quality": 0.0,
        "toxicity_confidence": 0.0
    }
    
    # Calculate emotion clarity
    if emotions and len(emotions) > 0:
        top_score = emotions[0]["score"]
        second_score = emotions[1]["score"] if len(emotions) > 1 else 0
        confidence_metrics["emotion_clarity"] = min(1.0, top_score - second_score + 0.2)
    else:
        confidence_metrics["emotion_clarity"] = 0.1  # Low confidence if no emotions
    
    # Calculate text quality factors
    if not text or not text.strip():
        confidence_metrics["text_quality"] = 0.0
    else:
        word_count = len(text.split())
        if word_count >= 5:
            confidence_metrics["text_quality"] = min(1.0, word_count / 10)
        elif word_count >= 2:
            confidence_metrics["text_quality"] = 0.6
        else:
            confidence_metrics["text_quality"] = 0.3  # Very short text
    
    # Check for context ambiguity markers
    ambiguous_phrases = ["i guess", "maybe", "sort of", "kind of", "perhaps", "might be"]
    if any(phrase in text.lower() for phrase in ambiguous_phrases):
        confidence_metrics["context_ambiguity"] = 0.4
    else:
        confidence_metrics["context_ambiguity"] = 0.8
    
    # Calculate toxicity confidence
    if toxicity and toxicity.get("is_toxic", False):
        confidence_metrics["toxicity_confidence"] = min(1.0, toxicity["score"])
    else:
        confidence_metrics["toxicity_confidence"] = 0.0
    
    # Calculate overall confidence
    base_confidence = (
        confidence_metrics["emotion_clarity"] * 0.4 +
        confidence_metrics["text_quality"] * 0.3 +
        confidence_metrics["context_ambiguity"] * 0.3
    )
    
    # Boost confidence if we have clear toxicity signals
    if confidence_metrics["toxicity_confidence"] > 0.7:
        base_confidence = max(base_confidence, 0.8)
    
    confidence_metrics["overall_confidence"] = min(1.0, base_confidence)
    
    return confidence_metrics

def get_emotional_summary(text: str) -> Dict:
    """
    Get a comprehensive emotional analysis summary for debugging and logging.
    
    Args:
        text: Text to analyze
    
    Returns:
        Complete analysis summary
    """
    emotions = detect_emotion(text)
    toxicity = detect_toxicity(text)
    confidence = analyze_sentiment_confidence(text, emotions, toxicity)
    
    return {
        "original_text": text,
        "processed_text": preprocess_text(text),
        "emotions": emotions,
        "toxicity": toxicity,
        "confidence": confidence,
        "primary_emotion": emotions[0]["label"] if emotions else "neutral",
        "primary_emotion_score": emotions[0]["score"] if emotions else 0.0,
        "is_toxic": toxicity.get("is_toxic", False),
        "max_toxicity_score": toxicity.get("score", 0.0)
    }

# CLI testing interface
def main():
    """Interactive testing interface for the emotion detection system."""
    print("🤖 Enhanced Emotion + Toxicity Classifier")
    print("✨ ML-powered sentiment analysis for hormone system")
    print("📝 Type 'quit' to exit, 'test' for sample analysis\n")
    
    while True:
        user_input = input("👤 You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        elif user_input.lower() == "test":
            # Run test cases
            test_cases = [
                "I love you so much!",
                "This is fucking terrible",
                "I'm feeling quite sad today",
                "You're amazing and wonderful",
                "I hate this stupid thing"
            ]
            
            print("\n🧪 Running test cases:")
            for test_text in test_cases:
                print(f"\n📝 Text: '{test_text}'")
                summary = get_emotional_summary(test_text)
                print(f" 🎭 Primary emotion: {summary['primary_emotion']} ({summary['primary_emotion_score']:.3f})")
                if summary['is_toxic']:
                    print(f" ⚠️ Toxicity detected: {summary['max_toxicity_score']:.3f}")
                print(f" 📊 Overall confidence: {summary['confidence']['overall_confidence']:.3f}")
            continue
        
        if not user_input.strip():
            continue
        
        # Get complete analysis
        summary = get_emotional_summary(user_input)
        
        print(f"\n💬 Analysis Results:")
        print(f" 📝 Processed: '{summary['processed_text']}'")
        
        # Show top emotions
        if summary['emotions']:
            print(f" 🎭 Emotions detected:")
            for i, emotion in enumerate(summary['emotions'][:3]):  # Top 3
                print(f"  {i+1}. {emotion['label'].title()} ({emotion['score']:.3f})")
        else:
            print(f" 🎭 No strong emotions detected")
        
        # Show toxicity
        if summary['toxicity']['is_toxic']:
            print(f" ⚠️ Toxicity detected: {summary['toxicity']['label']} ({summary['toxicity']['score']:.3f})")
        else:
            print(f" ✅ No toxicity detected")
        
        # Show confidence metrics
        conf = summary['confidence']
        print(f" 📊 Confidence: Overall {conf['overall_confidence']:.3f} | "
              f"Emotion {conf['emotion_clarity']:.3f} | "
              f"Quality {conf['text_quality']:.3f}")
        
        if conf['overall_confidence'] < 0.5:
            print(f" ⚠️ Note: Low confidence analysis")
        
        print()

if __name__ == "__main__":
    main()