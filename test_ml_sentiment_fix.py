# test_ml_working.py
"""
Test script to verify the fixed ML sentiment analysis is working correctly.
This version handles all the import errors and parsing issues.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

def test_ml_sentiment():
    """Test the fixed ML sentiment analysis pipeline"""
    
    print("🧪 Testing FIXED ML Sentiment Analysis Pipeline\n")
    
    # Import after adding to path
    try:
        from persona.emotion_nsfw_checker import detect_emotion, detect_toxicity, analyze_sentiment_confidence
        from persona.hormone_adjuster import apply_contextual_hormone_adjustments
        from persona.mood_tracker import update_mood_from_hormones
        print("✅ All imports successful!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test cases
    test_inputs = [
        "fuck you bitch",
        "I love you so much",
        "kiss me",
        "I hate this stupid thing", 
        "You're amazing and wonderful",
        "shut up",
        "hi there",
        "I'm feeling sad today"
    ]
    
    print("=" * 60)
    print("TESTING EMOTION DETECTION")
    print("=" * 60)
    
    for text in test_inputs:
        print(f"\n📝 Input: '{text}'")
        
        try:
            # Test emotion detection
            emotions = detect_emotion(text)
            toxicity = detect_toxicity(text)
            
            # Format emotions for display
            if emotions:
                emotion_list = []
                for e in emotions[:2]:  # Top 2 emotions
                    emotion_list.append(f"{e['label']}({e['score']:.2f})")
                emotions_str = ", ".join(emotion_list)
            else:
                emotions_str = "None detected"
            
            print(f"   🎭 Emotions: {emotions_str}")
            print(f"   ☢️  Toxicity: {toxicity.get('is_toxic', False)} (score: {toxicity.get('score', 0):.3f})")
            
            # Test hormone adjustments
            print(f"   🧪 Testing hormone adjustments...")
            
            # Apply hormone changes
            hormones_after = apply_contextual_hormone_adjustments(text)
            print(f"   🧪 New Hormone Levels: {hormones_after}")
            
            # Test mood update
            mood, intensity, context = update_mood_from_hormones(reason="test_analysis")
            mood_flags = []
            if context.get("is_hybrid"):
                mood_flags.append("HYBRID")
            if context.get("is_emergent"):
                mood_flags.append("EMERGENT")
            mood_type = f" [{'/'.join(mood_flags)}]" if mood_flags else ""
            
            print(f"   🎭 Resulting Mood: {mood}{mood_type} (intensity: {intensity:.2f})")
            print("   ✅ Test completed successfully!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE - Check for working hormone/mood changes above!")
    print("=" * 60)

if __name__ == "__main__":
    test_ml_sentiment()