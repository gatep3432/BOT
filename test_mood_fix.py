# test_mood_fix.py
"""
Test script to verify the fixed mood calculation system is working.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

def test_mood_calculation():
    """Test the mood calculation pipeline"""
    
    print("🧪 Testing FIXED Mood Calculation System\n")
    
    # Import after adding to path
    from persona.hormone_api import load_hormone_levels, save_hormone_levels
    from persona.mood_tracker import update_mood_from_hormones, get_current_mood
    
    # Test different hormone scenarios
    test_scenarios = [
        {
            "name": "High Cortisol (Stress/Anger)",
            "hormones": {"dopamine": 0.45, "serotonin": 0.47, "cortisol": 0.72, "oxytocin": 0.50}
        },
        {
            "name": "High Oxytocin + Dopamine (Love/Joy)",
            "hormones": {"dopamine": 0.65, "serotonin": 0.54, "cortisol": 0.50, "oxytocin": 0.68}
        },
        {
            "name": "Low Serotonin + High Cortisol (Depression)",
            "hormones": {"dopamine": 0.48, "serotonin": 0.35, "cortisol": 0.68, "oxytocin": 0.50}
        },
        {
            "name": "High Dopamine + Normal Others (Excitement)",
            "hormones": {"dopamine": 0.75, "serotonin": 0.55, "cortisol": 0.48, "oxytocin": 0.52}
        },
        {
            "name": "All Neutral (Baseline)",
            "hormones": {"dopamine": 0.50, "serotonin": 0.50, "cortisol": 0.50, "oxytocin": 0.50}
        }
    ]
    
    print("=" * 70)
    print("TESTING MOOD CALCULATION FROM HORMONE LEVELS")
    print("=" * 70)
    
    for scenario in test_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Hormone Levels: {scenario['hormones']}")
        
        try:
            # Set the hormone levels
            save_hormone_levels(scenario['hormones'])
            
            # Calculate mood from these hormones
            mood, intensity, context = update_mood_from_hormones(reason="test_scenario")
            
            # Get the current mood data
            current_mood_data = get_current_mood()
            
            # Display results
            mood_flags = []
            if context.get("is_hybrid"):
                mood_flags.append("HYBRID")
            if context.get("is_emergent"):
                mood_flags.append("EMERGENT")
            mood_type = f" [{'/'.join(mood_flags)}]" if mood_flags else ""
            
            print(f"   🎭 Calculated Mood: {mood}{mood_type}")
            print(f"   📊 Intensity: {intensity:.3f}")
            print(f"   🔬 Stability: {context.get('stability', 'unknown')}")
            print(f"   ✅ Test completed successfully!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("MOOD CALCULATION TEST COMPLETE")
    print("=" * 70)
    
    print("\n🎯 Expected Results:")
    print("- High cortisol scenarios should show 'anxious', 'stressed', or 'restless'")
    print("- High oxytocin scenarios should show 'loving', 'affectionate', or 'caring'") 
    print("- Low serotonin scenarios should show 'depressed', 'sad', or 'melancholic'")
    print("- High dopamine scenarios should show 'cheerful', 'energetic', or 'euphoric'")
    print("- Neutral hormones should show 'neutral' or 'contemplative'")
    print("\n❌ If all results show 'neutral (intensity: 1.00)', the mood calculation is still broken!")
    print("✅ If results show different moods with varying intensities, the fix worked!")

if __name__ == "__main__":
    test_mood_calculation()
