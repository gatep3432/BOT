import json
from pathlib import Path
import streamlit as st

from utils.session_id import get_or_create_session_file, save_turn_to_session
from utils.ui_helpers import render_message
from core.context_block_builder import (
    build_session_init_prompt,
    build_turn_prompt,
)
from core.fact_extractor import store_fact, load_facts
from memory.turn_memory import dump_turn, load_memory
from memory.session_summarizer import summarize_session
from persona.mood_tracker import apply_sentiment_to_mood, get_current_mood, update_mood

# ------------------------------------------------------------------ #
#  Streamlit Page Config
# ------------------------------------------------------------------ #
st.set_page_config(page_title="Memory Architect Bot", layout="wide")
st.title("🧠 Your Memory-Based AI Agent (FAISS + Persona)")

# ------------------------------------------------------------------ #
#  Session bootstrap + Mood Initialization
# ------------------------------------------------------------------ #
session_file = get_or_create_session_file(st.session_state)
session_id = st.session_state["session_id"]

if "turns" not in st.session_state:
    st.session_state.turns = []

if "session_prompt_printed" not in st.session_state:
    st.session_state.session_prompt_printed = False

if "mood_initialized" not in st.session_state:
    mood_file = Path("persona/mood_adjustments.json")
    if not mood_file.exists():
        try:
            mood_file.parent.mkdir(exist_ok=True)
            default_mood = {
                "current_mood": "neutral",
                "intensity": 0.5,
                "context": {
                    "is_hybrid": False,
                    "is_emergent": False,
                    "stability": "medium"
                },
                "last_updated": "2025-01-01T00:00:00.000000"
            }
            with open(mood_file, "w", encoding="utf-8") as f:
                json.dump(default_mood, f, indent=2)
            print("[Mood Init]: Created default neutral mood")
        except Exception as e:
            print(f"[Mood Init Error]: {e}")
    st.session_state.mood_initialized = True

# Reload past messages (UI only)
if not st.session_state.turns and Path(session_file).exists():
    try:
        with open(session_file, encoding="utf-8") as f:
            history = json.load(f)
            for turn in history:
                st.session_state.turns.append({"role": "user", "content": turn["user"]})
                st.session_state.turns.append(
                    {"role": "assistant", "content": turn["assistant"]}
                )
    except Exception as e:
        st.error(f"❌ Error loading session: {e}")

# Display prior messages
for m in st.session_state.turns:
    render_message(m)

# ------------------------------------------------------------------ #
#  Session-level injection
# ------------------------------------------------------------------ #
if not st.session_state.session_prompt_printed:
    session_block = build_session_init_prompt(session_id)
    with st.chat_message("assistant"):
        st.code(session_block, language="markdown")
    st.session_state.session_prompt_printed = True

# ------------------------------------------------------------------ #
#  Main chat input
# ------------------------------------------------------------------ #
if user_msg := st.chat_input("Type your message…"):
    st.session_state.turns.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    try:
        store_fact(user_msg)
    except Exception as e:
        st.warning(f"⚠️ Fact storage error: {e}")

    try:
        print(f"\n=== MOOD PROCESSING START ===")
        print(f"[App.py]: Processing user input: '{user_msg}'")
        apply_sentiment_to_mood(user_msg)
        print(f"[App.py]: Mood processing completed")
        print(f"=== MOOD PROCESSING END ===\n")
    except Exception as e:
        st.warning(f"⚠️ Mood processing error: {e}")
        print(f"[Mood Error]: {e}")

    # 🔄 FAISS memory state update
    try:
        from persona.faiss_memory_writer import update_faiss_memory_state_from_session
        update_faiss_memory_state_from_session(session_id)
    except Exception as e:
        st.warning(f"⚠️ FAISS memory write error: {e}")
        print(f"[FAISS Error]: {e}")

    # 🔄 TINY MODEL update
    try:
        from persona.tiny_model_writer import update_tiny_model_state_from_session
        update_tiny_model_state_from_session(session_id)
    except Exception as e:
        st.warning(f"⚠️ Tiny model write error: {e}")
        print(f"[Tiny Model Error]: {e}")

    turn_block = build_turn_prompt(user_msg, session_id)
    with st.chat_message("assistant"):
        st.code(turn_block, language="markdown")

    clean_placeholder = "Prompt generated"
    save_turn_to_session({"user": user_msg, "assistant": clean_placeholder}, st.session_state)
    dump_turn({"user": user_msg, "assistant": clean_placeholder})

# ------------------------------------------------------------------ #
#  Sidebar
# ------------------------------------------------------------------ #
st.sidebar.header("🧠 Memory Management")
if st.sidebar.button("End Chat & Save to Long-Term Memory"):
    try:
        with open(session_file, encoding="utf-8") as f:
            session_data = json.load(f)
        summary = summarize_session(session_id, session_data)
        st.sidebar.success("✅ Session summarized & stored.")
        st.sidebar.json(summary)
    except Exception as e:
        st.sidebar.error(f"❌ Could not summarize session: {e}")

st.sidebar.header("🎭 Mood Status")
try:
    current_mood_data = get_current_mood()
    st.sidebar.write(f"**Current Mood:** {current_mood_data['current_mood']}")
    st.sidebar.write(f"**Intensity:** {current_mood_data['intensity']:.2f}")
    
    mood_context = current_mood_data.get('context', {})
    if mood_context.get('is_hybrid'):
        st.sidebar.write("🔀 **Hybrid State**")
    if mood_context.get('is_emergent'):
        st.sidebar.write("⚡ **Emergent State**")
    
    st.sidebar.write(f"**Stability:** {mood_context.get('stability', 'medium')}")

    # Use neutral API instead of direct import from hormone_adjuster
    from persona.hormone_api import load_hormone_levels, save_hormone_levels
    hormones = load_hormone_levels()
    with st.sidebar.expander("🧪 Hormone Levels"):
        for hormone, level in hormones.items():
            if level > 0.7:
                st.sidebar.write(f"🔴 {hormone.title()}: {level:.2f}")
            elif level < 0.3:
                st.sidebar.write(f"🔵 {hormone.title()}: {level:.2f}")
            else:
                st.sidebar.write(f"⚪ {hormone.title()}: {level:.2f}")

    with st.sidebar.expander("🧪 Manual Mood Testing"):
        if st.button("Test Positive"):
            apply_sentiment_to_mood("I love this amazing wonderful experience")
            st.rerun()
        if st.button("Test Negative"):
            apply_sentiment_to_mood("I hate this terrible awful situation")
            st.rerun()
        if st.button("Reset to Neutral"):
            update_mood("neutral", 0.5, "manual_reset")
            # Use save_hormone_levels from neutral API
            save_hormone_levels({"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5})
            st.rerun()
except Exception as e:
    st.sidebar.error(f"❌ Mood status error: {e}")

st.sidebar.header("🔧 Debug")
st.sidebar.write(f"Session ID : `{session_id[:8]}...`")
st.sidebar.write(f"Turn count : {len(st.session_state.turns)}")
st.sidebar.write(f"Global memory turns: {len(load_memory())}")

facts = load_facts()
if facts:
    st.sidebar.write(f"Stored facts: {len(facts)}")
    with st.sidebar.expander("View last 5 facts"):
        for i, fct in enumerate(facts[-5:], 1):
            st.sidebar.write(f"{i}. {fct}")
