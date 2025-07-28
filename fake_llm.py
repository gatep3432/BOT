# fake_llm.py
def fake_llm(prompt: str) -> str:
    print("\n🧪  Mock LLM prompt\n", prompt)
    return "🧠 [Stubbed LLM response — OK for tests]"

__all__ = ["fake_llm"]
