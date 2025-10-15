# Dummy replacement for Emergent's internal AI module
class LlmChat:
    def __init__(self, *args, **kwargs):
        print("⚠️  Using dummy LlmChat (no real Emergent backend).")

    def chat(self, messages, *args, **kwargs):
        # You can replace this with your own OpenAI API call later
        return {"response": "AI simulation: this would be a real AI reply."}


class UserMessage:
    def __init__(self, content: str):
        self.content = content
