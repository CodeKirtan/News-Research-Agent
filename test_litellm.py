import os
from crewai import Agent
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = "fake_key_just_for_init_test"

print("Instantiating Agent with litellm groq model...")
try:
    agent = Agent(
        role="tester",
        goal="test",
        backstory="test",
        llm="groq/llama-3.3-70b-versatile"
    )
    print("Standard instantiation worked with litellm!")
except Exception as e:
    print(f"Standard failed: {e}")
