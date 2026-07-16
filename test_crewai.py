import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

print("Instantiating ChatGroq with prefix...")
try:
    chat = ChatGroq(api_key=os.environ.get("GROQ_API_KEY") or "fake", model="groq/llama-3.3-70b-versatile", temperature=0.0)
    agent = Agent(
        role="tester",
        goal="say hello",
        backstory="test",
        llm=chat
    )
    task = Task(description="say hello", expected_output="hello", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    print("Kicking off...")
    crew.kickoff()
    print("Kickoff succeeded!")
except Exception as e:
    print(f"Standard failed: {e}")
