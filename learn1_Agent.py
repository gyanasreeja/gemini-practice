import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

@tool
def word_count(text: str) -> int:
    """Counts the number of words in the given text."""
    words = text.split(" ")
    return len(words)

@tool
def is_even(number: int) -> bool:
    """Checks if a number is even. Returns True if even, False if odd."""
    return number % 2 == 0

tools = [word_count, is_even]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful study assistant. Use tools when needed for word counting or checking even/odd numbers. Otherwise, answer normally."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

# Custom callback to capture token usage from each LLM call
class TokenTracker(BaseCallbackHandler):
    def __init__(self):
        self.total_input = 0
        self.total_output = 0

    def on_llm_end(self, response, **kwargs):
        for generation in response.generations[0]:
            usage = generation.message.usage_metadata
            if usage:
                self.total_input += usage.get("input_tokens", 0)
                self.total_output += usage.get("output_tokens", 0)

print("Study Helper started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    try:
        tracker = TokenTracker()
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        result = agent_executor.invoke({"input": user_input}, config={"callbacks": [tracker]})

        print("Assistant:", result['output'])
        print(f"[Tokens → input: {tracker.total_input}, output: {tracker.total_output}, total: {tracker.total_input + tracker.total_output}]\n")

    except Exception as e:
        print("Something went wrong:", e, "\n")