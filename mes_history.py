import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash",
                             
                             google_api_key = os.getenv("GEMINI_API_KEY"))
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = llm | prompt
store = {}
def get_session_history(session_id):
    if get_session_history not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
# wrap the chain with the memory managment
chain_with_memory = RunnableWithMessageHistory(chain, get_session_history, input_messages_key= "input",
                                               history_messages_key= "history")
print("start the chat type exit to quit from the chat")
session_id = "user_123"
while True:
    user_input = input("You :")
    if user_input.lower()=="exit":
        break
    response = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    print(response.content)  
    