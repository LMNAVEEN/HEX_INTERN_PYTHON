# Run: python 01_langchain_rag/memory_chat.py
# Tech: LangChain + Google Gemini + InMemoryChatMessageHistory
# Purpose: Multi-turn conversation memory using RunnableWithMessageHistory

from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Create Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant."),
        ("human", "{input}")
    ]
)

# Output Parser
parser = StrOutputParser()

# LCEL Chain
chain = prompt | llm | parser

store = {}

# Function to get chat history
def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]
    
# Add Memory to Chain
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input"
)

config = {
    "configurable": {
        "session_id": "user_1"
    }
}

response_1 = chain_with_memory.invoke(
    {
        "input": "Hi, what's your name?"
    },
    config=config
)

print("\nAI:", response_1)

# Conversation 2
response_2 = chain_with_memory.invoke(
    {
        "input": "My name is Naveen my other name is also naveen."
    },
    config=config
)

print("\nAI:", response_2)

# Conversation 3
response_3 = chain_with_memory.invoke(
    {
        "input": "What's my other name?"
    },
    config=config
)

print("\nAI:", response_3)

# Conversation 4
response_4 = chain_with_memory.invoke(
    {
        "input": "My hobby is Reading."
    },
    config=config
)

print("\nAI:", response_4)

# Conversation 5
response_5 = chain_with_memory.invoke(
    {
        "input": "What is my hobby?"
    },
    config=config
)

response_6 = chain_with_memory.invoke(
    {
        "input": "my friends are Karthik Raja and Rubis, my mentor name is Nalla"
    },
    config= config

)
response_7 = chain_with_memory.invoke(
    {
        "input": "what is my mentor name?"
    },
    config= config

)

print("\nAI:", response_7)

#####