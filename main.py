from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from groq import Groq
import os

load_dotenv()


llm = init_chat_model("google_genai:gemini-2.0-flash")




class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


def chatbot(
    state: State):
    """Chatbot function to handle user input and generate a response."""

    if 'messages' in state:
        return { 'messages': [llm.invoke(state['messages']) ]}
    else:
        print("The 'messages' key is not present in the state.")

graph_builder.add_node('chatbot', chatbot)
graph_builder.add_edge(START, end_key='chatbot')
graph_builder.add_edge(start_key='chatbot',end_key=END)
graph = graph_builder.compile()


user_input = input("You: ") 
state = graph.invoke({'messages': [{'role': 'user', 'content': user_input}]} ) 


print("Assistant:", state['messages'][-1].content)
print("Assistant:", state['messages'])