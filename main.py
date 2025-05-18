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


class MessageClassifier(BaseModel):
    message_type: Literal['emotional', 'logical'] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
     )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    last_message = state['messages'][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ])
    return {"message_type":result.message_type}


def router(state: State) -> State:
    pass


def therapist_agent(state: State):
    pass


def logic_agent(state: State):
    pass


graph_builder = StateGraph(State)


def chatbot(
    state: State):
    """Chatbot function to handle user input and generate a response."""

    return { 'messages': [llm.invoke(state['messages']) ]}

graph_builder.add_node('chatbot', chatbot)
graph_builder.add_edge(START, end_key='chatbot')
graph_builder.add_edge(start_key='chatbot',end_key=END)
graph = graph_builder.compile()


user_input = input("You: ") 
state = graph.invoke({'messages': [{'role': 'user', 'content': user_input}]} ) 


print("Assistant:", state['messages'][-1].content)
print("Assistant:", state['messages'])