from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Generator chain
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a compelling LinkedIn post. Be specific. Use concrete details. Show impact."),
    MessagesPlaceholder(variable_name='messages'),
])
generate_chain = generation_prompt | ChatOpenAI(model='gpt-4o-mini')

# Critique chain
critique_prompt = ChatPromptTemplate.from_messages([
    ("system", "Review the LinkedIn post. Identify what makes it weak. Point out missing details, unclear sections, and areas lacking specificity."),
    MessagesPlaceholder(variable_name='messages'),
])
critique_chain = critique_prompt | ChatOpenAI(model='gpt-4o-mini')

def generation_node(state: State) -> dict:
    """Generate or revise the post from current message history."""
    ai_msg = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [ai_msg]}

def critique_node(state: State) -> dict:
    """Critique the post. Return feedback as a HumanMessage."""
    msgs = state["messages"]

    # Role swap: treat the model's own output as something to critique
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [msgs[0]] + [cls_map[m.type](content=m.content) for m in msgs[1:]]

    # Get the critique
    feedback = critique_chain.invoke({"messages": translated})

    # Return as HumanMessage so the generator sees it as user feedback
    return {"messages": [HumanMessage(content=feedback.content)]}

MAX_ITERATIONS = 3

def should_continue(state: State):
    if len(state["messages"]) > MAX_ITERATIONS:
        return END
    return "critique"

from langgraph.graph import StateGraph, END

builder = StateGraph(state_schema=State)
builder.add_node("generate", generation_node)
builder.add_node("critique", critique_node)

builder.set_entry_point("generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("critique", "generate")

graph = builder.compile()

from langchain_core.messages import HumanMessage

inputs: State = {"messages": [HumanMessage(content="Write a LinkedIn post about shipping an API caching layer")]}

for event in graph.stream(input=inputs):
    for node, state in event.items():
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                print(f"\n--- {node} ---")
                print(msg.content)
                print("\n" + "-" * 80 + "\n")
