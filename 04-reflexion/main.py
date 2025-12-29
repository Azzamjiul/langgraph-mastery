from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

from tavily import TavilyClient

tavily = TavilyClient()

class WriterState(TypedDict):
    topic: str
    outline: str
    output: str
    feedback: str
    sources: List[str]
    iteration: int
    total_iterations: int

PLAN_PROMPT = """You are an expert writer. Create a detailed outline for an essay on the given topic.
Include main sections and key points to cover."""

RESEARCH_PROMPT = """Generate 3 search queries to gather information for writing an essay.
Make queries specific and factual."""

WRITER_PROMPT = """Write a well-structured essay based on the outline and sources provided.
Be clear, accurate, and thorough.

Context:
{content}"""

REVIEW_PROMPT = """Critique this essay. Identify gaps, weak arguments, missing evidence,
and areas needing improvement. Be specific about what needs to change."""

RESEARCH_CRITIQUE_PROMPT = """Generate 3 search queries to find information that addresses
the critiques mentioned. Focus on filling the gaps identified."""

from pydantic import BaseModel

class Queries(BaseModel):
    queries: List[str]

def plan_node(state: WriterState):
    """Create outline for the essay."""
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['topic'])
    ]
    response = model.invoke(messages)
    return {"outline": response.content}

def research_plan_node(state: WriterState):
    """Generate search queries based on topic."""
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PROMPT),
        HumanMessage(content=state['topic'])
    ])

    sources = state.get('sources', [])
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            sources.append(r['content'])

    return {"sources": sources}

def write_node(state: WriterState):
    """Write or revise the essay."""
    content = "\n\n".join(state['sources'] or [])

    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        HumanMessage(content=f"{state['topic']}\n\nOutline:\n{state['outline']}")
    ]

    response = model.invoke(messages)
    return {
        "output": response.content,
        "iteration": state.get("iteration", 0) + 1
    }

def review_node(state: WriterState):
    """Critique the essay."""
    messages = [
        SystemMessage(content=REVIEW_PROMPT),
        HumanMessage(content=state['output'])
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}

def research_critique_node(state: WriterState):
    """Search for information to address critique."""
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['feedback'])
    ])

    sources = state.get('sources', [])
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            sources.append(r['content'])

    return {"sources": sources}

def should_continue(state: WriterState):
    """Stop if we've hit max iterations."""
    if state['iteration'] > state['total_iterations']:
        return END
    return 'review'

builder = StateGraph(WriterState)

builder.add_node('plan', plan_node)
builder.add_node('research_plan', research_plan_node)
builder.add_node('write', write_node)
builder.add_node('review', review_node)
builder.add_node('research_critique', research_critique_node)

builder.set_entry_point('plan')

builder.add_edge('plan', 'research_plan')
builder.add_edge('research_plan', 'write')
builder.add_conditional_edges('write', should_continue)

builder.add_edge('review', 'research_critique')
builder.add_edge('research_critique', 'write')

graph = builder.compile(checkpointer=MemorySaver())

thread: RunnableConfig = {'configurable': {'thread_id': '1'}}

inputs: WriterState = {
    'topic': 'The impact of renewable energy on climate change',
    'total_iterations': 2,
    'iteration': 1,
    'sources': [],
    'outline': '',
    'output': '',
    'feedback': ''
}

events = graph.stream(inputs, thread)
for event in events:
    print(event)
    print('-' * 80)

final_state = graph.get_state(thread).values
print("\nFinal Essay:")
print(final_state['output'])
