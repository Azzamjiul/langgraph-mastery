from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
load_dotenv()

class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(
    temperature=0.7,
    model='gpt-4o-mini',
)

def dialogue_agent(state: ConversationState):
    response_content = ""
    print("Bot: ", end="", flush=True)
    for chunk in llm.stream(state["messages"]):
        if isinstance(chunk.content, str):
            print(chunk.content, end="", flush=True)
            response_content += chunk.content
    print("\n")

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response_content)]}

# chatbot_graph = (
#     StateGraph(ConversationState)
#     .add_node("agent", dialogue_agent)
#     .set_entry_point("agent")
#     .set_finish_point("agent")
#     .compile()
# )

##### Step 6: Interact with Your Chatbot
# user_query = "Explain quantum entanglement in simple terms"
# print(f"User: {user_query}\n")

# for event in chatbot_graph.stream({"messages": [HumanMessage(content=user_query)]}):
#     if "agent" in event:
#         messages = event["agent"]["messages"]
#         print(f"Bot: {messages[-1].content}")

##### Mermaid Diagram
# diagram = chatbot_graph.get_graph().draw_mermaid_png()
# with open("graph_diagram.png", "wb") as f:
#     f.write(diagram)
# print("Graph diagram saved as 'graph_diagram.png'\n")

##### ASCII Representation
# print(chatbot_graph.get_graph().draw_ascii())

##### Running the ChatBot
# print("\n" + "=" * 60)
# print("Interactive Chatbot")
# print("=" * 60)
# print("Type 'exit', 'quit', or 'bye' to exit.\n")

# messages = []

# while True:
#     try:
#         user_input = input("You: ").strip()

#         if not user_input:
#             continue

#         if user_input.lower() in ['exit', 'quit', 'bye']:
#             print("\nChatbot: Thank you for chatting. Goodbye!")
#             break

#         messages.append(HumanMessage(content=user_input))
#         for _ in chatbot_graph.stream({"messages": messages}):
#             pass

#     except KeyboardInterrupt:
#         print("\n\nChatbot: Session interrupted. Goodbye!")
#         break
#     except Exception as e:
#         print(f"Error: {e}\n")

##### Tavily AI
from tavily import TavilyClient

client = TavilyClient()

# results = client.search(query='Latest developments in renewable energy 2025')
# for result in results['results'][:3]:
#     print(f"Title: {result['title']}")
#     print(f"URL: {result['url']}")
#     print(f"Snippet: {result['content'][:200]}...\n")

# result = client.search(
#     query='What are the emerging trends in AI safety?',
#     include_answer=True
# )
# print("Direct Answer:")
# print(result['answer'])

##### Connect Tools to Your Agent
# from langchain_tavily import TavilySearch
# from langgraph.prebuilt import ToolNode, tools_condition

# search_tool = TavilySearch(max_results=3)
# tools = [search_tool]

# llm_with_tools = llm.bind_tools(tools)

# def tool_aware_agent(state: ConversationState):
#     response = llm_with_tools.invoke(state["messages"])
#     return {"messages": [response]}

# agent_with_tools = (
#     StateGraph(ConversationState)
#     .add_node("agent", tool_aware_agent)
#     .add_node("tools", ToolNode(tools=[search_tool]))
#     .add_conditional_edges(
#         "agent",
#         tools_condition,
#     )
#     .add_edge("tools", "agent")
#     .set_entry_point("agent")
#     .compile()
# )

# user_query = "Compare the latest GPU offerings from NVIDIA and AMD"

# print(f"User: {user_query}\n")

# for event in agent_with_tools.stream({"messages": [HumanMessage(content=user_query)]}):
#     if "agent" in event:
#         messages = event["agent"]["messages"]
#         print(f"Bot: {messages[-1].content}")

##### Adding Memory to the ChatBot
# Checkpointing allows your agent to remember previous conversations
# by persisting the state at each node execution

from langgraph.checkpoint.memory import MemorySaver

# Create an in-memory checkpointer
checkpointer = MemorySaver()

# Rebuild the graph with checkpointing enabled
chatbot_graph_with_memory = (
    StateGraph(ConversationState)
    .add_node("agent", dialogue_agent)
    .set_entry_point("agent")
    .set_finish_point("agent")
    .compile(checkpointer=checkpointer)
)

# Example: Using thread_id to maintain separate conversations
# Each thread_id maintains its own conversation history

# Conversation 1 (thread_id="user_1")
print("=== Conversation 1 (User 1) ===")
response_1 = chatbot_graph_with_memory.invoke(
    {"messages": [HumanMessage(content="What is machine learning? shortly in 100 chars.")]},
    config={"configurable": {"thread_id": "user_1"}}
)

# Conversation 2 (thread_id="user_2") - separate conversation history
print("=== Conversation 2 (User 2) ===")
response_2 = chatbot_graph_with_memory.invoke(
    {"messages": [HumanMessage(content="Tell me about quantum computing. shortly in 100 chars.")]},
    config={"configurable": {"thread_id": "user_2"}}
)

# Continue the same conversation (same thread_id)
print("=== Conversation 1 (User 1) ===")
response_3 = chatbot_graph_with_memory.invoke(
    {"messages": [HumanMessage(content="Can you explain neural networks? shortly in 100 chars.")]},
    config={"configurable": {"thread_id": "user_1"}}
)

##### Check What Your Agent Remembers
# Use get_state() to retrieve the conversation history and next steps
print("\n=== Agent Memory State (User 1) ===")
state_snapshot = chatbot_graph_with_memory.get_state(
    config={"configurable": {"thread_id": "user_1"}}
)
print(f"Last message: {state_snapshot.values['messages'][-1]}")
print(f"Next step: {state_snapshot.next}")

print("\n=== Agent Memory State (User 2) ===")
state_snapshot = chatbot_graph_with_memory.get_state(
    config={"configurable": {"thread_id": "user_2"}}
)
print(f"Last message: {state_snapshot.values['messages'][-1]}")
print(f"Next step: {state_snapshot.next}")
