from dotenv import load_dotenv
load_dotenv()
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from typing import Literal

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
memory = SqliteSaver.from_conn_string(":memory:")
graph = graph_builder.compile(
    checkpointer=memory,
    # This is new!
    interrupt_before=["tools"],
    # Note: can also interrupt **after** actions, if desired.
    # interrupt_after=["tools"]
)

user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream({"messages": [("user", user_input)]}, config)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# snapshot = graph.get_state(config)
# existing_message = snapshot.values["messages"]
# tool_message = existing_message[-1].content[0].get([0])
# tool_id_message = existing_message[-2]
# tool_call_message = existing_message.content[0]

snapshot = graph.get_state(config)
tool_object = snapshot.values["messages"][-1]
tool_object.type = 'tool'

print(snapshot.values['messages'])

breakpoint()

# tool_call_data = existing_message.content[1]
# print(tool_call_data)


# existing_message.pretty_print()
# This looks exactly the same as the previous example, but now we are 
# going to add an AI Message to the state, manually.

from langchain_core.messages import AIMessage
answer = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs."
)

new_messages = [
    # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
    ToolMessage(content=tool_call_message, tool_call_id=tool_call_data['id']),
    # And then directly "put words in the LLM's mouth" by populating its response.
    AIMessage(content=answer),
]

# new_messages[-1].pretty_print()

graph.update_state(
    # Which state to update
    config,
    # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
    # to the existing state. We will review how to update existing messages in the next section!
    {"messages": new_messages},
)

# print("\n\nLast 2 messages;")

# This actually makes a lot of sense.
# .get_state() returns a snapshot of the state at the time of the call, for config.
# config is where we have been storing our memory, so its important we pass this to fetch 
# the correct instance of the memory.
# print(graph.get_state(config).values["messages"][-2:])