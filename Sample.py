# Use OpenAI Whisper for Speech to Text conversion
# Use 11Labs model for Text to Speech conversion

# Sangeen's Review

# Use LangSmith
# Instead of providing format in the prompt, create a separate function and provide a fixed JSON format. [Forced approach]
# There's a possibility that the LLM might hallucinate while making multiple Tool calls.


from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os, re, json
from serpapi import GoogleSearch

load_dotenv()

# This is the global variable to store document content
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_topics: str
    next_agent: str
    router: str


@tool
def web_search(query: str):
    """Finds general knowledge information about different topics using Google search. Can also be used
    to augment more 'general' knowledge to a previous specialist query."""

    serpapi_params = {
    "engine": "google",
    "api_key": os.getenv("SERP_API_KEY")
    }

    search = GoogleSearch({
        **serpapi_params,
        "q": query,
        "num": 1
    })

    results = search.get_dict()["organic_results"]
    contexts = "\n---\n".join(
        ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
    )

    return contexts

tools = [web_search]


model = ChatGroq(
    groq_api_key = os.getenv("GROQ_API_KEY"),
    model_name = "llama-3.1-8b-instant"
    ).bind_tools(tools)

def coordination(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are the Coordinator Agent in a multi-agent drafting system. Your role is to act as the central orchestrator for user requests 
    to draft or refine text, such as emails, essays, or other written content. Analyze the user's input to determine the task type, 
    required steps, and which agents to involve (Researcher, Finalizer). Maintain conversation flow, incorporate user 
    feedback, and route tasks accordingly.

    Key responsibilities:

    Parse user request: Identify if research is needed (e.g., factual topics), or finalizing.
    Route workflow: Decide sequence or parallelism (e.g., research first if facts are required).
    Handle iterations: If user provides feedback, decide if it requires re-research.
    User interaction: Respond conversationally, present drafts for approval, and ask clarifying questions if needed.
    End task: When user approves, route to Finalizer.

    Always be helpful, concise, and focused on progressing the draft. Do not perform research, drafting, or editing yourself—delegate 
    to specialized agents.

    The current document content is:{document_content}
    
    Output Format:
    Always respond in JSON with the following format:

    "next_agent": "Researcher | Finalizer",
    "description":  "Provide whatever instructions/details you have to pass on to the next agent (Researcher, Finalizer) or
                    if you have to provide an answer to the User.",
    "notes": "Provide notes if required as this is optional.",
    "research_topics": "Name ONLY 2 - 3 topics regarding the question to search them on Google using the tool web_search."

    """)

    if not state["messages"]:
        user_input = input("🤖 AI: Welcome to the Drafting Crew. What would you like to create?\n")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\n 🤖 AI: What would you like to do with the document?\n")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 Coordinator: {response.content}")
    print()

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message] + [response], 
            "router": "research"}


def should_continue(state: AgentState) -> str:

    message = state["messages"]

    for i in reversed(message[-2:]):
        if isinstance(i, AIMessage):
            if hasattr(i, "tool_calls") and i.tool_calls:
                return "Tool"
            else:
                return "Save"

def router_func(state: AgentState) -> str:
    if state["router"] == "research":
        return "research"
    elif state["router"] == "draft":
        return "draft"
    elif state["router"] == "edit":
        return "edit"
    else:
        return "end"


def research(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
    You are the Researcher Agent in a multi-agent drafting system. Your role is to gather relevant information, facts, data, 
    or references needed for the draft based on the user's request and shared state. Use tools like web search etc. to collect 
    accurate, summarized notes. Output only the research summary in a structured format (e.g., bullet points with sources) to 
    be used by the Drafter.

    Key responsibilities:
    - Analyze the task: Focus on key topics, questions, or gaps in knowledge from the user request.
    - Conduct research: Query reliable sources, summarize findings without bias, and cite origins.
    - Relevance: Only include info directly applicable to the draft; keep it concise (aim for 200-500 words).

    Do not draft or edit text, your output is purely informational support. If no research is needed, return an empty summary.
        """)

    messages = state["messages"]

    for message in reversed(messages):
        if (isinstance(message, ToolMessage)):
            tool_msg = message
            break

# System Prompt + [User question & Coordinator Response] + Response From Tool
    all_messages = [system_prompt] + list(state["messages"]) + [tool_msg]

    response = model.invoke(all_messages)

    print(f"\n🧪 Researcher: {response.content}")
    print()

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

#                       [User question & Coordinator Response] + Researcher Response
    return {"messages": list(state["messages"]) + [response], 
            "router": "draft"}


    # response = model.invoke([system_prompt] + list(state["messages"]) + Tool_Messages)

def finalize(state: AgentState) -> AgentState:
    print("You are in Finalize Node, SAVE you Work NOW!!!")
    state["next_agent"] = "Draft"
    return state

def drafting(state: AgentState) -> AgentState:
    print("You are in drafting Node, draft you Work NOW!!!")

    return state

def editing(state: AgentState) -> AgentState:
    print("You are in editing Node, edit you Work NOW!!!")

    return state


graph = StateGraph(AgentState)

graph.add_node("Coordinate_node", coordination)
graph.add_node("Research_node", research)
graph.add_node("Draft_node", drafting)
graph.add_node("Edit_node", editing)
graph.add_node("Finalize_node", finalize)
graph.add_node("Tools_node", ToolNode(tools))


graph.set_entry_point("Coordinate_node")

graph.add_conditional_edges(
    "Coordinate_node",
    should_continue,
    {
        "Tool": "Tools_node",
        "Save": "Finalize_node"
    }
)

graph.add_conditional_edges(
    "Tools_node",
    router_func,
    {
        "research": "Research_node",
        "draft": "Draft_node",
        "edit": "Edit_node",
        "end": END
    }
)

graph.add_edge("Research_node", "Tools_node")
graph.add_edge("Draft_node", "Tools_node")
graph.add_edge("Edit_node", "Tools_node")



# graph.add_conditional_edges(
#     "Research_node",
#     should_research,
#     {
#         "Tool": "Tools_node",
#         "Draft": "Draft_node"
#     }
# )

# graph.add_conditional_edges(
#     "Draft_node",
#     should_draft,
#     {
#         "Tool": "Tools_node",
#         "Edit": "Edit_node"
#     }
# )

# graph.add_conditional_edges(
#     "Edit_node",
#     should_edit,
#     {
#         "Tool": "Tools_node",
#         "AI_Response": "Coordinate_node"
#     }
# )




# graph.add_edge("Finalize_node", END)


app = graph.compile()


def print_tool_messages(messages):

    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️  TOOL RESULT: {message.content}")

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        # This checks if the step dictionary contains a key named "messages"
        if "messages" in step: 
            print_tool_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()