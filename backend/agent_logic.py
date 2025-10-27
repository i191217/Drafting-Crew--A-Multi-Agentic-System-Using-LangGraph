from typing import Annotated, Sequence, TypedDict, Dict, Any, Iterable
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.messages import messages_from_dict, messages_to_dict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from serpapi import GoogleSearch, GoogleScholarSearch
import os
import json

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    router: str
    coordinator_instructions: str
    research_summary: str
    draft_text: str
    final_response: str


@tool
def web_search(query: str) -> str:
    """Find general knowledge information using Google search."""
    serpapi_params = {"engine": "google", "api_key": os.getenv("SERP_API_KEY")}
    search = GoogleSearch({**serpapi_params, "q": query, "num": 1})
    results = search.get_dict().get("organic_results", [])
    contexts = "\n---\n".join(["\n".join([x.get("title", ""), x.get("snippet", ""), x.get("link", "")]) for x in results])
    return contexts


@tool
def google_scholar(query: str) -> str:
    """
    Search Google Scholar for academic articles.
    
    Args:
        query (str): The search query.
    
    Returns:
        List[Dict]: A list of academic papers with title, authors, abstract, and link.
    """
    serpapi_params = {"api_key": os.getenv("SERP_API_KEY")}
    search = GoogleScholarSearch({**serpapi_params, "q": query, "num": 1})
    results = search.get_dict().get("organic_results", [])
    formatted_results = []
    for result in results:
        article_info = {
            "title": result.get("title", "N/A"),
            "authors": result.get("publication_info", {}).get("authors", "N/A"),
            "abstract": result.get("snippet", "N/A"),
            "link": result.get("link", "N/A"),
        }
        formatted_results.append(article_info)
    return json.dumps(formatted_results)


@tool
def save(filename: str) -> str:
    """Save the final response to a text file."""
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    print("Actual writing will be decided by the coordinator; this tool simply acknowledges.")
    # Actual writing will be decided by the coordinator; this tool simply acknowledges.
    return f"Ready to save as '{filename}'. Send final content to client to persist."


TOOLS = [web_search, google_scholar]
COORDINATION_TOOLS = [save]


def _get_models():
    model = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Specify the OpenAI model (e.g., gpt-4o-mini, gpt-4o, etc.)
        temperature=0.2
        ).bind_tools(TOOLS, tool_choice="any")

    coordinator_model = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Specify the OpenAI model (e.g., gpt-4o-mini, gpt-4o, etc.)
        temperature=0.2
        ).bind_tools(COORDINATION_TOOLS)

    model_no_tools = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Specify the OpenAI model (e.g., gpt-4o-mini, gpt-4o, etc.)
        temperature=0.2
        )

# # Model for all Nodes except Coordinate node
#     model = ChatGoogleGenerativeAI(
#         # groq_api_key = os.getenv("GROQ_API_KEY"),
#         # model_name = "llama-3.1-8b-instant",
#         google_api_key = os.getenv("GOOGLE_API_KEY"),
#         model = "gemini-1.5-flash-latest",
#         api_version="v1",
#         temperature = 0.2 
#         ).bind_tools(TOOLS, tool_choice="any")

# # Model for Coordinate node
#     coordinator_model = ChatGoogleGenerativeAI(
#        google_api_key = os.getenv("GOOGLE_API_KEY"),
#         model = "gemini-1.5-flash-latest", 
#         api_version="v1",
#         temperature = 0.2 
#         ).bind_tools(COORDINATION_TOOLS)

# # Model without Tools
#     model_no_tools = ChatGoogleGenerativeAI(
#         google_api_key = os.getenv("GOOGLE_API_KEY"),
#         model = "gemini-1.5-flash-latest",  
#         api_version="v1",
#         temperature = 0.2
#         )

    return model, coordinator_model, model_no_tools


def coordination(state: AgentState) -> AgentState:
    _, coordinator_model, _ = _get_models()
    current_document = state.get("final_response") or state.get("draft_text") or ""
    system_prompt = SystemMessage(content=f"""
    You are the Coordinator Agent in a multi-agent drafting system. Your role is to act as the central orchestrator for user requests 
    to draft or refine text, such as emails, essays, or other written content. Analyze the user's input to determine the task type, 
    required steps, and which agents to involve. Maintain conversation flow, incorporate user feedback, and route tasks accordingly.

    Key responsibilities:

    Parse user request: Identify if research is needed (e.g., factual topics), or finalizing.
    Route workflow: Decide sequence or parallelism (e.g., research first if facts are required).
    Handle iterations: If user provides feedback, decide if it requires re-research.
    User interaction: Respond conversationally, present drafts for approval, and ask clarifying questions if needed.
    End task: When user wants to save and finish the draft, invoke the 'save' tool to store the final content.

    Always be helpful, concise, and focused on progressing the draft. Do not perform research, drafting, or editing yourself, delegate 
    to specialized agents.   

    You are an agent with access to only this tool: save. Do not invent or call any other tools, including parse_task, brave_search or 
    similar. If a task requires decomposition, reason step-by-step in your response instead of using a tool.

    The current document content is:{current_document}
    
    Output Format:
    Always respond in JSON with the following format:

    "description":  "Provide whatever instructions/details you have to pass on to the next agent or if you have to provide an answer 
                    to the User.",
    "notes": "Provide notes if required as this is optional."

    """)
    if (state["messages"][-1]) and isinstance(state["messages"][-1], ToolMessage):
        return state

    # Expect that the latest user message is already in state["messages"].
    all_messages = [system_prompt] + list(state["messages"])  # no input() calls
    response = coordinator_model.invoke(all_messages)

    print(f"\n🤖 Coordinator: {response.content}")
    print()

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    new_state: AgentState = {
        "messages": list(state["messages"]) + [response],
        "router": "coordinate",
        "coordinator_instructions": response.content,
        "research_summary": state.get("research_summary", ""),
        "draft_text": state.get("draft_text", ""),
        "final_response": state.get("final_response", "")
    }
    return new_state


def tools_node(state: AgentState) -> AgentState:
    tools = {t.name: t for t in (TOOLS + COORDINATION_TOOLS)}
    last_msg = state["messages"][-1] if state.get("messages") else None
    tool_messages: list[ToolMessage] = []
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        for call in last_msg.tool_calls:
            name = call.get("name")
            args = call.get("args", {}) or {}
            tool = tools.get(name)
            if tool is None:
                output = f"Tool '{name}' not found."
                print("Tool Message:", output)
            else:
                try:
                    output = tool.invoke(args)
                    print("Tool Message:", output)
                except Exception as exc:
                    output = f"Tool '{name}' failed: {exc}"
                    print("Tool Message:", output)
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call.get("id", name or "tool")))

    new_messages = list(state.get("messages", [])) + tool_messages
    return {
        "messages": new_messages,
        "router": state.get("router", "coordinate"),
        "coordinator_instructions": state.get("coordinator_instructions", ""),
        "research_summary": state.get("research_summary", ""),
        "draft_text": state.get("draft_text", ""),
        "final_response": state.get("final_response", ""),
    }


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "Save"
        return "Continue"
    if isinstance(last_msg, ToolMessage):
        return "End"
    return "Continue"


def should_progress(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "Tool"
    return "No Tool"


def router_func(state: AgentState) -> str:
    router = state.get("router", "coordinate")
    if router in ("coordinate", "research", "draft", "edit"):
        return router
    return "coordinate"


def research(state: AgentState) -> AgentState:
    model, _, model_no_tools = _get_models()
    system_prompt = SystemMessage(content=f"""
    You are the Researcher Agent in a multi-agent drafting system. Your role is to gather relevant information, facts, data, 
    or references needed for the draft based on the user's request and shared state. Use relevant tools to collect 
    accurate, summarized information. Output only the research summary in a structured format (e.g., bullet points with sources) to 
    be used by the Drafter.

    Key responsibilities:
    - Analyze the task: Focus on key topics, questions, or gaps in knowledge from the user request.
    - Conduct research: Query reliable sources, summarize findings without bias, and cite origins.
    - Relevance: Only include info directly applicable to the draft; keep it concise (aim for 200-500 words).

    Coordinator instructions: {state.get('coordinator_instructions', '')}

    You are an agent with access to only this tool: web_search, google_scholar. Do not invent or call any other tools, including parse_task, brave_search or 
    similar. If a task requires decomposition, reason step-by-step in your response instead of using a tool.

    Do not draft or edit text, your output is purely informational support. 
        """)

    last_msg = state["messages"][-1]
    all_messages = [system_prompt] + list(state["messages"])
    response = model_no_tools.invoke(all_messages) if isinstance(last_msg, ToolMessage) else model.invoke(all_messages)

    print(f"\n🧪 Researcher: {response.content}")
    print()

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print()
        print("No Tool call made")
        print()

    new_state: AgentState = {
        "messages": list(state["messages"]) + [response],
        "router": "research",
        "coordinator_instructions": state.get("coordinator_instructions", ""),
        "research_summary": response.content,
        "draft_text": state.get("draft_text", ""),
        "final_response": state.get("final_response", ""),
    }
    return new_state


def drafting(state: AgentState) -> AgentState:
    model, _, model_no_tools = _get_models()
    system_prompt = SystemMessage(content = f"""
    You are the Drafter Agent in a multi-agent drafting system. Your role is to generate the initial or revised draft of the 
    requested text (e.g., email, essay etc.) using the user's request, research notes from the Researcher, and any prior feedback. 
    Produce creative, coherent, and tailored content. Support iterative refinements based on user or Editor input.

    Key responsibilities:
    - Incorporate inputs: Blend user details, research notes, and style preferences (e.g., formal, concise).
    - Generate draft: Write complete, well-structured text; for essays, include intro/body/conclusion.
    - Iterations: If feedback is provided, revise accordingly (e.g., "make it shorter" or "add examples").

    Research summary: {state.get('research_summary', '')}

    You are an agent with access to only this tool: web_search, google_scholar. Do not invent or call any other tools, including parse_task, brave_search or 
    similar. If a task requires decomposition, reason step-by-step in your response instead of using a tool.

    Be versatile across writing types. Do not research or edit for grammar—focus on content creation. If the draft is initial, keep 
    it as a solid starting point.
    """)

    last_msg = state["messages"][-1]
    all_messages = [system_prompt] + list(state["messages"])
    response = model_no_tools.invoke(all_messages) if isinstance(last_msg, ToolMessage) else model.invoke(all_messages)

    print(f"\n📝 Drafter: {response.content}")
    print()

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print()
        print("No Tool call made")
        print()

    new_state: AgentState = {
        "messages": list(state["messages"]) + [response],
        "router": "draft",
        "coordinator_instructions": state.get("coordinator_instructions", ""),
        "research_summary": state.get("research_summary", ""),
        "draft_text": response.content,
        "final_response": state.get("final_response", ""),
    }
    return new_state


def editing(state: AgentState) -> AgentState:
    model, _, model_no_tools = _get_models()
    system_prompt = SystemMessage(content = f"""
    You are the Editor Agent in a multi-agent drafting system. Your role is to review and refine the current draft for quality, 
    focusing on grammar, style, coherence, clarity, and improvements. Provide suggestions and a revised version if needed. Act 
    as a critical eye to enhance the draft without changing core meaning unless specified.

    Current draft: {state.get('draft_text', '')}

    Key responsibilities:
    - Review draft: Check for errors (spelling, grammar), flow, consistency, and engagement.
    - Suggest changes: Output a list of improvements (e.g., "Rephrase sentence X for clarity") followed by the edited draft.
    - Incorporate feedback: Apply user or Coordinator notes (e.g., "make it more persuasive").

    You are an agent with access to only this tool: web_search, google_scholar. Do not invent or call any other tools, including parse_task, brave_search or 
    similar. If a task requires decomposition, reason step-by-step in your response instead of using a tool.

    Maintain the original intent and length unless instructed. Do not add new content or research—focus on polishing. If the draft 
    is already strong, suggest minimal changes.

    """)

    last_msg = state["messages"][-1]
    all_messages = [system_prompt] + list(state["messages"])
    response = model_no_tools.invoke(all_messages) if isinstance(last_msg, ToolMessage) else model.invoke(all_messages)

    print(f"\n📝 Editor: {response.content}")
    print()

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print()
        print("No Tool call made")
        print()

    new_state: AgentState = {
        "messages": list(state["messages"]) + [response],
        "router": "edit",
        "coordinator_instructions": state.get("coordinator_instructions", ""),
        "research_summary": state.get("research_summary", ""),
        "draft_text": state.get("draft_text", ""),
        "final_response": response.content,
    }
    return new_state


def build_app():
    graph = StateGraph(AgentState)
    graph.add_node("Coordinate_node", coordination)
    graph.add_node("Research_node", research)
    graph.add_node("Draft_node", drafting)
    graph.add_node("Edit_node", editing)
    # graph.add_node("Tools_node", ToolNode([web_search, google_scholar, save]))
    graph.add_node("Tools_node", tools_node)

    graph.set_entry_point("Coordinate_node")

    graph.add_conditional_edges(
        "Coordinate_node",
        should_continue,
        {
            "Continue": "Research_node",
            "Save": "Tools_node",
            "End": END,
        },
    )

    graph.add_conditional_edges(
        "Research_node",
        should_progress,
        {"Tool": "Tools_node", "No Tool": "Draft_node"},
    )

    graph.add_conditional_edges(
        "Draft_node",
        should_progress,
        {"Tool": "Tools_node", "No Tool": "Edit_node"},
    )

    graph.add_conditional_edges(
        "Edit_node",
        should_progress,
        {"Tool": "Tools_node", "No Tool": "Coordinate_node"},
    )

    graph.add_conditional_edges(
        "Tools_node",
        router_func,
        {
            "coordinate": "Coordinate_node",
            "research": "Research_node",
            "draft": "Draft_node",
            "edit": "Edit_node",
        },
    )

    return graph.compile()


def serialize_state(state: AgentState) -> Dict[str, Any]:
    return {
        "messages": messages_to_dict(list(state.get("messages", []))),
        "router": state.get("router", "coordinate"),
        "coordinator_instructions": state.get("coordinator_instructions", ""),
        "research_summary": state.get("research_summary", ""),
        "draft_text": state.get("draft_text", ""),
        "final_response": state.get("final_response", ""),
    }


def deserialize_state(payload: Dict[str, Any]) -> AgentState:
    messages_dicts = payload.get("messages", [])
    return {
        "messages": messages_from_dict(messages_dicts),
        "router": payload.get("router", "coordinate"),
        "coordinator_instructions": payload.get("coordinator_instructions", ""),
        "research_summary": payload.get("research_summary", ""),
        "draft_text": payload.get("draft_text", ""),
        "final_response": payload.get("final_response", ""),
    }


def empty_state() -> AgentState:
    return {
        "messages": [],
        "router": "coordinate",
        "coordinator_instructions": "",
        "research_summary": "",
        "draft_text": "",
        "final_response": "",
    }


