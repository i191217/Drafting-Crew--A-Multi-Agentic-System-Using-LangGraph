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
from serpapi import GoogleScholarSearch

load_dotenv()

# Will store the final AI response in ai_response as well
ai_response = ""
instructions = ""
summary = ""
draft = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_topics: str
    next_agent: str
    router: str
    calls_tool: int


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

@tool
def google_scholar(query: str):
    """
    Search Google Scholar for academic articles.
    
    Args:
        query (str): The search query.
    
    Returns:
        List[Dict]: A list of academic papers with title, authors, abstract, and link.
    """
    
    serpapi_params = {
    "api_key": os.getenv("SERP_API_KEY")
    }

    search = GoogleScholarSearch({
        **serpapi_params,
        "q": query,
        "num": 1
        })
    results = search.get_dict()["organic_results"]

    # Extract relevant information
    formatted_results = []
    for result in results:
        article_info = {
            "title": result.get("title", "N/A"),
            "authors": result.get("publication_info", {}).get("authors", "N/A"),
            "abstract": result.get("snippet", "N/A"),
            "link": result.get("link", "N/A")
        }
        formatted_results.append(article_info)

    for i, result in enumerate(formatted_results, 1):
        print("-" * 80)
        print(f"\nResult {i}:")
        print()
        print(f"Title: {result['title']}")
        print()
        print(f"Authors: {result['authors']}")
        print()
        print(f"Abstract: {result['abstract']}")
        print()
        print(f"Link: {result['link']}")
        print()
        print("-" * 80)
    
    return formatted_results


@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """

    global ai_response

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"


    try:
        with open(filename, 'w') as file:
            file.write(ai_response)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"


tools = [web_search, google_scholar]
coordination_tools = [save]


model = ChatGroq(
    groq_api_key = os.getenv("GROQ_API_KEY"),
    model_name = "llama-3.1-8b-instant"
    ).bind_tools(tools)

coordinator_model = ChatGroq(
    groq_api_key = os.getenv("GROQ_API_KEY"),
    model_name = "llama-3.1-8b-instant"
    ).bind_tools(coordination_tools)

def coordination(state: AgentState) -> AgentState:
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

    Always be helpful, concise, and focused on progressing the draft. Do not perform research, drafting, or editing yourself—delegate 
    to specialized agents.   

    The current document content is:{ai_response}
    
    Output Format:
    Always respond in JSON with the following format:

    "description":  "Provide whatever instructions/details you have to pass on to the next agent (Researcher, Finalizer) or
                    if you have to provide an answer to the User.",
    "notes": "Provide notes if required as this is optional."

    """)

    if not state["messages"]:
        user_input = input("🤖 AI: Welcome to the Drafting Crew. What would you like to create?\n")

    elif isinstance(state["messages"][-1], ToolMessage):
        return state 

    else:
        user_input = input("\n 🤖 AI: What would you like to do with the document?\n")


    print(f"\n👤 USER: {user_input}")
    user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = coordinator_model.invoke(all_messages)

    print(f"\n🤖 Coordinator: {response.content}")
    print()
    
    global instructions 
    instructions = response.content

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message] + [response],
            "router": "coordinate"}


def should_continue(state: AgentState) -> str:
    message = state["messages"]
    last_msg = message[-1]

    if isinstance(last_msg, AIMessage):
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "Save"
        else:
            return "Continue"
    elif isinstance(last_msg, ToolMessage):
        return "End"
    
def should_progress(state: AgentState) -> str:
    message = state["messages"]
    last_msg = message[-1]

    if isinstance(last_msg, AIMessage):
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "Tool"
        else:
            return "No Tool"

def router_func(state: AgentState) -> str:
    if state["router"] == "coordinate":
        return "coordinate"
    elif state["router"] == "research":
        return "research"
    elif state["router"] == "draft":
        return "draft"
    else:
        return "edit"


def research(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are the Researcher Agent in a multi-agent drafting system. Your role is to gather relevant information, facts, data, 
    or references needed for the draft based on the user's request and shared state. Use relevant tools to collect 
    accurate, summarized information. Output only the research summary in a structured format (e.g., bullet points with sources) to 
    be used by the Drafter.

    Key responsibilities:
    - Analyze the task: Focus on key topics, questions, or gaps in knowledge from the user request.
    - Conduct research: Query reliable sources, summarize findings without bias, and cite origins.
    - Relevance: Only include info directly applicable to the draft; keep it concise (aim for 200-500 words).

    The instructions from your Coordinator are {instructions}

    Do not draft or edit text, your output is purely informational support. 
        """)

    # messages = state["messages"]
    # tool_msg = []
    # count = 0  
    # calls = 0

    # for message in reversed(messages):
    #     if (isinstance(message, ToolMessage)):
    #         tool_msg.append(message)
    #         count += 1
    #         if count == state["calls_tool"]:
    #             break

#   System Prompt + [User question & Coordinator Response] 
    all_messages = [system_prompt] + list(state["messages"])
    # if tool_msg:  
    #     all_messages += tool_msg

    response = model.invoke(all_messages)

    print(f"\n🧪 Researcher: {response.content}")
    print()

    global summary 
    summary = response.content 

    if hasattr(response, "tool_calls") and response.tool_calls:
        # calls = len(response.tool_calls)
        # print("Tool calls:", calls)
        # print()

        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

#          [User question & Coordinator Response] + Researcher Response
    return {"messages": list(state["messages"]) + [response],
            "router":"research" }


def drafting(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content = f"""
    You are the Drafter Agent in a multi-agent drafting system. Your role is to generate the initial or revised draft of the 
    requested text (e.g., email, essay etc.) using the user's request, research notes from the Researcher, and any prior feedback. 
    Produce creative, coherent, and tailored content. Support iterative refinements based on user or Editor input.

    Key responsibilities:
    - Incorporate inputs: Blend user details, research notes, and style preferences (e.g., formal, concise).
    - Generate draft: Write complete, well-structured text; for essays, include intro/body/conclusion.
    - Iterations: If feedback is provided, revise accordingly (e.g., "make it shorter" or "add examples").

    The research summary from the Research_node is {summary}

    Be versatile across writing types. Do not research or edit for grammar—focus on content creation. If the draft is initial, keep 
    it as a solid starting point.
    """)

    # messages = state["messages"]
    # tool_msg = []
    # count = 0  
    # calls = 0

    # for message in reversed(messages):
    #     if (isinstance(message, ToolMessage)):
    #         tool_msg.append(message)
    #         count += 1
    #         if count == state["calls_tool"]:
    #             break

#   System Prompt + [User question & Coordinator + Researcher Response] 
    all_messages = [system_prompt] + list(state["messages"])
    # if tool_msg:  
    #     all_messages += tool_msg

    response = model.invoke(all_messages)

    print(f"\n📝 Drafter: {response.content}")
    print()

    global draft 
    draft = response.content

    if hasattr(response, "tool_calls") and response.tool_calls:
        # calls = len(response.tool_calls)
        # print("Tool calls:", calls)
        # print()

        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

        #  [User question & Coordinator + Researcher Response] + Drafter Response
    return {"messages": list(state["messages"]) + [response], 
            "router": "draft"}


def editing(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content = f"""
    You are the Editor Agent in a multi-agent drafting system. Your role is to review and refine the current draft for quality, 
    focusing on grammar, style, coherence, clarity, and improvements. Provide suggestions and a revised version if needed. Act 
    as a critical eye to enhance the draft without changing core meaning unless specified.

    Here is the current draft {draft}

    Key responsibilities:
    - Review draft: Check for errors (spelling, grammar), flow, consistency, and engagement.
    - Suggest changes: Output a list of improvements (e.g., "Rephrase sentence X for clarity") followed by the edited draft.
    - Incorporate feedback: Apply user or Coordinator notes (e.g., "make it more persuasive").

    Maintain the original intent and length unless instructed. Do not add new content or research—focus on polishing. If the draft 
    is already strong, suggest minimal changes.

    """)

    all_messages = [system_prompt] + list(state["messages"])
    # if tool_msg:  
    #     all_messages += tool_msg

    response = model.invoke(all_messages)

    print(f"\n🔍 Editor: {response.content}")
    print()

    global ai_response 
    ai_response = response.content

    if hasattr(response, "tool_calls") and response.tool_calls:
        # calls = len(response.tool_calls)
        # print("Tool calls:", calls)
        # print()

        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

        #  [User question & Coordinator + Researcher Response] + Drafter Response
    return {"messages": list(state["messages"]) + [response], 
            "router": "edit"}



graph = StateGraph(AgentState)

graph.add_node("Coordinate_node", coordination)
graph.add_node("Research_node", research)
graph.add_node("Draft_node", drafting)
graph.add_node("Edit_node", editing)
graph.add_node("Tools_node", ToolNode(tools))


graph.set_entry_point("Coordinate_node")

graph.add_conditional_edges(
    "Coordinate_node",
    should_continue,
    {
        "Continue": "Research_node",
        "Save": "Tools_node",
        "End": END
    }
)

graph.add_conditional_edges(
    "Research_node",
    should_progress,
    {
        "Tool": "Tools_node",
        "No Tool": "Draft_node"
    }
)

graph.add_conditional_edges(
    "Draft_node",
    should_progress,
    {
        "Tool": "Tools_node",
        "No Tool": "Edit_node"
    }
)

graph.add_conditional_edges(
    "Edit_node",
    should_progress,
    {
        "Tool": "Tools_node",
        "No Tool": "Coordinate_node"
    }
)

graph.add_conditional_edges(
    "Tools_node",
    router_func,
    {
        "coordinate": "Coordinate_node",
        "research": "Research_node",
        "draft": "Draft_node",
        "edit": "Edit_node"
    }
)


app = graph.compile()


def print_tool_messages(messages):

    """Function I made to print the TOOL messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-2:]:
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