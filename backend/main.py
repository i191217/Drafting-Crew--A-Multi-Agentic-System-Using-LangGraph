from typing import Dict, Any, AsyncIterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent_logic import build_app, deserialize_state, serialize_state, empty_state


print("🚀 Starting FastAPI server initialization...")
api = FastAPI(title="Agent Backend")
print("📊 Building LangGraph app...")
app_graph = build_app()
print("✅ LangGraph app built successfully!")


class ChatRequest(BaseModel):
    user_input: str
    conversation_state: Dict[str, Any] | None = None


async def stream_chat(state_payload: Dict[str, Any]) -> AsyncIterator[str]:
    for step in app_graph.stream(state_payload, stream_mode="values"):
        yield f"data: {serialize_state(step)}\n\n"


@api.post("/chat")
async def chat_handler(req: ChatRequest):
    print(f"📨 Received chat request: '{req.user_input[:50]}{'...' if len(req.user_input) > 50 else ''}'")
    print("🔄 Deserializing state...")
    state = deserialize_state(req.conversation_state or empty_state())
    print("💬 Adding user message to state...")
    updated_messages = list(state["messages"]) + [HumanMessage(content=req.user_input)]
    state["messages"] = updated_messages
    print(f"📊 Current state has {len(state['messages'])} messages")

    async def iterator():
        print("🔄 Starting graph stream...")
        step_count = 0
        for step in app_graph.stream(state, stream_mode="values"):
            step_count += 1
            print(f"📈 Stream step {step_count}: {step.get('router', 'unknown')} node")
            yield (serialize_state(step) | {"event": "step"}).__repr__() + "\n"
        print("🏁 Stream completed, getting final state...")
        # return final state
        final_state = app_graph.invoke(state)
        print("✅ Final state obtained, sending to client...")
        yield (serialize_state(final_state) | {"event": "final"}).__repr__() + "\n"
        print("🎉 Response sent successfully!")

    return StreamingResponse(iterator(), media_type="text/event-stream")


