# import json, os
# import gradio as gr
# import requests
# from typing import Dict, Any, Tuple


# # API_URL = "http://127.0.0.1:8000/chat"
# API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000") + "/chat"


# def submit_message(user_text: str, chat_history: list[Tuple[str, str]], client_state: Dict[str, Any]):
#     print(f"🎯 Gradio: Submitting message: '{user_text[:50]}{'...' if len(user_text) > 50 else ''}'")
#     payload = {"user_input": user_text, "conversation_state": client_state or {}}
#     print(f"📤 Gradio: Sending request to {API_URL}")
    
#     try:
#         with requests.post(API_URL, json=payload, stream=True) as r:
#             print(f"📡 Gradio: Got response with status {r.status_code}")
#             response_text = ""
#             line_count = 0
#             for line in r.iter_lines(decode_unicode=True):
#                 line_count += 1
#                 if not line:
#                     print(f"📭 Gradio: Empty line {line_count}")
#                     continue
#                 print(f"📄 Gradio: Processing line {line_count}: {line[:100]}{'...' if len(line) > 100 else ''}")
#                 try:
#                     event = eval(line)  # using repr on server; safe for our controlled payload
#                     print(f"✅ Gradio: Parsed event: {event.get('event', 'unknown')}")
#                 except Exception as e:
#                     print(f"❌ Gradio: Failed to parse line: {e}")
#                     continue
#                 if event.get("event") == "step":
#                     print("🔄 Gradio: Received step event")
#                     # Optionally, we can surface incremental messages; for now, ignore.
#                     pass
#                 elif event.get("event") == "final":
#                     print("🏁 Gradio: Received final event!")
#                     final_state = event
#                     # Remove event key
#                     final_state.pop("event", None)
#                     client_state = final_state
#                     # Find latest AI text from final_state messages if present
#                     response_text = ""
#                     messages = final_state.get("messages", [])
#                     print(f"💬 Gradio: Processing {len(messages)} messages from final state")
#                     if messages:
#                         # last element may be AI
#                         last = messages[-1]
#                         if last.get("type") == "ai":
#                             response_text = last.get("data", {}).get("content", "")
#                             print(f"🤖 Gradio: Extracted AI response: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")
#                     chat_history = chat_history + [(user_text, response_text)]
#                     print("✅ Gradio: Message processing completed successfully!")
#                     return "", chat_history, client_state
#             print("⚠️ Gradio: Stream ended without final event")
#     except Exception as e:
#         print(f"❌ Gradio: Request failed: {e}")
    
#     print("🔄 Gradio: Returning with no changes")
#     return "", chat_history, client_state


# print("🎨 Gradio: Creating UI components...")
# with gr.Blocks(title="Agent Chat") as demo:
#     gr.Markdown("## Multi-Agent Drafting Chat")
#     chatbot = gr.Chatbot(label="Conversation", type="messages")
#     txt = gr.Textbox(label="Your message")
#     state = gr.State({})

#     txt.submit(submit_message, inputs=[txt, chatbot, state], outputs=[txt, chatbot, state])
#     print("✅ Gradio: UI components created successfully!")


# if __name__ == "__main__":
#     print("🚀 Gradio: Starting Gradio app...")
#     demo.launch(server_name="0.0.0.0", server_port=7860)





import json
import os
import logging
import gradio as gr
import requests
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load BACKEND_URL
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000") + "/chat"
logger.debug(f"Using API_URL: {API_URL}")

def submit_message(user_text: str, chat_history: List[Tuple[str, str]], client_state: Dict[str, Any]):
    logger.debug(f"Submitting message: '{user_text[:50]}{'...' if len(user_text) > 50 else ''}'")
    payload = {"user_input": user_text, "conversation_state": client_state or {}}
    logger.debug(f"Sending request to {API_URL} with payload: {payload}")

    try:
        with requests.post(API_URL, json=payload, stream=True) as r:
            logger.debug(f"Got response with status {r.status_code}")
            response_text = ""
            line_count = 0
            for line in r.iter_lines(decode_unicode=True):
                line_count += 1
                if not line:
                    logger.debug(f"Empty line {line_count}")
                    continue
                logger.debug(f"Processing line {line_count}: {line[:100]}{'...' if len(line) > 100 else ''}")
                try:
                    event = eval(line)  # Note: eval is risky; consider json.loads if possible
                    logger.debug(f"Parsed event: {event.get('event', 'unknown')}")
                except Exception as e:
                    logger.error(f"Failed to parse line: {e}")
                    continue
                if event.get("event") == "step":
                    logger.debug("Received step event")
                    # Optionally handle incremental updates
                    pass
                elif event.get("event") == "final":
                    logger.debug("Received final event!")
                    final_state = event
                    final_state.pop("event", None)
                    client_state = final_state
                    response_text = ""
                    messages = final_state.get("messages", [])
                    logger.debug(f"Processing {len(messages)} messages from final state")
                    if messages:
                        last = messages[-1]
                        if last.get("type") == "ai":
                            response_text = last.get("data", {}).get("content", "")
                            logger.debug(f"Extracted AI response: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")
                    chat_history = chat_history + [(user_text, response_text)]
                    logger.debug("Message processing completed successfully!")
                    return "", chat_history, client_state
            logger.warning("Stream ended without final event")
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
    
    logger.debug("Returning with no changes")
    return "", chat_history, client_state

logger.info("Creating UI components...")
with gr.Blocks(title="Agent Chat") as demo:
    gr.Markdown("## Multi-Agent Drafting Chat")
    chatbot = gr.Chatbot(label="Conversation", type="messages")
    txt = gr.Textbox(label="Your message", placeholder="Type your message here...")
    state = gr.State(value={})
    submit_btn = gr.Button("Send")  # Added explicit button for clarity

    # Bind submit to both Textbox (Enter key) and Button click
    txt.submit(submit_message, inputs=[txt, chatbot, state], outputs=[txt, chatbot, state])
    submit_btn.click(submit_message, inputs=[txt, chatbot, state], outputs=[txt, chatbot, state])
    logger.info("UI components created successfully!")

if __name__ == "__main__":
    logger.info("Starting Gradio app...")
    demo.launch(server_name="0.0.0.0", server_port=7860)