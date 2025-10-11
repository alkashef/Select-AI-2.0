from PIL import Image
import streamlit as st
from dotenv import load_dotenv

import os
import asyncio
import datetime as dt
from typing import List

from backend import EventLoopThread, Message, Agent, get_ai, logger
from constants import ENV_PATH
from constants import TERADATA_LOGO_PATH, CHARTS_PATH

load_dotenv(ENV_PATH)

MAX_MESSAGES: int = 100  # Cap in-memory history length

def get_or_create_event_loop():
    """Get or create the persistent event loop thread."""
    if "event_loop" not in st.session_state:
        elt = EventLoopThread()
        elt.start()
        st.session_state["event_loop"] = elt

    return st.session_state["event_loop"]


# -------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize session state with messages, logger, and AI instance."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # type: List[Message] # type: ignore

    # Initialize ai_instance key first to avoid KeyError
    if "ai_instance" not in st.session_state:
        st.session_state["ai_instance"] = None

    # Initialize event loop
    get_or_create_event_loop()
    
    # Only initialize AI if it hasn't been initialized yet
    if st.session_state["ai_instance"] is None:
        try:
            loop = st.session_state["event_loop"]
            ai_impl = loop.run_coroutine(get_ai())
            st.session_state["ai_instance"] = ai_impl
            # Warmup: optional
            warmup_func = getattr(ai_impl, "warmup", None)
            if warmup_func and asyncio.iscoroutinefunction(warmup_func):
                loop = st.session_state["event_loop"]
                loop.run_coroutine(warmup_func())
            elif warmup_func:
                warmup_func()
            logger.event("ai.init", backend=ai_impl.__class__.__name__)
        except Exception as e:
            logger.event("ai.init.error", error=str(e))
            st.session_state["ai_instance"] = None


# -------------------------------------------------------------------------
# SIDEBAR RENDERING
# -------------------------------------------------------------------------
def render_sidebar() -> None:
    """Render the left sidebar with branding and description."""
    with st.sidebar:
        if TERADATA_LOGO_PATH.exists():
            st.image(str(TERADATA_LOGO_PATH))

        st.title("Select AI 2.0")

        st.markdown(
            (
                "<div class=\"sidebar-desc\" style=\"color:#888;\">"
                    "<p>"
                    "Select AI 2.0 is an AI for BI assistant that replaces dashboards, "
                    "letting executives and business users ask questions in plain English and get instant answers from enterprise data, "
                    "seamlessly connected to Vantage via TD MCP server."
                    "</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------------
# CHAT RENDERING
# -------------------------------------------------------------------------
def render_chat(messages: List[Message]) -> None:
    """Render chat messages in Streamlit UI."""
    import re

    def _normalize(text: str) -> str:
        s = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        return re.sub(r"\n{3,}", "\n\n", s)

    for msg in messages:
        role = msg.get("role", "ai")
        content = _normalize(msg.get("content", ""))
        chat_role = "user" if role == "user" else "assistant"
        
        with st.chat_message(chat_role):
            st.markdown(content)
            
            # If this AI message has an associated chart, display it
            if role == "ai" and "chart" in msg:
                st.image(msg["chart"], width=650)


# -------------------------------------------------------------------------
# AI MESSAGE HANDLING
# -------------------------------------------------------------------------
async def generate_ai_reply(backend: Agent, history: List[Message]) -> tuple[str, bool]:
    """Generate a reply from the AI backend for the provided history."""

    logger.event("ai.call.start", count=str(len(history)))
    reply_text, is_plot = await backend.generate_reply(history)
    logger.event("ai.call.end", chars=str(len(reply_text or "")))
    return reply_text, is_plot


def handle_user_input(prompt: str) -> None:
    """Handle user input and update session state."""
    text = prompt.strip()
    if not text:
        return

    # Check if AI is initialized before processing
    if st.session_state.get("ai_instance") is None:
        st.error("AI backend is still initializing. Please wait a moment and try again.")
        return

    # Add user message
    user_msg = {
        "role": "user",
        "content": text,
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    st.session_state["messages"].append(user_msg)
    logger.log(user_msg["role"], user_msg["content"])

    loop_thread = st.session_state.get("event_loop")
    ai_backend = st.session_state.get("ai_instance")

    if ai_backend is None or loop_thread is None:
        st.error("AI backend not initialized. Please refresh the app.")
        return

    chart_image = None
    history = list(st.session_state["messages"])
    
    try:
        with st.spinner("Thinking..."):
            reply, is_plot = loop_thread.run_coroutine(
                generate_ai_reply(ai_backend, history)
            )
            
            # Handle chart generation
            if is_plot:
                try:
                    charts = os.listdir(CHARTS_PATH)
                    if charts:
                        # Get the most recently created chart
                        chart_files = [os.path.join(str(CHARTS_PATH), f) for f in charts]
                        chart_path = max(chart_files, key=os.path.getctime)
                        
                        # Load the image using PIL
                        with Image.open(chart_path) as img:
                            chart_image = img.copy()
                        
                        # Optionally clean up the file after loading
                        try:
                            os.remove(chart_path)
                        except Exception as cleanup_error:
                            print(f"Warning: Could not delete chart file: {cleanup_error}")
                            
                except Exception as chart_error:
                    print(f"Error loading chart: {chart_error}")
                    st.warning(f"Could not load chart: {chart_error}")

    except Exception as e:
        st.error(f"Couldn't get a reply: {e}")
        ai_msg = {
            "role": "ai",
            "content": f"[error] {e}",
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
    else:
        ai_msg = {
            "role": "ai",
            "content": reply if (reply and reply.strip()) else "[empty response]",
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        
        # Attach the chart to the message if one was generated
        if chart_image is not None:
            ai_msg["chart"] = chart_image

    st.session_state["messages"].append(ai_msg)
    logger.log(ai_msg["role"], ai_msg["content"])

    # Enforce message cap
    if len(st.session_state["messages"]) > MAX_MESSAGES:
        st.session_state["messages"] = st.session_state["messages"][-MAX_MESSAGES:]

    st.rerun()


# -------------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Select AI 2.0", page_icon="💬", layout="wide")
    init_session_state()

    render_sidebar()
    
    # Show loading state if AI is not ready
    if st.session_state.get("ai_instance") is None:
        with st.spinner("Initializing AI backend..."):
            st.info("Please wait while the AI backend is being initialized.")
    
    render_chat(st.session_state["messages"])

    prompt = st.chat_input("Type a message and press Enter")
    if prompt:
        handle_user_input(prompt)


if __name__ == "__main__":
    main()