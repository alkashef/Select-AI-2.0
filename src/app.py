from PIL import Image
import streamlit as st

import os
import asyncio
import datetime as dt
from typing import List

from modules.logger import logger
from utils import get_multi_agent
from agents.base_agent import Message
from constants import TERADATA_LOGO_PATH, CHARTS_PATH
from modules.event_loop_thread import EventLoopThread


MAX_MESSAGES: int = 100  # Cap in-memory history length

def get_or_create_event_loop():
    """Get or create the persistent event loop thread.

    Returns
    -------
    EventLoopThread
        The global EventLoopThread instance stored in Streamlit session state.
    """
    if "event_loop" not in st.session_state:
        elt = EventLoopThread()
        elt.start()
        st.session_state["event_loop"] = elt

    return st.session_state["event_loop"]


# -------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize Streamlit session state keys used by the app.

    Initializes keys: ``messages``, ``ai_instance``, ``event_loop`` and
    ``init_attempted`` so other code can rely on their presence.

    Returns
    -------
    None
    """
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # type: List[Message] # type: ignore

    # Initialize ai_instance key first to avoid KeyError
    if "ai_instance" not in st.session_state:
        st.session_state["ai_instance"] = None

    # Initialize event loop (lightweight operation)
    get_or_create_event_loop()

    # Flag to track if we've attempted initialization
    if "init_attempted" not in st.session_state:
        st.session_state["init_attempted"] = False


def initialize_ai_backend() -> bool:
    """Initialize the AI backend and attach it to session state.

    Returns
    -------
    bool
        True on successful initialization, False on failure. On success
        the initialized AI implementation is stored at
        ``st.session_state['ai_instance']``.
    """
    try:
        loop = st.session_state["event_loop"]
        ai_impl = loop.run_coroutine(get_multi_agent())
        st.session_state["ai_instance"] = ai_impl

        # Warmup: optional
        warmup_func = getattr(ai_impl, "warmup", None)
        if warmup_func and asyncio.iscoroutinefunction(warmup_func):
            loop = st.session_state["event_loop"]
            loop.run_coroutine(warmup_func())
        elif warmup_func:
            warmup_func()

        logger.event("ai.init", backend=ai_impl.__class__.__name__)
        # logger.event("ai.type", model=ai_impl.model)
        st.session_state["init_attempted"] = True
        return True

    except Exception as e:
        logger.event("ai.init.error", error=str(e))
        st.session_state["ai_instance"] = None
        st.session_state["init_attempted"] = True
        st.error(f"Failed to initialize AI backend: {e}")
        return False


# -------------------------------------------------------------------------
# SIDEBAR RENDERING
# -------------------------------------------------------------------------
def render_sidebar() -> None:
    """Render the left sidebar including logo and short description.

    Returns
    -------
    None
    """
    with st.sidebar:
        if TERADATA_LOGO_PATH.exists():
            st.image(str(TERADATA_LOGO_PATH))

        st.title("Select AI 2.0")

        st.markdown(
            (
            "<div class=\"sidebar-desc\" style=\"color:#888;\">"
                "<p>"
                "Select AI 2.0 is an AI assistant for BI that replaces dashboards,"
                "letting executives and business users ask questions in plain English and get instant answers from enterprise data,"
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
    """Render chat messages in the Streamlit UI.

    Parameters
    ----------
    messages:
        List of message dictionaries to render in the chat area.

    Returns
    -------
    None
    """

    # def _normalize(text: str) -> str:
    #     return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    for msg in messages:
        role = msg.get("role", "ai")
        # content = _normalize(msg.get("content", ""))
        content = msg.get("content", "")
        chat_role = "user" if role == "user" else "assistant"

        with st.chat_message(chat_role):
            if "**SQL Commands:**" in content:
                splitted_text = content.split("**SQL Commands:**")
                message = splitted_text[0]
                sql = splitted_text[1]
                message = message.replace(" ", "&nbsp;").replace("\n", "  \n")
                content = "**SQL Commands:**".join([message, sql])
            else:
                content = content.replace(" ", "&nbsp;").replace("\n", "  \n")

            st.markdown(content)

            # If this AI message has an associated chart, display it
            if role == "ai" and "chart" in msg:
                st.image(msg["chart"], width=550)


# -------------------------------------------------------------------------
# AI MESSAGE HANDLING
# -------------------------------------------------------------------------
async def generate_ai_reply() -> tuple[str, bool]:
    """Generate a reply from the AI backend.

    Returns
    -------
    tuple[str, bool]
        A tuple of (reply_text, is_plot) where ``is_plot`` indicates if the
        response produced a visualization that should be displayed.
    """
    backend = st.session_state.get("ai_instance")

    if backend is None:
        raise RuntimeError("AI backend not initialized")

    logger.event("ai.call.start", count=str(len(st.session_state["messages"])))
    user_query = st.session_state["messages"][-1].get("content", "")
    state = await backend.run(user_query)
    reply_text = state.get("response", "")
    is_plot = state.get("is_plot", False)
    sql_queries = state.get("sql_queries", None)
    if sql_queries is not None:
        reply_text +=  sql_queries

    logger.event("ai.call.end", chars=str(len(reply_text or "")))
    return reply_text, is_plot


def handle_user_input(prompt: str) -> None:
    """Process a user's chat prompt, call the AI backend and update state.

    Parameters
    ----------
    prompt:
        The raw text input typed by the user in the chat box.

    Returns
    -------
    None
    """
    text = prompt.strip()
    if not text:
        return

    # Add user message
    user_msg = {
        "role": "user",
        "content": text,
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    st.session_state["messages"].append(user_msg)
    logger.log(user_msg["role"], user_msg["content"])

    # Immediately render the user message (so it stays visible)
    with st.chat_message("user"):
        st.markdown(text)

    loop_thread = st.session_state.get("event_loop")
    ai_backend = st.session_state.get("ai_instance")

    if ai_backend is None or loop_thread is None:
        st.error("AI backend not initialized. Please refresh the app.")
        return

    chart_image = None

    try:
        # Display the spinner *under* the user’s message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, is_plot = loop_thread.run_coroutine(generate_ai_reply())

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

        # Display AI reply immediately
        st.markdown(ai_msg["content"])
        if chart_image is not None:
            st.image(chart_image, width=550)

    st.session_state["messages"].append(ai_msg)
    logger.log(ai_msg["role"], ai_msg["content"])

    # Enforce message cap
    if len(st.session_state["messages"]) > MAX_MESSAGES:
        st.session_state["messages"] = st.session_state["messages"][-MAX_MESSAGES:]

    # Optional: rerun only if needed to refresh full chat view
    st.rerun()


# -------------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Select AI 2.0", page_icon="💬", layout="wide", )
    
    # Initialize basic session state (lightweight)
    init_session_state()

    if not st.session_state.get("init_attempted", False):
        with st.spinner("Initializing AI backend..."):
            success = initialize_ai_backend()
            if not success:
                st.error("Failed to initialize AI backend. Please refresh the page.")
                return

    # Render sidebar
    render_sidebar()

    # Render chat history
    render_chat(st.session_state["messages"])

    # Show welcome message if no messages yet
    if len(st.session_state["messages"]) == 0:
        st.info("Welcome to Select AI 2.0! Ask me anything about your enterprise data.")

    # Chat input (always visible)
    prompt = st.chat_input("Type a message and press Enter")
    if prompt:
        handle_user_input(prompt)


if __name__ == "__main__":
    main()