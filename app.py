"""Select AI 2.0 Streamlit chat app.

Responsibilities:
- UI: Render a minimal chat interface and branded sidebar.
- State: Manage session-level messages, logger, and AI instance.
- Backend: Obtain an `AI` implementation via ``ai.factory.get_ai()`` and call
    ``AI.generate_reply``; the concrete backend adheres to the ``AI`` interface.
- Telemetry: Emit lightweight events around initialization and AI calls.

Notes:
- Sidebar includes optional environment diagnostics to help identify interpreter
    and package issues (Python path/version and OpenAI SDK availability).
- Styling is loaded from ``assets/styles.css`` if present.
"""

from __future__ import annotations
import datetime as dt
import html
import re
from typing import Dict, List
from pathlib import Path as _Path
from ai.base import Message, AI
import streamlit as st
from ai import get_ai
from logger import ChatLogger


# --- Page setup ---
st.set_page_config(
    page_title="Select AI 2.0",
    page_icon="💬", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Constants ---
MAX_MESSAGES: int = 100  # Cap in-memory history length

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # type: List[Message]
if "logger" not in st.session_state:
    st.session_state["logger"] = ChatLogger()
if "ai_instance" not in st.session_state:
    try:
        ai_impl = get_ai()
        st.session_state["ai_instance"] = ai_impl
        # Warmup: pre-load schema snapshot if backend supports it
        try:
            getattr(ai_impl, "warmup", lambda: None)()
        except Exception:
            pass
        st.session_state["logger"].event(
            "ai.init", backend=ai_impl.__class__.__name__
        )
    except Exception as _e:
        st.session_state["logger"].event("ai.init.error", error=str(_e))
        st.session_state["ai_instance"] = None

# --- Left sidebar (collapsible) ---
with st.sidebar:
    # Logo at the top of the sidebar
    _logo_path = _Path(__file__).parent / "assets" / "td_new_trans.png"
    if _logo_path.exists():
        st.image(str(_logo_path))

    st.title("Select AI 2.0")
    st.markdown(
        (
            "<div class=\"sidebar-desc\" style=\"color:#888;\">"
            "<p>Select AI is an intelligent business assistant that replaces traditional dashboards by allowing executives and business users to get instant insights through conversation. Instead of navigating reports, users simply ask questions in plain English, and Select AI generates answers directly from enterprise data—powered by Teradata’s MCP.</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

# --- Styles (align user right, assistant left; no avatars) ---
# Load external CSS if present
_css_path = _Path(__file__).parent / "assets" / "styles.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    
# Optional: place for page-specific CSS hooks (kept minimal)

def _render_chat(messages: List[Message]) -> None:
    """Render chat using Streamlit's chat_message with Markdown.

    Benefits:
    - Proper Markdown rendering (lists, code blocks, inline formatting).
    - Streamlit handles spacing sensibly, reducing excessive blank lines.
    """
    def _normalize(text: str) -> str:
        # Keep a light normalization to avoid massive blank sections
        s = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        return re.sub(r"\n{3,}", "\n\n", s)

    for msg in messages:
        role = msg.get("role", "ai")
        content = _normalize(msg.get("content", ""))
        # Map roles for Streamlit chat UI
        chat_role = "user" if role == "user" else "assistant"
        with st.chat_message(chat_role):
            st.markdown(content)

# Initial chat render
_render_chat(st.session_state["messages"])  # type: ignore[arg-type]

# --- Input & send ---
prompt = st.chat_input("Type a message and press Enter")
if prompt is not None:
    text = prompt.strip()
    if text:
        # Append user message
        user_msg = {
            "role": "user",
            "content": text,
            # Timestamp is currently unused in UI but kept for future needs
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        st.session_state["messages"].append(user_msg)
        st.session_state["logger"].log(user_msg["role"], user_msg["content"])

        # Generate AI reply with narrower exception surface
        if st.session_state.get("ai_instance") is None:
            try:
                st.session_state["ai_instance"] = get_ai()
            except Exception as e:
                st.session_state["logger"].event("ai.init.error", error=str(e))
                st.error(f"Couldn't initialize AI backend: {e}")
                st.rerun()

        try:
            with st.spinner("Thinking..."):
                logger = st.session_state["logger"]
                logger.event("ai.call.start", count=str(len(st.session_state["messages"])) )
                backend: AI = st.session_state["ai_instance"]  # type: ignore[assignment]
                reply = backend.generate_reply(st.session_state["messages"], context=None)  # type: ignore[arg-type]
                logger.event("ai.call.end", chars=str(len(reply or "")))
        except Exception as e:
            st.error(f"Couldn't get a reply: {e}")
            # Also append an AI message so the chat always shows something
            ai_msg = {
                "role": "ai",
                "content": f"[error] {e}",
                "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            }
            st.session_state["messages"].append(ai_msg)
            st.session_state["logger"].log(ai_msg["role"], ai_msg["content"])
            # Re-render and stop
            st.rerun()
        else:
            ai_msg = {
                "role": "ai",
                "content": reply if (reply and reply.strip()) else "[empty response]",
                "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            }
            st.session_state["messages"].append(ai_msg)
            st.session_state["logger"].log(ai_msg["role"], ai_msg["content"])

            # Enforce cap (keep most recent messages)
            if len(st.session_state["messages"]) > MAX_MESSAGES:
                st.session_state["messages"] = st.session_state["messages"][-MAX_MESSAGES:]

        # Re-render immediately so the new messages show up
        st.rerun()
