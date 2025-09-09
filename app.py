"""Pinkman Streamlit app.

Responsibilities:
- UI: Render a minimal chat interface and sidebar copy.
- State: Manage session-level messages, logger, and AI instance.
- Backend: Route messages to AI via ai.factory.get_ai().
- Telemetry: Emit lightweight events around init and AI calls.
"""

from __future__ import annotations
import datetime as dt
import html
from typing import Dict, List
import streamlit as st
from ai import get_ai
from logger import ChatLogger


# --- Page setup ---
st.set_page_config(
    page_title="pinkman",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Constants ---
MAX_MESSAGES: int = 100  # Cap in-memory history length

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "logger" not in st.session_state:
    st.session_state["logger"] = ChatLogger()
if "ai_instance" not in st.session_state:
    try:
        ai_impl = get_ai()
        st.session_state["ai_instance"] = ai_impl
        st.session_state["logger"].event(
            "ai.init", backend=ai_impl.__class__.__name__
        )
    except Exception as _e:
        st.session_state["logger"].event("ai.init.error", error=str(_e))
        st.session_state["ai_instance"] = None

# --- Left sidebar (collapsible) ---
with st.sidebar:
    st.title("Pinkman")
    st.markdown(
        (
            "<div class=\"sidebar-desc\" style=\"color:#888;\">"
            "<p><strong>Jessy Pinkman</strong> is a no-frills Streamlit chat UI—fast, simple, and setup-free. It lets you plug directly into your AI backend with a clean single-page interface. Perfect for makers, hackers, and data scientists who want to prototype quickly without boilerplate: lighter than a CLI, richer than raw text, and built for fast demos, sharing, and iteration.</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

# --- Styles (align user right, assistant left; no avatars) ---
# Load external CSS if present
from pathlib import Path as _Path
_css_path = _Path(__file__).parent / "assets" / "styles.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# --- Chat feed ---

_bubbles: list[str] = []
for msg in st.session_state["messages"]:
    role = msg.get("role", "ai")
    content = msg.get("content", "")
    safe = html.escape(content)
    cls = "user" if role == "user" else "ai"
    _bubbles.append(f'<div class="msg {cls}"><div class="content">{safe}</div></div>')

st.markdown(
    f'<div class="chat-feed">{"".join(_bubbles)}</div>',
    unsafe_allow_html=True,
)

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

        # Generate AI reply
        try:
            if st.session_state["ai_instance"] is None:
                st.session_state["ai_instance"] = get_ai()
            with st.spinner("Thinking..."):
                logger = st.session_state["logger"]
                logger.event("ai.call.start", count=str(len(st.session_state["messages"])) )
                reply = st.session_state["ai_instance"].generate_reply(
                    st.session_state["messages"], context=None
                )
                logger.event("ai.call.end", chars=str(len(reply or "")))
        except Exception as e:  # noqa: BLE001 - surface any AI error to the UI
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
