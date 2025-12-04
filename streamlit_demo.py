"""
streamlit_demo.py - Web Interface for CRUCIBLE

A Streamlit-based chat interface for material identification. Provides a clean
web UI with conversation history, example queries, and visual tool call details.

The interface mirrors the CLI behavior but adds:
    - Persistent chat history across interactions
    - Expandable tool execution details
    - Example query buttons in the sidebar
    - Session metrics

Usage:
    streamlit run streamlit_demo.py
"""

import os
import json
import streamlit as st
from llama_cpp import Llama
from demo_tools import TOOL_SCHEMA, execute_tool


# --- Page Configuration ---
st.set_page_config(
    page_title="CRUCIBLE AI Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
MODEL_PATH = "./models/Phi-3.5-mini-instruct-Q4_K_M.gguf"

SYSTEM_PROMPT = """You are CRUCIBLE, a material science assistant that helps identify materials from spectroscopic data.

Your main capability is identifying materials using Raman spectroscopy data. You have access to a tool called 'identify_material' that requires three parameters:
- peak_1: First Raman peak in cm^-1
- peak_2: Second Raman peak in cm^-1  
- formation_energy: Formation energy in eV/atom

When users provide this data, use the tool to identify the material. If data is missing, politely ask for it.

Be concise, scientific, and helpful. Explain your identifications briefly."""

EXAMPLE_QUERIES = [
    "Identify material with peaks at 465 and 610 cm⁻¹, formation energy -11.2 eV/atom",
    "What material has Raman peaks at 144 and 399?",
    "I have peaks at 520 and 950 with formation energy 0.0",
    "What is Raman spectroscopy?",
    "Tell me about Ceria"
]


# --- Model Loading (cached) ---
@st.cache_resource
def load_model():
    """
    Load the LLM model. Cached so it persists across Streamlit reruns.
    
    Returns:
        Llama instance ready for inference.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.info("Download the model first using the download script.")
        st.stop()
    
    with st.spinner("Loading CRUCIBLE model..."):
        return Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=6,
            n_gpu_layers=0,
            verbose=False
        )


# --- Core Chat Logic ---
def get_response(user_message: str) -> tuple[str, list[dict]]:
    """
    Generate a response to the user's message, handling tool calls.
    
    Args:
        user_message: The user's input text.
    
    Returns:
        Tuple of (response_text, list_of_tool_call_info).
    """
    # Build message history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in st.session_state.messages:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    
    tool_calls_info = []
    
    # Tool-calling loop
    for _ in range(3):
        response = st.session_state.llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            tools=[TOOL_SCHEMA],
            tool_choice="auto"
        )
        
        message = response["choices"][0]["message"]
        
        if message.get("tool_calls"):
            tool_call = message["tool_calls"][0]
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_result = execute_tool(tool_name, tool_args)
            
            tool_calls_info.append({
                "name": tool_name,
                "args": tool_args,
                "result": json.loads(tool_result)
            })
            
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result
            })
        else:
            return message["content"], tool_calls_info
    
    return "I had trouble processing that. Could you rephrase?", tool_calls_info


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = load_model()


# --- Sidebar ---
with st.sidebar:
    st.title("🔬 CRUCIBLE")
    st.caption(
        "**C**omputational **R**epository for **U**nified **C**lassification "
        "and **I**nteractive **B**ase **L**earning **E**xpert"
    )
    
    st.divider()
    st.subheader("Example Queries")
    
    for i, query in enumerate(EXAMPLE_QUERIES, 1):
        if st.button(f"Example {i}", key=f"ex_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        Provide spectroscopic data:
        - Raman Peak 1 (cm⁻¹)
        - Raman Peak 2 (cm⁻¹)
        - Formation Energy (eV/atom)
        
        Ask naturally: "Identify this material..." or "What is...?"
        """)
    
    with st.expander("⚙️ Technical Details"):
        st.markdown(f"""
        **Model:** Phi-3.5 Mini Instruct (Q4_K_M)  
        **Parameters:** 3.8B  
        **Context:** 2,048 tokens  
        **Status:** ✅ Loaded
        """)


# --- Main Chat Interface ---
st.title("🔬 CRUCIBLE Material Identification")
st.caption("Powered by Phi-3.5 Mini + ML Classification")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg.get("tool_calls"):
            for tool in msg["tool_calls"]:
                with st.expander("🔧 Tool Details"):
                    st.json({
                        "tool": tool["name"],
                        "arguments": tool["args"],
                        "result": tool["result"]
                    })

# Chat input
if prompt := st.chat_input("Ask about materials or provide spectroscopic data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, tool_calls = get_response(prompt)
        
        st.markdown(response)
        
        if tool_calls:
            for tool in tool_calls:
                with st.expander("🔧 Tool Details"):
                    st.json({
                        "tool": tool["name"],
                        "arguments": tool["args"],
                        "result": tool["result"]
                    })
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "tool_calls": tool_calls if tool_calls else []
        })


# --- Footer Metrics ---
st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Messages", len(st.session_state.messages))
col2.metric("Tool Calls", sum(1 for m in st.session_state.messages if m.get("tool_calls")))
col3.metric("Model", "🟢 Active")
