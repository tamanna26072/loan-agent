import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import queue
import threading
from typing import Any, Dict, List, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from agent import run_loan_agent

# Page Config
st.set_page_config(
    page_title="LoanSense AI",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS — Dark Brutalist / Financial Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
}

.stApp {
    background: #0d0f14;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1e2230;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f0f2f8;
    letter-spacing: -1px;
    margin: 0;
}
.hero-title span {
    color: #4ade80;
}
.hero-sub {
    font-size: 0.95rem;
    color: #6b7280;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Input Panel */
.input-panel {
    background: #13161f;
    border: 1px solid #1e2230;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.panel-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 1.2rem;
}

/* Streamlit widget overrides */
.stTextInput input, .stNumberInput input {
    background: #1a1e2a !important;
    border: 1px solid #4b5563 !important;
    color: #f0f2f8 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    caret-color: #4ade80 !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #4ade80 !important;
    box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.15) !important;
    background: #1e2333 !important;
}
.stTextInput input::placeholder, .stNumberInput input::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

label, .stTextInput label, .stNumberInput label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label {
    color: #9ca3af !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}

/* Number input stepper buttons */
.stNumberInput button {
    background: #2a2f3e !important;
    border-color: #4b5563 !important;
    color: #e8eaf0 !important;
}

/* Button */
.stButton > button {
    background: #4ade80 !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: #22c55e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(74, 222, 128, 0.3) !important;
}

/* Expander (reasoning steps) */
.streamlit-expanderHeader {
    background: #13161f !important;
    border: 1px solid #1e2230 !important;
    border-radius: 8px !important;
    color: #9ca3af !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
}
.streamlit-expanderContent {
    background: #0d0f14 !important;
    border: 1px solid #1e2230 !important;
    border-top: none !important;
}

/* Step cards inside expander */
.step-card {
    background: #13161f;
    border-left: 3px solid #374151;
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    font-size: 0.85rem;
    line-height: 1.6;
    color: #e2e8f0;
}
.step-card.thought  { border-left-color: #818cf8; background: #1a1d2e; }
.step-card.action   { border-left-color: #f59e0b; background: #1a1a0f; }
.step-card.observe  { border-left-color: #22d3ee; background: #0a1a1f; }
.step-card code { color: #fbbf24; font-size: 0.82rem; }
.step-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    opacity: 0.6;
}
.step-card.thought  .step-label { color: #818cf8; }
.step-card.action   .step-label { color: #f59e0b; }
.step-card.observe  .step-label { color: #22d3ee; }

/* Decision Card */
.decision-card {
    border-radius: 12px;
    padding: 2rem;
    margin-top: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.decision-card.approve {
    background: linear-gradient(135deg, #052e16 0%, #0f2e1a 100%);
    border: 1px solid #15803d;
}
.decision-card.reject {
    background: linear-gradient(135deg, #2d0a0a 0%, #2e1010 100%);
    border: 1px solid #991b1b;
}
.decision-badge {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
}
.decision-card.approve .decision-badge { color: #4ade80; }
.decision-card.reject  .decision-badge { color: #f87171; }
.decision-reason {
    font-size: 0.95rem;
    line-height: 1.7;
    color: #d1d5db;
    max-width: 540px;
    margin: 0 auto;
    white-space: pre-wrap;
}
.decision-divider {
    width: 40px;
    height: 2px;
    margin: 1rem auto;
}
.decision-card.approve .decision-divider { background: #4ade80; }
.decision-card.reject  .decision-divider { background: #f87171; }

/* st.container(border=True) — match dark theme */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #13161f !important;
    border: 1px solid #1e2230 !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #4ade80 !important; }

/* Status messages */
.stAlert { border-radius: 8px !important; }

/* Divider */
hr { border-color: #1e2230 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a2f3e; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# Callback Handler — capture agent steps
class StepCaptureHandler(BaseCallbackHandler):
    def __init__(self):
        self.steps: List[Dict] = []

    def on_agent_action(self, action, **kwargs):
        self.steps.append({
            "type": "action",
            "tool": action.tool,
            "input": str(action.tool_input),
        })

    def on_tool_end(self, output: str, **kwargs):
        self.steps.append({
            "type": "observation",
            "output": output,
        })

    def on_llm_end(self, response: LLMResult, **kwargs):
        try:
            text = response.generations[0][0].text
            if "Thought:" in text:
                thought = text.split("Thought:")[-1].split("Action:")[0].strip()
                if thought:
                    self.steps.append({"type": "thought", "text": thought})
        except Exception:
            pass

# Helpers
def render_step_card(step: Dict):
    if step["type"] == "thought":
        st.markdown(f"""
        <div class="step-card thought">
            <div class="step-label">💭 Thought</div>
            {step['text']}
        </div>""", unsafe_allow_html=True)

    elif step["type"] == "action":
        st.markdown(f"""
        <div class="step-card action">
            <div class="step-label">⚡ Action → {step['tool']}</div>
            <code style="color:#fbbf24; font-size:0.82rem;">{step['input']}</code>
        </div>""", unsafe_allow_html=True)

    elif step["type"] == "observation":
        st.markdown(f"""
        <div class="step-card observe">
            <div class="step-label">📡 Observation</div>
            {step['output']}
        </div>""", unsafe_allow_html=True)


def render_decision(final_answer: str):
    is_approve = "APPROVE" in final_answer.upper() and "REJECT" not in final_answer.upper()
    card_class = "approve" if is_approve else "reject"
    badge = "✅ APPROVED" if is_approve else "❌ REJECTED"

    # Strip the DECISION line from reason display
    lines = final_answer.strip().split("\n")
    reason_lines = [l for l in lines if not l.startswith("DECISION:")]
    reason_text = "\n".join(reason_lines).strip()

    st.markdown(f"""
    <div class="decision-card {card_class}">
        <div class="decision-badge">{badge}</div>
        <div class="decision-divider"></div>
        <div class="decision-reason">{reason_text}</div>
    </div>
    """, unsafe_allow_html=True)

# App Layout
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Loan<span>Sense</span> AI</div>
    <div class="hero-sub">Autonomous Loan Underwriting · Powered by ReAct Agent</div>
</div>
""", unsafe_allow_html=True)

# Load API key from .env only
api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart.")
    st.stop()

# Info badges
st.markdown("""
<div style="display:flex; justify-content:center; gap:0.6rem;
            flex-wrap:wrap; margin-bottom:1.8rem;">
    <span style="background:#13161f; border:1px solid #1e2230; border-radius:20px;
                 padding:0.3rem 0.9rem; font-family:'Space Mono',monospace;
                 font-size:0.7rem; color:#4ade80; letter-spacing:0.06em;">
        &#9889; Model: Llama 3 (Groq)
    </span>
    <span style="background:#13161f; border:1px solid #1e2230; border-radius:20px;
                 padding:0.3rem 0.9rem; font-family:'Space Mono',monospace;
                 font-size:0.7rem; color:#818cf8; letter-spacing:0.06em;">
        &#128279; Agent: LangChain ReAct
    </span>
    <span style="background:#13161f; border:1px solid #1e2230; border-radius:20px;
                 padding:0.3rem 0.9rem; font-family:'Space Mono',monospace;
                 font-size:0.7rem; color:#f59e0b; letter-spacing:0.06em;">
        &#128295; Tools: 4 active
    </span>
</div>
""", unsafe_allow_html=True)

# Input Form
st.markdown('<div class="panel-label" style="margin-bottom:0.8rem;">📋 Application Details</div>', unsafe_allow_html=True)

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        user_id = st.text_input("User ID", value="U123", placeholder="e.g. U123")
        monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=45000.0, step=500.0)
    with col2:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=620)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, value=200000.0, step=1000.0)

# Sample IDs hint
st.markdown(
    '<p style="font-size:0.75rem; color:#4b5563; text-align:center; margin-bottom:1rem;">'
    '🔍 Try User IDs: <code style="color:#818cf8">U123</code> (clean) · '
    '<code style="color:#f87171">U456</code> (flagged) · '
    '<code style="color:#f87171">U999</code> (flagged)</p>',
    unsafe_allow_html=True
)

run_btn = st.button("🚀 Evaluate Loan Application", use_container_width=True)

# Agent Execution
if run_btn:
    handler = StepCaptureHandler()
    result = None
    error = None

    with st.spinner("Agent is reasoning through the application..."):
        try:
            result = run_loan_agent(
                user_id=user_id,
                credit_score=int(credit_score),
                monthly_income=float(monthly_income),
                loan_amount=float(loan_amount),
                groq_api_key=api_key,
                callbacks=[handler],
            )
        except Exception as e:
            error = str(e)

    if error:
        st.error(f"Agent encountered an error: {error}")
        st.stop()

    # Reasoning Steps
    if handler.steps:
        with st.expander("🧠 Agent Reasoning Steps", expanded=True):
            for step in handler.steps:
                render_step_card(step)

    # Final Decision
    if result and result.get("output"):
        st.markdown("---")
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace; font-size:0.7rem; '
            'letter-spacing:0.12em; color:#4b5563; text-transform:uppercase; '
            'text-align:center; margin-bottom:0.5rem;">Final Decision</p>',
            unsafe_allow_html=True
        )
        render_decision(result["output"])
    else:
        st.warning("Agent did not return a clear decision. Check reasoning steps above.")

# Footer
st.markdown("""
<hr style="margin-top:3rem;">
<p style="text-align:center; font-size:0.72rem; color:#374151; padding-bottom:1rem;">
    LoanSense AI · Built with LangChain ReAct + Llama 3 (Groq) · For demonstration purposes only
</p>
""", unsafe_allow_html=True)