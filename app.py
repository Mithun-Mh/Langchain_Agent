import os
from pathlib import Path

import certifi
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(
    page_title="MAI — Intelligence Assistant",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0B1020;
        --panel: #11182B;
        --border: #27334D;
        --text: #F2F5FF;
        --muted: #98A6C5;
        --violet: #9B8CFF;
        --cyan: #65E8E9;
    }

    .stApp {
        background: radial-gradient(ellipse at 50% 0%, #20234B 0%, transparent 48%), #0B1020;
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 850px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, .brand-title {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.7px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2.5rem;
    }

    .brand-icon {
        width: 44px;
        height: 44px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #8C7BFF, #4CD9E4);
        color: #0B1020;
        font-size: 25px;
        font-weight: 700;
        box-shadow: 0 0 28px #8C7BFF30;
    }

    .brand-title {
        font-size: 19px;
        color: #F2F5FF;
        font-weight: 700;
        line-height: 1.3;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #98A6C5;
        margin-top: 3px;
    }

    .eyebrow {
        color: #AFA4FF;
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(35px, 6vw, 54px);
        line-height: 1.12;
        font-weight: 700;
        letter-spacing: -2px;
        color: #F5F6FF;
        margin-bottom: 16px;
    }

    .gradient-text {
        background: linear-gradient(90deg, #B5A9FF, #65E8E9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-copy {
        color: #A3AFCA;
        font-size: 15px;
        line-height: 1.8;
        max-width: 610px;
        margin-bottom: 26px;
    }

    .section-label {
        font-size: 12px;
        font-weight: 600;
        color: #A3AFCA;
        margin: 24px 0 10px 0;
    }

    .feature-card {
        background: linear-gradient(145deg, #151D34, #101629);
        border: 1px solid #27334D;
        border-radius: 16px;
        padding: 17px;
        min-height: 112px;
    }

    .feature-icon { font-size: 20px; margin-bottom: 10px; }
    .feature-title { color: #EDF0FF; font-weight: 600; font-size: 13px; margin-bottom: 5px; }
    .feature-copy { color: #8F9DBB; font-size: 12px; line-height: 1.6; }

    div[data-testid="stChatMessage"] {
        background: #11182B;
        border: 1px solid #27334D;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: #65E8E9 !important;
    }

    /* Centered message input with dark frame and readable input text */
    div[data-testid="stForm"] {
        background: #11182B;
        border: 1px solid #27334D;
        border-radius: 16px;
        padding: 12px;
        margin-top: 22px;
        margin-bottom: 20px;
    }

    div[data-testid="stForm"] input {
        background: #FFFFFF !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 10px !important;
    }

    div[data-testid="stForm"] input::placeholder {
        color: #666666 !important;
        -webkit-text-fill-color: #666666 !important;
        opacity: 1 !important;
    }

    div[data-testid="stForm"] button {
        background: linear-gradient(135deg, #8C7BFF, #4CD9E4) !important;
        color: #0B1020 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 42px;
    }

    .stButton > button {
        border: 1px solid #343B5C;
        border-radius: 11px;
        background: #151D34;
        color: #DDE4FF;
        font-weight: 500;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #9B8CFF;
        color: white;
        background: #222443;
    }

    div[data-testid="stAlert"] { border-radius: 12px; }

    .footer {
        border-top: 1px solid #27334D;
        margin-top: 42px;
        padding-top: 20px;
        display: flex;
        justify-content: flex-end;
        color: #8492B1;
        font-size: 12px;
        width: 100%;
    }

    .footer strong { color: #D5D9F0; font-weight: 600; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load environment variables from the .env file beside app.py.
os.environ["SSL_CERT_FILE"] = certifi.where()
env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(env_file, override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


@tool
def get_weather_data(city: str) -> str:
    """Get current temperature, weather conditions, humidity, and wind speed for a given city."""
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return "Weather service is not configured."

    try:
        response = requests.get(
            "https://api.weatherstack.com/current",
            params={"access_key": api_key.strip(), "query": city},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data or "current" not in data:
            info = data.get("error", {}).get("info", "Unknown weather service error")
            return f"Could not retrieve weather for {city}: {info}"

        current = data["current"]
        location = data["location"]
        conditions = (current.get("weather_descriptions") or ["Unknown"])[0]
        return (
            f"City: {location['name']}, {location.get('country', '')}\n"
            f"Temperature: {current['temperature']} °C\n"
            f"Conditions: {conditions}\n"
            f"Humidity: {current['humidity']}%\n"
            f"Wind: {current.get('wind_speed', 0)} km/h"
        )
    except requests.RequestException as exc:
        return f"Weather request failed: {type(exc).__name__}."


@st.cache_resource(show_spinner=False)
def build_agent():
    missing = [
        name
        for name, value in [
            ("GOOGLE_API_KEY", GOOGLE_API_KEY),
            ("TAVILY_API_KEY", TAVILY_API_KEY),
            ("WEATHERSTACK_API_KEY", WEATHERSTACK_API_KEY),
        ]
        if not value
    ]
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))

    search_tool = TavilySearchResults(max_results=3)
    tools = [search_tool, get_weather_data]
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
        max_retries=1,
    )
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        stop_sequence=False,
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6,
    )


# Header
st.markdown(
    """
    <div class="brand">
        <div class="brand-icon">✦</div>
        <div>
            <div class="brand-title">M <span style="color:#AFA4FF">AI</span></div>
            <div class="brand-subtitle">Intelligence · Search · Weather</div>
        </div>
    </div>
    <div class="eyebrow">YOUR PERSONAL AI ASSISTANT</div>
    <div class="hero-title">
        Ask anything.<br>
        <span class="gradient-text">Discover more.</span>
    </div>
    <div class="hero-copy">
        Research current events, explore topics across the web,
        and get real-time weather insights — all in one place.
    </div>
    """,
    unsafe_allow_html=True,
)

# Capability cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """<div class="feature-card"><div class="feature-icon">⌕</div>
        <div class="feature-title">Web intelligence</div>
        <div class="feature-copy">Search current information using Tavily.</div></div>""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """<div class="feature-card"><div class="feature-icon">☼</div>
        <div class="feature-title">Live weather</div>
        <div class="feature-copy">Get conditions and temperature for a city.</div></div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """<div class="feature-card"><div class="feature-icon">✧</div>
        <div class="feature-title">AI reasoning</div>
        <div class="feature-copy">Combine tool results into useful answers.</div></div>""",
        unsafe_allow_html=True,
    )

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

# Prompt area sits in normal page flow (not fixed to the browser bottom).
st.markdown('<div class="section-label">ASK NEXUS AI</div>', unsafe_allow_html=True)

if not st.session_state.messages:
    st.caption("Try asking")
    examples = [
        "What is the current weather in Colombo?",
        "What is the latest news about Sri Lanka?",
        "Find today's major AI developments.",
    ]
    example_cols = st.columns(3)
    for index, example in enumerate(examples):
        with example_cols[index]:
            if st.button(example, key=f"example_{index}", use_container_width=True):
                st.session_state.pending_prompt = example
                st.rerun()

with st.form("chat_form", clear_on_submit=True):
    typed_prompt = st.text_input(
        "Your message",
        placeholder="Ask about the news, weather, or anything else...",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Send  ↑", use_container_width=True)

pending_prompt = st.session_state.pending_prompt
if pending_prompt:
    st.session_state.pending_prompt = ""

user_prompt = pending_prompt or (typed_prompt.strip() if submitted else "")

# Process a new prompt, then display both the user prompt and response below the input.
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    try:
        with st.spinner("Researching and preparing your answer..."):
            if "agent_executor" not in st.session_state:
                st.session_state.agent_executor = build_agent()
            result = st.session_state.agent_executor.invoke({"input": user_prompt})
            answer = result.get("output", "The agent did not return an answer.")
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "I couldn't complete that request. Check your API configuration, "
                    "model availability, service quotas, and installed package versions, then try again. "
                    f"(Error type: {type(exc).__name__})"
                ),
            }
        )

if st.session_state.messages:
    st.markdown('<div class="section-label">CONVERSATION</div>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.button("Clear conversation", key="clear_conversation"):
        st.session_state.messages = []
        st.rerun()

# Footer: only the owner credit, aligned right.
st.markdown(
    """
    <div class="footer">
        <span>Owned by <strong>Mithun Appuhamy</strong></span>
    </div>
    """,
    unsafe_allow_html=True,
)
