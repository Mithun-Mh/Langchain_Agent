# %%
import os
import sys
import certifi
import requests
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from typing import cast
from langchain import hub
from langchain.tools import BaseTool, tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

# %%
# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv(find_dotenv(), override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

print("Environment loaded successfully.")
print(f"GOOGLE_API_KEY present: {bool(GOOGLE_API_KEY)}")
print(f"TAVILY_API_KEY present: {bool(TAVILY_API_KEY)}")
print(f"WEATHERSTACK_API_KEY present: {bool(WEATHERSTACK_API_KEY)}")

# %%
# ==========================================
# DEFINE TOOLS
# ==========================================

# 1. Tavily Web Search Tool
search_tool = TavilySearchResults(max_results=2)

# 2. WeatherStack Weather Tool
@tool
def get_weather_data(city: str) -> str:
    """Get the current temperature, weather conditions, and humidity for a given city."""
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return "Weather API key is not configured."

    try:
        # Note: WeatherStack free tier uses HTTP
        response = requests.get(
            "http://api.weatherstack.com/current",
            params={
                "access_key": api_key.strip(),
                "query": city,
            },
            timeout=15,
        )

        data = response.json()

        if "error" in data or "current" not in data:
            error_info = data.get("error", {}).get("info", "Unknown error")
            return f"Could not retrieve weather for {city}: {error_info}"

        current = data["current"]
        conditions = (current.get("weather_descriptions") or ["Unknown"])[0]

        return (
            f"City: {data['location']['name']}, {data['location'].get('country', '')}\n"
            f"Temperature: {current['temperature']} °C\n"
            f"Conditions: {conditions}\n"
            f"Humidity: {current['humidity']}%\n"
            f"Wind: {current.get('wind_speed', 0)} km/h"
        )

    except requests.RequestException as exc:
        return f"Weather request failed: {type(exc).__name__}."


tools = [search_tool, get_weather_data]

# %%
# ==========================================
# TEST INDIVIDUAL TOOLS
# ==========================================
print("\n--- Testing Weather Tool ---")
print(cast(BaseTool, get_weather_data).invoke({"city": "Colombo"}))

print("\n--- Testing Search Tool ---")
search_test = search_tool.invoke("Current weather in Tokyo")
print("Search Results count:", len(search_test))

# %%
# ==========================================
# LLM (Fast, Low-Latency Model)
# ==========================================
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    max_retries=1
)

# %%
# ==========================================
# PROMPT TEMPLATE
# ==========================================
prompt = hub.pull("hwchase17/react")

# %%
# ==========================================
# CREATE REACT AGENT
# ==========================================
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# %%
# ==========================================
# AGENT EXECUTOR
# ==========================================
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# %%
# ==========================================
# RUN AGENT
# ==========================================
print("\n--- Running Agent ---")
response = agent_executor.invoke({
    "input": "What is the current weather in Colombo, and what is the latest news about Sri Lanka?"
})

# %%
print("\n================ FINAL ANSWER ================")
print(response["output"])
