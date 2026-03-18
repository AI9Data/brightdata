
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Literal
from urllib.parse import urlencode

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
# webmcp-langgraph-demo.py file

SYSTEM_PROMPT = """You are a web research assistant.

Task:
- Research the user's topic using Google search results and a few sources.
- Return 6–10 simple bullet points.
- Add a short "Sources:" list with only the URLs you used.

How to use tools:
- First call the search tool to get Google results.
- Select 3–5 reputable results and scrape them.
- If scraping fails, try a different result.

Constraints:
- Use at most 5 sources.
- Prefer official docs or primary sources.
- Keep it quick: no deep crawling.
"""

def make_llm_call_node(llm_with_tools):
    async def llm_call(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        ai_message = await llm_with_tools.ainvoke(messages)
        return {"messages": [ai_message]}
    return llm_call

def make_tool_node(tools_by_name: dict):
    async def tool_node(state: MessagesState):
        last_ai_msg = state["messages"][-1]
        tool_results = []

        for tool_call in last_ai_msg.tool_calls:
            tool = tools_by_name.get(tool_call["name"])

            if not tool:
                tool_results.append(
                    ToolMessage(
                        content=f"Tool not found: {tool_call['name']}",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            # MCP tools are typically async
            observation = (
                await tool.ainvoke(tool_call["args"])
                if hasattr(tool, "ainvoke")
                else tool.invoke(tool_call["args"])
            )

            tool_results.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": tool_results}
    return tool_node

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END
async def main():
    # Load environment variables from .env
    load_dotenv()

    # Read Bright Data token
    bd_token = os.getenv("BRIGHTDATA_TOKEN")
    if not bd_token:
        raise ValueError("Missing BRIGHTDATA_TOKEN")

    # Connect to Bright Data Web MCP server
    client = MultiServerMCPClient({
        "bright_data": {
            "url": f"https://mcp.brightdata.com/mcp?token={bd_token}",
            "transport": "streamable_http",
        }
    })
    #&groups=advanced_scraping,browser
    # Fetch all available MCP tools (search, scrape, etc.)
    tools = await client.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}
    
    print(f"Available tools: {list(tools_by_name.keys())}")  # Debug: print available tool names
    
    # Initialize the LLM and allow it to call MCP tools

    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-all", temperature=0, api_key=openai_api_key, base_url="https://poloapi.top/v1",)
    llm_with_tools = llm.bind_tools(tools)

    # Build the LangGraph agent
    graph = StateGraph(MessagesState)

    graph.add_node("llm_call", make_llm_call_node(llm_with_tools))
    graph.add_node("tool_node", make_tool_node(tools_by_name))

    # Graph flow:
    # START → LLM → (tools?) → LLM → END
    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph.add_edge("tool_node", "llm_call")

    agent = graph.compile()

    # Example research query
    topic = "使用工具访问https://quotes.toscrape.com/scroll，想办法拿到30条quotes提取 quote、author、tags"  # You can change this topic as needed

    # Run the agent
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=f"Research this topic:\n{topic}")
            ]
        },
        # Prevent infinite loops
        config={"recursion_limit": 50}
    )

    # Print the final response
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())