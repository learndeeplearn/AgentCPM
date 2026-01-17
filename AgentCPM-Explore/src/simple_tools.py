#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Tools Implementation

Provides basic search and web browsing tools without MCP infrastructure.
Uses DuckDuckGo for search and requests/BeautifulSoup for web browsing.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

logger = logging.getLogger("simple_tools")

# Tool definitions in OpenAI format
SIMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Returns top search results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch and extract text content from a webpage URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to fetch"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum length of content to return (default: 5000)",
                        "default": 5000
                    }
                },
                "required": ["url"]
            }
        }
    }
]


async def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Dict with search results
    """
    try:
        # Try using duckduckgo_search library
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=min(num_results, 10)))
            
            formatted_results = []
            for i, r in enumerate(results, 1):
                formatted_results.append({
                    "index": i,
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", ""))
                })
            
            return {
                "status": "success",
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results)
            }
            
        except ImportError:
            logger.warning("duckduckgo_search not installed, using fallback")
            
        # Fallback: Use httpx to query DuckDuckGo HTML
        import httpx
        
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            
        if response.status_code != 200:
            return {"status": "error", "error": f"Search failed: HTTP {response.status_code}"}
        
        # Simple HTML parsing
        html = response.text
        results = []
        
        # Extract results (simplified parsing)
        import re
        
        # Find result blocks
        result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
        snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]*)</a>'
        
        urls = re.findall(result_pattern, html)
        snippets = re.findall(snippet_pattern, html)
        
        for i, (url, title) in enumerate(urls[:num_results]):
            snippet = snippets[i] if i < len(snippets) else ""
            results.append({
                "index": i + 1,
                "title": title.strip(),
                "url": url,
                "snippet": snippet.strip()
            })
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"status": "error", "error": str(e)}


async def fetch_webpage(url: str, max_length: int = 5000) -> Dict[str, Any]:
    """
    Fetch and extract text content from a webpage.
    
    Args:
        url: URL to fetch
        max_length: Maximum content length
        
    Returns:
        Dict with page content
    """
    try:
        import httpx
        
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
        
        if response.status_code != 200:
            return {"status": "error", "error": f"Failed to fetch: HTTP {response.status_code}"}
        
        content_type = response.headers.get("content-type", "")
        
        if "text/html" in content_type:
            # Try to extract text using BeautifulSoup
            try:
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Remove script and style elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                
                # Get title
                title = soup.title.string if soup.title else ""
                
                # Get main content
                text = soup.get_text(separator="\n", strip=True)
                
                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = "\n".join(lines)
                
                if len(text) > max_length:
                    text = text[:max_length] + "\n...(truncated)"
                
                return {
                    "status": "success",
                    "url": url,
                    "title": title,
                    "content": text,
                    "length": len(text)
                }
                
            except ImportError:
                logger.warning("BeautifulSoup not installed, returning raw text")
                
                # Fallback: simple HTML tag removal
                import re
                text = re.sub(r'<[^>]+>', ' ', response.text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                if len(text) > max_length:
                    text = text[:max_length] + "...(truncated)"
                
                return {
                    "status": "success",
                    "url": url,
                    "content": text,
                    "length": len(text)
                }
        else:
            # Non-HTML content
            text = response.text[:max_length]
            return {
                "status": "success",
                "url": url,
                "content_type": content_type,
                "content": text,
                "length": len(text)
            }
            
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return {"status": "error", "error": str(e)}


async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool by name with given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        
    Returns:
        Tool execution result
    """
    if tool_name == "web_search":
        return await web_search(
            query=arguments.get("query", ""),
            num_results=arguments.get("num_results", 5)
        )
    elif tool_name == "fetch_webpage":
        return await fetch_webpage(
            url=arguments.get("url", ""),
            max_length=arguments.get("max_length", 5000)
        )
    else:
        return {"status": "error", "error": f"Unknown tool: {tool_name}"}


class SimpleToolHandler:
    """
    Simple tool handler that provides search and web browsing without MCP.
    """
    
    def __init__(self):
        self.openai_tools = SIMPLE_TOOLS
        self.initialized = True
    
    async def initialize(self) -> bool:
        """Initialize the tool handler (always succeeds)."""
        logger.info("SimpleToolHandler initialized with 2 tools: web_search, fetch_webpage")
        return True
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool."""
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")
        result = await execute_tool(tool_name, arguments)
        logger.info(f"Tool result status: {result.get('status')}")
        return result
    
    async def close(self):
        """Close the handler (no-op)."""
        pass


# Quick test
if __name__ == "__main__":
    async def test():
        handler = SimpleToolHandler()
        await handler.initialize()
        
        # Test search
        print("Testing web_search...")
        result = await handler.call_tool("web_search", {"query": "python programming", "num_results": 3})
        print(json.dumps(result, indent=2))
        
        # Test fetch
        print("\nTesting fetch_webpage...")
        result = await handler.call_tool("fetch_webpage", {"url": "https://example.com", "max_length": 500})
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())
