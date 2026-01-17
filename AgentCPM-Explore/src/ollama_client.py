#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Ollama Client

Direct HTTP client for Ollama - no openai package dependency needed.
Uses Ollama's native API at http://localhost:11434/api/chat
"""

import json
import logging
import re
import random
from typing import List, Dict, Any, Optional
import httpx

logger = logging.getLogger("ollama_client")


def extract_thinking(text: str) -> tuple:
    """
    Extract thinking/reasoning content from response text.
    DeepSeek models include <think>...</think> tags.
    
    Returns: (thinking_content, cleaned_response)
    """
    if not text:
        return "", ""
    
    thinking = ""
    cleaned = text
    
    # Pattern 1: <think>...</think> tags
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
    match = think_pattern.search(text)
    if match:
        thinking = match.group(1).strip()
        cleaned = text.replace(match.group(0), "").strip()
    
    return thinking, cleaned


def parse_tool_calls(text: str) -> tuple:
    """
    Parse tool calls from response text.
    
    Supports formats:
    - JSON with {"name": "...", "arguments": {...}}
    - Function call syntax
    
    Returns: (tool_calls_list, cleaned_text)
    """
    if not text:
        return [], text
    
    tool_calls = []
    
    # Try to find JSON tool calls
    # Pattern: {"name": "tool_name", "arguments": {...}}
    json_pattern = re.compile(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}', re.DOTALL)
    
    for match in json_pattern.finditer(text):
        try:
            tool_name = match.group(1)
            args_str = match.group(2)
            
            tool_calls.append({
                "id": f"call_{random.randint(10000, 99999)}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": args_str
                }
            })
        except Exception:
            pass
    
    return tool_calls, text


class OllamaClient:
    """
    Simple Ollama client using direct HTTP requests.
    No openai package dependency.
    """
    
    def __init__(
        self,
        model: str = "deepseek-r1:1.5b",
        base_url: str = "http://localhost:11434",
        timeout: float = 600.0
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., deepseek-r1:1.5b)
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        logger.info(f"Initialized Ollama client: model={model}, url={base_url}")
    
    def create_completion(
        self,
        messages: List[Dict],
        tools: List = None,
        stream: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using Ollama's native API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: List of tool definitions (optional)
            stream: Whether to stream (not implemented, always False)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with 'response', 'thought', 'tool_calls', etc.
        """
        # Build request payload for Ollama native API
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,  # We'll handle streaming separately if needed
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        # Add tools to system prompt if provided
        if tools:
            tool_descriptions = self._format_tools_for_prompt(tools)
            # Prepend tool info to system message or create one
            system_msg = None
            for msg in messages:
                if msg.get("role") == "system":
                    system_msg = msg
                    break
            
            tool_prompt = f"""You have access to the following tools:

{tool_descriptions}

To use a tool, respond with a JSON object in this format:
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}

Only use tools when necessary. After using a tool, wait for the result before continuing."""

            if system_msg:
                system_msg["content"] = tool_prompt + "\n\n" + system_msg["content"]
            else:
                messages.insert(0, {"role": "system", "content": tool_prompt})
        
        logger.info(f"Calling Ollama API: {self.base_url}/api/chat")
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "response": "",
                        "thought": "",
                        "tool_calls": [],
                        "error": error_msg
                    }
                
                result = response.json()
                
        except httpx.TimeoutException:
            error_msg = "Ollama request timed out"
            logger.error(error_msg)
            return {
                "response": "",
                "thought": "",
                "tool_calls": [],
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Ollama request failed: {str(e)}"
            logger.error(error_msg)
            return {
                "response": "",
                "thought": "",
                "tool_calls": [],
                "error": error_msg
            }
        
        # Extract response content
        message = result.get("message", {})
        content = message.get("content", "")
        
        # Extract thinking/reasoning
        thought, cleaned_content = extract_thinking(content)
        
        # Parse tool calls from response
        tool_calls, _ = parse_tool_calls(cleaned_content)
        
        # Build response
        return {
            "response": cleaned_content,
            "thought": thought,
            "tool_calls": tool_calls if tool_calls else None,
            "raw_message": message,
            "raw_response": result,
            "usage": {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            }
        }
    
    def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """Format tools for inclusion in prompt."""
        tool_strs = []
        
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "No description")
            params = func.get("parameters", {})
            
            param_strs = []
            properties = params.get("properties", {})
            required = params.get("required", [])
            
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                req = " (required)" if param_name in required else " (optional)"
                param_strs.append(f"  - {param_name} ({param_type}){req}: {param_desc}")
            
            params_text = "\n".join(param_strs) if param_strs else "  No parameters"
            tool_strs.append(f"**{name}**: {desc}\nParameters:\n{params_text}")
        
        return "\n\n".join(tool_strs)


# Factory function for easy creation
def create_ollama_client(
    model: str = "deepseek-r1:1.5b",
    base_url: str = "http://localhost:11434",
    **kwargs
) -> OllamaClient:
    """Create an Ollama client instance."""
    return OllamaClient(model=model, base_url=base_url, **kwargs)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    client = create_ollama_client()
    
    result = client.create_completion(
        messages=[
            {"role": "user", "content": "What is 2+2? Think step by step."}
        ],
        temperature=0.0
    )
    
    print("Thought:", result.get("thought", "")[:200])
    print("\nResponse:", result.get("response", "")[:500])
    print("\nUsage:", result.get("usage"))
