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


def parse_tool_calls(text: str, valid_tools: List[str] = None) -> tuple:
    """
    Parse tool calls from response text.
    
    Supports multiple formats:
    1. <tool_call>{"name": "...", "arguments": {...}}</tool_call> (XML tags)
    2. {"name": "...", "arguments": {...}} (raw JSON)
    
    Args:
        text: Response text to parse
        valid_tools: List of valid tool names (default: web_search, fetch_webpage)
    
    Returns: (tool_calls_list, cleaned_text)
    """
    if not text:
        return [], text
    
    if valid_tools is None:
        valid_tools = ["web_search", "fetch_webpage"]
    
    tool_calls = []
    
    # Method 1: Parse <tool_call>...</tool_call> XML tags (preferred)
    tool_call_pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL | re.IGNORECASE)
    
    for match in tool_call_pattern.finditer(text):
        try:
            json_str = match.group(1).strip()
            parsed = json.loads(json_str)
            
            if isinstance(parsed, dict) and parsed.get("name") in valid_tools:
                tool_calls.append({
                    "id": f"call_{random.randint(10000, 99999)}",
                    "type": "function",
                    "function": {
                        "name": parsed.get("name"),
                        "arguments": json.dumps(parsed.get("arguments", {}), ensure_ascii=False)
                    }
                })
        except Exception as e:
            logging.warning(f"Failed to parse tool_call: {e}")
    
    # Method 2: If no XML tags found, try raw JSON format
    if not tool_calls:
        for tool_name in valid_tools:
            pattern = re.compile(
                r'\{\s*"name"\s*:\s*"' + re.escape(tool_name) + r'"\s*,\s*"arguments"\s*:\s*(\{[^{}]*\})\s*\}',
                re.DOTALL
            )
            
            for match in pattern.finditer(text):
                try:
                    args_str = match.group(1)
                    json.loads(args_str)  # Validate JSON
                    
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


def extract_answer(text: str) -> Optional[str]:
    """
    Extract final answer from <answer>...</answer> tags.
    
    Returns the answer content or None if no answer tags found.
    """
    if not text:
        return None
    
    # Pattern for <answer>...</answer>
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
    match = answer_pattern.search(text)
    
    if match:
        return match.group(1).strip()
    
    return None


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
        
        # Clean up base URL - remove /v1 suffix if present (that's for OpenAI-compatible API)
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        self.base_url = base_url
        self.timeout = timeout
        
        logger.info(f"Initialized Ollama client: model={model}, url={self.base_url}")
    
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
            
            tool_prompt = f"""# Available Tools

{tool_descriptions}

# How to Use Tools

To call a tool, output it in this XML format:
<tool_call>
{{"name": "web_search", "arguments": {{"query": "your search query"}}}}
</tool_call>

# How to Provide Final Answer

When you have gathered enough information, wrap your final answer in:
<answer>
Your complete answer here
</answer>

# Important Rules
- Make ONE tool call per response
- After receiving tool results, analyze them and continue
- Only output <answer> when you have completed ALL research needed"""

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
