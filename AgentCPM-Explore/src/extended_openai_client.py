#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenAI Client Extension

This version introduces the LLMClientManager class, used to create and manage 
multiple language model client instances with different configurations in a 
single evaluation task.
"""

import os
import logging
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv
import httpx
from openai import OpenAI


script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

load_dotenv()

class BaseLLMClient:
    """LLM Client Base Class"""
    def create_completion(self, messages: List[Dict], tools: List = None, stream: bool = False, temperature: float = 0.0) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement create_completion method")

class ExtendedOpenAIClient(BaseLLMClient):
    """
    Extended OpenAI client that handles API calls, streaming response parsing, and logging.
    """
    def __init__(self, 
                 model: str, 
                 api_key: Optional[str], 
                 base_url: Optional[str],
                 timeout: float,
                 tool_start_tag: Optional[str],
                 tool_end_tag: Optional[str]):
        
        self.model = model
        # For Ollama (localhost:11434), use a dummy API key since it's not required
        is_ollama = base_url and "localhost:11434" in base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ("ollama" if is_ollama else None)
        if not self.api_key:
            raise ValueError(f"Model '{model}' has no OPENAI_API_KEY environment variable set or api_key parameter provided")
        
        self.tool_start_tag = tool_start_tag
        self.tool_end_tag = tool_end_tag
        
        long_timeout = httpx.Timeout(timeout, connect=60.0)
        
        client_args = {
            "api_key": self.api_key,
            "timeout": long_timeout
        }
        
        if base_url:
            client_args["base_url"] = base_url
        
        self.client = OpenAI(**client_args)
        
        config_info = {
            "model": self.model,
            "base_url": base_url or "Default URL",
            "timeout": timeout
        }
        logging.info(f"Initialized extended OpenAI client: {config_info}")

    def _parse_tool_calls_from_text(self, text: str) -> Optional[Tuple[List[Dict], str]]:
        if not text:
            return None

        # Pattern 1: Prioritize handling the new format for DeepSeek V3.1
        # <｜tool calls begin｜><｜tool call begin｜>search<｜tool sep｜>{"query": "..."}<｜tool call end｜><｜tool calls end｜>
        ds_pattern = re.compile(r'<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>', re.DOTALL)
        ds_match = ds_pattern.search(text)
        
        if ds_match:
            logging.info("Fallback parsing: Detected DeepSeek V3.1 tool call format...")
            inner_content = ds_match.group(1)
            
            tool_calls = []
            # Multiple tool calls may exist within a single block
            call_pattern = re.compile(r'<｜tool▁call▁begin｜>(.*?)<｜tool▁call▁end｜>', re.DOTALL)
            
            for call_match in call_pattern.finditer(inner_content):
                call_body = call_match.group(1).strip()
                # Use <｜tool sep｜> to split tool name and arguments
                parts = call_body.split('<｜tool▁sep｜>', 1)
                
                if len(parts) == 2:
                    tool_name = parts[0].strip()
                    arguments_str = parts[1].strip()
                    try:
                        # Verify if arguments are valid JSON
                        json.loads(arguments_str) 
                        tool_calls.append({
                            "id": f"call_{random.randint(10000, 99999)}",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": arguments_str}
                        })
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse DeepSeek tool call arguments JSON: {e}")
                        continue
            
            if tool_calls:
                
                cleaned_text = text.replace(ds_match.group(0), "").strip()
                return tool_calls, cleaned_text

        # Pattern 2: If no match above, execute the original general tag logic (e.g., <tool_call>{...}</tool_call>)
        if not self.tool_start_tag or not self.tool_end_tag:
            return None

        start_tag = re.escape(self.tool_start_tag)
        end_tag = re.escape(self.tool_end_tag)
        pattern = re.compile(f"{start_tag}(.*?){end_tag}", re.DOTALL)
        
        match = pattern.search(text)
        if match:
            logging.info(f"Fallback parsing: Detected custom tool call format: {self.tool_start_tag}...")

            block = match.group(1).strip()

            # Compatible with PythonInterpreter: JSON followed by <code>...</code>
            code_str = None
            json_part = block

            if "<code>" in block:
                json_part = block.split("<code>", 1)[0].strip()
                if "</code>" in block:
                    code_str = block.split("<code>", 1)[1].split("</code>", 1)[0].strip()
                else:
                    code_str = block.split("<code>", 1)[1].strip()

            try:
                # Auto-fix common missing right brace errors (only for JSON part)
                if json_part.startswith('{') and not json_part.endswith('}'):
                    json_part += '}'

                parsed_json = json.loads(json_part)

                # Only support single tool call
                if isinstance(parsed_json, dict) and ("name" in parsed_json):
                    args = parsed_json.get("arguments", {})
                    # Allow arguments to be missing; but unify to dict -> str
                    if args is None:
                        args = {}
                    arguments_str = json.dumps(args, ensure_ascii=False)

                    tool_call = {
                        "id": f"call_{random.randint(10000, 99999)}",
                        "type": "function",
                        "function": {"name": parsed_json["name"], "arguments": arguments_str},
                    }

                    # Hang code separately (for use when executing in data_test later)
                    if code_str is not None:
                        tool_call["code"] = code_str

                    return [tool_call], text  

            except (json.JSONDecodeError, TypeError) as e:
                logging.warning(f"Failed to parse JSON within custom tag '{self.tool_start_tag}': {e}")

        return None


    def _parse_think_from_text(self, text: str) -> Optional[Tuple[str, str]]:
        """
        [New] Parse <think>...</think> tags from 'content' text.
        
        Args:
            text: Original content string containing <think> tags.

        Returns:
            A tuple (thought_content, cleaned_text) or None.
            - thought_content: Text inside <think> tags.
            - cleaned_text: Text after removing the <think> tag block.
        """
        if not text:
            return None
        
        
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        match = pattern.search(text)
        
        if match:
            thought_content = match.group(1).strip()
            
            cleaned_text = text.replace(match.group(0), "").strip()
            logging.info("Fallback parsing: Extracted <think>...</think> tags from 'content'.")
            return thought_content, cleaned_text
        
        return None  

    def create_completion(self, messages: List[Dict], tools: List = None, stream: bool = False, temperature: float = 0.0, top_p: Optional[float] = None, presence_penalty: Optional[float] = None, stop: Optional[List[str]] = None, max_tokens: Optional[int] = None, reasoning_effort: Optional[str] = None, thinking_budget: Optional[int] = None) -> Dict[str, Any]:
        log_dir = os.path.join(os.getcwd(), "openai_logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            request_data["tools"] = tools
            request_data["tool_choice"] = "auto"
        if top_p is not None:
            request_data["top_p"] = top_p
            
        if presence_penalty is not None:
            request_data["presence_penalty"] = presence_penalty
            
        if stop is not None:
            request_data["stop"] = stop
            
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens

        if reasoning_effort is not None:
            request_data["reasoning_effort"] = reasoning_effort
            
        if thinking_budget is not None:
            
            if "extra_body" not in request_data:
                request_data["extra_body"] = {}
            request_data["extra_body"]["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }


        req_file = os.path.join(log_dir, f"request_{timestamp}.json")
        with open(req_file, "w", encoding="utf-8") as f:
            json.dump(request_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Request recorded to: {req_file}")
        retry_num = 0
        max_retries = 5
        retry_delay = 10
        
        while retry_num < max_retries:
        
            try:
                stream_resp = self.client.chat.completions.create(
                    **request_data,
                    stream=True
                )

                content_parts = []
                tool_deltas = {}
                reasoning_content_parts = []
                all_logprob_tokens = []
                final_usage = None

                for chunk in stream_resp:
                    if chunk.usage:
                        
                        final_usage = chunk.usage.model_dump() 
                        logging.debug(f"Captured Usage information: {final_usage}")
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    if delta is None:
                        
                        continue

                    content = getattr(delta, "content", None)
                    if content:
                        content_parts.append(content)

                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        reasoning_content_parts.append(reasoning)

                    tool_calls = getattr(delta, "tool_calls", None)
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_deltas:
                                tool_deltas[idx] = {"id": tc.id or "", "type": "function", "function": {"name": "", "arguments": ""}}
                            if tc.function:
                                if tc.function.name: tool_deltas[idx]["function"]["name"] += tc.function.name
                                if tc.function.arguments: tool_deltas[idx]["function"]["arguments"] += tc.function.arguments

                    if chunk.choices and chunk.choices[0].logprobs and chunk.choices[0].logprobs.content:
                        for logprob_item in chunk.choices[0].logprobs.content:
                            all_logprob_tokens.append(logprob_item.model_dump())

                final_content = "".join(content_parts)
                final_tool_calls = [tool_deltas[k] for k in sorted(tool_deltas.keys())]
                final_reasoning = "".join(reasoning_content_parts)

                # ORIGINAL CODE BEHAVIOR: Do NOT extract <think> tags from content
                # Keep the full response including "Thinking..." or <think> tags in content
                # Only use reasoning_content from API if explicitly provided (DeepSeek API)
                # The _parse_think_from_text was extracting and REMOVING thinking from content
                # which differs from original data_test_copy.py behavior
                
                # If we have reasoning from API streaming (reasoning_content field), 
                # prepend it to content in "Thinking...done thinking" format for consistency
                if final_reasoning and not final_content.startswith("Thinking"):
                    # Format like original: "Thinking...\n{reasoning}\n...done thinking.\n\n{content}"
                    final_content = f"Thinking...\n{final_reasoning}\n...done thinking.\n\n{final_content}"
                    final_reasoning = ""  # Clear since it's now in content

                if not final_tool_calls:
                    parsed_result = self._parse_tool_calls_from_text(final_content)
                    if parsed_result:
                        final_tool_calls, final_content = parsed_result
                    else:
                        parsed_result_reasoning = self._parse_tool_calls_from_text(final_reasoning)
                        if parsed_result_reasoning:
                            final_tool_calls, final_reasoning = parsed_result_reasoning
                else:
                    if final_content:
                        _, cleaned_content = self._parse_tool_calls_from_text(final_content) or (None, final_content)
                        final_content = cleaned_content
                    if final_reasoning:
                        _, cleaned_reasoning = self._parse_tool_calls_from_text(final_reasoning) or (None, final_reasoning)
                        final_reasoning = cleaned_reasoning

                final_logprobs_obj = {"content": all_logprob_tokens, "refusal": None}
                final_message_obj = {
                    "role": "assistant",
                    "content": final_content,
                    "tool_calls": final_tool_calls or None
                }
                if final_reasoning:
                    final_message_obj['thought'] = final_reasoning

                final_choice = {"index": 0, "finish_reason": "stop", "message": final_message_obj}
                final_response_dict = { "choices": [final_choice], "logprobs": final_logprobs_obj, "usage": final_usage }

                resp_file = os.path.join(log_dir, f"response_{timestamp}.json")
                with open(resp_file, "w", encoding="utf-8") as f:
                    json.dump(final_response_dict, f, ensure_ascii=False, indent=2)
                logging.info(f"Response recorded to: {resp_file}")

                return {
                    "response": final_content,
                    "tool_calls": final_tool_calls,
                    "thought": final_reasoning,
                    "raw_message": final_message_obj,
                    "raw_response": final_response_dict,
                    "logprobs": final_logprobs_obj,
                    "usage": final_usage
                }

            except Exception as e:
                logging.error(f"OpenAI API Error: {e}", exc_info=True)
                # raise ValueError(f"OpenAI API Error: {e}")
                retry_num += 1
                if retry_num < max_retries:
                    logging.info(f"Waiting {retry_delay * retry_num} seconds to retry... (Attempt {retry_num} of {max_retries})")
                    import time
                    time.sleep(retry_delay  * retry_num)
                else:
                    logging.error("Max retries reached, operation failed.")
                    raise ValueError(f"OpenAI API Error: {e}")

class OllamaLLMClient(BaseLLMClient):
    def __init__(self, model: str = "llama3"):
        import ollama
        self.model = model
        self.client = ollama
        logging.info(f"Initialized Ollama client, model: {model}")
    def create_completion(self, messages: List[Dict], tools: List = None) -> Dict[str, Any]:
        
        pass


class LLMClientManager:
    """
    A manager for creating and managing multiple LLM clients with different configurations.
    """
    def __init__(self):
        self._clients: Dict[str, BaseLLMClient] = {}

    def create_client(
        self,
        client_name: str,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 1800.0,
        tool_start_tag: Optional[str] = None,
        tool_end_tag: Optional[str] = None
    ) -> BaseLLMClient:
        """
        Creates a new LLM client instance based on the provided configuration and caches it.
        If a client with the same name already exists, returns the existing instance to avoid duplicate creation.
        """
        if client_name in self._clients:
            logging.info(f"Client named '{client_name}' already exists, returning existing instance.")
            return self._clients[client_name]

        logging.info(f"Creating new LLM client for '{client_name}' -> Provider: {provider}, Model: {model}, Base URL: {base_url or 'N/A'}")
        
        client: BaseLLMClient
        if provider == "openai":
            client = ExtendedOpenAIClient(
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                tool_start_tag=tool_start_tag,
                tool_end_tag=tool_end_tag
            )
        elif provider == "ollama":
            client = OllamaLLMClient(model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self._clients[client_name] = client
        return client

    def get_client(self, client_name: str) -> Optional[BaseLLMClient]:
        """Gets a previously created client."""
        return self._clients.get(client_name)


def get_extended_llm_client(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 1800.0,
    tool_start_tag: Optional[str] = None,
    tool_end_tag: Optional[str] = None
) -> BaseLLMClient:
    """
    Factory function to create an LLM client instance.
    
    Args:
        provider: LLM provider ("openai" or "ollama")
        model: Model name (default: gpt-4o-mini)
        api_key: API key (not required for Ollama)
        base_url: API base URL (default: OpenAI, or http://localhost:11434/v1 for Ollama)
        timeout: Request timeout in seconds
        tool_start_tag: Tag to identify start of tool calls in response
        tool_end_tag: Tag to identify end of tool calls in response
    
    Returns:
        BaseLLMClient: An LLM client instance
    """
    if provider == "openai":
        return ExtendedOpenAIClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            tool_start_tag=tool_start_tag,
            tool_end_tag=tool_end_tag
        )
    elif provider == "ollama":
        return OllamaLLMClient(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")