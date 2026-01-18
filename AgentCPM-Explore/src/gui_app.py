#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AgentCPM-MCP GUI Application

A Gradio-based GUI for interacting with the AgentCPM-MCP system.
Shows step-by-step progress from initial prompt through thinking to final result.
"""

import os
import sys
import asyncio
import json
import logging
import io
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
import gradio as gr

# Ensure AgentCPM-MCP module can be imported
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from ollama_client import OllamaClient, create_ollama_client, extract_answer
from simple_tools import SimpleToolHandler, SIMPLE_TOOLS

# Configure logging to capture logs for GUI display
log_buffer = io.StringIO()
log_handler = logging.StreamHandler(log_buffer)
log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%H:%M:%S"))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), log_handler]
)
logger = logging.getLogger("gui_app")


def extract_thinking_from_response(response_text: str) -> tuple:
    """
    Extract thinking/reasoning content from response text.
    
    Supports multiple formats:
    1. <think>...</think> tags (DeepSeek style)
    2. "Thinking..." ... "...done thinking." format (AgentCPM/quickstart style)
    3. **Thinking:** sections
    
    Returns: (thinking_content, cleaned_response)
    """
    if not response_text:
        return "", ""
    
    thinking = ""
    cleaned = response_text
    
    # Pattern 1: <think>...</think> tags
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
    match = think_pattern.search(response_text)
    if match:
        thinking = match.group(1).strip()
        cleaned = response_text.replace(match.group(0), "").strip()
        return thinking, cleaned
    
    # Pattern 2: "Thinking..." ... "...done thinking." format (from quickstart.py example)
    thinking_done_pattern = re.compile(r'Thinking\.\.\.?\s*(.*?)\s*\.\.\.done thinking\.?', re.DOTALL | re.IGNORECASE)
    match = thinking_done_pattern.search(response_text)
    if match:
        thinking = match.group(1).strip()
        cleaned = response_text.replace(match.group(0), "").strip()
        return thinking, cleaned
    
    # Pattern 3: Just starts with "Thinking..." (model thinking out loud)
    if response_text.strip().lower().startswith("thinking"):
        # Find where the actual response/conclusion begins
        lines = response_text.split('\n')
        thinking_lines = []
        response_lines = []
        found_response = False
        
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            
            # Check for markers that indicate end of thinking
            if any(marker in line_lower for marker in [
                'done thinking', '...done', 'conclusion:', 'to find', 'to identify',
                'the ideal', 'the niche', 'sustainable', 'in summary', 'here\'s why',
                '### high demand', '### low', '### building', '<answer>'
            ]):
                found_response = True
            
            if not found_response and i < len(lines) - 3:  # Keep some context
                thinking_lines.append(line)
            else:
                response_lines.append(line)
        
        if thinking_lines and len(thinking_lines) > 1:
            thinking = '\n'.join(thinking_lines)
            cleaned = '\n'.join(response_lines).strip()
            return thinking, cleaned
    
    # Pattern 4: **Thinking:** or **Reasoning:** sections
    section_pattern = re.compile(r'\*\*(Thinking|Reasoning|Analysis):\*\*\s*(.*?)(?=\*\*(?:Answer|Response|Result|Conclusion):\*\*|$)', re.DOTALL | re.IGNORECASE)
    match = section_pattern.search(response_text)
    if match:
        thinking = match.group(2).strip()
        return thinking, cleaned
    
    return thinking, cleaned


def get_logs():
    """Get accumulated logs from buffer."""
    log_buffer.seek(0)
    logs = log_buffer.read()
    log_buffer.seek(0)
    log_buffer.truncate(0)
    return logs


class AgentGUI:
    """GUI wrapper for AgentCPM interactions with Ollama."""
    
    def __init__(self):
        self.tool_handler = None  # SimpleToolHandler
        self.current_client = None  # OllamaClient
        self.conversation_history = []
        self.step_counter = 0
        self._client_config = {}  # Store config to detect changes
        
    def initialize_client(
        self,
        model: str,
        base_url: str,
        api_key: str = None  # Not used for Ollama, kept for compatibility
    ) -> str:
        """Initialize or reinitialize the Ollama client."""
        try:
            # Check if we need to reinitialize
            new_config = {"model": model, "base_url": base_url}
            if self.current_client and self._client_config == new_config:
                return f"✅ Client already initialized: {model}"
            
            # Create new Ollama client (no openai dependency!)
            self.current_client = create_ollama_client(
                model=model,
                base_url=base_url if base_url else "http://localhost:11434",
                timeout=600.0
            )
            self._client_config = new_config
            return f"✅ Client initialized: {model} @ {base_url or 'http://localhost:11434'}"
        except Exception as e:
            return f"❌ Failed to initialize client: {str(e)}"

    async def initialize_tools(self, use_mcp: bool = False, manager_url: str = None) -> str:
        """Initialize tool handler - uses simple built-in tools."""
        try:
            # Just use simple built-in tools (no MCP dependency)
            self.tool_handler = SimpleToolHandler()
            await self.tool_handler.initialize()
            tool_names = [t["function"]["name"] for t in self.tool_handler.openai_tools]
            return f"✅ Tools ready: {', '.join(tool_names)}"
            
        except Exception as e:
            return f"❌ Tool init error: {str(e)}"

    async def initialize_mcp(self, manager_url: str) -> str:
        """Initialize MCP handler (legacy, now also initializes simple tools as fallback)."""
        return await self.initialize_tools(use_mcp=True, manager_url=manager_url)

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.step_counter = 0
        return "", "Conversation reset."

    def format_section(self, title: str, content: str, icon: str = "") -> str:
        """Format a section for display."""
        if not content or content.strip() == "":
            return ""
        return f"\n\n---\n### {icon} {title}\n\n{content}\n"

    def process_prompt(
        self,
        prompt: str,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        manager_url: str,
        use_tools: bool,
        max_iterations: int = 30,
        max_consecutive_no_op: int = 3,
        return_thought: bool = True,
        use_browser_processor: bool = False,
        use_context_manager: bool = False,
        max_context_tokens: int = 15000,
        log_settings: dict = None,
        enabled_tools: dict = None
    ) -> Generator[tuple, None, None]:
        """
        Process a user prompt and yield step-by-step updates.
        
        Yields tuples of (combined_output, status)
        """
        self.step_counter = 1
        output_parts = []
        
        # Default log settings
        if log_settings is None:
            log_settings = {"raw_output": True, "tool_calls": True, "errors": True, "debug": False}
        
        # Structured log storage - each entry has level and content
        all_logs = []  # List of {"level": "INFO|RAW|TOOL|ERROR|DEBUG", "content": "..."}
        
        def add_log(level: str, content: str):
            """Add a log entry with level prefix."""
            all_logs.append({"level": level, "content": content})
        
        def get_filtered_logs():
            """Get logs filtered by current log_settings."""
            filtered = []
            for log in all_logs:
                level = log["level"]
                content = log["content"]
                
                # Check if this level is enabled
                show = False
                if level == "RAW" and log_settings.get("raw_output"):
                    show = True
                elif level == "TOOL" and log_settings.get("tool_calls"):
                    show = True
                elif level == "ERROR" and log_settings.get("errors"):
                    show = True
                elif level == "DEBUG" and log_settings.get("debug"):
                    show = True
                elif level == "INFO":  # Always show INFO
                    show = True
                
                if show:
                    # Format with level prefix
                    filtered.append(f"[{level}] {content}")
            
            return "\n".join(filtered)
        
        def build_output(input_text="", thinking_text="", tool_calls_text="", response_text="", status_text=""):
            """Build combined output from all sections."""
            sections = []
            
            # Status at the top
            if status_text:
                sections.append(f"**⏳ Status:** {status_text}\n")
            
            if input_text:
                sections.append(f"### 📝 INPUT\n\n{input_text}")
            
            if thinking_text:
                sections.append(f"### 🧠 MODEL RESPONSES (Thinking/Reasoning)\n\n{thinking_text}")
            
            if tool_calls_text:
                sections.append(f"### 🔧 TOOL CALLS\n\n{tool_calls_text}")
            
            if response_text:
                sections.append(f"### 💬 FINAL RESPONSE\n\n{response_text}")
            
            # LOGS section - filtered by log_settings
            logs_text = get_filtered_logs()
            if logs_text:
                sections.append(f"### 📋 LOGS\n\n```\n{logs_text}\n```")
            
            return "\n\n---\n\n".join(sections) if sections else "*Processing...*"
        
        input_text = f"**User Prompt:**\n```\n{prompt}\n```"
        thinking_text = ""
        tool_calls_text = ""
        response_text = ""
        current_status = "🔄 Processing input..."
        
        # Step 1: Input received
        yield (build_output(input_text=input_text, status_text=current_status), current_status)
        
        # Initialize client if needed
        if not self.current_client:
            current_status = "🔄 Initializing LLM client..."
            add_log("INFO", f"Initializing LLM client: {model} @ {base_url}")
            yield (build_output(input_text=input_text, status_text=current_status), current_status)
            
            init_result = self.initialize_client(model, base_url)
            if "❌" in init_result:
                add_log("ERROR", f"Client initialization failed: {init_result}")
                yield (build_output(input_text=input_text, status_text=init_result), init_result)
                return
            add_log("INFO", "Client initialized successfully")
            current_status = "✅ Client initialized"
            yield (build_output(input_text=input_text, status_text=current_status), current_status)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user message (ORIGINAL CODE FORMAT - line 1457 in data_test_copy.py)
        # Original: self.historyx.add_message({"role": "user", "content": f"Your task is to answer the user's question: {query}"})
        user_message = f"Your task is to answer the user's question: {prompt}"
        messages.append({"role": "user", "content": user_message})
        
        # Log the actual message being sent (like original code)
        add_log("INFO", f"User message: {user_message[:150]}...")
        
        # Log RAW initial prompt (original code format)
        initial_messages_json = json.dumps(messages, indent=2, ensure_ascii=False)
        add_log("RAW", f"Initial messages to model:\n{initial_messages_json}")
        
        input_text += f"\n\n**Context:** {len(messages)} messages in conversation"
        current_status = "🔄 Preparing request..."
        yield (build_output(input_text=input_text, status_text=current_status), current_status)
        
        # Get tools if enabled
        tools = None
        # Default enabled_tools if not specified
        if enabled_tools is None:
            enabled_tools = {"web_search": True, "fetch_webpage": True}
        
        if use_tools:
            # Initialize tools if not already done
            if not self.tool_handler:
                current_status = "🔧 Initializing tools..."
                yield (build_output(input_text=input_text, status_text=current_status), current_status)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                init_result = loop.run_until_complete(self.initialize_tools(use_mcp=True, manager_url=manager_url))
                loop.close()
                
                if "❌" in init_result:
                    add_log("ERROR", f"Tool initialization failed: {init_result}")
                    input_text += f"\n**Tools:** {init_result}"
                    current_status = init_result
                    yield (build_output(input_text=input_text, status_text=current_status), current_status)
                else:
                    add_log("INFO", f"Tools initialized: {init_result}")
                    input_text += f"\n**Tools:** {init_result}"
                    current_status = init_result
                    yield (build_output(input_text=input_text, status_text=current_status), current_status)
            
            if self.tool_handler:
                # Filter tools based on enabled_tools checkboxes
                all_tools = self.tool_handler.openai_tools
                tools = [t for t in all_tools if t["function"]["name"] in enabled_tools and enabled_tools.get(t["function"]["name"], False)]
                
                if tools:
                    tool_names = [t["function"]["name"] for t in tools]
                    input_text += f"\n**Enabled tools:** {', '.join(tool_names)}"
                    current_status = f"🔧 {len(tools)} tools enabled"
                    yield (build_output(input_text=input_text, status_text=current_status), current_status)
                else:
                    input_text += f"\n**Tools:** No tools enabled"
                    current_status = "⚠️ No tools enabled"
                    yield (build_output(input_text=input_text, status_text=current_status), current_status)
                    tools = None  # No tools to use
        
        # Agent loop - iterate until <answer> tags found or max iterations
        import time
        start_time = time.time()
        
        all_thinking = []
        all_tool_calls = []
        all_steps = []  # Track all steps/subtasks
        final_response = ""
        final_answer = None
        iteration = 0
        consecutive_no_tool = 0
        MAX_NO_TOOL = max_consecutive_no_op  # Original: MAX_CONSECUTIVE_NO_OP = 3
        context_tokens = 0  # Track token count for context manager
        
        # Patterns for detecting generic/unhelpful filler responses
        GENERIC_PATTERNS = [
            "I'm here to provide",
            "I'm here to help",
            "Please let me know",
            "I'd be happy to help",
            "What would you like",
            "How can I assist",
            "What topic or question",
            "feel free to ask",
            "Let me know if",
            "I can help you with",
        ]
        
        # Stats tracking
        stats = {
            "tool_calls": [],
            "total_tool_calls": 0,
            "execution_time": 0,
            "interactions": 0,
            "max_interactions_reached": False,
            "total_tokens": 0,
            "thinking_iterations": 0,
            "context_compressions": 0
        }
        
        # Log active settings
        logger.info(f"Agent settings: MAX_ITERATIONS={max_iterations}, MAX_NO_OP={max_consecutive_no_op}, "
                   f"RETURN_THOUGHT={return_thought}, BROWSER_PROCESSOR={use_browser_processor}, "
                   f"CONTEXT_MANAGER={use_context_manager}, MAX_CTX_TOKENS={max_context_tokens}")
        
        try:
            while iteration < max_iterations:
                iteration += 1
                current_status = f"🔄 Step {iteration}/{max_iterations}: Analyzing task..."
                logger.info(f"Starting iteration {iteration}/{max_iterations}")
                print(f"[DEBUG] === ITERATION {iteration}/{max_iterations} ===")
                
                # USE_CONTEXT_MANAGER: Check if context exceeds limit (original lines 1473-1525)
                if use_context_manager and context_tokens > max_context_tokens:
                    logger.warning(f"Context manager: {context_tokens} tokens exceeds {max_context_tokens}, compressing...")
                    current_status = f"🗜️ Step {iteration}: Compressing context ({context_tokens} tokens)..."
                    yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                                       tool_calls_text="\n\n---\n\n".join(all_tool_calls), status_text=current_status), current_status)
                    
                    # Simple compression: Keep system prompt, first user message, and last N messages
                    if len(messages) > 6:
                        compressed = [messages[0]]  # System prompt
                        if len(messages) > 1:
                            compressed.append(messages[1])  # First user message
                        # Add summary of middle messages
                        middle_count = len(messages) - 6
                        compressed.append({
                            "role": "system",
                            "content": f"[Context compressed: {middle_count} earlier messages summarized to save tokens]"
                        })
                        # Keep last 4 messages
                        compressed.extend(messages[-4:])
                        messages = compressed
                        context_tokens = max_context_tokens // 2  # Reset estimate
                        stats["context_compressions"] += 1
                        logger.info(f"Context manager: Compressed to {len(messages)} messages")
                
                yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                                   tool_calls_text="\n\n---\n\n".join(all_tool_calls), status_text=current_status), current_status)
                
                # Call LLM
                print(f"[DEBUG] Calling LLM with {len(messages)} messages...")
                result = self.current_client.create_completion(
                    messages=messages,
                    tools=tools,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens if max_tokens > 0 else None
                )
                add_log("DEBUG", f"LLM response received for step {iteration}")
                
                # Check for errors
                if result.get("error"):
                    add_log("ERROR", f"LLM Error at step {iteration}: {result['error']}")
                    current_status = f"❌ LLM Error: {result['error']}"
                    yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                                       tool_calls_text="\n\n---\n\n".join(all_tool_calls), 
                                       status_text=current_status), current_status)
                    break
                
                # Process thinking/reasoning
                # ORIGINAL CODE: Does NOT extract "Thinking..." from content
                # The full response including "Thinking..." is kept in content
                # Only uses separate thought field when LLM explicitly returns it (like DeepSeek <think> tags)
                thinking_text = result.get("thought", "")
                raw_response = result.get("response", "")
                tool_calls = result.get("tool_calls", None)
                
                # Log raw output in original format (like data_test_copy.py) - FULL content, no truncation
                raw_message = {
                    "role": "assistant",
                    "content": raw_response
                }
                if thinking_text:
                    raw_message["thought"] = thinking_text
                if tool_calls:
                    raw_message["tool_calls"] = tool_calls
                
                # Format as JSON like original code output - FULL RAW OUTPUT
                raw_json = json.dumps(raw_message, indent=2, ensure_ascii=False)
                add_log("RAW", f"Step {iteration} model response:\n{raw_json}")
                
                # Debug log
                add_log("DEBUG", f"Step {iteration}: response length={len(raw_response)}, has_tool_calls={tool_calls is not None}")
                
                # Log this step - ORIGINAL FORMAT: Show full raw response (not extracted thinking)
                # Original code shows: "Thinking...\n[reasoning]\n...done thinking.\n\n[response]"
                step_info = f"**Step {iteration}:**"
                
                # For display, show the FULL response to match original output format
                # ORIGINAL CODE: Does NOT truncate - shows complete model response
                if raw_response:
                    all_thinking.append(f"{step_info}\n{raw_response}")
                    current_status = f"🧠 Step {iteration}: Model responding..."
                    yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking), 
                                       status_text=current_status), current_status)
                
                # Check for final answer in response
                final_answer = extract_answer(raw_response)
                add_log("DEBUG", f"Step {iteration} has <answer> tags: {final_answer is not None}")
                
                # Check if answer is generic/unhelpful (model sometimes outputs filler responses)
                is_generic = False
                if final_answer:
                    answer_lower = final_answer.lower()
                    for pattern in GENERIC_PATTERNS:
                        if pattern.lower() in answer_lower and len(final_answer) < 500:
                            is_generic = True
                            add_log("DEBUG", f"Step {iteration}: Detected generic answer, ignoring")
                            break
                
                if final_answer and not is_generic:
                    # Original logic: if <answer> found, we're done
                    logger.info(f"✅ Final answer detected at step {iteration}")
                    print(f"[DEBUG] ✅ Final answer found at step {iteration}")
                    print(f"[DEBUG] Final answer content: {final_answer[:200]}...")
                    
                    final_response = final_answer
                    all_steps.append(f"Step {iteration}: ✅ Final answer provided")
                    
                    # Update conversation history
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": raw_response})
                    
                    # Yield immediately to show final answer (before breaking)
                    add_log("INFO", f"Final answer found at step {iteration}")
                    current_status = f"✅ Final answer found at step {iteration}"
                    yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                                       tool_calls_text="\n\n---\n\n".join(all_tool_calls),
                                       response_text=final_response,
                                       status_text=current_status), current_status)
                    break
                elif final_answer and is_generic:
                    # Generic answer detected - continue iterating or use previous good content
                    add_log("INFO", f"Step {iteration}: Generic answer in <answer> tags ignored, continuing")
                    all_steps.append(f"Step {iteration}: ⚠️ Generic answer ignored")
                
                # Track usage/tokens
                usage = result.get("usage", {})
                if usage:
                    stats["total_tokens"] += usage.get("total_tokens", 0)
                
                # Check for tool calls
                tool_calls = result.get("tool_calls", [])
                
                if tool_calls and self.tool_handler:
                    # Original code (line 1785): reset consecutive_no_op_count = 0
                    consecutive_no_tool = 0
                    
                    # Original code (line 1786): add assistant message ONCE before tool loop
                    messages.append({"role": "assistant", "content": raw_response})
                    
                    iteration_tools = []
                    
                    for i, tool_call in enumerate(tool_calls):
                        func_name = tool_call.get("function", {}).get("name", "unknown")
                        func_args_str = tool_call.get("function", {}).get("arguments", "{}")
                        
                        # Skip invalid or disabled tool names
                        if func_name not in ["web_search", "fetch_webpage"]:
                            logger.warning(f"Skipping invalid tool: {func_name}")
                            continue
                        
                        if not enabled_tools.get(func_name, False):
                            logger.warning(f"Skipping disabled tool: {func_name}")
                            iteration_tools.append(f"**Tool:** `{func_name}` (DISABLED - skipped)")
                            continue
                        
                        tool_entry = f"**Tool:** `{func_name}`\n```json\n{func_args_str}\n```"
                        iteration_tools.append(tool_entry)
                        all_steps.append(f"Step {iteration}: 🔧 Calling {func_name}")
                        
                        # Log tool call request
                        add_log("TOOL", f"Step {iteration} calling {func_name} with args:\n{func_args_str}")
                        
                        current_status = f"🔧 Step {iteration}: Executing {func_name}..."
                        yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking), 
                                           tool_calls_text="\n\n---\n\n".join(all_tool_calls + [f"**Step {iteration}:**\n" + "\n".join(iteration_tools)]),
                                           status_text=current_status), current_status)
                        
                        # Execute tool
                        try:
                            func_args = json.loads(func_args_str)
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            tool_result = loop.run_until_complete(
                                self.tool_handler.call_tool(func_name, func_args)
                            )
                            loop.close()
                            
                            result_str = json.dumps(tool_result, indent=2, ensure_ascii=False)
                            
                            # Log tool result (full, no truncation)
                            add_log("TOOL", f"Step {iteration} {func_name} result:\n{result_str}")
                            
                            # USE_BROWSER_PROCESSOR: Summarize long web content (original lines 2030-2097)
                            if use_browser_processor and func_name in ["fetch_webpage", "web_search"]:
                                if len(result_str) > 2000:
                                    logger.info(f"Browser processor: Summarizing {len(result_str)} chars from {func_name}")
                                    summary_prompt = f"Summarize the key information from this {func_name} result that is relevant to answering a question. Be concise but include all important facts, numbers, and details:\n\n{result_str[:8000]}"
                                    summary_result = self.current_client.create_completion(
                                        messages=[{"role": "user", "content": summary_prompt}],
                                        tools=None,
                                        stream=True,
                                        temperature=0.3,
                                        max_tokens=1000
                                    )
                                    if not summary_result.get("error"):
                                        result_str = f"[Summarized by browser processor]\n{summary_result.get('response', result_str)}"
                                        logger.info(f"Browser processor: Reduced to {len(result_str)} chars")
                            
                            # Original code does NOT truncate tool results
                            # max_context_tokens in original is 15000000 (essentially unlimited)
                            
                            iteration_tools.append(f"**Result:**\n```\n{result_str}\n```")
                            
                            # Track token usage for context manager
                            context_tokens += len(result_str) // 4  # Rough estimate
                            
                            # Original format (lines 2152-2154): <tool_response>...</tool_response>
                            messages.append({
                                "role": "user", 
                                "content": f"<tool_response>\n{result_str}\n</tool_response>"
                            })
                            
                            current_status = f"✅ Step {iteration}: {func_name} completed"
                            all_steps.append(f"Step {iteration}: ✅ {func_name} returned results")
                            
                            # Track tool call in stats
                            stats["tool_calls"].append({
                                "name": func_name,
                                "arguments": func_args,
                                "step": iteration,
                                "status": "success"
                            })
                            stats["total_tool_calls"] += 1
                            
                        except Exception as e:
                            error_str = str(e)
                            iteration_tools.append(f"**Error:** {error_str}")
                            
                            # Log tool error
                            add_log("ERROR", f"Step {iteration} {func_name} failed: {error_str}")
                            
                            # Original format for error (lines 2122-2124)
                            error_content = json.dumps({"error": error_str})
                            messages.append({
                                "role": "user",
                                "content": f"<tool_response>\n{error_content}\n</tool_response>"
                            })
                            
                            current_status = f"⚠️ Step {iteration}: {func_name} failed"
                            all_steps.append(f"Step {iteration}: ❌ {func_name} failed: {error_str[:50]}")
                            
                            # Track failed tool call
                            stats["tool_calls"].append({
                                "name": func_name,
                                "arguments": func_args_str,
                                "step": iteration,
                                "status": "error",
                                "error": error_str[:100]
                            })
                            stats["total_tool_calls"] += 1
                        
                        yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking), 
                                           tool_calls_text="\n\n---\n\n".join(all_tool_calls + [f"**Step {iteration}:**\n" + "\n".join(iteration_tools)]),
                                           status_text=current_status), current_status)
                    
                    if iteration_tools:
                        all_tool_calls.append(f"**Step {iteration}:**\n" + "\n".join(iteration_tools))
                    
                    # Continue to next iteration
                    continue
                
                # No tool calls and no answer - model is thinking/reasoning
                # This is NORMAL - like the original code, the model may iterate multiple times
                consecutive_no_tool += 1
                stats["thinking_iterations"] += 1
                
                logger.info(f"Step {iteration}: No tool call or answer - thinking iteration {consecutive_no_tool}/{MAX_NO_TOOL}")
                print(f"[DEBUG] Step {iteration}: No tool/answer, consecutive_no_tool={consecutive_no_tool}/{MAX_NO_TOOL}")
                
                # Response already added to all_thinking above - just update status
                all_steps.append(f"Step {iteration}: 🧠 Reasoning/thinking")
                print(f"[DEBUG] Thinking iteration {iteration}, response preview: {raw_response[:200] if raw_response else 'empty'}...")
                
                current_status = f"🧠 Step {iteration}: Model reasoning... ({consecutive_no_tool}/{MAX_NO_TOOL} before force)"
                yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                                   tool_calls_text="\n\n---\n\n".join(all_tool_calls),
                                   status_text=current_status), current_status)
                
                # Add assistant response to history
                # Original: if return_thought, wrap thinking in <think> tags (lines 1593-1595)
                if return_thought and thinking_text:
                    content_with_thought = f"<think>{thinking_text}</think>\n{raw_response}"
                    messages.append({"role": "assistant", "content": content_with_thought})
                else:
                    messages.append({"role": "assistant", "content": raw_response})
                
                # Original code logic (lines 1765-1781):
                # if consecutive_no_op_count < MAX_CONSECUTIVE_NO_OP: just continue
                # else: insert force prompt AND reset count to 0, then continue
                
                if consecutive_no_tool < MAX_NO_TOOL:
                    # Just continue to next iteration, no prompt added
                    logger.info(f"NO-OP {consecutive_no_tool}/{MAX_NO_TOOL}: continuing to next round")
                    print(f"[DEBUG] NO-OP {consecutive_no_tool}/{MAX_NO_TOOL}: continuing to next round")
                    all_steps.append(f"Step {iteration}: NO-OP {consecutive_no_tool}/{MAX_NO_TOOL}, continuing...")
                else:
                    # Original: "3rd time: insert system forced answer once"
                    logger.warning(f"NO-OP has reached threshold ({MAX_NO_TOOL}): inserting forced answer system prompt")
                    print(f"[DEBUG] ⚠️ NO-OP threshold reached - inserting force prompt")
                    
                    # Insert force prompt (exactly like original NO_OP_FORCE_SYSTEM_PROMPT)
                    force_msg = {
                        "role": "system",
                        "content": "You have repeatedly failed to produce a tool call or a final answer. Do NOT make any tool calls. Provide your best-guess final answer NOW, strictly wrapped in <answer></answer> tags."
                    }
                    messages.append(force_msg)
                    all_steps.append(f"Step {iteration}: ⚠️ Force prompt inserted")
                    
                    # CRITICAL: Reset count to avoid immediate re-injection (original line 1780)
                    consecutive_no_tool = 0
                    
                    current_status = f"⚠️ Step {iteration}: Force prompt inserted, continuing..."
                    yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                                       tool_calls_text="\n\n---\n\n".join(all_tool_calls),
                                       status_text=current_status), current_status)
                
                # Continue to next iteration (original always continues here)
                continue
            
            # Loop ended - check if we need forced final answer (like original code)
            add_log("INFO", f"Agent loop ended at iteration {iteration}")
            max_iterations_reached = (iteration >= max_iterations)
            
            # Check if we have an answer from the last message
            # Original code: checks last_message_content for <answer> tags
            last_assistant_content = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_assistant_content = msg.get("content", "")
                    break
            
            has_answer_in_last = bool(extract_answer(last_assistant_content))
            
            # ORIGINAL CODE BEHAVIOR: If max iterations reached AND no answer, make ONE MORE call
            if max_iterations_reached and not has_answer_in_last and not final_answer:
                logger.warning(f"Maximum iterations reached ({max_iterations}), but no answer found. Forcing model summarization.")
                print(f"[DEBUG] ⚠️ MAX ITERATIONS REACHED - Forcing final summarization...")
                
                current_status = f"⚠️ Max iterations reached - Forcing final answer..."
                yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                                   tool_calls_text="\n\n---\n\n".join(all_tool_calls),
                                   status_text=current_status), current_status)
                
                # Add force answer prompt (exactly like original code line 2188-2194)
                force_answer_prompt = (
                    "You have now reached the maximum interaction limit. "
                    "You MUST stop making tool calls. "
                    "Based on all the information you have gathered so far, "
                    "you must synthesize and provide what you consider the most likely answer "
                    "in the required format: <answer>your answer</answer>"
                )
                
                messages.append({"role": "system", "content": force_answer_prompt})
                all_steps.append(f"Step {iteration + 1}: 🔴 System forcing final synthesis")
                
                # Make ONE MORE LLM call for final synthesis (original code line 2220-2226)
                final_result = self.current_client.create_completion(
                    messages=messages,
                    tools=None,  # No tools for final synthesis
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens if max_tokens > 0 else None
                )
                
                if not final_result.get("error"):
                    final_content = final_result.get("response", "")
                    
                    # Try to extract answer from forced response
                    forced_answer = extract_answer(final_content)
                    if forced_answer:
                        final_response = forced_answer
                        logger.info("Forced synthesis produced <answer> tags")
                        print(f"[DEBUG] ✅ Forced synthesis produced answer")
                    else:
                        # Use the whole response as answer (original behavior)
                        final_response = final_content
                        logger.info("Forced synthesis did not produce <answer> tags, using full response")
                        print(f"[DEBUG] Using full forced response as answer")
                    
                    # ORIGINAL BEHAVIOR: No truncation - show full response
                    all_thinking.append(f"**Forced Final Synthesis:**\n{final_content}")
                    iteration += 1  # Count this as an iteration
                else:
                    final_response = f"Failed to generate final synthesis: {final_result.get('error')}"
            
            # If still no final_response, find the best substantive response from all assistant messages
            if not final_response:
                # Look through all assistant messages and find the longest/most substantive one
                best_response = ""
                best_length = 0
                
                for msg in messages:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        # Skip generic responses
                        is_generic = False
                        content_lower = content.lower()
                        for pattern in GENERIC_PATTERNS:
                            if pattern.lower() in content_lower and len(content) < 500:
                                is_generic = True
                                break
                        
                        if not is_generic and len(content) > best_length:
                            # Check if it has an answer tag, extract it
                            extracted = extract_answer(content)
                            if extracted and len(extracted) > best_length:
                                best_response = extracted
                                best_length = len(extracted)
                            elif len(content) > best_length:
                                best_response = content
                                best_length = len(content)
                
                if best_response:
                    final_response = best_response
                    add_log("INFO", f"Using best substantive response (length: {best_length})")
                    logger.info(f"Using best substantive assistant message as final response (length: {best_length})")
                elif last_assistant_content:
                    # Fallback to last message if nothing better found
                    extracted = extract_answer(last_assistant_content)
                    final_response = extracted if extracted else last_assistant_content
                    add_log("INFO", "Fallback: Using last assistant message as final response")
                    logger.info("Fallback: Using last assistant message as final response")
            
            # Finalize stats
            stats["execution_time"] = round(time.time() - start_time, 2)
            stats["interactions"] = iteration
            stats["max_interactions_reached"] = max_iterations_reached
            
            # Build stats display
            stats_display = f"""
---
### 📊 Execution Statistics

| Metric | Value |
|--------|-------|
| **Execution Time** | {stats['execution_time']}s |
| **Total Interactions** | {stats['interactions']} |
| **Thinking Iterations** | {stats['thinking_iterations']} |
| **Tool Calls** | {stats['total_tool_calls']} |
| **Max Iterations Reached** | {stats['max_interactions_reached']} |
| **Total Tokens** | {stats['total_tokens'] or 'N/A'} |
"""
            if stats["tool_calls"]:
                stats_display += "\n**Tool Call History:**\n"
                for tc in stats["tool_calls"]:
                    status_icon = "✅" if tc.get("status") == "success" else "❌"
                    stats_display += f"- Step {tc['step']}: {status_icon} `{tc['name']}`\n"
            
            if not final_response:
                # Still no response - shouldn't happen but handle it
                final_response = f"No final answer generated after {iteration} steps.\n\n**Steps taken:**\n" + "\n".join(all_steps)
            
            print(f"[DEBUG] Building final output with response length: {len(final_response)}")
            print(f"[DEBUG] Final response preview: {final_response[:300]}...")
            
            final_response = final_response + stats_display
            
            if max_iterations_reached:
                status = f"⚠️ Completed via forced synthesis ({iteration} steps) | {stats['execution_time']}s"
            else:
                status = f"✅ Complete ({iteration} step{'s' if iteration > 1 else ''}) | {stats['execution_time']}s | {stats['total_tool_calls']} tool calls"
            
            add_log("INFO", f"Completed: {status}")
            yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                               tool_calls_text="\n\n---\n\n".join(all_tool_calls), 
                               response_text=final_response,
                               status_text=status), status)
            
        except Exception as e:
            error_msg = f"Error during agent loop: {str(e)}"
            logger.error(error_msg, exc_info=True)
            add_log("ERROR", f"Exception: {error_msg}")
            current_status = f"❌ Error: {error_msg[:80]}"
            yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                               tool_calls_text="\n\n---\n\n".join(all_tool_calls), 
                               status_text=current_status), current_status)


def create_gui() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    
    agent = AgentGUI()
    
    with gr.Blocks(title="AgentCPM-MCP GUI") as demo:
        
        gr.Markdown("""
        # 🤖 AgentCPM-MCP Interactive GUI
        
        Interactive interface for the AgentCPM-MCP system with step-by-step visualization.
        """)
        
        with gr.Row():
            # Left column: Configuration
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                with gr.Accordion("Model Settings", open=True):
                    model_input = gr.Textbox(
                        label="Model Name",
                        value="deepseek-r1:1.5b",
                        placeholder="e.g., deepseek-r1:1.5b, llama3, mistral"
                    )
                    base_url_input = gr.Textbox(
                        label="Ollama URL",
                        value="http://localhost:11434",
                        placeholder="Ollama server URL (default: http://localhost:11434)"
                    )
                
                with gr.Accordion("Generation Settings", open=True):
                    temperature_slider = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,  # Original code uses 1.0 for natural "thinking" style
                        step=0.1,
                        info="1.0 = natural thinking style (original), 0.0 = deterministic"
                    )
                    max_tokens_slider = gr.Slider(
                        label="Max Tokens (0 = no limit)",
                        minimum=0,
                        maximum=32768,
                        value=16384,  # Original code uses 16384
                        step=1024,
                        info="Original code uses 16384 for full thinking chains"
                    )
                
                with gr.Accordion("Tools Settings", open=True):
                    manager_url_input = gr.Textbox(
                        label="MCP Manager URL (optional)",
                        value="http://localhost:8000/mcpapi",
                        placeholder="MCP Manager API URL (leave default for simple tools)"
                    )
                    gr.Markdown("**Built-in Tools** (select which to enable):")
                    
                    with gr.Row():
                        web_search_checkbox = gr.Checkbox(
                            label="🔍 web_search",
                            value=True,
                            info="Search the web using DuckDuckGo"
                        )
                        fetch_webpage_checkbox = gr.Checkbox(
                            label="🌐 fetch_webpage",
                            value=True,
                            info="Fetch and extract content from URLs"
                        )
                    
                    init_mcp_btn = gr.Button("🔧 Initialize Tools", variant="secondary")
                    mcp_status = gr.Textbox(
                        label="Tools Status",
                        value="Not initialized (will auto-init on first use)",
                        interactive=False
                    )
                    gr.Markdown("""
                    *Tools are initialized automatically when you send a message with any tool enabled.*
                    """)
                
                with gr.Accordion("Agent Settings (Original Flags)", open=True):
                    max_iterations_slider = gr.Slider(
                        label="MAX_INTERACTIONS",
                        minimum=5,
                        maximum=100,
                        value=30,
                        step=5,
                        info="Max rounds before forced synthesis (original: 30)"
                    )
                    
                    consecutive_no_op_slider = gr.Slider(
                        label="MAX_CONSECUTIVE_NO_OP",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        info="Force prompt after N consecutive no-ops (original: 3)"
                    )
                    
                    return_thought_checkbox = gr.Checkbox(
                        label="RETURN_THOUGHT_TO_LLM",
                        value=True,
                        info="Feed model's <think> reasoning back in history (original: true)"
                    )
                    
                    use_browser_processor_checkbox = gr.Checkbox(
                        label="USE_BROWSER_PROCESSOR",
                        value=False,
                        info="Summarize web content with LLM before feeding to agent (original: true)"
                    )
                    
                    use_context_manager_checkbox = gr.Checkbox(
                        label="USE_CONTEXT_MANAGER",
                        value=False,
                        info="Compress history when context exceeds limit (original: false)"
                    )
                    
                    max_context_tokens_slider = gr.Slider(
                        label="Max Context Tokens (for context manager)",
                        minimum=4000,
                        maximum=128000,
                        value=15000,
                        step=1000,
                        info="Trigger compression when exceeded (original: 15000000)",
                        visible=False  # Only show when context manager enabled
                    )
                    
                    # Toggle visibility of max_context_tokens based on context manager
                    use_context_manager_checkbox.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[use_context_manager_checkbox],
                        outputs=[max_context_tokens_slider]
                    )
                    
                with gr.Accordion("Logging Settings", open=False):
                    gr.Markdown("**Select which logs to display:**")
                    
                    log_raw_output_checkbox = gr.Checkbox(
                        label="Log Raw Output (JSON)",
                        value=True,
                        info="Show full assistant messages in original JSON format"
                    )
                    
                    log_tool_calls_checkbox = gr.Checkbox(
                        label="Log Tool Calls",
                        value=True,
                        info="Show tool call requests and results"
                    )
                    
                    log_errors_checkbox = gr.Checkbox(
                        label="Log Errors",
                        value=True,
                        info="Show error messages"
                    )
                    
                    log_debug_checkbox = gr.Checkbox(
                        label="Log Debug Info",
                        value=False,
                        info="Show detailed debug information"
                    )
                    
                    gr.Markdown("""
                    **Notes:**
                    - **Raw Output**: Shows messages like original code: `{"role": "assistant", "content": "Thinking..."}`
                    - **Tool Calls**: Shows function calls and their results
                    - **Errors**: Shows any errors that occur
                    - **Debug**: Shows internal state and processing details
                    """)
                
                with gr.Accordion("System Prompt", open=False):
                    # Preset system prompts - matching original format from data_test_copy.py
                    SYSTEM_PROMPTS = {
                        "AgentCPM Original": """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query. You are provided with functions:

<tools>
{"type": "function", "function": {"name": "web_search", "description": "Search the internet for current information on any topic. Returns search results with titles, URLs, and snippets.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query to find information about"}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "fetch_webpage", "description": "Fetch and extract the main content from a webpage URL. Use this to read articles, documentation, or any web page.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The URL of the webpage to fetch"}}, "required": ["url"]}}}
</tools>

IMPORTANT: ALWAYS adhere to this exact format for tool use:
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>""",

                        "Step-by-Step Research": """You are a deep research assistant. You accomplish tasks iteratively, breaking them into clear steps.

## Task Strategy
1. Analyze the user's request, break it into sub-goals, arrange them logically.
2. Develop a step-by-step plan (1., 2., 3.), each step for a specific sub-goal.
3. Call ONE tool per step to gather information.
4. After each tool result, extract key information and plan next step.
5. Keep iterating until you have enough information.
6. When ready to give final answer, wrap it in <answer>YOUR ANSWER</answer> tags.

## Tool Usage
To call a tool, use this XML format:
<tool_call>
{"name": "web_search", "arguments": {"query": "your search query"}}
</tool_call>

Or:
<tool_call>
{"name": "fetch_webpage", "arguments": {"url": "https://example.com"}}
</tool_call>

## Important Rules
- Call tools to gather REAL data - don't make up information
- One tool call per response
- After tool results, analyze and plan next step
- Only output <answer>...</answer> when you have completed ALL research""",

                        "Thinking Agent": """You are a deep thinking research assistant. For each response:

1. First, think through the problem step by step (prefix with "Thinking...")
2. Consider what information you need to gather
3. Use tools to research: <tool_call>{"name": "web_search", "arguments": {"query": "..."}}</tool_call>
4. After getting results, think about what you learned (prefix with "Thinking...")
5. Continue researching until you have enough data
6. Provide final answer in <answer>YOUR COMPLETE ANSWER</answer> tags

Always show your reasoning process. Think out loud before acting.""",

                        "Minimal": """You are a research assistant with access to web_search and fetch_webpage tools.

Use <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call> to call tools.
Wrap final answer in <answer>...</answer> tags.""",

                        "Custom": ""
                    }
                    
                    prompt_dropdown = gr.Dropdown(
                        label="Select Preset Prompt",
                        choices=list(SYSTEM_PROMPTS.keys()),
                        value="AgentCPM Original",
                        interactive=True
                    )
                    
                    system_prompt_input = gr.Textbox(
                        label="System Prompt (editable)",
                        value=SYSTEM_PROMPTS["AgentCPM Original"],
                        lines=12,
                        placeholder="Enter or modify system prompt..."
                    )
                    
                    def update_prompt(choice):
                        return SYSTEM_PROMPTS.get(choice, "")
                    
                    prompt_dropdown.change(
                        fn=update_prompt,
                        inputs=[prompt_dropdown],
                        outputs=[system_prompt_input]
                    )
            
            # Right column: Main interaction area
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Conversation")
                
                # Default prompt
                default_prompt = "Find a current business niche based on trends and demands from trend analysis with low competition and entry costs, high demand and bill. Validate it, analyse competitions and build MVP code"
                
                # Input area
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="Your Prompt",
                        value=default_prompt,
                        placeholder="Enter your question or request...",
                        lines=3,
                        scale=4
                    )
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("🚀 Send", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                # Status display
                status_display = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
                
                # Combined output display with better height
                gr.Markdown("### 📋 Output (Input → Thinking → Tool Calls → Response → Logs)")
                output_display = gr.Markdown(
                    value="*Click Send to start processing...*",
                    elem_id="output-display",
                    height=500
                )
        
        # Event handlers
        def process_wrapper(prompt, model, base_url, temp, max_tok, max_iter, max_no_op, 
                           return_thought, use_browser_proc, use_ctx_mgr, max_ctx_tokens,
                           sys_prompt, mgr_url, use_web_search, use_fetch_webpage,
                           log_raw, log_tools, log_errors, log_debug):
            """Wrapper to handle the generator output."""
            if not prompt.strip():
                yield ("*Please enter a prompt*", "⚠️ Please enter a prompt")
                return
            
            # Pack logging settings
            log_settings = {
                "raw_output": log_raw,
                "tool_calls": log_tools,
                "errors": log_errors,
                "debug": log_debug
            }
            
            # Pack enabled tools
            enabled_tools = {
                "web_search": use_web_search,
                "fetch_webpage": use_fetch_webpage
            }
            
            # Check if any tools are enabled
            use_tools = use_web_search or use_fetch_webpage
            
            for result in agent.process_prompt(
                prompt, model, base_url, temp, max_tok, sys_prompt, mgr_url, use_tools, 
                int(max_iter), int(max_no_op), return_thought,
                use_browser_proc, use_ctx_mgr, int(max_ctx_tokens),
                log_settings=log_settings,
                enabled_tools=enabled_tools
            ):
                yield result
        
        async def init_tools_wrapper(manager_url):
            """Wrapper for tools initialization."""
            result = await agent.initialize_tools(use_mcp=True, manager_url=manager_url)
            return result
        
        def clear_wrapper():
            """Clear conversation and outputs."""
            agent.reset_conversation()
            # Reset the client so new settings take effect
            agent.current_client = None
            agent.tool_handler = None
            return (
                default_prompt,  # Reset to default prompt
                "*Click Send to start processing...*",
                "Ready"
            )
        
        # Connect events
        submit_btn.click(
            fn=process_wrapper,
            inputs=[
                prompt_input, model_input, base_url_input,
                temperature_slider, max_tokens_slider, max_iterations_slider,
                consecutive_no_op_slider, return_thought_checkbox,
                use_browser_processor_checkbox, use_context_manager_checkbox, max_context_tokens_slider,
                system_prompt_input, manager_url_input, web_search_checkbox, fetch_webpage_checkbox,
                log_raw_output_checkbox, log_tool_calls_checkbox, log_errors_checkbox, log_debug_checkbox
            ],
            outputs=[output_display, status_display]
        )
        
        prompt_input.submit(
            fn=process_wrapper,
            inputs=[
                prompt_input, model_input, base_url_input,
                temperature_slider, max_tokens_slider, max_iterations_slider,
                consecutive_no_op_slider, return_thought_checkbox,
                use_browser_processor_checkbox, use_context_manager_checkbox, max_context_tokens_slider,
                system_prompt_input, manager_url_input, web_search_checkbox, fetch_webpage_checkbox,
                log_raw_output_checkbox, log_tool_calls_checkbox, log_errors_checkbox, log_debug_checkbox
            ],
            outputs=[output_display, status_display]
        )
        
        init_mcp_btn.click(
            fn=init_tools_wrapper,
            inputs=[manager_url_input],
            outputs=[mcp_status]
        )
        
        clear_btn.click(
            fn=clear_wrapper,
            inputs=[],
            outputs=[prompt_input, output_display, status_display]
        )
    
    return demo





def main():
    """Main entry point for the GUI application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AgentCPM-MCP GUI Application")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=12000, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    
    args = parser.parse_args()
    
    logger.info(f"Starting AgentCPM-MCP GUI on {args.host}:{args.port}")
    logger.info("Built-in tools available: web_search, fetch_webpage")
    logger.info("Tools will be initialized automatically when used.")
    
    demo = create_gui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=not args.no_browser  # Auto-open browser
    )


if __name__ == "__main__":
    main()
