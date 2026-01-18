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
        max_iterations: int = 5
    ) -> Generator[tuple, None, None]:
        """
        Process a user prompt and yield step-by-step updates.
        
        Yields tuples of (combined_output, status)
        """
        self.step_counter = 1
        output_parts = []
        logs_accumulated = []
        
        def build_output(input_text="", thinking_text="", tool_calls_text="", response_text="", logs_text="", status_text=""):
            """Build combined output from all sections."""
            sections = []
            
            # Status at the top
            if status_text:
                sections.append(f"**⏳ Status:** {status_text}\n")
            
            if input_text:
                sections.append(f"### 📝 INPUT\n\n{input_text}")
            
            if thinking_text:
                sections.append(f"### 🧠 THINKING/REASONING\n\n{thinking_text}")
            
            if tool_calls_text:
                sections.append(f"### 🔧 TOOL CALLS\n\n{tool_calls_text}")
            
            if response_text:
                sections.append(f"### 💬 FINAL RESPONSE\n\n{response_text}")
            
            if logs_text:
                sections.append(f"### 📋 LOGS\n\n```\n{logs_text}\n```")
            
            return "\n\n---\n\n".join(sections) if sections else "*Processing...*"
        
        # Collect logs
        def collect_logs():
            logs = get_logs()
            if logs:
                logs_accumulated.append(logs)
            return "\n".join(logs_accumulated)
        
        input_text = f"**User Prompt:**\n```\n{prompt}\n```"
        thinking_text = ""
        tool_calls_text = ""
        response_text = ""
        logs_text = ""
        current_status = "🔄 Processing input..."
        
        # Step 1: Input received
        yield (build_output(input_text=input_text, status_text=current_status), current_status)
        
        # Initialize client if needed
        if not self.current_client:
            current_status = "🔄 Initializing LLM client..."
            yield (build_output(input_text=input_text, status_text=current_status), current_status)
            
            init_result = self.initialize_client(model, base_url)
            logs_text = collect_logs()
            if "❌" in init_result:
                logs_text += f"\n{init_result}"
                yield (build_output(input_text=input_text, logs_text=logs_text, status_text=init_result), init_result)
                return
            current_status = "✅ Client initialized"
            yield (build_output(input_text=input_text, logs_text=logs_text, status_text=current_status), current_status)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": prompt})
        
        input_text += f"\n\n**Context:** {len(messages)} messages in conversation"
        current_status = "🔄 Preparing request..."
        yield (build_output(input_text=input_text, status_text=current_status), current_status)
        
        # Get tools if enabled
        tools = None
        if use_tools:
            # Initialize tools if not already done
            if not self.tool_handler:
                current_status = "🔧 Initializing tools..."
                yield (build_output(input_text=input_text, status_text=current_status), current_status)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                init_result = loop.run_until_complete(self.initialize_tools(use_mcp=True, manager_url=manager_url))
                loop.close()
                
                logs_text = collect_logs()
                if "❌" in init_result:
                    input_text += f"\n**Tools:** {init_result}"
                    current_status = init_result
                    yield (build_output(input_text=input_text, logs_text=logs_text, status_text=current_status), current_status)
                else:
                    input_text += f"\n**Tools:** {init_result}"
                    current_status = init_result
                    yield (build_output(input_text=input_text, logs_text=logs_text, status_text=current_status), current_status)
            
            if self.tool_handler:
                tools = self.tool_handler.openai_tools
                tool_names = [t["function"]["name"] for t in tools]
                input_text += f"\n**Available tools:** {', '.join(tool_names)}"
                current_status = f"🔧 {len(tools)} tools available"
                yield (build_output(input_text=input_text, status_text=current_status), current_status)
        
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
        MAX_NO_TOOL = 4  # Max consecutive responses without tool calls before forcing (like original)
        
        # Stats tracking
        stats = {
            "tool_calls": [],
            "total_tool_calls": 0,
            "execution_time": 0,
            "interactions": 0,
            "max_interactions_reached": False,
            "total_tokens": 0,
            "thinking_iterations": 0
        }
        
        try:
            while iteration < max_iterations:
                iteration += 1
                current_status = f"🔄 Step {iteration}/{max_iterations}: Analyzing task..."
                logger.info(f"Starting iteration {iteration}/{max_iterations}")
                print(f"[DEBUG] === ITERATION {iteration}/{max_iterations} ===")
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
                print(f"[DEBUG] LLM response received")
                
                logs_text = collect_logs()
                
                # Check for errors
                if result.get("error"):
                    current_status = f"❌ LLM Error: {result['error']}"
                    yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                                       tool_calls_text="\n\n---\n\n".join(all_tool_calls), 
                                       logs_text=logs_text, status_text=current_status), current_status)
                    break
                
                # Process thinking/reasoning
                thinking_text = result.get("thought", "")
                raw_response = result.get("response", "")
                
                # Extract thinking from response if not explicit
                if not thinking_text and raw_response:
                    extracted_thinking, cleaned_response = extract_thinking_from_response(raw_response)
                    if extracted_thinking:
                        thinking_text = extracted_thinking
                        result["response"] = cleaned_response
                        raw_response = cleaned_response
                
                # Log this step
                step_info = f"**Step {iteration}:**"
                if thinking_text:
                    # Truncate long thinking for display
                    display_thinking = thinking_text[:800] + "..." if len(thinking_text) > 800 else thinking_text
                    all_thinking.append(f"{step_info}\n{display_thinking}")
                    current_status = f"🧠 Step {iteration}: Reasoning..."
                    yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking), 
                                       logs_text=logs_text, status_text=current_status), current_status)
                
                # Check for final answer in response
                # But ONLY accept answer after at least some iterations OR if tools were used
                final_answer = extract_answer(raw_response)
                
                # Log raw response for debugging
                logger.info(f"Step {iteration} raw response length: {len(raw_response)}")
                logger.info(f"Step {iteration} has <answer> tags: {final_answer is not None}")
                print(f"[DEBUG] Step {iteration}: response length={len(raw_response)}, has_answer={final_answer is not None}")
                
                if final_answer:
                    # Only accept answer if we've done some work OR it's forced
                    # This prevents model from giving answer without research
                    if iteration >= 2 or stats["total_tool_calls"] > 0 or consecutive_no_tool >= MAX_NO_TOOL:
                        logger.info(f"✅ Final answer detected at step {iteration}")
                        print(f"[DEBUG] ✅ Accepting final answer at step {iteration}")
                        final_response = final_answer
                        all_steps.append(f"Step {iteration}: ✅ Final answer provided")
                        
                        # Update conversation history
                        self.conversation_history.append({"role": "user", "content": prompt})
                        self.conversation_history.append({"role": "assistant", "content": final_response})
                        break
                    else:
                        # Model gave answer too early - treat it as thinking and continue
                        logger.info(f"⚠️ Model provided answer at step {iteration} but needs more iterations")
                        print(f"[DEBUG] ⚠️ Answer too early at step {iteration}, continuing...")
                        
                        # Add to thinking instead
                        all_thinking.append(f"**Iteration {iteration} (Early answer, needs research):**\n{raw_response[:1500]}...")
                        all_steps.append(f"Step {iteration}: ⚠️ Early answer - needs research first")
                        stats["thinking_iterations"] += 1
                        
                        messages.append({"role": "assistant", "content": raw_response})
                        messages.append({
                            "role": "user",
                            "content": "Before providing your final answer, you need to research this topic. Please use the web_search tool to gather real data first. Call the tool like this:\n<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"your search query\"}}\n</tool_call>"
                        })
                        
                        current_status = f"🔄 Step {iteration}: Answer too early - prompting for research..."
                        yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                                           tool_calls_text="\n\n---\n\n".join(all_tool_calls),
                                           logs_text=logs_text, status_text=current_status), current_status)
                        continue
                
                # Track usage/tokens
                usage = result.get("usage", {})
                if usage:
                    stats["total_tokens"] += usage.get("total_tokens", 0)
                
                # Check for tool calls
                tool_calls = result.get("tool_calls", [])
                
                if tool_calls and self.tool_handler:
                    consecutive_no_tool = 0  # Reset counter
                    iteration_tools = []
                    
                    for i, tool_call in enumerate(tool_calls):
                        func_name = tool_call.get("function", {}).get("name", "unknown")
                        func_args_str = tool_call.get("function", {}).get("arguments", "{}")
                        
                        # Skip invalid tool names
                        if func_name not in ["web_search", "fetch_webpage"]:
                            logger.warning(f"Skipping invalid tool: {func_name}")
                            continue
                        
                        tool_entry = f"**Tool:** `{func_name}`\n```json\n{func_args_str}\n```"
                        iteration_tools.append(tool_entry)
                        all_steps.append(f"Step {iteration}: 🔧 Calling {func_name}")
                        
                        current_status = f"🔧 Step {iteration}: Executing {func_name}..."
                        yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking), 
                                           tool_calls_text="\n\n---\n\n".join(all_tool_calls + [f"**Step {iteration}:**\n" + "\n".join(iteration_tools)]),
                                           logs_text=logs_text, status_text=current_status), current_status)
                        
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
                            # Keep more results for analysis
                            if len(result_str) > 4000:
                                result_str = result_str[:4000] + "\n... (truncated)"
                            
                            iteration_tools.append(f"**Result:**\n```\n{result_str}\n```")
                            
                            # Add tool result to messages for next iteration
                            messages.append({"role": "assistant", "content": raw_response})
                            messages.append({
                                "role": "user", 
                                "content": f"""Tool '{func_name}' returned the following results:

{result_str}

Based on these results:
1. Extract key information relevant to the task
2. Determine if you need more information (use another tool call)
3. If you have gathered enough information, provide your final answer wrapped in <answer>YOUR ANSWER</answer> tags

Continue with the next step of your plan."""
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
                            messages.append({"role": "assistant", "content": raw_response})
                            messages.append({
                                "role": "user",
                                "content": f"Tool '{func_name}' failed with error: {error_str}\n\nPlease try a different approach or search query."
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
                        
                        logs_text = collect_logs()
                        yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking), 
                                           tool_calls_text="\n\n---\n\n".join(all_tool_calls + [f"**Step {iteration}:**\n" + "\n".join(iteration_tools)]),
                                           logs_text=logs_text, status_text=current_status), current_status)
                    
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
                
                # Add this iteration's reasoning to thinking display
                if raw_response:
                    # Show full response as thinking for this iteration
                    display_response = raw_response[:2000] + "..." if len(raw_response) > 2000 else raw_response
                    all_thinking.append(f"**Iteration {iteration} (Thinking):**\n{display_response}")
                    all_steps.append(f"Step {iteration}: 🧠 Reasoning/thinking")
                    print(f"[DEBUG] Added thinking iteration {iteration}, response preview: {raw_response[:200]}...")
                
                current_status = f"🧠 Step {iteration}: Model reasoning... ({consecutive_no_tool}/{MAX_NO_TOOL} before force)"
                yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                                   tool_calls_text="\n\n---\n\n".join(all_tool_calls),
                                   logs_text=logs_text, status_text=current_status), current_status)
                
                # Add assistant response to history
                messages.append({"role": "assistant", "content": raw_response})
                
                # Check how many consecutive iterations without tool/answer
                if consecutive_no_tool >= MAX_NO_TOOL:
                    # Force the model to provide a final answer (like original code)
                    logger.info(f"Forcing final answer after {consecutive_no_tool} consecutive no-ops")
                    print(f"[DEBUG] ⚠️ FORCING FINAL ANSWER after {consecutive_no_tool} no-ops")
                    messages.append({
                        "role": "system",
                        "content": "You have repeatedly failed to produce a tool call or a final answer. Do NOT make any tool calls. Provide your best-guess final answer NOW, strictly wrapped in <answer></answer> tags."
                    })
                    all_steps.append(f"Step {iteration}: ⚠️ System forcing final answer")
                    
                    current_status = f"⚠️ Step {iteration}: Forcing final answer after {consecutive_no_tool} reasoning iterations..."
                    yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                                       tool_calls_text="\n\n---\n\n".join(all_tool_calls),
                                       logs_text=logs_text, status_text=current_status), current_status)
                else:
                    # Let the model continue thinking
                    logger.info(f"Continuing to next iteration...")
                    print(f"[DEBUG] Continuing to iteration {iteration + 1}...")
                    all_steps.append(f"Step {iteration}: Continuing to next iteration...")
                
                # Continue to next iteration
                continue
            
            # Loop ended - either by answer or max iterations
            logs_text = collect_logs()
            
            # Finalize stats
            stats["execution_time"] = round(time.time() - start_time, 2)
            stats["interactions"] = iteration
            stats["max_interactions_reached"] = (iteration >= max_iterations and not final_answer)
            
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
            
            if not final_answer and not final_response:
                # Max iterations reached without answer
                final_response = f"Task incomplete after {iteration} steps.\n\n**Steps taken:**\n" + "\n".join(all_steps)
                final_response += stats_display
                status = f"⚠️ Max iterations ({max_iterations}) reached without final answer"
            else:
                final_response = final_response + stats_display
                status = f"✅ Complete ({iteration} step{'s' if iteration > 1 else ''}) | {stats['execution_time']}s | {stats['total_tool_calls']} tool calls"
            
            yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                               tool_calls_text="\n\n---\n\n".join(all_tool_calls), 
                               response_text=final_response,
                               logs_text=logs_text, status_text=status), status)
            
        except Exception as e:
            error_msg = f"Error during agent loop: {str(e)}"
            logger.error(error_msg, exc_info=True)
            logs_text = collect_logs()
            logs_text += f"\n\n**Exception:** {error_msg}"
            current_status = f"❌ Error: {error_msg[:80]}"
            yield (build_output(input_text=input_text, thinking_text="\n\n".join(all_thinking),
                               tool_calls_text="\n\n---\n\n".join(all_tool_calls), 
                               logs_text=logs_text, status_text=current_status), current_status)


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
                        value=0.0,
                        step=0.1
                    )
                    max_tokens_slider = gr.Slider(
                        label="Max Tokens (0 = no limit)",
                        minimum=0,
                        maximum=8192,
                        value=2048,
                        step=256
                    )
                
                with gr.Accordion("Tools Settings", open=True):
                    manager_url_input = gr.Textbox(
                        label="MCP Manager URL (optional)",
                        value="http://localhost:8000/mcpapi",
                        placeholder="MCP Manager API URL (leave default for simple tools)"
                    )
                    use_tools_checkbox = gr.Checkbox(
                        label="Enable Tools (web_search, fetch_webpage)",
                        value=True
                    )
                    init_mcp_btn = gr.Button("🔧 Initialize Tools", variant="secondary")
                    mcp_status = gr.Textbox(
                        label="Tools Status",
                        value="Not initialized (will auto-init on first use)",
                        interactive=False
                    )
                    gr.Markdown("""
                    **Built-in Tools** (always available):
                    - 🔍 **web_search** - Search the web using DuckDuckGo
                    - 🌐 **fetch_webpage** - Fetch and extract content from URLs
                    
                    *Tools are initialized automatically when you send a message with "Enable Tools" checked.*
                    """)
                
                with gr.Accordion("Agent Settings", open=True):
                    max_iterations_slider = gr.Slider(
                        label="Max Iterations (for complex tasks)",
                        minimum=1,
                        maximum=30,
                        value=15,
                        step=1,
                        info="How many think→tool→think cycles (more = deeper research)"
                    )
                
                with gr.Accordion("System Prompt", open=False):
                    # Preset system prompts
                    SYSTEM_PROMPTS = {
                        "AgentCPM Original": """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query. You are provided with functions:

<tools>
- web_search: Search the internet for information
- fetch_webpage: Read content from a specific URL
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
{"name": "web_search", "arguments": {"query": "your search"}}
</tool_call>

Or:
<tool_call>
{"name": "fetch_webpage", "arguments": {"url": "https://example.com"}}
</tool_call>

## Important Rules
- Call tools to gather REAL data - don't make up information
- One tool call per response
- After tool results, analyze and call another tool if needed
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
        def process_wrapper(prompt, model, base_url, temp, max_tok, max_iter, sys_prompt, mgr_url, use_tools):
            """Wrapper to handle the generator output."""
            if not prompt.strip():
                yield ("*Please enter a prompt*", "⚠️ Please enter a prompt")
                return
            
            for result in agent.process_prompt(
                prompt, model, base_url, temp, max_tok, sys_prompt, mgr_url, use_tools, int(max_iter)
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
                system_prompt_input, manager_url_input, use_tools_checkbox
            ],
            outputs=[output_display, status_display]
        )
        
        prompt_input.submit(
            fn=process_wrapper,
            inputs=[
                prompt_input, model_input, base_url_input,
                temperature_slider, max_tokens_slider, max_iterations_slider,
                system_prompt_input, manager_url_input, use_tools_checkbox
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
