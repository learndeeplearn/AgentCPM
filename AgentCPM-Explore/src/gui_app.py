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

from ollama_client import OllamaClient, create_ollama_client
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
    DeepSeek models often include <think>...</think> tags.
    
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
    
    # Pattern 2: **Thinking:** or **Reasoning:** sections
    if not thinking:
        section_pattern = re.compile(r'\*\*(Thinking|Reasoning|Analysis):\*\*\s*(.*?)(?=\*\*(?:Answer|Response|Result|Conclusion):\*\*|$)', re.DOTALL | re.IGNORECASE)
        match = section_pattern.search(response_text)
        if match:
            thinking = match.group(2).strip()
    
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
        """Initialize tool handler (Simple or MCP)."""
        try:
            if use_mcp and manager_url:
                # Try MCP first
                try:
                    from mcp_handler import MCPHandler
                    mcp_handler = MCPHandler(
                        server_name="all",
                        manager_url=manager_url
                    )
                    if await mcp_handler.initialize():
                        self.tool_handler = mcp_handler
                        tool_count = len(self.tool_handler.openai_tools)
                        return f"✅ MCP initialized with {tool_count} tools"
                except Exception as e:
                    logger.warning(f"MCP init failed: {e}, falling back to simple tools")
            
            # Use simple built-in tools
            self.tool_handler = SimpleToolHandler()
            await self.tool_handler.initialize()
            tool_names = [t["function"]["name"] for t in self.tool_handler.openai_tools]
            return f"✅ Simple tools initialized: {', '.join(tool_names)}"
            
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
        use_tools: bool
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
        
        current_status = "🔄 Calling LLM (this may take a while)..."
        yield (build_output(input_text=input_text, status_text=current_status), current_status)
        
        # Call LLM
        try:
            result = self.current_client.create_completion(
                messages=messages,
                tools=tools,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens > 0 else None
            )
            
            logs_text = collect_logs()
            current_status = "✅ LLM response received"
            yield (build_output(input_text=input_text, logs_text=logs_text, status_text=current_status), current_status)
            
            # Process thinking/reasoning - check multiple sources
            thinking_text = result.get("thought", "")
            raw_response = result.get("response", "")
            
            # If no explicit thought, try to extract from response
            if not thinking_text and raw_response:
                extracted_thinking, cleaned_response = extract_thinking_from_response(raw_response)
                if extracted_thinking:
                    thinking_text = extracted_thinking
                    # Update response to cleaned version
                    result["response"] = cleaned_response
            
            if thinking_text:
                current_status = "🧠 Reasoning/thinking extracted"
                yield (build_output(input_text=input_text, thinking_text=thinking_text, logs_text=logs_text, status_text=current_status), 
                       current_status)
            
            # Process tool calls
            if result.get("tool_calls"):
                tool_entries = []
                for i, tool_call in enumerate(result["tool_calls"]):
                    func_name = tool_call.get("function", {}).get("name", "unknown")
                    func_args = tool_call.get("function", {}).get("arguments", "{}")
                    
                    tool_entry = f"**Tool {i+1}:** `{func_name}`\n**Arguments:**\n```json\n{func_args}\n```"
                    tool_entries.append(tool_entry)
                    
                    current_status = f"🔧 Executing tool: {func_name}"
                    tool_calls_text = "\n\n".join(tool_entries)
                    yield (build_output(input_text=input_text, thinking_text=thinking_text, 
                                       tool_calls_text=tool_calls_text, logs_text=logs_text, status_text=current_status),
                           current_status)
                    
                    # Execute tool call if tool handler available
                    if self.tool_handler:
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            tool_result = loop.run_until_complete(
                                self.tool_handler.call_tool(func_name, json.loads(func_args))
                            )
                            loop.close()
                            
                            result_str = json.dumps(tool_result, indent=2, ensure_ascii=False)
                            if len(result_str) > 3000:
                                result_str = result_str[:3000] + "\n... (truncated)"
                            
                            tool_entries.append(f"**Result:**\n```json\n{result_str}\n```")
                            current_status = f"✅ Tool {func_name} completed"
                        except Exception as e:
                            tool_entries.append(f"**Error:** {str(e)}")
                            current_status = f"❌ Tool {func_name} failed: {str(e)[:50]}"
                        
                        tool_calls_text = "\n\n".join(tool_entries)
                        logs_text = collect_logs()
                        yield (build_output(input_text=input_text, thinking_text=thinking_text, 
                                           tool_calls_text=tool_calls_text, logs_text=logs_text, status_text=current_status),
                               current_status)
                
                current_status = "🔧 All tool calls processed"
                yield (build_output(input_text=input_text, thinking_text=thinking_text, 
                                   tool_calls_text=tool_calls_text, logs_text=logs_text, status_text=current_status),
                       current_status)
            
            # Process final response
            response_text = result.get("response", "")
            logs_text = collect_logs()
            
            if response_text:
                current_status = "✅ Response ready"
                yield (build_output(input_text=input_text, thinking_text=thinking_text,
                                   tool_calls_text=tool_calls_text, response_text=response_text,
                                   logs_text=logs_text, status_text=current_status),
                       current_status)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({
                "role": "assistant", 
                "content": response_text,
                "tool_calls": result.get("tool_calls")
            })
            
            # Final yield with complete status
            usage = result.get('usage', {})
            status = f"✅ Complete"
            if usage:
                status += f" | Tokens: {usage}"
            yield (build_output(input_text=input_text, thinking_text=thinking_text,
                               tool_calls_text=tool_calls_text, response_text=response_text,
                               logs_text=logs_text, status_text=status), status)
            
        except Exception as e:
            error_msg = f"Error during LLM call: {str(e)}"
            logger.error(error_msg, exc_info=True)
            logs_text = collect_logs()
            logs_text += f"\n\n**Exception:** {error_msg}"
            current_status = f"❌ Error: {error_msg[:80]}"
            yield (build_output(input_text=input_text, thinking_text=thinking_text,
                               tool_calls_text=tool_calls_text, logs_text=logs_text, status_text=current_status),
                   current_status)


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
                        placeholder="e.g., gpt-4o-mini, deepseek-r1:1.5b"
                    )
                    base_url_input = gr.Textbox(
                        label="Base URL",
                        value="http://localhost:11434/v1",
                        placeholder="Leave empty for OpenAI, or http://localhost:11434/v1 for Ollama"
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
                
                with gr.Accordion("System Prompt", open=False):
                    system_prompt_input = gr.Textbox(
                        label="System Prompt",
                        value="You are a helpful AI assistant that thinks step-by-step and explains your reasoning clearly.",
                        lines=3,
                        placeholder="Enter system prompt..."
                    )
            
            # Right column: Main interaction area
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Conversation")
                
                # Default prompt
                default_prompt = "Find a current business niche based on trends and demands from reddit analysis with low competition and entry costs, high demand and bill. Validate it, analyse competitions and build MVP code"
                
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
        def process_wrapper(prompt, model, base_url, temp, max_tok, sys_prompt, mgr_url, use_tools):
            """Wrapper to handle the generator output."""
            if not prompt.strip():
                yield ("*Please enter a prompt*", "⚠️ Please enter a prompt")
                return
            
            for result in agent.process_prompt(
                prompt, model, base_url, temp, max_tok, sys_prompt, mgr_url, use_tools
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
                temperature_slider, max_tokens_slider, system_prompt_input,
                manager_url_input, use_tools_checkbox
            ],
            outputs=[output_display, status_display]
        )
        
        prompt_input.submit(
            fn=process_wrapper,
            inputs=[
                prompt_input, model_input, base_url_input,
                temperature_slider, max_tokens_slider, system_prompt_input,
                manager_url_input, use_tools_checkbox
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
