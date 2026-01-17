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

from extended_openai_client import (
    get_extended_llm_client, LLMClientManager
)

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
    """GUI wrapper for AgentCPM-MCP interactions."""
    
    def __init__(self):
        self.client_manager = LLMClientManager()
        self.mcp_handler = None
        self.current_client = None
        self.conversation_history = []
        self.step_counter = 0
        
    def initialize_client(
        self,
        model: str,
        base_url: str,
        api_key: str = None
    ) -> str:
        """Initialize or reinitialize the LLM client."""
        try:
            # Always use openai provider (works with Ollama's OpenAI-compatible API)
            client_name = f"openai_{model}"
            self.current_client = self.client_manager.create_client(
                client_name=client_name,
                provider="openai",
                model=model,
                api_key=api_key if api_key else None,
                base_url=base_url if base_url else None,
                timeout=1800.0
            )
            return f"✅ Client initialized: {model} @ {base_url or 'default'}"
        except Exception as e:
            return f"❌ Failed to initialize client: {str(e)}"

    async def initialize_mcp(self, manager_url: str) -> str:
        """Initialize MCP handler."""
        try:
            from mcp_handler import MCPHandler
            self.mcp_handler = MCPHandler(
                server_name="all",
                manager_url=manager_url
            )
            if await self.mcp_handler.initialize():
                tool_count = len(self.mcp_handler.openai_tools)
                return f"✅ MCP initialized with {tool_count} tools"
            else:
                return "❌ MCP initialization failed"
        except Exception as e:
            return f"❌ MCP error: {str(e)}"

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
        
        def build_output(input_text="", thinking_text="", tool_calls_text="", response_text="", logs_text=""):
            """Build combined output from all sections."""
            sections = []
            
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
            
            return "\n\n---\n\n".join(sections)
        
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
        
        # Step 1: Input received
        yield (build_output(input_text=input_text), "🔄 Processing input...")
        
        # Initialize client if needed
        if not self.current_client:
            init_result = self.initialize_client(model, base_url)
            logs_text = collect_logs()
            if "❌" in init_result:
                logs_text += f"\n{init_result}"
                yield (build_output(input_text=input_text, logs_text=logs_text), init_result)
                return
            yield (build_output(input_text=input_text, logs_text=logs_text), "🔄 Client initialized...")
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": prompt})
        
        input_text += f"\n\n**Context:** {len(messages)} messages in conversation"
        yield (build_output(input_text=input_text), "🔄 Calling LLM...")
        
        # Get tools if enabled
        tools = None
        if use_tools and self.mcp_handler:
            tools = self.mcp_handler.openai_tools
            input_text += f"\n**Tools:** {len(tools)} tools available"
            yield (build_output(input_text=input_text), "🔄 Tools loaded...")
        
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
                yield (build_output(input_text=input_text, thinking_text=thinking_text, logs_text=logs_text), 
                       "🧠 Reasoning complete...")
            
            # Process tool calls
            if result.get("tool_calls"):
                tool_entries = []
                for i, tool_call in enumerate(result["tool_calls"]):
                    func_name = tool_call.get("function", {}).get("name", "unknown")
                    func_args = tool_call.get("function", {}).get("arguments", "{}")
                    
                    tool_entry = f"**Tool {i+1}:** `{func_name}`\n**Arguments:**\n```json\n{func_args}\n```"
                    tool_entries.append(tool_entry)
                    
                    # Execute tool call if MCP handler available
                    if self.mcp_handler:
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            tool_result = loop.run_until_complete(
                                self.mcp_handler.call_tool(func_name, json.loads(func_args))
                            )
                            loop.close()
                            
                            result_str = json.dumps(tool_result, indent=2, ensure_ascii=False)
                            if len(result_str) > 1000:
                                result_str = result_str[:1000] + "\n... (truncated)"
                            
                            tool_entries.append(f"**Result:**\n```json\n{result_str}\n```")
                        except Exception as e:
                            tool_entries.append(f"**Error:** {str(e)}")
                
                tool_calls_text = "\n\n".join(tool_entries)
                logs_text = collect_logs()
                yield (build_output(input_text=input_text, thinking_text=thinking_text, 
                                   tool_calls_text=tool_calls_text, logs_text=logs_text),
                       "🔧 Tool calls processed")
            
            # Process final response
            response_text = result.get("response", "")
            logs_text = collect_logs()
            
            if response_text:
                yield (build_output(input_text=input_text, thinking_text=thinking_text,
                                   tool_calls_text=tool_calls_text, response_text=response_text,
                                   logs_text=logs_text),
                       "✅ Response complete")
            
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
                               logs_text=logs_text), status)
            
        except Exception as e:
            error_msg = f"Error during LLM call: {str(e)}"
            logger.error(error_msg, exc_info=True)
            logs_text = collect_logs()
            logs_text += f"\n\n**Exception:** {error_msg}"
            yield (build_output(input_text=input_text, thinking_text=thinking_text,
                               tool_calls_text=tool_calls_text, logs_text=logs_text),
                   f"❌ {error_msg[:100]}")


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
                
                with gr.Accordion("MCP Tools Settings", open=True):
                    manager_url_input = gr.Textbox(
                        label="MCP Manager URL",
                        value="http://localhost:8000/mcpapi",
                        placeholder="MCP Manager API URL"
                    )
                    use_tools_checkbox = gr.Checkbox(
                        label="Enable MCP Tools",
                        value=True
                    )
                    init_mcp_btn = gr.Button("🔌 Initialize MCP", variant="secondary")
                    mcp_status = gr.Textbox(
                        label="MCP Status",
                        value="Not initialized",
                        interactive=False
                    )
                    gr.Markdown("""
                    **Available MCP Tools** (when connected):
                    - 🔍 Search (web search via multiple engines)
                    - 🌐 Browse (web page content extraction)
                    - 📄 Read File (enhanced file reading)
                    - And more depending on MCP server configuration
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
        
        async def init_mcp_wrapper(manager_url):
            """Wrapper for MCP initialization."""
            result = await agent.initialize_mcp(manager_url)
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
            fn=init_mcp_wrapper,
            inputs=[manager_url_input],
            outputs=[mcp_status]
        )
        
        clear_btn.click(
            fn=clear_wrapper,
            inputs=[],
            outputs=[prompt_input, output_display, status_display]
        )
    
    return demo


def start_mcp_server(config_path: str = None, port: int = 8000):
    """
    Start the MCP server in a subprocess.
    
    Args:
        config_path: Path to MCP config file
        port: Port to run MCP server on
    
    Returns:
        subprocess.Popen or None
    """
    import subprocess
    import time
    
    # Find the MCP server main.py
    mcp_server_paths = [
        project_root / "AgentDock" / "agentdock-node-explore" / "main.py",
        project_root / "AgentDock" / "node" / "main.py",
        script_dir.parent / "AgentDock" / "agentdock-node-explore" / "main.py",
    ]
    
    mcp_main = None
    for path in mcp_server_paths:
        if path.exists():
            mcp_main = path
            break
    
    if not mcp_main:
        logger.warning("MCP server main.py not found. MCP tools will not be available.")
        logger.warning(f"Searched paths: {mcp_server_paths}")
        return None
    
    # Find config file
    if not config_path:
        config_paths = [
            mcp_main.parent / "config.toml",
            project_root / "AgentDock" / "agentdock-node-explore" / "config.toml",
        ]
        for path in config_paths:
            if path.exists():
                config_path = str(path)
                break
    
    if not config_path:
        logger.warning("MCP config file not found.")
        return None
    
    logger.info(f"Starting MCP server from: {mcp_main}")
    logger.info(f"Using config: {config_path}")
    
    try:
        # Set environment and start MCP server
        env = os.environ.copy()
        env["CONFIG_FILE_PATH"] = config_path
        
        # Create log file for MCP server output
        mcp_log_path = script_dir / "mcp_server.log"
        mcp_log = open(mcp_log_path, "w")
        
        process = subprocess.Popen(
            ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(port)],
            cwd=str(mcp_main.parent),
            env=env,
            stdout=mcp_log,
            stderr=subprocess.STDOUT
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        if process.poll() is None:
            logger.info(f"MCP server started on port {port}")
            logger.info(f"MCP server logs: {mcp_log_path}")
            return process
        else:
            # Read log file to get error details
            mcp_log.close()
            with open(mcp_log_path, "r") as f:
                error_log = f.read()
            logger.error(f"MCP server failed to start. Log output:\n{error_log}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        return None


def main():
    """Main entry point for the GUI application."""
    import argparse
    import atexit
    
    parser = argparse.ArgumentParser(description="AgentCPM-MCP GUI Application")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=12000, help="Port to run on")
    parser.add_argument("--mcp-port", type=int, default=8000, help="Port for MCP server")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--no-mcp", action="store_true", help="Don't auto-start MCP server")
    
    args = parser.parse_args()
    
    mcp_process = None
    
    # Auto-start MCP server
    if not args.no_mcp:
        logger.info("Attempting to start MCP server...")
        mcp_process = start_mcp_server(port=args.mcp_port)
        
        if mcp_process:
            # Register cleanup on exit
            def cleanup_mcp():
                if mcp_process and mcp_process.poll() is None:
                    logger.info("Shutting down MCP server...")
                    mcp_process.terminate()
                    mcp_process.wait(timeout=5)
            
            atexit.register(cleanup_mcp)
    
    logger.info(f"Starting AgentCPM-MCP GUI on {args.host}:{args.port}")
    
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
