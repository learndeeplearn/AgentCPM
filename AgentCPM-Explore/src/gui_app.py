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
    MODEL_NAME, BASE_URL, get_extended_llm_client, LLMClientManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("gui_app")


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
        api_key: str,
        provider: str = "openai"
    ) -> str:
        """Initialize or reinitialize the LLM client."""
        try:
            client_name = f"{provider}_{model}"
            self.current_client = self.client_manager.create_client(
                client_name=client_name,
                provider=provider,
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
        return "", [], "Conversation reset."

    def format_step(self, step_type: str, content: str, step_num: int = None) -> str:
        """Format a step for display."""
        if step_num is None:
            step_num = self.step_counter
            self.step_counter += 1
        
        icons = {
            "input": "📝",
            "thinking": "🧠",
            "tool_call": "🔧",
            "tool_result": "📊",
            "response": "💬",
            "error": "❌",
            "info": "ℹ️"
        }
        icon = icons.get(step_type, "•")
        
        return f"\n### {icon} Step {step_num}: {step_type.upper()}\n\n{content}\n"

    def process_prompt(
        self,
        prompt: str,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        manager_url: str,
        use_tools: bool
    ) -> Generator[tuple, None, None]:
        """
        Process a user prompt and yield step-by-step updates.
        
        Yields tuples of (steps_display, thinking_display, response_display, tool_calls_display, status)
        """
        self.step_counter = 1
        steps_log = []
        thinking_content = ""
        response_content = ""
        tool_calls_log = []
        
        def add_step(step_type: str, content: str):
            formatted = self.format_step(step_type, content, self.step_counter)
            steps_log.append(formatted)
            self.step_counter += 1
            return "\n".join(steps_log)
        
        # Step 1: Input received
        steps_display = add_step("input", f"**User Prompt:**\n```\n{prompt}\n```")
        yield (steps_display, thinking_content, response_content, 
               "\n".join(tool_calls_log), "🔄 Processing input...")
        
        # Initialize client if needed
        if not self.current_client:
            init_result = self.initialize_client(model, base_url, api_key)
            if "❌" in init_result:
                steps_display = add_step("error", init_result)
                yield (steps_display, thinking_content, response_content,
                       "\n".join(tool_calls_log), init_result)
                return
            steps_display = add_step("info", init_result)
            yield (steps_display, thinking_content, response_content,
                   "\n".join(tool_calls_log), "🔄 Client initialized...")
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": prompt})
        
        steps_display = add_step("info", f"**Messages prepared:** {len(messages)} messages in context")
        yield (steps_display, thinking_content, response_content,
               "\n".join(tool_calls_log), "🔄 Calling LLM...")
        
        # Get tools if enabled
        tools = None
        if use_tools and self.mcp_handler:
            tools = self.mcp_handler.openai_tools
            steps_display = add_step("info", f"**Tools available:** {len(tools)} tools loaded")
            yield (steps_display, thinking_content, response_content,
                   "\n".join(tool_calls_log), "🔄 Tools loaded...")
        
        # Call LLM
        try:
            result = self.current_client.create_completion(
                messages=messages,
                tools=tools,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens > 0 else None
            )
            
            # Process thinking/reasoning
            if result.get("thought"):
                thinking_content = result["thought"]
                steps_display = add_step("thinking", f"```\n{thinking_content}\n```")
                yield (steps_display, thinking_content, response_content,
                       "\n".join(tool_calls_log), "🧠 Reasoning complete...")
            
            # Process tool calls
            if result.get("tool_calls"):
                for i, tool_call in enumerate(result["tool_calls"]):
                    func_name = tool_call.get("function", {}).get("name", "unknown")
                    func_args = tool_call.get("function", {}).get("arguments", "{}")
                    
                    tool_info = f"**Tool:** `{func_name}`\n**Arguments:**\n```json\n{func_args}\n```"
                    tool_calls_log.append(f"### Tool Call {i+1}\n{tool_info}")
                    steps_display = add_step("tool_call", tool_info)
                    yield (steps_display, thinking_content, response_content,
                           "\n".join(tool_calls_log), f"🔧 Tool call: {func_name}")
                    
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
                            
                            tool_calls_log.append(f"**Result:**\n```json\n{result_str}\n```")
                            steps_display = add_step("tool_result", f"```json\n{result_str}\n```")
                            yield (steps_display, thinking_content, response_content,
                                   "\n".join(tool_calls_log), f"📊 Tool result received")
                        except Exception as e:
                            error_msg = f"Tool execution error: {str(e)}"
                            tool_calls_log.append(f"**Error:** {error_msg}")
                            steps_display = add_step("error", error_msg)
                            yield (steps_display, thinking_content, response_content,
                                   "\n".join(tool_calls_log), f"❌ {error_msg}")
            
            # Process final response
            response_content = result.get("response", "")
            if response_content:
                steps_display = add_step("response", response_content)
                yield (steps_display, thinking_content, response_content,
                       "\n".join(tool_calls_log), "✅ Response complete")
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({
                "role": "assistant", 
                "content": response_content,
                "tool_calls": result.get("tool_calls")
            })
            
            # Final yield with complete status
            status = f"✅ Complete | Tokens: {result.get('usage', {})}"
            yield (steps_display, thinking_content, response_content,
                   "\n".join(tool_calls_log), status)
            
        except Exception as e:
            error_msg = f"Error during LLM call: {str(e)}"
            logger.error(error_msg, exc_info=True)
            steps_display = add_step("error", f"```\n{error_msg}\n```")
            yield (steps_display, thinking_content, response_content,
                   "\n".join(tool_calls_log), f"❌ {error_msg}")


CUSTOM_CSS = """
.step-container { 
    max-height: 500px; 
    overflow-y: auto; 
    border: 1px solid #ddd; 
    border-radius: 8px; 
    padding: 10px;
}
.thinking-box {
    background-color: #f0f7ff;
    border-left: 4px solid #0066cc;
    padding: 10px;
    margin: 5px 0;
}
.response-box {
    background-color: #f0fff0;
    border-left: 4px solid #00cc66;
    padding: 10px;
    margin: 5px 0;
}
.tool-box {
    background-color: #fff7f0;
    border-left: 4px solid #cc6600;
    padding: 10px;
    margin: 5px 0;
}
"""


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
                        value=MODEL_NAME,
                        placeholder="e.g., deepseek-r1-1.5b"
                    )
                    base_url_input = gr.Textbox(
                        label="Base URL",
                        value=BASE_URL,
                        placeholder="e.g., http://localhost:11434/v1"
                    )
                    api_key_input = gr.Textbox(
                        label="API Key (optional for Ollama)",
                        value="",
                        type="password",
                        placeholder="Leave empty for Ollama"
                    )
                    provider_input = gr.Dropdown(
                        label="Provider",
                        choices=["openai", "ollama"],
                        value="openai"
                    )
                
                with gr.Accordion("Generation Settings", open=True):
                    temperature_slider = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1
                    )
                    max_tokens_slider = gr.Slider(
                        label="Max Tokens (0 = no limit)",
                        minimum=0,
                        maximum=8192,
                        value=2048,
                        step=256
                    )
                
                with gr.Accordion("MCP Tools Settings", open=False):
                    manager_url_input = gr.Textbox(
                        label="MCP Manager URL",
                        value="http://localhost:8000/mcpapi",
                        placeholder="MCP Manager API URL"
                    )
                    use_tools_checkbox = gr.Checkbox(
                        label="Enable MCP Tools",
                        value=False
                    )
                    init_mcp_btn = gr.Button("🔌 Initialize MCP", variant="secondary")
                    mcp_status = gr.Textbox(
                        label="MCP Status",
                        value="Not initialized",
                        interactive=False
                    )
                
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
                
                # Input area
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="Your Prompt",
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
                
                # Output tabs
                with gr.Tabs():
                    with gr.TabItem("📋 Steps Log"):
                        steps_output = gr.Markdown(
                            value="*Steps will appear here as processing progresses...*",
                            elem_classes=["step-container"]
                        )
                    
                    with gr.TabItem("🧠 Thinking/Reasoning"):
                        thinking_output = gr.Markdown(
                            value="*Reasoning content will appear here...*",
                            elem_classes=["thinking-box"]
                        )
                    
                    with gr.TabItem("💬 Final Response"):
                        response_output = gr.Markdown(
                            value="*Final response will appear here...*",
                            elem_classes=["response-box"]
                        )
                    
                    with gr.TabItem("🔧 Tool Calls"):
                        tools_output = gr.Markdown(
                            value="*Tool calls and results will appear here...*",
                            elem_classes=["tool-box"]
                        )
        
        # Examples section
        gr.Markdown("### 📝 Example Prompts")
        gr.Examples(
            examples=[
                ["Explain the concept of recursion with a simple example."],
                ["What are the key differences between Python lists and tuples?"],
                ["Write a haiku about artificial intelligence."],
                ["Solve this step by step: If a train travels 120 km in 2 hours, what is its average speed?"],
                ["What is the weather like today? (requires MCP tools)"],
            ],
            inputs=prompt_input
        )
        
        # Event handlers
        def process_wrapper(prompt, model, base_url, api_key, temp, max_tok, sys_prompt, mgr_url, use_tools):
            """Wrapper to handle the generator output."""
            if not prompt.strip():
                yield ("", "", "", "", "⚠️ Please enter a prompt")
                return
            
            for result in agent.process_prompt(
                prompt, model, base_url, api_key, temp, max_tok, sys_prompt, mgr_url, use_tools
            ):
                yield result
        
        async def init_mcp_wrapper(manager_url):
            """Wrapper for MCP initialization."""
            result = await agent.initialize_mcp(manager_url)
            return result
        
        def clear_wrapper():
            """Clear conversation and outputs."""
            agent.reset_conversation()
            return (
                "",  # prompt
                "*Steps will appear here as processing progresses...*",
                "*Reasoning content will appear here...*",
                "*Final response will appear here...*",
                "*Tool calls and results will appear here...*",
                "Ready"
            )
        
        # Connect events
        submit_btn.click(
            fn=process_wrapper,
            inputs=[
                prompt_input, model_input, base_url_input, api_key_input,
                temperature_slider, max_tokens_slider, system_prompt_input,
                manager_url_input, use_tools_checkbox
            ],
            outputs=[steps_output, thinking_output, response_output, tools_output, status_display]
        )
        
        prompt_input.submit(
            fn=process_wrapper,
            inputs=[
                prompt_input, model_input, base_url_input, api_key_input,
                temperature_slider, max_tokens_slider, system_prompt_input,
                manager_url_input, use_tools_checkbox
            ],
            outputs=[steps_output, thinking_output, response_output, tools_output, status_display]
        )
        
        init_mcp_btn.click(
            fn=init_mcp_wrapper,
            inputs=[manager_url_input],
            outputs=[mcp_status]
        )
        
        clear_btn.click(
            fn=clear_wrapper,
            inputs=[],
            outputs=[prompt_input, steps_output, thinking_output, response_output, tools_output, status_display]
        )
    
    return demo


def main():
    """Main entry point for the GUI application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AgentCPM-MCP GUI Application")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=12000, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    
    args = parser.parse_args()
    
    logger.info(f"Starting AgentCPM-MCP GUI on {args.host}:{args.port}")
    
    demo = create_gui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CUSTOM_CSS
    )


if __name__ == "__main__":
    main()
