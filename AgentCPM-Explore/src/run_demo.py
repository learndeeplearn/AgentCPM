#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AgentCPM-MCP Demo Script

This script demonstrates how to use AgentCPM-MCP for conversation and tool calling.
"""

import os
import sys
import asyncio
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Ensure AgentCPM-MCP module can be imported
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from task import AgentTask
from client import MCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("demo")

async def demo_conversation(args):
    """
    Run demo conversation
    
    Args:
        args: Command line arguments
    """
    try:
        # Set result save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = project_root / "gaia_results" / timestamp
        os.makedirs(result_dir, exist_ok=True)
        
        # Initialize MCP client
        client = MCPClient(
            model=args.model,
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            is_api=True,  # Use API mode
            server_name=args.server,
            config_file=args.config,
            manager_url=args.manager_url,
            result_dir=result_dir
        )
        
        # Initialize client
        if not await client.initialize():
            logger.error("Client initialization failed")
            return
        
        # Prepare conversation
        conversation = [
            {
                "role": "system",
                "content": "You are a smart assistant that can answer questions and use various tools to complete tasks."
            },
            {
                "role": "user",
                "content": args.prompt
            }
        ]
        
        # Create task
        task = AgentTask(
            task_id="demo_task",
            conversation_history=conversation
        )
        
        # Execute task
        logger.info("Starting task execution...")
        result = await client.solve_agent_task(task)
        
        # Display results
        if result["success"]:
            logger.info("Task execution successful!")
            final_response = result.get("model_response", "")
            if final_response:
                print("\n====== Final Response ======")
                print(final_response)
                print("============================\n")
                
            tool_calls = result.get("tool_calls", [])
            if tool_calls:
                print(f"\nUsed {len(tool_calls)} tool calls:")
                for i, tool_call in enumerate(tool_calls):
                    print(f"{i+1}. {tool_call['function']['name']}")
                    
            print(f"\nConversation history saved at: {result.get('logs_path', 'Unknown')}")
        else:
            logger.error(f"Task execution failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error running demo: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close client
        if 'client' in locals():
            await client.close()
            
def main():
    parser = argparse.ArgumentParser(description="AgentCPM-MCP Demo Script")
    
    # Basic configuration
    parser.add_argument("--prompt", type=str, default="What's the weather like today? Can you also introduce me to some tourist attractions in Beijing?", 
                        help="User prompt")
    parser.add_argument("--server", default="all", help="MCP server name to use")
    parser.add_argument("--config", default="config.toml", help="MCP server configuration file path")
    parser.add_argument("--manager-url", default="http://localhost:8000/mcpapi", 
                        help="MCPManager API URL")
    
    # LLM configuration
    parser.add_argument("--provider", default="openai", help="LLM provider (openai, ollama, etc.)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name to use")
    parser.add_argument("--api-key", help="API Key (uses env var by default)")
    parser.add_argument("--base-url", help="API BASE URL (uses standard URL by default)")
    
    args = parser.parse_args()
    
    # Run demo
    asyncio.run(demo_conversation(args))
    
if __name__ == "__main__":
    main() 
