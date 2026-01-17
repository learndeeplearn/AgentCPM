#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP Server Tool Evaluation Script (AgentCPM-MCP Version)

Supports direct evaluation using MCPManager API
"""

import os
import sys
import asyncio
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Add project root directory to Python path
file_path = Path(__file__).resolve()
script_dir = file_path.parent  # src/tool_evaluation/
src_dir = script_dir.parent    # src/
project_root = src_dir.parent  # AgentCPM-MCP/
sys.path.insert(0, str(project_root))

# Import ToolEvaluator
from src.tool_evaluation.tool_evaluator import ToolEvaluator

logger = logging.getLogger("mcp_evaluation")

async def save_server_tools_mapping(evaluator: ToolEvaluator, results: dict):
    """
    Save server-to-tools mapping based on evaluation results

    Args:
        evaluator: ToolEvaluator instance
        results: Evaluation results (usually output from evaluate_all_servers)
    """
    try:
        server_to_tools_map = {}
        
        # evaluator.tools_by_server is already in format {server_name: [tool_schema1, ...]}
        for server_name, server_tools in evaluator.tools_by_server.items():
            # Get all tool names in this server
            tool_names = [tool.get("function", {}).get("name", "unknown") for tool in server_tools]
            
            # Add to mapping
            server_to_tools_map[server_name] = {
                "server_name": server_name,
                "tools": tool_names
            }
            
            # Filter out successful tools from results
            if "results" in results and server_name in results["results"]:
                server_result = results["results"][server_name]
                if "results" in server_result:
                    # Get evaluation results for tools
                    working_tools = []
                    for tool_result in server_result["results"]:
                        if tool_result.get("success_rate", 0) > 0:
                            tool_name = tool_result.get("tool_name", "unknown")
                            working_tools.append(tool_name)
                            
                    # Update tool list, keep only valid tools        
                    server_to_tools_map[server_name]["working_tools"] = working_tools
                    server_to_tools_map[server_name]["working_ratio"] = len(working_tools) / len(tool_names) if tool_names else 0
        
        # Save mapping
        output_dir = evaluator.results_dir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"server_tools_mapping_{timestamp}.json")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(server_to_tools_map, f, ensure_ascii=False, indent=2)
            
        # Also save a fixed-name version for reference by other modules
        fixed_output_file = os.path.join(src_dir, "server_eval", "server_to_tools.json")
        os.makedirs(os.path.dirname(fixed_output_file), exist_ok=True)
        
        with open(fixed_output_file, "w", encoding="utf-8") as f:
            json.dump(server_to_tools_map, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Server tools mapping saved to file: {output_file}")
        logger.info(f"Fixed version saved to: {fixed_output_file}")
    except Exception as e:
        logger.error(f"Error saving server tools mapping: {e}")
        import traceback
        logger.error(traceback.format_exc())

# Same functionality as the function above, kept for compatibility purposes
async def save_server_tools_mapping_mcp_mgr(evaluator: ToolEvaluator, results: dict):
    """
    Save server-to-tools mapping based on evaluation results (MCPManager version)

    Args:
        evaluator: ToolEvaluator instance
        results: Evaluation results (usually output from evaluate_all_servers)
    """
    return await save_server_tools_mapping(evaluator, results)

# Main function
async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP Server Tool Evaluation Script (AgentCPM-MCP Version)")
    parser.add_argument("--manager-url", default="http://localhost:8000/mcpapi", help="MCPManager API URL")
    parser.add_argument("--server", default="all", help="Server to evaluate ('all' means all servers)")
    parser.add_argument("--tool", help="Specify tool to evaluate")
    parser.add_argument("--iterations", type=int, default=3, help="Number of evaluation iterations per tool")
    parser.add_argument("--save-mapping", action="store_true", help="Generate and save server-to-tools mapping file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--data", help="GAIA data file path (for compatibility with old commands)")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum number of samples (for compatibility with old commands)")
    
    # LLM related options
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"], help="LLM provider")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--base-url", help="API base URL")
    
    args = parser.parse_args()
    
    # Set log level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create evaluator
    evaluator = ToolEvaluator(
        manager_url=args.manager_url,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        results_dir=args.output_dir,
        log_level=args.log_level
    )
    
    evaluation_results = None
    try:
        logger.info("Initializing evaluator...")
        if not await evaluator.initialize():
            logger.error("Evaluator initialization failed.")
            return 1
        
        # Handle GAIA data file parameter (for compatibility with old commands)
        if args.data:
            logger.info(f"GAIA data file parameter detected, please use gaia_test.py script for GAIA testing")
            logger.info(f"Example: python -m src.gaia_test.gaia_api_test --gaia-file {args.data} --max-samples {args.max_samples}")
            return 1
        
        if args.tool:
            # Evaluate specific tool on specified server or all servers
            if args.server and args.server.lower() != "all":
                logger.info(f"Evaluating tool '{args.tool}' on server '{args.server}'...")
                server, tool_def = evaluator.find_tool_by_name(args.tool, args.server)
                
                if server and tool_def:
                    evaluation_results = await evaluator.evaluate_tool(
                        server, args.tool, args.iterations
                    )
                    
                    # Save individual tool evaluation results
                    tool_output_dir = evaluator.results_dir
                    os.makedirs(tool_output_dir, exist_ok=True)
                    output_file = os.path.join(tool_output_dir, f"tool_{args.tool}_{server}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
                    logger.info(f"Evaluation results saved to: {output_file}")
                    
                    # Create simplified results for generating reports and mappings
                    mini_result = {
                        "results": {
                            server: {
                                "results": [evaluation_results]
                            }
                        }
                    }
                    
                    # Save mapping if requested
                    if args.save_mapping:
                        await save_server_tools_mapping(evaluator, mini_result)
                else:
                    logger.error(f"Tool '{args.tool}' not found on server '{args.server if args.server != 'all' else 'any server'}'.")
                    return 1
        
        elif args.server.lower() == "all":
            logger.info("Starting evaluation of all servers obtained from MCPManager...")
            evaluation_results = await evaluator.evaluate_all_servers(args.iterations)
            
            # Save mapping if requested
            if args.save_mapping:
                await save_server_tools_mapping(evaluator, evaluation_results)
            
        else:
            # Evaluate single server
            logger.info(f"Starting evaluation of server '{args.server}'...")
            evaluation_results = await evaluator.evaluate_server(
                args.server, args.iterations
            )
            
            # Save individual server evaluation results
            server_output_dir = evaluator.results_dir
            os.makedirs(server_output_dir, exist_ok=True)
            output_file = os.path.join(server_output_dir, f"server_{args.server}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Evaluation results saved to: {output_file}")
            
            # Create simplified results for generating reports and mappings
            mini_result = {
                "results": {
                    args.server: evaluation_results
                }
            }
            
            # Save mapping if requested
            if args.save_mapping:
                await save_server_tools_mapping(evaluator, mini_result)
                
            # Generate tool reports
            evaluator.generate_tool_reports(mini_result)
        
        logger.info(f"Detailed results saved to: {evaluator.results_dir}")
            
        return 0  # Operation successful
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
    except Exception as e:
        logger.error(f"Critical error occurred during evaluation: {str(e)}", exc_info=True)
        return 1 # Ensure error code is returned in other exception cases
    finally:
        logger.info("Closing evaluator resources...")
        await evaluator.close()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 