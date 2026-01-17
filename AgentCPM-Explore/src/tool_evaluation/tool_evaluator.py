#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP Server Tool Evaluator (AgentCPM-MCP Version)

Used to evaluate tool availability and performance on MCP servers
"""

import os
import sys
import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union

# Add project root directory to Python path
file_path = Path(__file__).resolve()
script_dir = file_path.parent
src_dir = script_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Import MCPManager
from src.mcp_manager import MCPManager
from src.extended_openai_client import get_extended_llm_client

# Configure logging
logger = logging.getLogger("tool_evaluator")

class ToolEvaluator:
    """
    MCP Server Tool Evaluator (AgentCPM-MCP Version)

    Used to evaluate tool availability and performance on MCP servers
    """
    
    def __init__(
        self,
        manager_url: str = "http://localhost:8000/mcpapi",
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        results_dir: str = None,
        log_level: str = "INFO",
    ):
        """
        Initialize evaluator

        Args:
            manager_url: MCPManager API URL
            provider: LLM provider
            model: LLM model name
            api_key: API key
            base_url: API base URL
            results_dir: Results save directory
            log_level: Log level
        """
        # Set log level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
        logger.setLevel(numeric_level)
        
        # Initialize MCPManager client
        self.manager = MCPManager(manager_url=manager_url)
        
        # Initialize LLM client
        self.llm_client = get_extended_llm_client(
            provider=provider, 
            model=model, 
            api_key=api_key, 
            base_url=base_url
        )
        
        # Save settings
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        
        # Initialize tool list
        self.all_tools = []
        self.tools_by_server = {}
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = project_root / "evaluation_results" / timestamp
        
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {self.results_dir}")
    
    async def initialize(self) -> bool:
        """
        Initialize evaluator, connect to MCPManager and load tool list

        Returns:
            bool: Whether initialization was successful
        """
        try:
            logger.info("Initializing MCPManager client...")
            if not await self.manager.initialize():
                logger.error("MCPManager client initialization failed")
                return False
            
            # Get all tools
            self.all_tools = self.manager.openai_tools
            logger.info(f"Retrieved {len(self.all_tools)} tools from MCPManager")
            
            # Organize tools by server
            servers = await self.manager.list_servers()
            for server in servers:
                server_tools = await self.manager.get_server_tools(server)
                self.tools_by_server[server] = server_tools
                logger.info(f"Server '{server}' has {len(server_tools)} tools")
                
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def close(self):
        """Close evaluator resources"""
        if hasattr(self, 'manager') and self.manager:
            await self.manager.close()
            logger.info("MCPManager client closed")
    
    def find_tool_by_name(self, tool_name: str, server_name: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Find tool by name

        Args:
            tool_name: Tool name
            server_name: Server name (optional)

        Returns:
            Tuple[Optional[str], Optional[Dict]]: Tuple containing server name and tool definition, returns (None, None) if not found
        """
        # If server is specified, search only in that server
        if server_name and server_name in self.tools_by_server:
            for tool in self.tools_by_server[server_name]:
                if "function" in tool and tool["function"].get("name") == tool_name:
                    return server_name, tool
        else:
            # Search in all servers
            for server, tools in self.tools_by_server.items():
                for tool in tools:
                    if "function" in tool and tool["function"].get("name") == tool_name:
                        return server, tool
        
        return None, None
    
    def generate_test_prompt(self, tool_schema: Dict) -> str:
        """
        Generate test prompt for tool

        Args:
            tool_schema: Tool schema

        Returns:
            str: Test prompt
        """
        try:
            # Extract tool information
            tool_name = tool_schema.get("function", {}).get("name", "unknown_tool")
            description = tool_schema.get("function", {}).get("description", "No description available")
            parameters = tool_schema.get("function", {}).get("parameters", {})
            
            # Build prompt
            prompt = f"I need to test a tool named '{tool_name}'.\n\n"
            prompt += f"Tool description: {description}\n\n"
            
            # Add parameter information
            if parameters:
                prompt += "This tool requires the following parameters:\n"
                
                # Get required parameters
                required_params = parameters.get("required", [])
                
                # Iterate through parameter properties
                properties = parameters.get("properties", {})
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "No description")
                    is_required = param_name in required_params
                    
                    prompt += f"- {param_name} ({param_type}): {param_desc}"
                    if is_required:
                        prompt += " (required)"
                    prompt += "\n"
            
            # Add test request
            prompt += "\nPlease generate a valid test case to call this tool. Provide reasonable parameter values and explain why these values are appropriate for testing this tool."
            
            return prompt
        except Exception as e:
            logger.error(f"Error generating test prompt: {str(e)}")
            return f"Please test the tool named '{tool_schema.get('function', {}).get('name', 'unknown_tool')}' and provide appropriate parameters."
    
    async def generate_test_parameters(self, tool_schema: Dict) -> Dict[str, Any]:
        """
        Use LLM to generate test parameters for tool

        Args:
            tool_schema: Tool schema

        Returns:
            Dict[str, Any]: Test parameters
        """
        try:
            # Get tool name and description
            tool_name = tool_schema.get("function", {}).get("name", "unknown_tool")
            tool_description = tool_schema.get("function", {}).get("description", "")
            
            # Build system prompt to let GPT generate appropriate tool calls
            system_prompt = f"""You are an AI assistant specialized in generating tool calls. Your task is to generate an appropriate tool call for the following tool.

Do not describe or explain tool calls in the response content, but generate tool calls directly through the tool_calls function. Your content should be empty (null) or briefly explain what you will execute.

Tool information:
Name: {tool_name}
Description: {tool_description}
"""
            
            if "parameters" in tool_schema.get("function", {}) and tool_schema["function"]["parameters"]:
                # Record parameter information for debugging
                logger.debug(f"Tool {tool_name} parameter schema: {json.dumps(tool_schema['function']['parameters'], ensure_ascii=False)[:100]}...")
                
                # Extract required parameters
                required_params = []
                if isinstance(tool_schema["function"]["parameters"], dict):
                    required_params = tool_schema["function"]["parameters"].get("required", [])
                
                # Add parameter schema to prompt
                system_prompt += f"\nParameter schema: {json.dumps(tool_schema['function']['parameters'], ensure_ascii=False, indent=2)}"
                
                # If there are required parameters, emphasize them
                if required_params:
                    system_prompt += f"\n\nRequired parameters: {', '.join(required_params)}"
            else:
                logger.warning(f"Tool {tool_name} has no parameter schema definition")
            
            system_prompt += """

Important notes:
1. Directly use the tool_calls function to generate tool calls, do not describe tool calls in content
2. Provide all required parameters, ensure parameter values meet schema requirements
3. Do not use placeholders, use actual, reasonable values
4. If the tool requires code parameters, provide valid, short code examples
5. Your response should have one tool call, and content field can be null or brief explanation"""

            # Generate user prompt for tool calls
            user_prompt = f"Please imagine a simple scenario corresponding to this tool and generate a valid tool call for tool '{tool_name}'. Remember to use the tool_calls function instead of describing in content."
            
            # Create conversation history
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM
            logger.info(f"Generating test parameters for tool '{tool_name}'...")
            response = self.llm_client.create_completion(
                messages=messages,
                tools=[tool_schema]
            )
            
            # Extract tool calls from response
            tool_calls = response.get("tool_calls", [])
            
            if not tool_calls:
                logger.warning(f"LLM did not generate tool calls, will use default parameters")
                return self._generate_default_parameters(tool_schema)
            
            # Extract the first tool call
            tool_call = tool_calls[0]
            
            # Extract tool name and parameters
            if isinstance(tool_call, dict) and "function" in tool_call:
                tool_name_called = tool_call["function"].get("name", "")
                raw_arguments = tool_call["function"].get("arguments", "{}")
                
                # Validate tool name
                if tool_name_called != tool_name:
                    logger.warning(f"Tool name mismatch: expected {tool_name}, got {tool_name_called}, will use default parameters")
                    return self._generate_default_parameters(tool_schema)
                
                # Parse parameters
                try:
                    parameters = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    return parameters
                except json.JSONDecodeError as e:
                    logger.warning(f"Parameter parsing error: {str(e)}, raw arguments: {raw_arguments}")
                    return self._generate_default_parameters(tool_schema)
            else:
                logger.warning("Unable to parse LLM tool calls, will use default parameters")
                return self._generate_default_parameters(tool_schema)
            
        except Exception as e:
            logger.error(f"Error generating test parameters: {str(e)}")
            return self._generate_default_parameters(tool_schema)
    
    def _generate_default_parameters(self, tool_schema: Dict) -> Dict[str, Any]:
        """
        Generate default parameters for tool

        Args:
            tool_schema: Tool schema

        Returns:
            Dict[str, Any]: Default parameters
        """
        parameters = {}
        
        # Get parameter definitions
        param_defs = tool_schema.get("function", {}).get("parameters", {}).get("properties", {})
        
        # Generate default values for each parameter
        for param_name, param_info in param_defs.items():
            param_type = param_info.get("type", "string")
            
            # Generate default values based on type
            if param_type == "string":
                parameters[param_name] = f"test_{param_name}"
            elif param_type == "number" or param_type == "integer":
                parameters[param_name] = 1
            elif param_type == "boolean":
                parameters[param_name] = True
            elif param_type == "array":
                parameters[param_name] = []
            elif param_type == "object":
                parameters[param_name] = {}
        
        return parameters
    
    async def evaluate_tool(self, server_name: str, tool_name: str, iterations: int = 3) -> Dict[str, Any]:
        """
        Evaluate single tool

        Args:
            server_name: Server name
            tool_name: Tool name
            iterations: Number of evaluation iterations

        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Find tool
        server, tool_schema = self.find_tool_by_name(tool_name, server_name)
        
        if not server or not tool_schema:
            logger.error(f"Tool '{tool_name}' not found on server '{server_name}'")
            return {
                "tool_name": tool_name,
                "server_name": server_name,
                "success": False,
                "success_rate": 0.0,
                "error": f"Tool '{tool_name}' not found on server '{server_name}'"
            }
        
        logger.info(f"Starting evaluation of tool: '{tool_name}' (server: '{server}')")
        
        # Create results directory
        tool_dir = self.results_dir / server / f"{tool_name}_logs"
        os.makedirs(tool_dir, exist_ok=True)
        
        # Generate test parameters
        parameters = await self.generate_test_parameters(tool_schema)
        
        # Save generated parameters
        params_file = tool_dir / "test_parameters.json"
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(parameters, f, ensure_ascii=False, indent=2)
        
        # Evaluation results
        results = []
        successful_calls = 0
        total_latency = 0
        
        # Perform multiple iteration tests
        for i in range(iterations):
            logger.info(f"Tool '{tool_name}' evaluation iteration {i+1}/{iterations}")
            
            start_time = time.time()
            call_result = {
                "iteration": i + 1,
                "success": False,
                "parameters": parameters,
                "latency": 0
            }
            
            try:
                # Call tool
                result = await self.manager.call_tool(tool_name, parameters, server)
                
                # Record latency
                latency = time.time() - start_time
                call_result["latency"] = latency
                
                # Check results
                if isinstance(result, dict) and result.get("status") == "error":
                    error_msg = result.get("content", {}).get("error", "Unknown error")
                    error_detail = result.get("content", {}).get("detail", "")
                    logger.warning(f"Tool '{tool_name}' call failed: {error_msg}")
                    
                    call_result["error"] = error_msg
                    if error_detail:
                        call_result["error_detail"] = error_detail
                    
                    # Save failed logs
                    log_file = tool_dir / f"iteration_{i+1}_failed.json"
                else:
                    # Call successful
                    successful_calls += 1
                    total_latency += latency
                    
                    logger.info(f"Tool '{tool_name}' call successful, latency: {latency:.2f}s")
                    
                    call_result["success"] = True
                    call_result["result"] = result
                    
                    # Save successful logs
                    log_file = tool_dir / f"iteration_{i+1}_success.json"
                
                # Save call results
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(call_result, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.error(f"Tool '{tool_name}' evaluation error: {str(e)}")
                
                latency = time.time() - start_time
                call_result["latency"] = latency
                call_result["error"] = str(e)
                
                # Save error logs
                log_file = tool_dir / f"iteration_{i+1}_error.json"
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(call_result, f, ensure_ascii=False, indent=2)
            
            # Add results
            results.append(call_result)
        
        # Calculate success rate and average latency
        success_rate = successful_calls / iterations if iterations > 0 else 0
        avg_latency = total_latency / successful_calls if successful_calls > 0 else 0
        
        # Build evaluation results
        evaluation_result = {
            "tool_name": tool_name,
            "server_name": server,
            "success": successful_calls > 0,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "total_latency": total_latency,
            "iterations": iterations,
            "successful_calls": successful_calls,
            "failed_calls": iterations - successful_calls,
            "parameters": parameters,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save evaluation results
        result_file = self.results_dir / server / f"{tool_name}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        return evaluation_result
    
    async def evaluate_server(self, server_name: str, iterations: int = 3) -> Dict[str, Any]:
        """
        Evaluate all tools on single server

        Args:
            server_name: Server name
            iterations: Number of evaluation iterations

        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info(f"Starting evaluation of server: '{server_name}'")
        
        # Get tools on server
        tools = []
        for server, tool_list in self.tools_by_server.items():
            if server == server_name:
                tools = tool_list
                break
        
        if not tools:
            logger.error(f"Server '{server_name}' has no available tools")
            return {
                "server_name": server_name,
                "success": False,
                "error": f"Server '{server_name}' has no available tools"
            }
        
        # Create server results directory
        server_dir = self.results_dir / server_name
        os.makedirs(server_dir, exist_ok=True)
        
        # Evaluation results
        tool_results = {}
        total_calls = 0
        successful_calls = 0
        total_latency = 0
        
        # Evaluate each tool
        for tool in tools:
            # Get tool name
            tool_name = None
            if isinstance(tool, dict):
                if "function" in tool:
                    # OpenAI format tools
                    tool_name = tool.get("function", {}).get("name", "")
                else:
                    # Plain dictionary format
                    tool_name = tool.get("name", "")
            elif hasattr(tool, "name"):
                # Object format
                tool_name = tool.name
            
            if not tool_name:
                logger.warning(f"Skipping unknown tool: {tool}")
                continue
            
            # Evaluate tool
            result = await self.evaluate_tool(server_name, tool_name, iterations)
            
            # Add to results
            tool_results[tool_name] = result
            
            # Update statistics
            total_calls += result.get("iterations", 0)
            successful_calls += result.get("successful_calls", 0)
            total_latency += result.get("total_latency", 0)
        
        # Calculate success rate and average latency
        success_rate = successful_calls / total_calls if total_calls > 0 else 0
        avg_latency = total_latency / successful_calls if successful_calls > 0 else 0
        
        # Build evaluation results
        evaluation_result = {
            "server_name": server_name,
            "success": successful_calls > 0,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "total_latency": total_latency,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "total_tools": len(tools),
            "tool_results": tool_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"server_{server_name}_{timestamp}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Server '{server_name}' evaluation completed, results saved to: {result_file}")
        
        return evaluation_result
    
    async def evaluate_all_servers(self, iterations: int = 3) -> Dict[str, Any]:
        """
        Evaluate all servers

        Args:
            iterations: Number of evaluation iterations

        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info("Starting evaluation of all servers")
        
        # Evaluation results
        server_results = {}
        total_tools = 0
        total_calls = 0
        successful_calls = 0
        total_latency = 0
        
        # Evaluate each server
        for server_name in self.tools_by_server.keys():
            # Evaluate server
            result = await self.evaluate_server(server_name, iterations)
            
            # Add to results
            server_results[server_name] = result
            
            # Update statistics
            total_tools += result.get("total_tools", 0)
            total_calls += result.get("total_calls", 0)
            successful_calls += result.get("successful_calls", 0)
            total_latency += result.get("total_latency", 0)
        
        # Calculate success rate and average latency
        success_rate = successful_calls / total_calls if total_calls > 0 else 0
        avg_latency = total_latency / successful_calls if successful_calls > 0 else 0
        
        # Build evaluation results
        evaluation_result = {
            "success": successful_calls > 0,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "total_latency": total_latency,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "total_servers": len(self.tools_by_server),
            "total_tools": total_tools,
            "server_results": server_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"all_servers_{timestamp}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        # Generate tool reports
        self.generate_tool_reports(evaluation_result)
        
        logger.info(f"All servers evaluation completed, results saved to: {result_file}")
        
        return evaluation_result
    
    def generate_tool_reports(self, evaluation_result: Dict[str, Any]) -> None:
        """
        Generate tool reports

        Args:
            evaluation_result: Evaluation results
        """
        logger.info("Generating tool reports")
        
        # Create reports directory
        reports_dir = self.results_dir / "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate overall report
        overall_report = f"""# MCP Tool Evaluation Report

## Overall Statistics

- Evaluation time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Number of servers: {evaluation_result.get("total_servers", 0)}
- Total tools: {evaluation_result.get("total_tools", 0)}
- Total calls: {evaluation_result.get("total_calls", 0)}
- Successful calls: {evaluation_result.get("successful_calls", 0)}
- Failed calls: {evaluation_result.get("failed_calls", 0)}
- Success rate: {evaluation_result.get("success_rate", 0) * 100:.2f}%
- Average latency: {evaluation_result.get("avg_latency", 0):.2f}s

## Server Statistics

| Server | Tool Count | Total Calls | Successful Calls | Success Rate | Average Latency(s) |
|--------|------------|-------------|------------------|--------------|-------------------|
"""
        
        # Add server statistics
        server_results = evaluation_result.get("server_results", {})
        for server_name, server_result in server_results.items():
            total_tools = server_result.get("total_tools", 0)
            total_calls = server_result.get("total_calls", 0)
            successful_calls = server_result.get("successful_calls", 0)
            success_rate = server_result.get("success_rate", 0) * 100
            avg_latency = server_result.get("avg_latency", 0)
            
            overall_report += f"| {server_name} | {total_tools} | {total_calls} | {successful_calls} | {success_rate:.2f}% | {avg_latency:.2f} |\n"
        
        # Save overall report
        with open(reports_dir / "overall_report.md", "w", encoding="utf-8") as f:
            f.write(overall_report)
        
        # Generate detailed report for each server
        for server_name, server_result in server_results.items():
            server_report = f"""# Server '{server_name}' Tool Evaluation Report

## Overall Statistics

- Evaluation time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Total tools: {server_result.get("total_tools", 0)}
- Total calls: {server_result.get("total_calls", 0)}
- Successful calls: {server_result.get("successful_calls", 0)}
- Failed calls: {server_result.get("failed_calls", 0)}
- Success rate: {server_result.get("success_rate", 0) * 100:.2f}%
- Average latency: {server_result.get("avg_latency", 0):.2f}s

## Tool Statistics

| Tool Name | Total Calls | Successful Calls | Success Rate | Average Latency(s) |
|-----------|-------------|------------------|--------------|-------------------|
"""
            
            # Add tool statistics
            tool_results = server_result.get("tool_results", {})
            for tool_name, tool_result in tool_results.items():
                iterations = tool_result.get("iterations", 0)
                successful_calls = tool_result.get("successful_calls", 0)
                success_rate = tool_result.get("success_rate", 0) * 100
                avg_latency = tool_result.get("avg_latency", 0)
                
                server_report += f"| {tool_name} | {iterations} | {successful_calls} | {success_rate:.2f}% | {avg_latency:.2f} |\n"
            
            # Save server report
            with open(reports_dir / f"server_{server_name}_report.md", "w", encoding="utf-8") as f:
                f.write(server_report)
        
        # Generate tool mapping file
        server_tools_mapping = {}
        for server_name, tools in self.tools_by_server.items():
            tool_names = []
            for tool in tools:
                tool_name = None
                if isinstance(tool, dict):
                    if "function" in tool:
                        # OpenAI format tools
                        tool_name = tool.get("function", {}).get("name", "")
                    else:
                        # Plain dictionary format
                        tool_name = tool.get("name", "")
                elif hasattr(tool, "name"):
                    # Object format
                    tool_name = tool.name
                
                if tool_name:
                    tool_names.append(tool_name)
            
            server_tools_mapping[server_name] = tool_names
        
        # Save tool mapping
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mapping_file = self.results_dir / f"server_tools_mapping_{timestamp}.json"
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(server_tools_mapping, f, ensure_ascii=False, indent=2)
        
        # Save fixed version to src/server_eval directory
        fixed_mapping_file = Path("src/server_eval/server_to_tools.json")
        with open(fixed_mapping_file, "w", encoding="utf-8") as f:
            json.dump(server_tools_mapping, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Tool reports generated in directory: {reports_dir}")
        logger.info(f"Overall report saved to: {reports_dir / 'overall_report.md'}")
        logger.info(f"Server tool mapping saved to file: {mapping_file}")
        logger.info(f"Fixed version saved to: {fixed_mapping_file}") 