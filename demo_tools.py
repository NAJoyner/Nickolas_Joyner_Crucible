"""
demo_tools.py - LLM Tool Interface for CRUCIBLE

Provides the bridge between the LLM's function-calling capability and the
existing material identification model. Defines the OpenAI-compatible tool
schema that describes available functions to the LLM, and handles execution
of those functions when called.

The tool schema follows the OpenAI function calling format, which llama-cpp-python
supports natively for compatible models like Phi-3.5.
"""

import json
from tools import identify_material


# Tool schema in OpenAI function-calling format.
# This JSON structure tells the LLM what tools exist and how to invoke them.
# The LLM uses this to decide when to call a tool and how to structure arguments.
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "identify_material",
        "description": (
            "Identify a material based on its Raman spectroscopy peaks and "
            "formation energy. Returns the predicted material name with confidence score."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "peak_1": {
                    "type": "number",
                    "description": "First Raman peak in wavenumbers (cm^-1), typically 100-2000."
                },
                "peak_2": {
                    "type": "number",
                    "description": "Second Raman peak in wavenumbers (cm^-1), typically 100-2000."
                },
                "formation_energy": {
                    "type": "number",
                    "description": (
                        "Formation energy in eV/atom, typically -15 to 0. "
                        "Negative values indicate stable compounds."
                    )
                }
            },
            "required": ["peak_1", "peak_2", "formation_energy"]
        }
    }
}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Execute a tool by name with the provided arguments.
    
    This function acts as a dispatcher, routing tool calls from the LLM to the
    appropriate backend function. Results are returned as JSON strings for
    easy parsing by the LLM.
    
    Args:
        tool_name: The function name requested by the LLM.
        arguments: Dictionary of parameter names to values.
    
    Returns:
        JSON string containing either {"success": True, "result": ...}
        or {"success": False, "error": ...}.
    """
    if tool_name != "identify_material":
        return json.dumps({"success": False, "error": f"Unknown tool: {tool_name}"})
    
    try:
        result = identify_material(
            arguments["peak_1"],
            arguments["peak_2"],
            arguments["formation_energy"]
        )
        return json.dumps({"success": True, "result": result})
    
    except KeyError as e:
        return json.dumps({"success": False, "error": f"Missing argument: {e}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


if __name__ == "__main__":
    # Quick smoke test with known Ceria parameters
    print("Testing demo_tools.py")
    print("-" * 50)
    
    test_args = {"peak_1": 465, "peak_2": 610, "formation_energy": -11.2}
    print(f"Input: {test_args}")
    
    result = execute_tool("identify_material", test_args)
    print(f"Output: {json.loads(result)}")
