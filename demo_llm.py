"""
demo_llm.py - CLI Chat Interface for CRUCIBLE

A command-line interface for conversational material identification. This module
handles the LLM interaction loop, including multi-turn conversation history and
automatic tool calling when the model requests it.

The conversation flow:
    1. User provides input (potentially including spectroscopy data)
    2. LLM decides whether to call identify_material or respond directly
    3. If tool called: execute it, feed result back to LLM, get final response
    4. Display response, repeat

Usage:
    python demo_llm.py
"""

import os
import json
from llama_cpp import Llama
from demo_tools import TOOL_SCHEMA, execute_tool


# Model configuration
MODEL_PATH = "./models/Phi-3.5-mini-instruct-Q4_K_M.gguf"

SYSTEM_PROMPT = """You are CRUCIBLE, a material science assistant that helps identify materials from spectroscopic data.

Your main capability is identifying materials using Raman spectroscopy data. You have access to a tool called 'identify_material' that requires three parameters:
- peak_1: First Raman peak in cm^-1
- peak_2: Second Raman peak in cm^-1  
- formation_energy: Formation energy in eV/atom

When users provide this data, use the tool to identify the material. If data is missing, politely ask for it.

Be concise, scientific, and helpful. Explain your identifications briefly."""


class CrucibleChat:
    """
    Manages conversation state and LLM interactions for CRUCIBLE.
    
    Handles loading the model, maintaining conversation history, and orchestrating
    the tool-calling loop when the LLM decides to invoke identify_material.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the LLM and prepare for conversation.
        
        Args:
            model_path: Path to the GGUF model file.
        """
        print("Loading CRUCIBLE...")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=6,
            n_gpu_layers=0,
            verbose=False
        )
        self.history = []
        
        print("Ready.\n")
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.
        
        Handles the tool-calling loop: if the LLM requests a tool, we execute it,
        append the result to the message chain, and let the LLM generate a final
        response incorporating the tool output.
        
        Args:
            user_message: The user's input text.
        
        Returns:
            The assistant's final response string.
        """
        self.history.append({"role": "user", "content": user_message})
        
        # Build full message list with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        
        # Tool-calling loop (max 3 iterations to prevent runaway)
        for _ in range(3):
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                tools=[TOOL_SCHEMA],
                tool_choice="auto"
            )
            
            message = response["choices"][0]["message"]
            
            # Check if LLM wants to call a tool
            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                print(f"\n[Tool: {tool_name}] {tool_args}")
                tool_result = execute_tool(tool_name, tool_args)
                print(f"[Result] {tool_result}\n")
                
                # Append tool interaction to message chain
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result
                })
                # Loop continues - LLM will now respond with tool result in context
            
            else:
                # No tool call - this is the final response
                assistant_response = message["content"]
                self.history.append({"role": "assistant", "content": assistant_response})
                return assistant_response
        
        return "I had trouble processing that request. Could you rephrase?"
    
    def clear_history(self):
        """Reset conversation history."""
        self.history = []
        print("Conversation cleared.\n")


def print_help():
    """Display example queries."""
    print("""
Example queries:
  - "Identify a material with peaks at 465 and 610 cm^-1, formation energy -11.2 eV/atom"
  - "I have Raman peaks at 144 and 399, what could it be?"
  - "What is Raman spectroscopy?"

Commands:
  exit/quit  - Exit CRUCIBLE
  clear      - Clear conversation history
  help       - Show this message
""")


def main():
    """Run the CLI chat loop."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Download the model first using the download script.")
        return
    
    try:
        crucible = CrucibleChat(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    print("CRUCIBLE - Material Identification Demo")
    print("Type 'help' for examples, 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                break
            if user_input.lower() == "clear":
                crucible.clear_history()
                continue
            if user_input.lower() == "help":
                print_help()
                continue
            
            response = crucible.chat(user_input)
            print(f"CRUCIBLE: {response}\n")
        
        except KeyboardInterrupt:
            print("\nType 'exit' to quit.\n")
        except EOFError:
            break


if __name__ == "__main__":
    main()
