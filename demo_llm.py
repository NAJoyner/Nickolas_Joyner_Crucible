"""
demo_llm.py

Main CRUCIBLE LLM demo - connects the AI model with your material identification tool.
This is a simple CLI chat interface where you can talk naturally to identify materials.
"""

from llama_cpp import Llama
import json
from demo_tools import TOOL_SCHEMA, execute_tool


class SimpleCrucible:
    """
    Simple CRUCIBLE LLM interface.
    Handles conversation and tool calling.
    """
    
    def __init__(self, model_path):
        """Initialize the LLM and set up the system prompt."""
        print("=" * 60)
        print("Loading CRUCIBLE...")
        print("=" * 60)
        
        # Load the model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,      # Context window size
            n_threads=6,     # CPU threads (adjust for your system)
            n_gpu_layers=0,  # CPU only
            verbose=False    # Don't show internal details
        )
        
        # System prompt - tells the LLM how to behave
        self.system_prompt = """You are CRUCIBLE, a material science assistant that helps identify materials from spectroscopic data.

Your main capability is identifying materials using Raman spectroscopy data. You have access to a tool called 'identify_material' that requires three parameters:
- peak_1: First Raman peak in cm^-1
- peak_2: Second Raman peak in cm^-1  
- formation_energy: Formation energy in eV/atom

When users provide this data, use the tool to identify the material. If data is missing, politely ask for it.

Be concise, scientific, and helpful. Explain your identifications briefly."""

        # Conversation history
        self.history = []
        
        print("CRUCIBLE ready!")
    
    def chat(self, user_message):
        """
        Process a user message and return a response.
        Handles tool calling automatically.
        
        Args:
            user_message: The user's input text
            
        Returns:
            The assistant's response as a string
        """
        # Add user message to history
        self.history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare messages with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.history
        
        # Generate response (with tool calling enabled)
        max_iterations = 3  # Prevent infinite loops
        
        for iteration in range(max_iterations):
            # Ask LLM for response
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                tools=[TOOL_SCHEMA],
                tool_choice="auto"
            )
            
            message = response['choices'][0]['message']
            
            # Check if LLM wants to use a tool
            if 'tool_calls' in message and message['tool_calls']:
                tool_call = message['tool_calls'][0]
                tool_name = tool_call['function']['name']
                tool_args = json.loads(tool_call['function']['arguments'])
                
                # Show what's happening
                print(f"\n🔧 [Tool Call] {tool_name}")
                print(f"   Arguments: {tool_args}")
                
                # Execute the tool
                tool_result = execute_tool(tool_name, tool_args)
                
                print(f"   Result: {tool_result}\n")
                
                # Add tool interaction to message history
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": tool_result
                })
                
                # Continue loop - LLM will now respond with the tool result
                
            else:
                # No tool call - we have the final response
                assistant_message = message['content']
                
                # Add to history
                self.history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                return assistant_message
        
        # If we hit max iterations
        return "I apologize, but I had trouble processing that request. Could you rephrase?"
    
    def reset(self):
        """Clear conversation history."""
        self.history = []
        print("Conversation history cleared.\n")


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 60)
    print("CRUCIBLE - Material Identification Demo")
    print("=" * 60)
    print("""
Computational Repository for Unified Classification 
and Interactive Base Learning Expert

I can identify materials from Raman spectroscopy data!

Commands:
  'exit' or 'quit' - Exit CRUCIBLE
  'clear'          - Clear conversation history
  'help'           - Show example queries
""")
    print("=" * 60 + "\n")


def print_help():
    """Print help with example queries."""
    print("\n" + "=" * 60)
    print("Example Queries:")
    print("=" * 60)
    print("""
1. "Identify a material with peaks at 465 and 610 cm^-1, 
    formation energy -11.2 eV/atom"

2. "I have Raman peaks at 144 and 399, what could it be?"
   (I'll ask for the formation energy)

3. "What material has peaks 520 and 950?"
   (I'll ask for the missing data)

4. "What is Raman spectroscopy?"
   (I can answer general questions too!)

5. "Tell me about Ceria"
   (Ask about materials)
""")
    print("=" * 60 + "\n")


def main():
    """Main application loop."""
    
    # Check if model exists
    model_path = "./models/Phi-3.5-mini-instruct-Q4_K_M.gguf"
    
    import os
    if not os.path.exists(model_path):
        print("ERROR: Model file not found!")
        print(f"Expected location: {model_path}")
        print("\nPlease download the model first.")
        return
    
    # Initialize CRUCIBLE
    try:
        crucible = SimpleCrucible(model_path)
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {e}")
        return
    
    # Show welcome message
    print_welcome()
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Skip empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye! Keep exploring materials!\n")
                break
            
            if user_input.lower() == 'clear':
                crucible.reset()
                continue
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            # Get response from LLM
            print("\nCRUCIBLE: ", end="", flush=True)
            
            try:
                response = crucible.chat(user_input)
                print(response + "\n")
                
            except Exception as e:
                print(f"Error: {e}\n")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit.\n")
            continue
        
        except EOFError:
            print("\nGoodbye!\n")
            break


if __name__ == "__main__":
    main()
