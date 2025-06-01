import requests
import json
import time

def query_vllm_server(prompt: str, endpoint_url: str, sampling_params: dict = None) -> str:
    """
    Sends a prompt to a vLLM server and returns the generated text.

    Args:
        prompt: The prompt string to send to the model.
        endpoint_url: The URL of the vLLM API server (e.g., "http://localhost:8001/generate").
        sampling_params: Dictionary of sampling parameters for vLLM.
                         Example: {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}

    Returns:
        The generated text from the model, or an error message if the request fails.
    """
    if sampling_params is None:
        sampling_params = {
            "temperature": 0.7, # Default temperature
            "top_p": 1.0,       # Default top_p
            "max_tokens": 2048, # Default max_tokens, adjust as needed for reward model
            "stop": ["<|im_end|>", "<|endoftext|>"] # Common stop tokens, adjust if needed
        }

    payload = {
        "prompt": prompt,
        **sampling_params
    }

    try:
        response = requests.post(endpoint_url, json=payload, timeout=120) # 120 seconds timeout
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
        # Assuming the response structure from vLLM's /generate endpoint
        # is like: {"text": ["generated_output_for_prompt_1"]}
        if "text" in result and isinstance(result["text"], list) and len(result["text"]) > 0:
            return result["text"][0]
        else:
            # Fallback for older vLLM or different output structures
            # This part might need adjustment based on actual vLLM server response
            if isinstance(result, dict) and "choices" in result and len(result["choices"]) > 0:
                 return result["choices"][0].get("text", "") # OpenAI compatible
            return json.dumps(result) # Return the whole JSON if structure is unexpected

    except requests.exceptions.RequestException as e:
        print(f"Error querying vLLM server at {endpoint_url}: {e}")
        return f"Error: {e}"
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from {endpoint_url}: {e}")
        print(f"Response content: {response.text}")
        return f"Error decoding JSON: {e}"

if __name__ == '__main__':
    # Example usage (assuming a vLLM server is running on port 8001)
    # To run this example, you'd first need to start a vLLM server:
    # python -m vllm.entrypoints.api_server --model "gpt2" --host localhost --port 8001
    
    print("Attempting to query a dummy vLLM server (e.g., gpt2 on port 8001)...")
    test_prompt = "San Francisco is a"
    test_endpoint = "http://localhost:8001/generate" # Default vLLM API endpoint
    
    # Give some time for the server to potentially start if run concurrently
    # time.sleep(10) 
    
    generated_text = query_vllm_server(test_prompt, test_endpoint, {"max_tokens": 50})
    print(f"Prompt: {test_prompt}")
    print(f"Generated Text: {generated_text}")

    # Example for reward model qualitative output
    # This is a hypothetical prompt structure for the reward model
    reward_prompt_example = """[INST] You are an accessibility expert.
Client Question: Create an accessible button.
Chatbot A Code: <div onclick='doSomething()'>Click me</div>
Generated Code: <button aria-label='Submit form'>Submit</button>
Based on the client question and Chatbot A's code, evaluate the accessibility of the "Generated Code".
Focus on 'aria_labels'.
Provide your assessment as one of these labels: "Excellent Accessibility", "Good Accessibility", "Minor Accessibility Issues", "Significant Accessibility Issues", "Poor Accessibility".
Assessment: [/INST]"""
    
    # Assuming the reward model would complete this with one of the labels
    # generated_assessment = query_vllm_server(reward_prompt_example, test_endpoint, {"max_tokens": 20})
    # print(f"\nReward Model Assessment Example (simulated):")
    # print(f"Prompt: {reward_prompt_example}")
    # print(f"Generated Assessment: {generated_assessment}")
    # if "Excellent Accessibility" in generated_assessment:
    #     print("Parsed: Excellent Accessibility")
