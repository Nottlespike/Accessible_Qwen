import requests
import json
import time
import os

# It's good practice to get the model name from an environment variable or config
# For the OpenAI API, the 'model' parameter in the request body is often required,
# even if the server is configured to serve a specific model.
# We'll try to make it configurable or use a placeholder if not critical for vLLM's OpenAI endpoint.
VLLM_SERVED_MODEL_NAME = os.getenv("REWARD_MODEL_NAME_FOR_CLIENT", "gaotang/RM-R1-DeepSeek-Distilled-Qwen-14B")


def query_reward_model_server(prompt: str, endpoint_url: str, sampling_params: dict = None) -> str:
    """
    Sends a prompt to a vLLM OpenAI-compatible server and returns the generated text.

    Args:
        prompt: The prompt string to send to the model.
        endpoint_url: The base URL of the vLLM OpenAI API server (e.g., "http://localhost:8001").
                      The client will append "/v1/completions".
        sampling_params: Dictionary of sampling parameters.
                         Example: {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512, "stop": ["\n"]}

    Returns:
        The generated text from the model, or an error message if the request fails.
    """
    # Default sampling parameters for OpenAI-compatible /v1/completions
    default_openai_sampling_params = {
        "temperature": 0.1,
        "top_p": 0.9,        # OpenAI API typically uses top_p
        "max_tokens": 4096,    # Renamed from max_new_tokens
        "stop": ["\n", "<|im_end|>", "<|endoftext|>"], # Renamed from stop_sequences
        # "do_sample" is not an OpenAI API parameter; temperature controls sampling.
        # If temperature is 0 or very low, it's effectively greedy.
    }

    current_sampling_params = {**default_openai_sampling_params, **(sampling_params or {})}

    # Construct the full API endpoint URL
    if endpoint_url.endswith('/'):
        completions_endpoint_url = f"{endpoint_url}v1/completions"
    else:
        completions_endpoint_url = f"{endpoint_url}/v1/completions"
    
    # OpenAI /v1/completions payload structure
    payload = {
        "model": VLLM_SERVED_MODEL_NAME, # vLLM might ignore this if it serves a single model
        "prompt": prompt,
        **current_sampling_params # Spread the sampling parameters directly
    }
    
    # Remove None values from payload as some servers might not like them
    payload = {k: v for k, v in payload.items() if v is not None}


    print(f"Querying vLLM OpenAI endpoint: {completions_endpoint_url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(completions_endpoint_url, json=payload, timeout=120) # 120 seconds timeout
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
        
        # OpenAI /v1/completions response structure:
        # {"choices": [{"text": "output_text_here", ...}], ...}
        if "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
            if "text" in result["choices"][0] and isinstance(result["choices"][0]["text"], str):
                return result["choices"][0]["text"].strip()
            # Sometimes, for chat models adapted to completion, it might be in message.content
            elif "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                 return result["choices"][0]["message"]["content"].strip()

        print(f"Unexpected OpenAI API response format from {completions_endpoint_url}: {result}")
        return json.dumps(result) # Return the whole JSON if structure is unexpected

    except requests.exceptions.RequestException as e:
        print(f"Error querying vLLM server at {completions_endpoint_url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        return f"Error: {e}"
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from {completions_endpoint_url}: {e}")
        print(f"Response content: {response.text}")
        return f"Error decoding JSON: {e}"

if __name__ == '__main__':
    print("Attempting to query a vLLM OpenAI-compatible server (e.g., on host port 8001)...")
    # Ensure REWARD_MODEL_NAME_FOR_CLIENT is set if your vLLM server expects a specific model name in requests.
    # For example: export REWARD_MODEL_NAME_FOR_CLIENT="gpt2" if vLLM serves gpt2.
    # If vLLM is started with --model "my_model_path", then that path or its HF ID might be used.
    
    # The model name used here in VLLM_SERVED_MODEL_NAME should ideally match
    # what the vLLM server was started with, or a name it recognizes.
    # If vLLM is serving a specific model, it might ignore the 'model' field in the request,
    # or it might require it to match.
    
    test_prompt = "San Francisco is a"
    test_endpoint = "http://localhost:8001" # Base URL for vLLM OpenAI server
    
    # Example sampling parameters for vLLM OpenAI API
    custom_sampling_params = {
        "max_tokens": 60,
        "temperature": 0.5
    }
    
    generated_text = query_reward_model_server(test_prompt, test_endpoint, custom_sampling_params)
    print(f"\n--- Test Query ---")
    print(f"Prompt: {test_prompt}")
    print(f"Generated Text: {generated_text}")

    reward_prompt_example = """[INST] You are an accessibility expert.
Client Question: Create an accessible button.
Chatbot A Code: <div onclick='doSomething()'>Click me</div>
Generated Code: <button aria-label='Submit form'>Submit</button>
Based on the client question and Chatbot A's code, evaluate the accessibility of the "Generated Code".
Focus on 'aria_labels'.
Provide your assessment as one of these labels: "Excellent Accessibility", "Good Accessibility", "Minor Accessibility Issues", "Significant Accessibility Issues", "Poor Accessibility".
Assessment: [/INST]""" # vLLM might prefer the prompt to end here if it's not a chat model.
    
    # For reward model, we want a short, specific label.
    reward_sampling_params = {
        "max_tokens": 20, # Should be enough for "Excellent Accessibility" etc.
        "temperature": 0.01, # Near-deterministic for reward assessment
        "stop": ["\n"] # Stop after the first line (the label)
    }
    
    # Update VLLM_SERVED_MODEL_NAME if needed for the specific reward model
    # For example, if your run_grpo_training.py uses 'gaotang/RM-R1-DeepSeek-Distilled-Qwen-14B'
    # VLLM_SERVED_MODEL_NAME = "gaotang/RM-R1-DeepSeek-Distilled-Qwen-14B" 
    # This is now handled by the global variable at the top.

    generated_assessment = query_reward_model_server(reward_prompt_example, test_endpoint, reward_sampling_params)
    print(f"\n--- Reward Model Assessment Example (simulated with vLLM) ---")
    print(f"Prompt (first 50 chars): {reward_prompt_example[:50]}...")
    print(f"Generated Assessment: {generated_assessment}")
    
    possible_labels = ["Excellent Accessibility", "Good Accessibility", "Minor Accessibility Issues", "Significant Accessibility Issues", "Poor Accessibility"]
    parsed_label = "Not Parsed"
    for label in possible_labels:
        if label.lower() in generated_assessment.lower():
            parsed_label = label
            break
    print(f"Parsed Label: {parsed_label}")
