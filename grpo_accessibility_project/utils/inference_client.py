# import requests # No longer needed for vLLM
import json
import time
import os
import google.generativeai as genai

# VLLM_SERVED_MODEL_NAME = os.getenv("REWARD_MODEL_NAME_FOR_CLIENT", "gaotang/RM-R1-DeepSeek-Distilled-Qwen-14B") # Commented out for Gemini

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)

# Model name as specified in the task
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20" 

def query_gemini_api(prompt: str, generation_config_override: dict = None) -> str:
    """
    Sends a prompt to the Google Gemini API and returns the generated text.

    Args:
        prompt: The prompt string to send to the model.
        generation_config_override: Dictionary of Gemini generation parameters to override defaults.
                                    Example: {"temperature": 0.2, "max_output_tokens": 50}

    Returns:
        The generated text from the model, or an error message if the request fails.
    """
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    # Default generation configuration for reward label generation
    default_generation_config = {
        "temperature": 0.1,       # Low temperature for deterministic label
        "top_p": 0.9,
        "max_output_tokens": 256, # Increased from 30. Sufficient for a short qualitative label, 
                                  # raised due to MAX_TOKENS errors indicating actual output might be longer 
                                  # or token counting is more inclusive.
        # "stop_sequences": ["\n"] # Gemini API might handle this differently or not need it for short outputs
    }

    current_generation_config = {**default_generation_config, **(generation_config_override or {})}

    print(f"Querying Gemini API model: {GEMINI_MODEL_NAME}")
    # print(f"Prompt (first 100 chars): {prompt[:100]}...") # For debugging if needed
    # print(f"Generation Config: {json.dumps(current_generation_config, indent=2)}")

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(**current_generation_config)
        )
        
        # Debugging the full response object
        # print(f"Full Gemini Response: {response}")

        if response.parts:
            # Assuming the first part contains the text
            generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            return generated_text.strip()
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
             # Handle cases where response.parts might be empty but candidates are populated
            generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            return generated_text.strip()
        else:
            # Fallback if the structure is unexpected or no text is found
            # Check for prompt feedback (e.g., safety blocks)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                error_message = f"Gemini API request blocked. Reason: {response.prompt_feedback.block_reason}."
                if response.prompt_feedback.safety_ratings:
                    error_message += f" Safety Ratings: {response.prompt_feedback.safety_ratings}"
                print(error_message)
                return f"Error: {error_message}"
            print(f"Unexpected Gemini API response format or empty content. Full response: {response}")
            return "Error: No text content found in Gemini response."


    except Exception as e:
        # Catching a broad exception type from the Gemini SDK if available, or general Exception
        # Example: google.api_core.exceptions.GoogleAPIError
        print(f"Error querying Gemini API: {e}")
        # Attempt to get more details if it's a known API error type
        # This part is speculative as the exact exception type might vary.
        # if hasattr(e, 'message'):
        #     print(f"Error details: {e.message}")
        return f"Error: {e}"


# def query_reward_model_server(prompt: str, endpoint_url: str, sampling_params: dict = None) -> str:
#     """
#     Sends a prompt to a vLLM OpenAI-compatible server and returns the generated text.
#     (Commented out as we are switching to Gemini)
#     """
#     # Default sampling parameters for OpenAI-compatible /v1/completions
#     default_openai_sampling_params = {
#         "temperature": 0.1,
#         "top_p": 0.9,        # OpenAI API typically uses top_p
#         "max_tokens": 4096,    # Renamed from max_new_tokens
#         "stop": ["\n", "<|im_end|>", "<|endoftext|>"], # Renamed from stop_sequences
#     }
#     current_sampling_params = {**default_openai_sampling_params, **(sampling_params or {})}
#     if endpoint_url.endswith('/'):
#         completions_endpoint_url = f"{endpoint_url}v1/completions"
#     else:
#         completions_endpoint_url = f"{endpoint_url}/v1/completions"
#     payload = {
#         "model": VLLM_SERVED_MODEL_NAME, 
#         "prompt": prompt,
#         **current_sampling_params 
#     }
#     payload = {k: v for k, v in payload.items() if v is not None}
#     print(f"Querying vLLM OpenAI endpoint: {completions_endpoint_url}")
#     print(f"Payload: {json.dumps(payload, indent=2)}")
#     try:
#         response = requests.post(completions_endpoint_url, json=payload, timeout=120) 
#         response.raise_for_status()  
#         result = response.json()
#         if "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
#             if "text" in result["choices"][0] and isinstance(result["choices"][0]["text"], str):
#                 return result["choices"][0]["text"].strip()
#             elif "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
#                  return result["choices"][0]["message"]["content"].strip()
#         print(f"Unexpected OpenAI API response format from {completions_endpoint_url}: {result}")
#         return json.dumps(result) 
#     except requests.exceptions.RequestException as e:
#         print(f"Error querying vLLM server at {completions_endpoint_url}: {e}")
#         if hasattr(e, 'response') and e.response is not None:
#             print(f"Response status: {e.response.status_code}")
#             print(f"Response content: {e.response.text}")
#         return f"Error: {e}"
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON response from {completions_endpoint_url}: {e}")
#         print(f"Response content: {response.text}")
#         return f"Error decoding JSON: {e}"

if __name__ == '__main__':
    print("Testing Gemini API client...")
    
    # Ensure GEMINI_API_KEY is set in your environment
    # export GEMINI_API_KEY="your_api_key_here" 
    # (The user was instructed to use AIzaSyCm_8F71yMN3IWyhVQyRYk6aN96qTmT8MM)

    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY environment variable is not set. Please set it to run this test.")
    else:
        test_prompt_gemini = "What is the capital of France?"
        custom_config = {
            "temperature": 0.7,
            "max_output_tokens": 50
        }
        
        generated_text = query_gemini_api(test_prompt_gemini, generation_config_override=custom_config)
        print(f"\n--- Gemini Test Query ---")
        print(f"Prompt: {test_prompt_gemini}")
        print(f"Generated Text: {generated_text}")

        reward_prompt_example_gemini = """[INST] Evaluate the accessibility of the "Generated Code" based on the "Client Question", "Chatbot A's Code", and the specified "Accessibility Category".
Focus on WCAG 2.1/3.0 principles for the category: 'aria_labels'.

Client Question:
Create an accessible button.

Chatbot A's Code (Less Accessible Baseline):
```html
<div onclick='doSomething()'>Click me</div>
```

Generated Code (to be evaluated):
```html
<button aria-label='Submit form'>Submit</button>
```

Accessibility Category: aria_labels

Your task is to provide ONLY ONE of the following qualitative labels. Do NOT add any other text, explanation, or formatting.
- "Excellent Accessibility"
- "Good Accessibility"
- "Minor Accessibility Issues"
- "Significant Accessibility Issues"
- "Poor Accessibility"

Label: [/INST]"""
        
        # For reward model, we want a short, specific label.
        reward_generation_config = {
            "max_output_tokens": 256, # Increased from 20. Should be ample for a qualitative label.
            "temperature": 0.01, 
        }
        
        generated_assessment = query_gemini_api(reward_prompt_example_gemini, generation_config_override=reward_generation_config)
        print(f"\n--- Gemini Reward Model Assessment Example ---")
        print(f"Prompt (first 200 chars): {reward_prompt_example_gemini[:200]}...")
        print(f"Generated Assessment: {generated_assessment}")
        
        possible_labels = ["Excellent Accessibility", "Good Accessibility", "Minor Accessibility Issues", "Significant Accessibility Issues", "Poor Accessibility"]
        parsed_label = "Not Parsed"
        if isinstance(generated_assessment, str): # Ensure it's a string before lowercasing
            for label in possible_labels:
                if label.lower() in generated_assessment.lower():
                    parsed_label = label
                    break
        print(f"Parsed Label: {parsed_label}")

# --- Old vLLM main block commented out ---
# if __name__ == '__main__':
#     print("Attempting to query a vLLM OpenAI-compatible server (e.g., on host port 8001)...")
#     # Ensure REWARD_MODEL_NAME_FOR_CLIENT is set if your vLLM server expects a specific model name in requests.
#     # For example: export REWARD_MODEL_NAME_FOR_CLIENT="gpt2" if vLLM serves gpt2.
#     # If vLLM is started with --model "my_model_path", then that path or its HF ID might be used.
    
#     # The model name used here in VLLM_SERVED_MODEL_NAME should ideally match
#     # what the vLLM server was started with, or a name it recognizes.
#     # If vLLM is serving a specific model, it might ignore the 'model' field in the request,
#     # or it might require it to match.
    
#     test_prompt = "San Francisco is a"
#     test_endpoint = "http://localhost:8001" # Base URL for vLLM OpenAI server
    
#     # Example sampling parameters for vLLM OpenAI API
#     custom_sampling_params = {
#         "max_tokens": 60,
#         "temperature": 0.5
#     }
    
#     generated_text = query_reward_model_server(test_prompt, test_endpoint, custom_sampling_params)
#     print(f"\n--- Test Query ---")
#     print(f"Prompt: {test_prompt}")
#     print(f"Generated Text: {generated_text}")

#     reward_prompt_example = """[INST] You are an accessibility expert.
# Client Question: Create an accessible button.
# Chatbot A Code: <div onclick='doSomething()'>Click me</div>
# Generated Code: <button aria-label='Submit form'>Submit</button>
# Based on the client question and Chatbot A's code, evaluate the accessibility of the "Generated Code".
# Focus on 'aria_labels'.
# Provide your assessment as one of these labels: "Excellent Accessibility", "Good Accessibility", "Minor Accessibility Issues", "Significant Accessibility Issues", "Poor Accessibility".
# Assessment: [/INST]""" # vLLM might prefer the prompt to end here if it's not a chat model.
    
#     # For reward model, we want a short, specific label.
#     reward_sampling_params = {
#         "max_tokens": 20, # Should be enough for "Excellent Accessibility" etc.
#         "temperature": 0.01, # Near-deterministic for reward assessment
#         "stop": ["\n"] # Stop after the first line (the label)
#     }
    
#     # Update VLLM_SERVED_MODEL_NAME if needed for the specific reward model
#     # For example, if your run_grpo_training.py uses 'gaotang/RM-R1-DeepSeek-Distilled-Qwen-14B'
#     # VLLM_SERVED_MODEL_NAME = "gaotang/RM-R1-DeepSeek-Distilled-Qwen-14B" 
#     # This is now handled by the global variable at the top.

#     generated_assessment = query_reward_model_server(reward_prompt_example, test_endpoint, reward_sampling_params)
#     print(f"\n--- Reward Model Assessment Example (simulated with vLLM) ---")
#     print(f"Prompt (first 50 chars): {reward_prompt_example[:50]}...")
#     print(f"Generated Assessment: {generated_assessment}")
    
#     possible_labels = ["Excellent Accessibility", "Good Accessibility", "Minor Accessibility Issues", "Significant Accessibility Issues", "Poor Accessibility"]
#     parsed_label = "Not Parsed"
#     for label in possible_labels:
#         if label.lower() in generated_assessment.lower():
#             parsed_label = label
#             break
#     print(f"Parsed Label: {parsed_label}")
