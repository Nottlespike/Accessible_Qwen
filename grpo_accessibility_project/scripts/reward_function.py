import re
# from grpo_accessibility_project.utils.inference_client import query_reward_model_server # Old import
from grpo_accessibility_project.utils.inference_client import query_gemini_api # New import
import concurrent.futures

def construct_reward_prompt(client_question: str, chatbot_a_code: str, generated_code: str, category: str) -> str:
    """
    Constructs the prompt for the reward model to evaluate accessibility.
    """
    # Define maximum character lengths for code sections to avoid exceeding model context limits
    # These are approximate; tokenization is more accurate but adds complexity here.
    # Assuming reward model context limit is ~4000 tokens, and other parts of prompt take ~200-300 tokens.
    # Max prompt tokens for variable parts: ~3700 tokens. Avg 4 chars/token => ~14800 chars.
    # Fixed prompt text is ~700 chars. client_question + category ~ a few hundred.
    # Budget for code snippets:
    max_chars_client_question = 1000 
    max_chars_chatbot_a = 4000 # Further reduced
    max_chars_generated_code = 4000 # Further reduced

    truncated_client_question = client_question
    if len(client_question) > max_chars_client_question:
        truncated_client_question = client_question[:max_chars_client_question] + "\\n...[TRUNCATED]"

    truncated_chatbot_a_code = chatbot_a_code
    if len(chatbot_a_code) > max_chars_chatbot_a:
        truncated_chatbot_a_code = chatbot_a_code[:max_chars_chatbot_a] + "\\n...[TRUNCATED CODE]"

    truncated_generated_code = generated_code
    if len(generated_code) > max_chars_generated_code:
        truncated_generated_code = generated_code[:max_chars_generated_code] + "\\n...[TRUNCATED CODE]"

    # Ensure the instruction for the reward model is clear and strict about the expected output format.
    prompt = f"""[INST] Evaluate the accessibility of the "Generated Code" based on the "Client Question", "Chatbot A's Code", and the specified "Accessibility Category".
Focus on WCAG 2.1/3.0 principles for the category: '{category}'.

Client Question:
{truncated_client_question}

Chatbot A's Code (Less Accessible Baseline):
```html
{truncated_chatbot_a_code}
```

Generated Code (to be evaluated):
```html
{truncated_generated_code}
```

Accessibility Category: {category}

Your task is to provide ONLY ONE of the following qualitative labels. Do NOT add any other text, explanation, or formatting.
- "Excellent Accessibility"
- "Good Accessibility"
- "Minor Accessibility Issues"
- "Significant Accessibility Issues"
- "Poor Accessibility"

Label: [/INST]"""
    return prompt

def parse_reward_model_output(text_output: str, reward_mapping: dict) -> int:
    """
    Parses the reward model's textual output to extract the qualitative label and map it to a score.
    """
    # Look for one of the predefined labels in the output.
    # Make the search case-insensitive and strip whitespace for robustness.
    normalized_text_output = text_output.strip().lower()
    
    for label, score in reward_mapping.items():
        if label == "Default": # Skip default during search
            continue
        if label.lower() in normalized_text_output:
            # print(f"Parsed label: '{label}' from output: '{text_output[:100]}...' -> Score: {score}")
            return score
            
    # If no specific label is found, try a more general search or return default.
    # This part can be made more sophisticated if needed.
    if "excellent" in normalized_text_output: return reward_mapping.get("Excellent Accessibility", reward_mapping["Default"])
    if "good" in normalized_text_output: return reward_mapping.get("Good Accessibility", reward_mapping["Default"])
    if "minor" in normalized_text_output: return reward_mapping.get("Minor Accessibility Issues", reward_mapping["Default"])
    if "significant" in normalized_text_output: return reward_mapping.get("Significant Accessibility Issues", reward_mapping["Default"])
    if "poor" in normalized_text_output: return reward_mapping.get("Poor Accessibility", reward_mapping["Default"])
    
    print(f"Warning: Could not parse a recognized qualitative label from reward model output: '{text_output[:200]}...'. Using default score.")
    return reward_mapping["Default"]

def _process_single_completion(
    completion_code: str, 
    client_question: str, 
    chatbot_a_code: str, 
    category_item: str, 
    reward_mapping: dict
) -> int:
    """
    Helper function to process a single completion item.
    This will be called by each thread in the ThreadPoolExecutor.
    """
    reward_prompt_str = construct_reward_prompt(
        client_question=client_question,
        chatbot_a_code=chatbot_a_code,
        generated_code=completion_code,
        category=category_item
    )
    
    gemini_generation_config = {
        "temperature": 0.1,
        "top_p": 0.9,
        "max_output_tokens": 28672, 
    }

    # print(f"\n--- Querying Gemini Reward Model ---") # Less verbose in parallel
    # print(f"Category: {category_item}")
    
    reward_model_output = query_gemini_api(
        prompt=reward_prompt_str,
        generation_config_override=gemini_generation_config
    )
    
    # print(f"Reward Model Raw Output for category {category_item}: {reward_model_output}")
    score = parse_reward_model_output(reward_model_output, reward_mapping)
    # print(f"Assigned Score for category {category_item}: {score}")
    return score

def calculate_accessibility_reward(
    prompts: list, # List of prompt dictionaries (as prepared by data_loader)
    completions: list, # List of generated code strings from Qwen3
    reward_mapping: dict,
    category: list = None,
    original_chatbot_A_code: list = None,
    client_question_text: list = None,
    max_workers: int = None, # Optional: Number of parallel workers, defaults to Python's choice
    **kwargs 
) -> list:
    """
    Calculates accessibility rewards for a batch of completions in parallel.
    This function will be called by the GRPOTrainer.
    """
    batch_size = len(completions)
    if not (len(category) == batch_size and \
            len(original_chatbot_A_code) == batch_size and \
            len(client_question_text) == batch_size):
        raise ValueError("Mismatch in batch sizes of completions and auxiliary data (category, original_chatbot_A_code, client_question_text).")

    results = [None] * batch_size # Initialize results list to store scores in original order

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        future_to_index = {
            executor.submit(
                _process_single_completion,
                completions[i],
                client_question_text[i],
                original_chatbot_A_code[i],
                category[i],
                reward_mapping # reward_mapping is shared across threads
            ): i 
            for i in range(batch_size)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                score = future.result()
                results[index] = score
            except Exception as exc:
                print(f'Completion at index {index} generated an exception during reward calculation: {exc}')
                results[index] = reward_mapping["Default"] # Assign default score on error

    return results

if __name__ == '__main__':
    # Example Usage (now uses Gemini API, ensure GEMINI_API_KEY is set)
    
    # Dummy data for testing
    test_prompts_data = [ # This structure matches what GRPOTrainer would get from our dataset
        {
            "prompt": [{"role":"system","content":"..."},{"role":"user","content":"Client Q1..."}], # Generator prompt
            "category": "aria_labels",
            "original_chatbot_A_code": "<div>Inaccessible Button A</div>",
            "client_question_text": "Create an accessible button."
        },
        {
            "prompt": [{"role":"system","content":"..."},{"role":"user","content":"Client Q2..."}],
            "category": "semantic_html",
            "original_chatbot_A_code": "<span>Not a heading</span>",
            "client_question_text": "Create a semantic heading."
        },
        {
            "prompt": [{"role":"system","content":"..."},{"role":"user","content":"Client Q3..."}],
            "category": "image_alt_text",
            "original_chatbot_A_code": "<img src='image.jpg'>",
            "client_question_text": "Make this image accessible."
        }
    ]
    # Extracting the fields as GRPOTrainer would pass them if they are columns
    test_category = [p["category"] for p in test_prompts_data]
    test_original_chatbot_A_code = [p["original_chatbot_A_code"] for p in test_prompts_data]
    test_client_question_text = [p["client_question_text"] for p in test_prompts_data]
    
    # Dummy completions from the generator model
    test_completions = [
        "<button aria-label='Accessible Button'>Click Me</button> <!-- Excellent Accessibility -->", 
        "<h1>Proper Semantic Heading</h1> <!-- Good Accessibility -->",
        "<img src='image.jpg' alt='A descriptive alt text'> <!-- Excellent Accessibility -->"
    ]

    dummy_reward_mapping = {
        "Excellent Accessibility": 10,
        "Good Accessibility": 7,
        "Minor Accessibility Issues": 2,
        "Significant Accessibility Issues": -5,
        "Poor Accessibility": -10,
        "Default": 0
    }

    print("Testing calculate_accessibility_reward function with Gemini API (parallel)...")
    print("Ensure GEMINI_API_KEY environment variable is set.")
    
    print("\nAttempting to calculate scores using Gemini (parallel)...")
    try:
        calculated_scores = calculate_accessibility_reward(
            prompts=test_prompts_data, # prompts are not directly used by _process_single_completion but passed for consistency
            completions=test_completions,
            reward_mapping=dummy_reward_mapping,
            category=test_category,
            original_chatbot_A_code=test_original_chatbot_A_code,
            client_question_text=test_client_question_text,
            max_workers=2 # Example: use 2 worker threads
        )
        print(f"\nCalculated scores (parallel): {calculated_scores}")
        # Expected: e.g. [10, 7, 10] if parsing works as before
    except Exception as e:
        print(f"Error during calculate_accessibility_reward test (parallel): {e}")
        print("This might be due to missing GEMINI_API_KEY or API issues.")

    # Test parsing logic (remains the same)
    print("\nTesting parsing logic:")
    outputs_to_test = {
        "Excellent Accessibility. The code uses all correct ARIA attributes.": 10,
        "  good accessibility - some minor improvements possible.": 7,
        "Assessment: Minor Accessibility Issues": 2,
        "This has Significant Accessibility Issues.": -5,
        "It's Poor Accessibility.": -10,
        "The model failed to follow instructions.": 0, # Default
        "Excellent": 10, # Test partial match
    }
    for output_str, expected_score in outputs_to_test.items():
        parsed = parse_reward_model_output(output_str, dummy_reward_mapping)
        print(f"Output: '{output_str}' -> Parsed Score: {parsed}, Expected: {expected_score} -> {'Pass' if parsed == expected_score else 'FAIL'}")
