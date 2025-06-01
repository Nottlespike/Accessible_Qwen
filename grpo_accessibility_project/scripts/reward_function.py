import re
from grpo_accessibility_project.utils.inference_client import query_reward_model_server # Updated import

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
    max_chars_chatbot_a = 6500
    max_chars_generated_code = 7000 # Slightly less to be safe with overall length

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
    prompt = f"""[INST] You are an expert in web accessibility, focusing on WCAG 2.1 and WCAG 3.0 standards.
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

Based on the Client Question and Chatbot A's Code, critically evaluate the accessibility of the "Generated Code".
Pay specific attention to the accessibility category: '{category}'.
Consider aspects like semantic HTML, ARIA usage, keyboard navigation, focus management, color contrast (if applicable from code), and overall adherence to WCAG principles for the given category.

Your response MUST be one of the following exact qualitative labels and nothing else:
- "Excellent Accessibility"
- "Good Accessibility"
- "Minor Accessibility Issues"
- "Significant Accessibility Issues"
- "Poor Accessibility"

Assessment: [/INST]"""
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


def calculate_accessibility_reward(
    prompts: list, # List of prompt dictionaries (as prepared by data_loader)
    completions: list, # List of generated code strings from Qwen3
    reward_model_endpoint_url: str,
    reward_mapping: dict,
    # GRPOTrainer passes these automatically if they are columns in the dataset
    category: list = None, 
    original_chatbot_A_code: list = None,
    client_question_text: list = None,
    **kwargs 
) -> list:
    """
    Calculates accessibility rewards for a batch of completions.
    This function will be called by the GRPOTrainer.
    'prompts' here is a list of the actual prompt strings/message lists fed to the generator.
    We need 'client_question_text' and 'original_chatbot_A_code' from the dataset items.
    """
    scores = []
    
    # Ensure all necessary auxiliary data has the same batch size as completions
    batch_size = len(completions)
    if not (len(category) == batch_size and \
            len(original_chatbot_A_code) == batch_size and \
            len(client_question_text) == batch_size):
        raise ValueError("Mismatch in batch sizes of completions and auxiliary data (category, original_chatbot_A_code, client_question_text).")

    for i in range(batch_size):
        gen_code = completions[i]
        
        # Extract necessary parts from the corresponding dataset item
        # The 'prompts' argument to this function is what the *generator* received.
        # We need the specific fields we stored in our processed dataset.
        current_client_question = client_question_text[i]
        current_chatbot_a_code = original_chatbot_A_code[i]
        current_category = category[i]

        reward_prompt_str = construct_reward_prompt(
            client_question=current_client_question,
            chatbot_a_code=current_chatbot_a_code,
            generated_code=gen_code,
            category=current_category
        )
        
        # Define sampling parameters for the reward model
        # We want a relatively deterministic, focused output for the assessment label.
        reward_sampling_params = {
            "temperature": 0.1, 
            "top_p": 0.9, # TGI uses top_p
            "max_new_tokens": 50, # TGI uses max_new_tokens
            "stop_sequences": ["\n", "<|im_end|>", "<|endoftext|>"], # TGI uses stop_sequences
            "do_sample": True # Ensure sampling is enabled for TGI
        }

        # print(f"\n--- Querying Reward Model (Sample {i+1}) ---")
        # print(f"Category: {current_category}")
        # print(f"Reward Prompt (first 200 chars): {reward_prompt_str[:200]}...")
        
        reward_model_output = query_reward_model_server(
            prompt=reward_prompt_str,
            endpoint_url=reward_model_endpoint_url, # This should be like http://host:port (client adds /generate)
            sampling_params=reward_sampling_params
        )
        
        # print(f"Reward Model Raw Output: {reward_model_output}")

        score = parse_reward_model_output(reward_model_output, reward_mapping)
        scores.append(score)
        # print(f"Assigned Score: {score}")
        # print("--------------------------------------")

    return scores

if __name__ == '__main__':
    # Example Usage (requires a running reward model server and vllm_client.py)
    
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
        }
    ]
    # Extracting the fields as GRPOTrainer would pass them if they are columns
    test_category = [p["category"] for p in test_prompts_data]
    test_original_chatbot_A_code = [p["original_chatbot_A_code"] for p in test_prompts_data]
    test_client_question_text = [p["client_question_text"] for p in test_prompts_data]
    
    # Dummy completions from the generator model
    test_completions = [
        "<button aria-label='Accessible Button'>Click Me</button> <!-- Excellent Accessibility -->", # Reward model might add its reasoning
        "<h1>Proper Semantic Heading</h1> <!-- Good Accessibility -->"
    ]

    # Configuration (normally from grpo_config.yaml)
    dummy_reward_model_endpoint = "http://localhost:8001/generate" # Replace if your server is elsewhere
    dummy_reward_mapping = {
        "Excellent Accessibility": 10,
        "Good Accessibility": 7,
        "Minor Accessibility Issues": 2,
        "Significant Accessibility Issues": -5,
        "Poor Accessibility": -10,
        "Default": 0
    }

    print("Testing calculate_accessibility_reward function...")
    print(f"Ensure a reward model server (e.g., gpt2 for testing) is running on {dummy_reward_model_endpoint}")
    
    # To run this test, you'd need to:
    # 1. Have inference_client.py in the grpo_accessibility_project/utils directory.
    # 2. Start an SGLang server, e.g.:
    #    python -m sglang.launch_server --model-path gpt2 --host localhost --port 8001
    #    (gpt2 won't give meaningful accessibility scores, but it will test the communication)

    # calculated_scores = calculate_accessibility_reward(
    #     prompts=test_prompts_data, 
    #     completions=test_completions,
    #     reward_model_endpoint_url=dummy_reward_model_endpoint,
    #     reward_mapping=dummy_reward_mapping,
    #     category=test_category,
    #     original_chatbot_A_code=test_original_chatbot_A_code,
    #     client_question_text=test_client_question_text
    # )
    # print(f"\nCalculated scores: {calculated_scores}")

    # Test parsing logic
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
