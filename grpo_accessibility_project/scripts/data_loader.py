import json
import re
from datasets import Dataset, DatasetDict

def extract_code_from_response(response_text: str, chatbot_marker: str) -> str:
    """
    Extracts code block for a specific chatbot from the combined response string.
    Example chatbot_marker: "[The Start of Chatbot A's Response]"
    """
    end_marker = chatbot_marker.replace("[The Start of", "[The End of")
    try:
        # Regex to find content between start and end markers
        # DOTALL flag allows . to match newlines
        match = re.search(rf"{re.escape(chatbot_marker)}(.*?){re.escape(end_marker)}", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            # print(f"Warning: Could not find markers for {chatbot_marker} in response.")
            # print(f"Response Text was: {response_text[:500]}...") # Print start of problematic text
            return "" # Return empty if not found, GRPOTrainer might need to handle this
    except Exception as e:
        # print(f"Error extracting code for {chatbot_marker}: {e}")
        return ""

def load_and_process_accessibility_dataset(dataset_path: str, tokenizer=None, max_prompt_length: int = 2048):
    """
    Loads the accessibility dataset and processes it for GRPO training.

    Args:
        dataset_path: Path to the accessibility_dataset.json file.
        tokenizer: (Optional) The tokenizer to use for checking prompt length.
        max_prompt_length: (Optional) Maximum token length for a prompt to be included.

    Returns:
        A Hugging Face Dataset object.
    """
    processed_data = []
    with open(dataset_path, 'r') as f:
        raw_dataset = json.load(f)

    for item in raw_dataset:
        if not item.get("context_messages") or len(item["context_messages"]) < 2:
            print(f"Skipping item due to missing or incomplete context_messages: {item.get('metadata', {}).get('category')}")
            continue

        system_message = item["context_messages"][0]
        user_message_obj = item["context_messages"][1]
        
        full_user_content = user_message_obj.get("content", "")

        # Extract Chatbot A's code
        chatbot_a_code = extract_code_from_response(full_user_content, "[The Start of Chatbot A's Response]")
        
        # The prompt for the generator model will be the client question part,
        # and Chatbot A's code will be part of the context given to the reward model.
        # We need to isolate the client question.
        client_question_match = re.search(r"\[Client Question\](.*?)(?:\[The Start of Chatbot A's Response\]|$)", full_user_content, re.DOTALL)
        if not client_question_match:
            print(f"Skipping item, could not extract client question: {item.get('metadata', {}).get('category')}")
            continue
        client_question = client_question_match.group(1).strip()

        # The prompt for the Qwen3 generator will be the system message + user client question
        # The GRPO trainer expects a list of messages for the 'prompt' field
        generator_prompt_messages = [
            system_message,
            {"role": "user", "content": client_question}
        ]
        
        # For the reward function, we'll also need Chatbot A's code and the category
        # These will be passed as part of the dataset item, and the GRPOTrainer
        # should make them available to the reward function.

        # Optional: Filter by tokenized prompt length if tokenizer is provided
        if tokenizer:
            # We need to format the prompt as the model would see it before tokenizing
            # This depends on the chat_template applied by the GRPOTrainer / Unsloth
            # For now, let's assume a simple concatenation for length check,
            # or rely on GRPOTrainer's internal handling.
            # A more accurate way would be to apply the chat_template here.
            # For simplicity, we'll just check the client_question length for now,
            # as the system prompt is fixed.
            tokenized_prompt = tokenizer.encode(client_question) # A rough estimate
            if len(tokenized_prompt) > max_prompt_length:
                # print(f"Skipping item due to prompt length: {len(tokenized_prompt)} > {max_prompt_length}")
                continue
        
        processed_item = {
            "prompt": generator_prompt_messages, # This is what Qwen3 generator sees
            "category": item.get("metadata", {}).get("category", "unknown"),
            "original_chatbot_A_code": chatbot_a_code,
            # "original_chatbot_B_code": extract_code_from_response(full_user_content, "[The Start of Chatbot B's Response]"), # For reference
            # "reference_issues": item.get("metadata", {}).get("accessibility_issues", []), # For reference
            "client_question_text": client_question # For reward model prompt construction
        }
        processed_data.append(processed_item)

    if not processed_data:
        raise ValueError("No data was processed. Check dataset format or filtering criteria.")
        
    # Convert to Hugging Face Dataset
    # hf_dataset = Dataset.from_pandas(pd.DataFrame(processed_data)) # If using pandas
    hf_dataset = Dataset.from_list(processed_data)
    return hf_dataset

if __name__ == '__main__':
    # Example usage:
    # You might need to install datasets: pip install datasets
    # from transformers import AutoTokenizer # To test with tokenizer

    # Dummy tokenizer for testing length filtering (replace with actual)
    class DummyTokenizer:
        def encode(self, text):
            return text.split() # Simple word split
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            if tokenize:
                return self.encode(" ".join([m["content"] for m in messages]))
            return " ".join([m["content"] for m in messages])


    # config_path = '../configs/grpo_config.yaml' # Adjust path as needed
    # import yaml
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    # dataset_file_path = config['dataset_path']
    # max_prompt_len_config = config.get('max_prompt_length', 2048)

    # For standalone testing without config:
    dataset_file_path = "/home/ubuntu/RM-R1/rm_r1/dataset/reasoning_chain_generation/accessibility_dataset_output/accessibility_dataset.json" # Replace with actual path if different
    max_prompt_len_config = 2048
    
    print(f"Loading dataset from: {dataset_file_path}")
    
    # To test with a real tokenizer:
    # tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Base")
    # accessibility_dataset = load_and_process_accessibility_dataset(dataset_file_path, tokenizer, max_prompt_len_config)
    
    # To test without a real tokenizer (for structure checking):
    accessibility_dataset = load_and_process_accessibility_dataset(dataset_file_path, None, max_prompt_len_config)
    
    print(f"Processed dataset with {len(accessibility_dataset)} samples.")
    if len(accessibility_dataset) > 0:
        print("\nFirst sample:")
        print(json.dumps(accessibility_dataset[0], indent=2))
        
        print("\nStructure of 'prompt' in the first sample:")
        for msg in accessibility_dataset[0]['prompt']:
            print(f"  Role: {msg['role']}, Content snippet: {msg['content'][:100]}...")

        # Verify all necessary keys are present
        required_keys = ["prompt", "category", "original_chatbot_A_code", "client_question_text"]
        for key in required_keys:
            if key not in accessibility_dataset[0]:
                print(f"ERROR: Key '{key}' missing from processed dataset sample!")
            else:
                print(f"Key '{key}' present.")

    # Example of how GRPOTrainer might use this:
    # train_dataset = accessibility_dataset
    # eval_dataset = None # Or a split

    # You can also split it:
    # dataset_dict = accessibility_dataset.train_test_split(test_size=0.1)
    # train_dataset = dataset_dict["train"]
    # eval_dataset = dataset_dict["test"]
    # print(f"\nTrain samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
