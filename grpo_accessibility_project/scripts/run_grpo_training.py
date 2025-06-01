import yaml
import subprocess
import time
import os
import sys

# Ensure the script can find other modules in the project when run from the 'scripts' directory.
# This adds the project's root directory (parent of 'grpo_accessibility_project') to sys.path.
# This is necessary because the script is in a subdirectory (scripts) and uses absolute-like imports
# starting with 'grpo_accessibility_project.'
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root) # Insert at the beginning to prioritize project modules

import signal # For terminating subprocess
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from vllm import SamplingParams # For generator model sampling if not using Unsloth's default
from functools import partial, update_wrapper

# Adjust import paths based on your project structure
from grpo_accessibility_project.scripts.data_loader import load_and_process_accessibility_dataset
from grpo_accessibility_project.scripts.reward_function import calculate_accessibility_reward
# Removed import for start_reward_model_server as it's no longer used with Gemini

# Global variable for reward server process is no longer needed
# Cleanup function for reward server is no longer needed

def main():
    # global reward_server_process # Removed as it's no longer needed

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'grpo_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- GPU Configuration ---
    import torch # For checking CUDA availability
    # reward_model_gpu_id is no longer needed as Gemini is used.
    policy_model_gpu_id = str(config.get('policy_model_gpu_id', 0)) # Default to 0 if only one GPU or for policy
    print(f"--- GPU Configuration ---")
    print(f"Policy Model & Training will be targeted to GPU: {policy_model_gpu_id}")

    # --- 1. Start Reward Model Server --- (Section removed as Gemini is used)
    # print("\n--- Reward Model Server is NOT started (using Gemini API) ---")
    # reward_model_endpoint is no longer needed.

    try:
        # --- Set CUDA_VISIBLE_DEVICES for Policy Model & Training ---
        print(f"\n--- Setting main process CUDA_VISIBLE_DEVICES to '{policy_model_gpu_id}' for Policy Model & Training ---")
        os.environ['CUDA_VISIBLE_DEVICES'] = policy_model_gpu_id
        
        # Force PyTorch to recognize the change immediately and set the device for subsequent operations
        if torch.cuda.is_available():
            if torch.cuda.device_count() >= 1: # Check if at least one GPU is visible
                torch.cuda.set_device(0) # PyTorch will see this as device 0 within the visible set
                print(f"PyTorch CUDA is available. Device count visible: {torch.cuda.device_count()}. Set to use visible device 0 (originally GPU {policy_model_gpu_id}).")
                print(f"Current PyTorch CUDA device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
            else: # No GPUs visible or an issue
                print("Error: No CUDA devices visible to PyTorch after setting CUDA_VISIBLE_DEVICES. Check GPU setup and policy_model_gpu_id.")
                # cleanup_reward_server() # No server to cleanup
                return
        else:
            print("Error: PyTorch CUDA is NOT available after setting CUDA_VISIBLE_DEVICES. Check GPU setup.")
            # cleanup_reward_server() # No server to cleanup
            return

        # --- 2. Initialize Generator Model and Tokenizer (Unsloth) ---
        print("\n--- Initializing Generator Model and Tokenizer (should be on GPU {policy_model_gpu_id}) ---")
        generator_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config['generator_model_name'],
            max_seq_length=config['max_seq_length'],
            load_in_4bit=False, # GRPO typically uses 16-bit for policy model
            max_lora_rank=config['lora_rank'], # Set if using PEFT immediately
            gpu_memory_utilization=0.3, # Reduced as a precaution
        )

        # Apply PEFT settings for LoRA
        generator_model = FastLanguageModel.get_peft_model(
            generator_model,
            r=config['lora_rank'],
            target_modules=config['lora_target_modules'],
            lora_alpha=config['lora_rank'] * config['lora_alpha_multiplier'],
            use_gradient_checkpointing="unsloth", # Unsloth's recommendation
            random_state=3407, # For reproducibility
        )
        
        # --- 3. Adapt Chat Template for Generator ---
        qwen_chat_template = (
            "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
                "{% elif message['role'] == 'user' %}"
                    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}"
                "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
        tokenizer.chat_template = qwen_chat_template
        print("Successfully set Qwen chat template for the tokenizer.")

        # --- 4. Load and Process Dataset ---
        print("\n--- Loading and Processing Dataset ---")
        train_dataset = load_and_process_accessibility_dataset(
            dataset_path=config['dataset_path'],
            tokenizer=tokenizer, 
            max_prompt_length=config['max_prompt_length']
        )
        print(f"Loaded dataset with {len(train_dataset)} samples.")
        if len(train_dataset) == 0:
            print("No data to train on. Exiting.")
            return

        # --- 5. Set up GRPO Configuration and Trainer ---
        print("\n--- Setting up GRPO Trainer ---")
        
        grpo_vllm_sampling_params = SamplingParams(
            temperature=1.0, 
            top_p=1.0,
            top_k=-1, 
            max_tokens=config['max_completion_length'],
            stop=[tokenizer.eos_token, "<|im_end|>"], 
            include_stop_str_in_output=True, 
        )

        training_args = GRPOConfig(
            output_dir=config['output_dir'],
            num_train_epochs=config.get('num_train_epochs', 1), 
            max_steps=config['max_steps'],
            per_device_train_batch_size=config['per_device_train_batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            learning_rate=config['learning_rate'],
            logging_steps=config['logging_steps'],
            save_steps=config['save_steps'],
            optim=config['optim'],
            weight_decay=config['weight_decay'],
            lr_scheduler_type=config['lr_scheduler_type'],
            warmup_ratio=config['warmup_ratio'],
            report_to="none", 
            remove_unused_columns=False, 
            
            num_generations=config['num_generations'],
            max_prompt_length=config['max_prompt_length'],
            max_completion_length=config['max_completion_length'],
            temperature=1.0, 
            
            vllm_sampling_params=grpo_vllm_sampling_params,
            dataloader_drop_last=True, 
        )

        # The reward function now uses Gemini and only needs reward_mapping.
        fixed_reward_args = {
            "reward_mapping": config['reward_mapping']
            # "reward_model_endpoint_url" is no longer needed
        }

        partial_calculate_accessibility_reward = partial(
            calculate_accessibility_reward,
            # reward_model_endpoint_url is removed
            reward_mapping=fixed_reward_args["reward_mapping"]
        )
        update_wrapper(partial_calculate_accessibility_reward, calculate_accessibility_reward)

        trainer = GRPOTrainer(
            model=generator_model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            reward_funcs=[partial_calculate_accessibility_reward], 
        )
        
        # --- 6. Run Training ---
        print("\n--- Starting GRPO Training ---")
        trainer.train()
        print("--- GRPO Training Finished ---")

        # --- 7. Save Final Model ---
        print("\n--- Saving Final Model ---")
        final_save_path = os.path.join(config['output_dir'], "final_model")
        trainer.save_model(final_save_path)
        print(f"Final model saved to {final_save_path}")

    except Exception as e:
        print(f"An error occurred during the training process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 8. Cleanup Reward Server --- (No longer needed)
        print("\n--- Training process finished/terminated. No local reward server to clean up. ---")
        # cleanup_reward_server() # Removed

if __name__ == "__main__":
    # sys.path modification moved to the top of the script.
    
    # Handle Ctrl+C for graceful shutdown (reward server part removed)
    def signal_handler(sig, frame):
        print('Ctrl+C detected. Exiting script...')
        # cleanup_reward_server() # Removed
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    main()
