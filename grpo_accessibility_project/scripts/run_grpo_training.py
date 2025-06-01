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
from grpo_accessibility_project.scripts.reward_model_server import start_reward_model_server # Renamed import

# Global variable to store the reward server process
reward_server_process = None

def cleanup_reward_server():
    global reward_server_process
    if reward_server_process and reward_server_process.poll() is None:
        print("Terminating reward model server...")
        # Send SIGTERM first, then SIGKILL if it doesn't terminate
        reward_server_process.terminate()
        try:
            reward_server_process.wait(timeout=10) # Wait 10 seconds
            print("Reward model server terminated gracefully.")
        except subprocess.TimeoutExpired:
            print("Reward model server did not terminate gracefully, sending SIGKILL.")
            reward_server_process.kill()
            reward_server_process.wait()
            print("Reward model server killed.")
        reward_server_process = None

def main():
    global reward_server_process

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'grpo_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- GPU Configuration ---
    import torch # For checking CUDA availability
    reward_model_gpu_id = str(config.get('reward_model_gpu_id', 0)) # Default to 0
    policy_model_gpu_id = str(config.get('policy_model_gpu_id', 1)) # Default to 1
    print(f"--- GPU Configuration ---")
    print(f"Reward Model will be targeted to GPU: {reward_model_gpu_id}")
    print(f"Policy Model & Training will be targeted to GPU: {policy_model_gpu_id}")

    # --- 1. Start Reward Model Server ---
    print("\n--- Starting Reward Model Server ---")
    # Updated to use start_reward_model_server with TGI parameters from config
    reward_server_process = start_reward_model_server(
        model_name=config['reward_model_name'],
        host=config['reward_model_server_host'],
        port=config['reward_model_server_port'],
        dtype=config['reward_model_dtype'],
        num_shard=config['tgi_num_shard'],
        max_total_tokens=config['tgi_max_total_tokens'],
        shm_size=config['tgi_shm_size'],
        log_level=config.get('tgi_log_level', "info"), # Add tgi_log_level to config or default
        target_gpu_id=reward_model_gpu_id, # Already a string, start_reward_model_server handles formatting for docker
        quantize=config.get('reward_model_quantization', None), # Get from config, default to None
        # Optional TGI params from config if they exist
        max_input_length=config.get('tgi_max_input_length', None),
        max_batch_prefill_tokens=config.get('tgi_max_batch_prefill_tokens', None)
    )
    if not reward_server_process:
        print("Failed to start TGI reward model server. Exiting.")
        return
    
    # The inference_client.py will append the correct OpenAI API path (e.g., /v1/completions)
    reward_model_endpoint = f"http://{config['reward_model_server_host']}:{config['reward_model_server_port']}"
    print(f"Reward model server base URL: {reward_model_endpoint}")

    try:
        # --- Set CUDA_VISIBLE_DEVICES for Policy Model & Training ---
        print(f"\n--- Setting main process CUDA_VISIBLE_DEVICES to '{policy_model_gpu_id}' for Policy Model & Training ---")
        os.environ['CUDA_VISIBLE_DEVICES'] = policy_model_gpu_id
        
        # Force PyTorch to recognize the change immediately and set the device for subsequent operations
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1: # Should be 1 after CUDA_VISIBLE_DEVICES is set
                torch.cuda.set_device(0) # PyTorch will see this as device 0 within the visible set
                print(f"PyTorch CUDA is available. Device count visible: {torch.cuda.device_count()}. Set to use visible device 0 (originally GPU {policy_model_gpu_id}).")
                print(f"Current PyTorch CUDA device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
            elif torch.cuda.device_count() > 1:
                print(f"Warning: CUDA_VISIBLE_DEVICES set to '{policy_model_gpu_id}', but PyTorch sees {torch.cuda.device_count()} devices. This might lead to unexpected behavior. Attempting to set to the first visible device.")
                torch.cuda.set_device(0) # Default to the first one it sees
                print(f"Current PyTorch CUDA device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
            else: # No GPUs visible or an issue
                print("Error: No CUDA devices visible to PyTorch after setting CUDA_VISIBLE_DEVICES. Check GPU setup and policy_model_gpu_id.")
                cleanup_reward_server()
                return
        else:
            print("Error: PyTorch CUDA is NOT available after setting CUDA_VISIBLE_DEVICES. Check GPU setup.")
            cleanup_reward_server()
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
        # The dataset provides context_messages[0] as system, context_messages[1] as user.
        # Qwen3 format:
        # <|im_start|>system
        # {system_message}<|im_end|>
        # <|im_start|>user
        # {user_message}<|im_end|>
        # <|im_start|>assistant
        # (This is where the model generates)
        # We need to ensure the tokenizer's chat_template handles this.
        # Unsloth's FastLanguageModel usually sets up a good default template.
        # If specific customization is needed:
        # Explicitly set the chat template for Qwen3
        # Reference: https://huggingface.co/docs/transformers/main/en/chat_templating
        # And Qwen1.5/Qwen2 model cards often specify this format.
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
        # For Qwen, often no explicit template change is needed if using HF auto-tokenization conventions.
        # The `prompt` field in our dataset is already a list of messages.
        
        # Set the main system prompt for the generator (will be used by GRPOTrainer if not overridden by dataset)
        # However, our dataset items already contain a system prompt in `context_messages`.
        # The GRPOTrainer will use the `prompt` column from the dataset, which is `generator_prompt_messages`.
        # So, a separate `config['generator_system_prompt']` might not be directly used by TRL's SFT/GRPO
        # if the dataset itself provides system messages. We'll rely on the dataset's system prompt.

        # --- 4. Load and Process Dataset ---
        print("\n--- Loading and Processing Dataset ---")
        train_dataset = load_and_process_accessibility_dataset(
            dataset_path=config['dataset_path'],
            tokenizer=tokenizer, # Pass tokenizer for length checking if desired
            max_prompt_length=config['max_prompt_length']
        )
        print(f"Loaded dataset with {len(train_dataset)} samples.")
        if len(train_dataset) == 0:
            print("No data to train on. Exiting.")
            return

        # --- 5. Set up GRPO Configuration and Trainer ---
        print("\n--- Setting up GRPO Trainer ---")
        
        # Define sampling parameters for the generator model during GRPO rollouts
        # These are passed to the `generate` call within GRPOTrainer
        # Note: Unsloth's GRPOTrainer might have its own way of handling vLLM sampling params.
        # For standard TRL GRPOTrainer, you'd set `generation_kwargs`.
        # For Unsloth, it might be through `vllm_sampling_params` in GRPOConfig.
        
        # Check TRL/Unsloth version for GRPOTrainer's vLLM integration.
        # Assuming `vllm_sampling_params` is the way for Unsloth's GRPOTrainer.
        grpo_vllm_sampling_params = SamplingParams(
            temperature=1.0, # Higher temperature for more diverse generations during training
            top_p=1.0,
            top_k=-1, # -1 means no top-k filtering
            max_tokens=config['max_completion_length'],
            stop=[tokenizer.eos_token, "<|im_end|>"], # Ensure model stops appropriately
            include_stop_str_in_output=True, # Usually True for GRPO
        )

        training_args = GRPOConfig(
            output_dir=config['output_dir'],
            num_train_epochs=config.get('num_train_epochs', 1), # Default to 1 epoch if not specified, max_steps will override
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
            report_to="none", # Or "wandb", "tensorboard"
            remove_unused_columns=False, # Important for passing custom columns to reward fn
            
            # GRPO specific arguments
            num_generations=config['num_generations'],
            max_prompt_length=config['max_prompt_length'],
            max_completion_length=config['max_completion_length'],
            temperature=1.0, # Rollout temperature for policy model
            
            # For Unsloth's GRPOTrainer with vLLM backend for generator
            vllm_sampling_params=grpo_vllm_sampling_params,
            # If not using Unsloth's vLLM integration directly in GRPOTrainer,
            # you might need `generation_kwargs` for HF generate.
            dataloader_drop_last=True, # Try dropping the last batch if incomplete
        )

        # The reward function needs access to the reward_model_endpoint and reward_mapping.
        # We will use functools.partial to bind these arguments to the reward function,
        # as Unsloth's GRPOTrainer might not pass reward_kwargs as positional arguments.
        
        fixed_reward_args = {
            "reward_model_endpoint_url": reward_model_endpoint,
            "reward_mapping": config['reward_mapping']
        }

        # Create a partial function with reward_model_endpoint_url and reward_mapping pre-filled.
        # The GRPOTrainer will then call this partial function with:
        # partial_calculate_accessibility_reward(prompts, completions, category=..., original_chatbot_A_code=..., etc.)
        # The other arguments (category, etc.) are expected to be passed by the trainer
        # if they are columns in the dataset.
        partial_calculate_accessibility_reward = partial(
            calculate_accessibility_reward,
            reward_model_endpoint_url=fixed_reward_args["reward_model_endpoint_url"],
            reward_mapping=fixed_reward_args["reward_mapping"]
        )
        # Copy metadata (like __name__) from the original function to the partial object
        # This is needed because UnslothGRPOTrainer inspects func.__name__
        update_wrapper(partial_calculate_accessibility_reward, calculate_accessibility_reward)

        trainer = GRPOTrainer(
            model=generator_model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            reward_funcs=[partial_calculate_accessibility_reward], # Use the partial function
            # reward_kwargs is commented out as its arguments are now bound by `partial`.
            # If GRPOTrainer uses reward_kwargs for other purposes, it could be reinstated,
            # but it was not correctly supplying these specific arguments to calculate_accessibility_reward.
            # reward_kwargs=fixed_reward_args, 
            # processing_class=tokenizer, # Unsloth's GRPOTrainer might need this
        )
        
        # --- 6. Run Training ---
        print("\n--- Starting GRPO Training ---")
        trainer.train()
        print("--- GRPO Training Finished ---")

        # --- 7. Save Final Model ---
        print("\n--- Saving Final Model ---")
        final_save_path = os.path.join(config['output_dir'], "final_model")
        trainer.save_model(final_save_path)
        # tokenizer.save_pretrained(final_save_path) # Unsloth's save_model might handle this
        print(f"Final model saved to {final_save_path}")

    except Exception as e:
        print(f"An error occurred during the training process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 8. Cleanup Reward Server ---
        print("\n--- Cleaning up Reward Model Server ---")
        cleanup_reward_server()

if __name__ == "__main__":
    # sys.path modification moved to the top of the script.
    
    # Handle Ctrl+C for graceful shutdown of reward server
    def signal_handler(sig, frame):
        print('Ctrl+C detected. Shutting down...')
        cleanup_reward_server()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    main()
