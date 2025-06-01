import subprocess
import time
import argparse
import requests # For health check
import os
import sys

def start_reward_model_server(
    model_name: str,
    host: str = "localhost",
    port: int = 8001,
    dtype: str = "fp8", # vLLM will use 'auto' or this if specified
    num_shard: int = 1,    # Corresponds to tensor_parallel_size for vLLM
    max_total_tokens: int = 8192, # vLLM uses max_model_len, often inferred
    shm_size: str = "1g",  # Not directly applicable to vLLM non-Docker
    log_level: str = "info", # vLLM has its own logging, not directly set this way
    target_gpu_id: str = None, # For CUDA_VISIBLE_DEVICES, e.g., "0" or "0,1"
    quantize: str = None, # vLLM supports quantization like awq, gptq. bitsandbytes is different.
    max_input_length: int = None, # vLLM's max_model_len covers this
    max_batch_prefill_tokens: int = None, # vLLM has related params like max_num_batched_tokens
):
    """
    Starts a vLLM OpenAI-compatible server for the reward model.

    Args:
        model_name: Name or path of the HuggingFace model.
        host: Host address for the server.
        port: Port for the server.
        dtype: Data type for the model (e.g., "bfloat16", "float16", "auto").
        num_shard: Number of GPU shards (tensor_parallel_size).
        max_total_tokens: Max model length (vLLM's max_model_len).
        shm_size: Not used by vLLM directly.
        log_level: Not directly used by vLLM in this way.
        target_gpu_id: Specific GPU ID(s) to target (e.g., "0", "0,1").
        quantize: vLLM quantization (e.g., 'awq', 'gptq').
        max_input_length: Max input length (part of max_model_len).
        max_batch_prefill_tokens: vLLM has `max_num_batched_tokens` and `max_num_seqs`.
    """
    
    command = [
        sys.executable, # Use the current Python interpreter
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(num_shard),
        "--dtype", dtype,
        "--max-model-len", str(max_total_tokens), 
    ]

    if quantize and quantize != "bitsandbytes-nf4": # vLLM doesn't use bitsandbytes in the same way
        # Common vLLM quantization options are 'awq', 'gptq', 'squeezellm'
        # TGI's 'bitsandbytes-nf4' is specific to TGI/Transformers.
        # If the model requires specific bitsandbytes, vLLM might not load it correctly
        # without further model conversion or if vLLM doesn't support that exact format.
        # We'll pass it if it's a known vLLM quant type.
        if quantize in ["awq", "gptq", "squeezellm", "aqlm", "gptq_marlin", "awq_marlin"]:
             command.extend(["--quantization", quantize])
        else:
            print(f"Warning: Quantization type '{quantize}' may not be directly supported by vLLM in this way. Proceeding without it.")
    
    # vLLM specific arguments if needed, e.g. for memory or specific model types
    # command.extend(["--gpu-memory-utilization", "0.90"]) # Example: use 90% of GPU memory

    env = os.environ.copy()
    if target_gpu_id:
        # Ensure target_gpu_id is a string of comma-separated integers, e.g., "0" or "0,1"
        # vLLM respects CUDA_VISIBLE_DEVICES.
        # If num_shard > 1, vLLM will distribute across the GPUs made visible by CUDA_VISIBLE_DEVICES.
        env["CUDA_VISIBLE_DEVICES"] = str(target_gpu_id)
        print(f"Setting CUDA_VISIBLE_DEVICES to: {target_gpu_id}")


    print(f"Starting vLLM server for reward model with command: {' '.join(command)}")
    
    try:
        # Start the vLLM server as a subprocess
        server_process = subprocess.Popen(command, env=env)
        print(f"vLLM server for {model_name} starting with PID: {server_process.pid}. Waiting for it to become healthy...")

        # Wait for the server to start
        # Health check for vLLM OpenAI-compatible server
        initial_delay_seconds = 15 # vLLM can take some time to load models
        print(f"Waiting {initial_delay_seconds} seconds before starting health checks...")
        time.sleep(initial_delay_seconds)

        max_retries = 60  # Try for 60 * 10 = 600 seconds (10 minutes)
        retries = 0
        # vLLM's OpenAI API server has a /health endpoint
        health_check_url = f"http://{host}:{port}/health" 

        while retries < max_retries:
            if server_process.poll() is not None:
                print(f"vLLM server process exited prematurely with code {server_process.returncode}. Check logs.")
                return None
            try:
                response = requests.get(health_check_url, timeout=15)
                if response.status_code == 200:
                    print(f"vLLM server for {model_name} is healthy on {health_check_url}.")
                    return server_process
                else:
                    print(f"Health check failed with status {response.status_code} at {health_check_url}. Response: {response.text}. Retrying...")
            except requests.exceptions.ConnectionError:
                print(f"Connection refused on {health_check_url}. Server might still be starting. Retrying...")
            except requests.exceptions.RequestException as e:
                print(f"Health check request failed for {health_check_url}: {e}. Retrying...")
            
            time.sleep(10)  # Wait 10 seconds before retrying
            retries += 1
        
        print(f"vLLM server for {model_name} did not become healthy after {max_retries * 10} seconds. Terminating process.")
        server_process.terminate()
        try:
            server_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("vLLM server did not terminate gracefully, sending SIGKILL.")
            server_process.kill()
            server_process.wait()
        return None

    except Exception as e:
        print(f"Failed to start vLLM server for {model_name}: {e}")
        if 'server_process' in locals() and server_process.poll() is None:
            server_process.kill()
            server_process.wait()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a vLLM OpenAI-compatible API server for the reward model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the HuggingFace model for the reward server.")
    parser.add_argument("--host", type=str, default="localhost", help="Host address for the server.")
    parser.add_argument("--port", type=int, default=8001, help="Port for the server.")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type for vLLM model (e.g., bfloat16, float16, auto).")
    parser.add_argument("--num_shard", type=int, default=1, help="Number of GPU shards (tensor_parallel_size for vLLM).")
    # max_total_tokens is more of a vLLM --max-model-len, let's not expose it directly unless needed
    parser.add_argument("--max_total_tokens", type=int, default=8192, help="Max model length for vLLM.")
    parser.add_argument("--target_gpu_id", type=str, default=None, help="Comma-separated list of GPU ID(s) for vLLM (e.g., '0' or '0,1'). Sets CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--quantize", type=str, default=None, help="Quantization mode for vLLM (e.g. 'awq', 'gptq')")
    
    args = parser.parse_args()

    server_process = start_reward_model_server(
        model_name=args.model_name,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        num_shard=args.num_shard,
        # max_total_tokens=args.max_total_tokens, # If added back
        target_gpu_id=args.target_gpu_id,
        quantize=args.quantize
    )

    if server_process:
        print(f"vLLM reward model server process started. PID: {server_process.pid}")
        print(f"To stop the server, Ctrl+C this script.")
        try:
            while True:
                if server_process.poll() is not None:
                    print(f"vLLM server process (PID {server_process.pid}) has exited with code {server_process.returncode}.")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Terminating vLLM reward model server...")
        finally:
            if server_process.poll() is None:
                print(f"Attempting to terminate vLLM server process (PID {server_process.pid})...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=30)
                    print("vLLM server process terminated.")
                except subprocess.TimeoutExpired:
                    print("vLLM server process did not terminate gracefully after SIGTERM, sending SIGKILL.")
                    server_process.kill()
                    server_process.wait()
                    print("vLLM server process killed.")
    else:
        print("Failed to start vLLM reward model server.")
