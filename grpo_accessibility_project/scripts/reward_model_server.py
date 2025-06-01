import subprocess
import time
import argparse
import requests # For health check

def start_vllm_server(
    model_name: str,
    host: str = "localhost",
    port: int = 8001,
    quantization: str = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = None,
    api_endpoint: str = "/generate" # Default for vLLM, OpenAI compatible uses /v1/completions or /v1/chat/completions
):
    """
    Starts a vLLM API server as a subprocess.

    Args:
        model_name: Name or path of the HuggingFace model.
        host: Host address for the server.
        port: Port for the server.
        quantization: Quantization method (e.g., "fp8", "awq").
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: GPU memory utilization limit.
        max_model_len: Maximum model length. If None, will be derived from the model.
        api_endpoint: The API endpoint vLLM server will use.
    """
    command = [
        "python", "-m", "vllm.entrypoints.api_server",
        "--model", model_name,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        # "--served-model-name", "reward-model" # Optional: if you want to give it a specific name
    ]

    if quantization:
        command.extend(["--quantization", quantization])
    
    if max_model_len:
        command.extend(["--max-model-len", str(max_model_len)])

    print(f"Starting vLLM server for reward model with command: {' '.join(command)}")
    
    try:
        # Start the server as a non-blocking subprocess
        server_process = subprocess.Popen(command)
        print(f"vLLM server for {model_name} started with PID: {server_process.pid}. Waiting for it to become healthy...")

        # Health check loop
        max_retries = 30 # Try for 30 * 5 = 150 seconds
        retries = 0
        health_check_url = f"http://{host}:{port}/health" # Standard vLLM health endpoint

        while retries < max_retries:
            try:
                response = requests.get(health_check_url, timeout=5)
                if response.status_code == 200:
                    print(f"vLLM server for {model_name} is healthy on port {port}.")
                    return server_process
                else:
                    print(f"Health check failed with status {response.status_code}. Retrying...")
            except requests.exceptions.ConnectionError:
                print(f"Connection refused on port {port}. Server might still be starting. Retrying...")
            except requests.exceptions.RequestException as e:
                print(f"Health check request failed: {e}. Retrying...")
            
            time.sleep(5) # Wait 5 seconds before retrying
            retries += 1
        
        print(f"vLLM server for {model_name} did not become healthy after {max_retries * 5} seconds. Terminating process.")
        server_process.terminate()
        server_process.wait()
        return None

    except Exception as e:
        print(f"Failed to start vLLM server for {model_name}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a vLLM API server for the reward model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the HuggingFace model for the reward server.")
    parser.add_argument("--host", type=str, default="localhost", help="Host address for the server.")
    parser.add_argument("--port", type=int, default=8001, help="Port for the server.")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization method (e.g., 'fp8', 'awq').")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="GPU memory utilization limit.")
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum model sequence length.")

    args = parser.parse_args()

    server_process = start_vllm_server(
        model_name=args.model_name,
        host=args.host,
        port=args.port,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )

    if server_process:
        print(f"Reward model server started successfully. PID: {server_process.pid}")
        print(f"To stop the server, manually kill the process with PID {server_process.pid} or Ctrl+C if this script is kept running.")
        try:
            # Keep the main script alive, so Ctrl+C can be used to kill the server
            # Or, this script can exit, and the server runs in the background.
            # For interactive use, keeping it alive is often better.
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Terminating reward model server...")
        finally:
            if server_process.poll() is None: # Check if process is still running
                server_process.terminate()
                server_process.wait()
                print("Reward model server terminated.")
    else:
        print("Failed to start reward model server.")
