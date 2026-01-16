import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


def run_command(command, cwd=None, check=True, capture_output=False):
    """Runs a command and prints its output."""
    print(f"Running command: {' '.join(command)}")
    try:
        # For commands like docker login, we want to see the output live
        if not capture_output:
            process = subprocess.run(
                command,
                cwd=cwd,
                check=check,
                encoding="utf-8",
                errors="replace",
            )
        else:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=cwd,
                check=check,
                encoding="utf-8",
                errors="replace",
            )
            if process.stdout:
                print(process.stdout)
            if process.stderr:
                print("Error output:")
                print(process.stderr)
        return process
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        # Captured output is in e.stdout/e.stderr if capture_output=True
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def _is_port_free(port: int, host: str = "0.0.0.0") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _choose_minio_ports() -> tuple[int, int]:
    candidates = [(9000, 9001), (19000, 19001), (29000, 29001), (39000, 39001)]
    for api_port, ui_port in candidates:
        if _is_port_free(api_port) and _is_port_free(ui_port):
            return api_port, ui_port
    raise RuntimeError(
        "No free MinIO port pair found (tried 9000/9001, 19000/19001, 29000/29001, 39000/39001)"
    )


def wait_ready(url: str, timeout_s: int = 1800, interval_s: int = 5):
    print(f"Waiting for {url} to be ready...")
    start = time.time()
    last_err = None
    while time.time() - start < timeout_s:
        try:
            # Use standard library to avoid external dependency
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    print(f"{url} is ready.")
                    return True
        except Exception as e:
            last_err = e
        time.sleep(interval_s)
    raise RuntimeError(f"Timed out waiting for ready: {url} (last_err={last_err})")


def get_data_root() -> Path:
    """
    Returns a writable path for storing data.
    Prefers /ephemeral if it exists and is writable, otherwise uses the user's home directory.
    """
    ephemeral_path = Path("/ephemeral")
    if ephemeral_path.exists() and os.access(ephemeral_path, os.W_OK):
        data_root = ephemeral_path / "vss-data"
        print(f"Using ephemeral storage: {data_root}")
    else:
        data_root = Path.home() / ".vss-data"
        print(f"Using home directory storage: {data_root}")

    data_root.mkdir(parents=True, exist_ok=True)
    return data_root.resolve()


def main():
    """
    Configures and deploys the VSS instance.
    """
    try:
        # 1. Prerequisites
        print("--- 1. Setting up prerequisites ---")
        ngc_api_key = os.environ.get("NGC_API_KEY")
        if not ngc_api_key or ngc_api_key == "***":
            print("ERROR: Please set the NGC_API_KEY environment variable.")
            return

        try:
            vss_repo_dir = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except Exception:
            vss_repo_dir = str(
                Path(__file__).resolve().parent.parent.parent
            )  # Assuming script is in deploy/scripts

        os.environ["VSS_REPO_DIR"] = vss_repo_dir
        compose_dir = str(Path(vss_repo_dir) / "deploy" / "docker" / "local_deployment_single_gpu")
        os.environ["VSS_COMPOSE_DIR"] = compose_dir

        print(f"VSS_REPO_DIR={vss_repo_dir}")
        print(f"VSS_COMPOSE_DIR={compose_dir}")

        # 2. Configure deployment settings
        print("\n--- 2. Configuring deployment settings ---")
        data_root = get_data_root()

        asset_dir = (data_root / "assets").resolve()
        milvus_dir = (data_root / "milvus").resolve()
        nim_cache_dir = (data_root / "nim-cache").resolve()
        ngc_model_cache_dir = (data_root / "ngc-model-cache").resolve()
        via_logs_dir = (data_root / "via-logs").resolve()
        trt_engine_dir = (data_root / "trt-engines").resolve()
        via_tmp_dir = (data_root / "via-tmp").resolve()

        for p in [
            asset_dir,
            milvus_dir,
            nim_cache_dir,
            ngc_model_cache_dir,
            via_logs_dir,
            trt_engine_dir,
            via_tmp_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(p, 0o777)
            except Exception as e:
                print(f"Warning: Could not set permissions on {p}: {e}")

        # Set Docker data root only if using ephemeral storage
        if "ephemeral" in str(data_root):
            os.environ["DOCKER_DATA_ROOT"] = "/ephemeral/docker"

        os.environ.setdefault("BACKEND_PORT", "8100")
        os.environ.setdefault("FRONTEND_PORT", "9100")

        if "MINIO_PORT" not in os.environ or "MINIO_WEBUI_PORT" not in os.environ:
            api_port, ui_port = _choose_minio_ports()
            os.environ.setdefault("MINIO_PORT", str(api_port))
            os.environ.setdefault("MINIO_WEBUI_PORT", str(ui_port))

        os.environ.setdefault("GRAPH_DB_USERNAME", "neo4j")
        os.environ.setdefault("GRAPH_DB_PASSWORD", "password")
        os.environ.setdefault("ARANGO_DB_USERNAME", "root")
        os.environ.setdefault("ARANGO_DB_PASSWORD", "password")

        os.environ["ASSET_STORAGE_DIR"] = str(asset_dir)
        os.environ["NGC_MODEL_CACHE"] = str(ngc_model_cache_dir)
        os.environ["VIA_LOG_DIR"] = str(via_logs_dir)
        os.environ["TRT_ENGINE_PATH"] = str(trt_engine_dir)
        os.environ["LOCAL_NIM_CACHE"] = str(nim_cache_dir)
        os.environ["VIA_TMP_DIR"] = str(via_tmp_dir)

        compose_yaml = (Path(compose_dir) / "compose.yaml").resolve()
        ca_rag_cfg = (Path(compose_dir) / "config.yaml").resolve()
        guardrails_dir_path = (Path(compose_dir) / "guardrails").resolve()

        if not compose_yaml.exists():
            raise FileNotFoundError(f"compose.yaml not found: {compose_yaml}")
        if not ca_rag_cfg.exists():
            raise FileNotFoundError(f"config.yaml not found: {ca_rag_cfg}")
        if not guardrails_dir_path.exists():
            raise FileNotFoundError(f"guardrails dir not found: {guardrails_dir_path}")

        os.environ["CA_RAG_CONFIG"] = str(ca_rag_cfg)
        os.environ["GUARDRAILS_CONFIG"] = str(guardrails_dir_path)
        os.environ["MILVUS_DATA_DIR"] = str(milvus_dir)
        os.environ["VLM_MODEL_TO_USE"] = "cosmos-reason1"
        os.environ.setdefault("MODEL_PATH", "git:https://huggingface.co/nvidia/Cosmos-Reason1-7B")
        os.environ["NUM_GPUS"] = "1"
        os.environ["NIM_GPU_DEVICE"] = "0"
        os.environ.setdefault("TRT_LLM_MEM_USAGE_FRACTION", "0.6")
        os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", "0.75")
        os.environ.setdefault("VLM_BATCH_SIZE", "1")
        os.environ.setdefault("VLM_MAX_MODEL_LEN", "8192")
        os.environ.setdefault("DISABLE_CV_PIPELINE", "true")
        os.environ.setdefault("ENABLE_AUDIO", "false")
        os.environ.setdefault("DISABLE_GUARDRAILS", "true")
        os.environ.setdefault("MAX_ASSET_STORAGE_SIZE_GB", "80")

        print("Configuration complete.")

        # 3. Log in to NGC
        print("\n--- 3. Logging in to NGC Docker registry ---")
        login_command = ["docker", "login", "nvcr.io", "-u", "$oauthtoken", "--password-stdin"]
        # Pipe the key directly to the command's stdin
        subprocess.run(
            login_command,
            input=ngc_api_key,
            text=True,
            check=True,
        )
        print("Docker login successful.")

        # 4. Start the stack
        print("\n--- 4. Starting the VSS stack with Docker Compose ---")
        run_command(
            ["docker", "compose", "up", "-d", "--quiet-pull"], cwd=compose_dir, capture_output=True
        )
        run_command(["docker", "compose", "ps"], cwd=compose_dir, capture_output=True)
        print("`docker compose up` command issued.")

        # 5. Wait for services to become ready
        print("\n--- 5. Waiting for services to become ready ---")
        backend_port = os.environ["BACKEND_PORT"]
        checks = [
            ("LLM NIM", "http://localhost:8000/v1/health/ready"),
            ("Embedding NIM", "http://localhost:8006/v1/health/ready"),
            ("Reranker NIM", "http://localhost:8005/v1/health/ready"),
            ("VSS Backend", f"http://localhost:{backend_port}/health/ready"),
        ]

        for name, url in checks:
            wait_ready(url)

        print("\n--- All services are ready! ---")
        print(f"Backend URL: http://localhost:{backend_port}")
        print(f"Frontend URL: http://localhost:{os.environ['FRONTEND_PORT']}")
        print("\nTo view logs, run:")
        print(f"cd {compose_dir} && docker compose logs -f")
        print("\nTo shut down the stack, run:")
        print(f"cd {compose_dir} && docker compose down")

    except Exception as e:
        print(f"\nAn error occurred during VSS configuration: {e}")


if __name__ == "__main__":
    main()
