"""Start the fixed multi-node experiment without command-line configuration."""

from nbiot_transport.runtime.sender import run as run_sender
from common.paths import PROJECT_ROOT

if __name__ == "__main__":
    run_sender(PROJECT_ROOT / "src" / "nbiot_transport" / "multi_node.json")
