"""Device selection helpers that prefer Ascend NPU, then CUDA, then CPU.

On Huawei ModelArts the accelerator is an Ascend NPU exposed through
`torch_npu`. Importing `torch_npu` registers the ``npu`` backend on ``torch``
and the ``npu:0`` device string. We try that first, fall back to CUDA on a
GPU box, and finally CPU so the same code runs locally and on the cluster.
"""
import torch

# Importing torch_npu (if installed) registers the NPU backend on torch.
try:  # pragma: no cover - depends on runtime environment
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except Exception:  # torch_npu not installed (e.g. local CPU/GPU machine)
    _HAS_NPU = False


def npu_available():
    """Return True when a usable Ascend NPU is visible."""
    if not _HAS_NPU:
        return False
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def get_device():
    """Return the best available torch.device: npu -> cuda -> cpu."""
    if npu_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_device_seed(seed, device):
    """Seed the active accelerator backend for reproducibility."""
    dev_type = device.type if isinstance(device, torch.device) else str(device)
    if dev_type == "npu" and npu_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
    elif dev_type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    """Block until the accelerator finishes queued work (for timing)."""
    dev_type = device.type if isinstance(device, torch.device) else str(device)
    if dev_type == "npu" and npu_available():
        torch.npu.synchronize()
    elif dev_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
