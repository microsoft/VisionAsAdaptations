import re
import numpy as np
import torch
import torch.nn as nn
from unilora.layer import Linear as UniLoRALinear


def _generate_unilora_index(total_length: int, theta_d_length: int, proj_seed: int) -> torch.Tensor:
    """Replicate the UniLoRA index generation logic.

    Uniformly distributes indices across the theta_d bank using the seed and shuffles.
    Returns a 1D LongTensor of length total_length.
    """
    base_count = total_length // theta_d_length
    remaining = total_length % theta_d_length
    rng = np.random.default_rng(proj_seed)
    data = np.repeat(np.arange(theta_d_length), base_count)
    if remaining > 0:
        extras = rng.choice(theta_d_length, size=remaining, replace=False)
        data = np.concatenate([data, extras])
    rng.shuffle(data)
    return torch.tensor(data, dtype=torch.long)


def inject_unilora(
    model: nn.Module,
    target_regex: str,
    r: int,
    theta_d_length: int,
    proj_seed: int,
    dropout: float = 0.0,
    init_theta_d_bound: float = 0.02,
    adapter_name: str = "default",
):
    """Replace Linear layers with UniLoRA Linear using a shared theta_d vector.

    Returns list of replaced module names.
    """
    pat = re.compile(target_regex)
    replaced, modules = [], []

    # Shared trainable vector
    sample_param = next(model.parameters())
    device, dtype = sample_param.device, sample_param.dtype
    theta_d_param = nn.Parameter(torch.empty(theta_d_length, device=device, dtype=dtype))
    nn.init.uniform_(theta_d_param, -init_theta_d_bound, init_theta_d_bound)
    shared_theta = nn.ParameterDict({adapter_name: theta_d_param})
    setattr(model, "unilora_theta_d", shared_theta)
    model.unilora_meta = {
        "adapter_name": adapter_name,
        "theta_d_length": theta_d_length,
        "proj_seed": proj_seed,
        "r": r,
        "dropout": dropout,
        "init_theta_d_bound": init_theta_d_bound,
    }

    # Replace modules
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and pat.search(name):
            parent_path = name.split(".")[:-1]
            child_name = name.split(".")[-1]
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            new_layer = UniLoRALinear(
                base_layer=module,
                unilora_theta_d=shared_theta,
                adapter_name=adapter_name,
                r=r,
                theta_d_length=theta_d_length,
                unilora_dropout=dropout,
            )
            setattr(parent, child_name, new_layer)
            replaced.append(name)
            modules.append(new_layer)

    if not modules:
        return replaced

    # Deterministic indices shared across all layers
    total_needed = sum(
        m.unilora_indices_A[adapter_name].numel() + m.unilora_indices_B[adapter_name].numel()
        for m in modules
    )
    all_elements = _generate_unilora_index(total_needed, theta_d_length, proj_seed).to(device)

    pointer = 0
    for m in modules:
        numel_a = m.unilora_indices_A[adapter_name].numel()
        chunk_a = all_elements[pointer: pointer + numel_a].view_as(m.unilora_indices_A[adapter_name]).to(device)
        m.unilora_indices_A[adapter_name] = chunk_a
        pointer += numel_a

        numel_b = m.unilora_indices_B[adapter_name].numel()
        chunk_b = all_elements[pointer: pointer + numel_b].view_as(m.unilora_indices_B[adapter_name]).to(device)
        m.unilora_indices_B[adapter_name] = chunk_b
        pointer += numel_b

    assert pointer == len(all_elements)

    counts = torch.bincount(all_elements, minlength=theta_d_length).to(device)
    sqrt_counts = 1.0 / torch.sqrt(counts.float().clamp_min(1.0))

    for m in modules:
        idx_a = m.unilora_indices_A[adapter_name].long()
        idx_b = m.unilora_indices_B[adapter_name].long()
        scale_a = sqrt_counts[idx_a].to(m.get_base_layer().weight.device, m.get_base_layer().weight.dtype)
        scale_b = sqrt_counts[idx_b].to(m.get_base_layer().weight.device, m.get_base_layer().weight.dtype)
        m.update_norm(adapter_name, scale_a, scale_b)

    return replaced


def collect_unilora_state_dict(model: nn.Module, adapter_name: str = "default") -> dict:
    sd = {}
    if hasattr(model, "unilora_theta_d"):
        sd["unilora_theta_d"] = model.unilora_theta_d[adapter_name].detach().cpu()
    # Omit meta to keep safetensors pure-tensor
    return sd


def load_unilora_state_dict(model: nn.Module, state_dict: dict, adapter_name: str = "default"):
    if not hasattr(model, "unilora_theta_d"):
        raise RuntimeError("UniLoRA parameters not found on model; inject UniLoRA first.")
    theta = model.unilora_theta_d[adapter_name]
    if "unilora_theta_d" not in state_dict:
        raise KeyError("State dict missing 'unilora_theta_d'.")
    new_theta = state_dict["unilora_theta_d"].to(theta.device, theta.dtype)
    if new_theta.numel() != theta.numel():
        raise ValueError(
            f"theta_d length mismatch: checkpoint {new_theta.numel()} vs model {theta.numel()}"
        )
    theta.data.copy_(new_theta)


    # print(theta.max(), theta.min())
    # hist_path = "theta_hist.png"
    # theta_cpu = theta.detach().cpu().view(-1)
    # try:
    #     import matplotlib.pyplot as plt

    #     plt.figure(figsize=(6, 4))
    #     plt.hist(theta_cpu.float().numpy(), bins=150, color="steelblue", edgecolor="black")
    #     plt.title("theta_d distribution")
    #     plt.xlabel("Value")
    #     plt.ylabel("Frequency")
    #     plt.tight_layout()
    #     plt.savefig(hist_path)
    #     plt.close()
    # except Exception as exc:
    #     print(f"Failed to save theta histogram: {exc}")


def build_lora_regex(
    total_blocks: int = 32,
    last_k: int | None = 8,
    preset: str = "qkvo",  # 'qkv' | 'qkvo' | 'qkvo_mlp'
) -> str:
    """Build regex pattern to match nn.Linear modules in QwenImage transformer."""
    if last_k is None or last_k >= total_blocks:
        block_pat = r"\d+"
    else:
        start = max(0, total_blocks - last_k)
        block_pat = "(" + "|".join(str(i) for i in range(start, total_blocks)) + ")"

    parts = []
    # QwenImageTransformer2DModel uses ModuleList name `transformer_blocks` (not `blocks`).
    parts.append(rf"transformer_blocks\.{block_pat}\.attn\.(?:to_q|to_k|to_v)$")

    if preset in ("qkvo", "qkvo_mlp"):
        parts.append(rf"transformer_blocks\.{block_pat}\.attn\.to_out\.0$")

    if preset == "qkvo_mlp":
        parts.append(rf"transformer_blocks\.{block_pat}\.(?:ff|ffn)\.net\.(?:0\.proj|2)$")
        parts.append(
            rf"transformer_blocks\.{block_pat}\.(?:img_mlp|txt_mlp)\.net\.(?:0\.proj|2)$"
        )

    return "(?:" + "|".join(parts) + ")"


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

