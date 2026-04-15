import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ._buffer_dict import BufferDict

class UniLoRALayer(nn.Module):
    # List all names of layers that may contain adapter weights
    # unilora_theta_d is a shared parameter.
    #But it is referenced within individual layers.
    adapter_layer_names = ("unilora_theta_d",)

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})


        self.unilora_indices_A = BufferDict({}, persistent=True)
        self.unilora_indices_B = BufferDict({}, persistent=True)


        self.unilora_scales_A = BufferDict({}, persistent=True)
        self.unilora_scales_B = BufferDict({}, persistent=True)

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

        # Track active adapter
        self._active_adapters: List[str] = []

    @property
    def active_adapters(self) -> List[str]:
        return self._active_adapters

    def set_adapter(self, adapters):
        if isinstance(adapters, str):
            adapters = [adapters]
        self._active_adapters = list(adapters)

    @property
    def disable_adapters(self) -> bool:
        return self._disable_adapters

    @disable_adapters.setter
    def disable_adapters(self, flag: bool):
        self._disable_adapters = bool(flag)

    def get_base_layer(self):
        return self.base_layer

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str):
        base_layer = self.get_base_layer()
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        if adapter_name in self.unilora_indices_A.keys():
            self.unilora_indices_A[adapter_name] = self.unilora_indices_A[adapter_name].to(device)
        if adapter_name in self.unilora_indices_B.keys():
            self.unilora_indices_B[adapter_name] = self.unilora_indices_B[adapter_name].to(device)
        if adapter_name in self.unilora_scales_A.keys():
            self.unilora_scales_A[adapter_name] = self.unilora_scales_A[adapter_name].to(device=device, dtype=dtype)
        if adapter_name in self.unilora_scales_B.keys():
            self.unilora_scales_B[adapter_name] = self.unilora_scales_B[adapter_name].to(device=device, dtype=dtype)

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_theta_d,
        r: int,
        theta_d_length: int,
        unilora_dropout: float = 0.0,
    ):
        if r <= 0:
            raise ValueError(f"`r` {r} should be a positive integer value")

        self.r[adapter_name] = r

        if unilora_dropout > 0.0:
            unilora_dropout_layer = nn.Dropout(p=unilora_dropout)
        else:
            unilora_dropout_layer = nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: unilora_dropout_layer}))

        self.unilora_theta_d = unilora_theta_d

        # Initialize indices and move to device
        self.reset_unilora_parameters(adapter_name, theta_d_length)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_unilora_parameters(self, adapter_name, theta_d_length):
        """
        Initializes the indices (pointers to theta_d) randomly.
        Renamed from reset_unilora_logits to ensure clarity.
        """
        if adapter_name in self.unilora_theta_d.keys():
            # Generate random indices pointing to the vector bank
            indices_A = torch.randint(0, theta_d_length, (self.r[adapter_name], self.in_features), dtype=torch.long)
            indices_B = torch.randint(0, theta_d_length, (self.out_features, self.r[adapter_name]), dtype=torch.long) 

            self.unilora_indices_A[adapter_name] = indices_A
            self.unilora_indices_B[adapter_name] = indices_B

    def update_norm(
        self,
        adapter_name: str,
        unilora_scales_A,
        unilora_scales_B,
    ):   
        """
        Updates the scaling factors. 
        Note: Method name kept as update_norm for compatibility if called externally, 
        but arguments updated to 'scales'.
        """
        if adapter_name in self.unilora_theta_d.keys():
            base_layer = self.get_base_layer()
            target_device = base_layer.weight.device
            target_dtype = base_layer.weight.dtype

            self.unilora_scales_A[adapter_name] = unilora_scales_A.to(
                device=target_device, dtype=target_dtype
            )
            self.unilora_scales_B[adapter_name] = unilora_scales_B.to(
                device=target_device, dtype=target_dtype
            )

class Linear(nn.Linear, UniLoRALayer):
    # UniLoRA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        unilora_theta_d,
        adapter_name: str,
        r: int,
        theta_d_length: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        UniLoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.set_adapter([adapter_name])
        self.update_layer(
            adapter_name, unilora_theta_d, r, theta_d_length, unilora_dropout,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def _get_lora_matrices(self, adapter, cast_to_fp32=False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Changed: Accessing the renamed buffers
        unilora_indices_A = self.unilora_indices_A[adapter] 
        unilora_indices_B = self.unilora_indices_B[adapter] 

        unilora_theta_d = self.unilora_theta_d[adapter].to(unilora_indices_A.device)

        if cast_to_fp32:
            unilora_theta_d = unilora_theta_d.float()

        # Changed: Replaced 'logits' with 'indices' and 'norm' with 'scales'
        # Core Logic: Retrieve vector from bank using indices, then scale it.
        # A_matrix = Bank[Indices] * Scale
        A = unilora_theta_d[unilora_indices_A.long()] * self.unilora_scales_A[adapter]
        B = unilora_theta_d[unilora_indices_B.long()] * self.unilora_scales_B[adapter]
        # Cast back if necessary (handled implicitly by torch usually, but good to be explicit if needed)
        if cast_to_fp32:
             A = A.float()
             B = B.float()

        return A, B

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.unilora_indices_A.keys():
                    continue

                A, B = self._get_lora_matrices(active_adapter)

                x = x.to(self.unilora_theta_d[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]

                # Standard LoRA calculation: x @ A @ B
                result = result + F.linear(F.linear(dropout(x), A), B)

        result = result.to(previous_dtype)
        return result
