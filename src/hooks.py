"""Hook attachment for TorchRec EmbeddingBagCollection modules.

Attaches forward pre-hooks and backward hooks to the underlying
nn.EmbeddingBag modules inside an EBC to capture embedding-update
statistics without modifying TorchRec internals.
"""
import warnings
import torch
import torch.nn as nn


class EBCHooks:
    """Captures per-step embedding statistics via PyTorch hooks.

    Hooks are attached to each nn.EmbeddingBag inside the EBC's
    ``embedding_bags`` ModuleDict. Statistics are buffered internally
    and harvested via ``collect()``.

    Captured stats per table per step:
        grad_norm:    L2 norm of gradient on embedding output
        grad_max:     max absolute gradient value
        n_accessed:   number of unique rows looked up in forward pass
        accessed_ids: list of unique row indices accessed
    """

    def __init__(self, ebc, table_names: list[str] | None = None):
        self._ebc = ebc
        self._table_names = table_names or list(ebc.embedding_bags.keys())
        self._handles: list = []
        self._forward_buffers: dict[str, dict] = {}
        self._backward_buffers: dict[str, dict] = {}
        self._attached = False

    def attach(self) -> None:
        if self._attached:
            return
        for name in self._table_names:
            eb = self._ebc.embedding_bags[name]
            self._forward_buffers[name] = {}
            self._backward_buffers[name] = {}

            fwd_handle = eb.register_forward_pre_hook(
                self._make_forward_hook(name), with_kwargs=True
            )
            self._handles.append(fwd_handle)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bwd_handle = eb.register_full_backward_hook(
                    self._make_backward_hook(name)
                )
            self._handles.append(bwd_handle)

        self._attached = True

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._forward_buffers.clear()
        self._backward_buffers.clear()
        self._attached = False

    def collect(self) -> dict[str, dict]:
        """Harvest stats from the last forward+backward pass.

        Returns {table_name: {stat_name: value}}.
        Resets internal buffers for next step.
        """
        result = {}
        for name in self._table_names:
            fwd = self._forward_buffers.get(name, {})
            bwd = self._backward_buffers.get(name, {})
            result[name] = {**fwd, **bwd}
            self._forward_buffers[name] = {}
            self._backward_buffers[name] = {}
        return result

    def _make_forward_hook(self, table_name: str):
        def hook(module, args, kwargs):
            indices = None
            if args and len(args) > 0:
                indices = args[0]
            elif kwargs and "input" in kwargs:
                indices = kwargs["input"]
            if indices is not None and isinstance(indices, torch.Tensor):
                unique = indices.unique()
                self._forward_buffers[table_name] = {
                    "n_accessed": len(unique),
                    "accessed_ids": unique.detach().cpu().tolist(),
                }
        return hook

    def _make_backward_hook(self, table_name: str):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                g = grad_output[0]
                stats = {
                    "grad_norm": g.norm().item(),
                    "grad_max": g.abs().max().item(),
                }

                # Per-row gradient analysis for distribution detectors
                if g.dim() == 2:
                    row_norms = g.norm(dim=1)
                    if row_norms.numel() > 1:
                        mean = row_norms.mean()
                        var = row_norms.var()
                        if var > 1e-12:
                            # Excess kurtosis: m4/m2^2 - 3
                            m4 = ((row_norms - mean) ** 4).mean()
                            stats["grad_kurtosis"] = (m4 / (var ** 2) - 3).item()
                        else:
                            stats["grad_kurtosis"] = 0.0
                        median = row_norms.median()
                        if median > 1e-12:
                            stats["grad_concentration"] = (row_norms.max() / median).item()
                        else:
                            stats["grad_concentration"] = 0.0

                self._backward_buffers[table_name] = stats
        return hook

    @property
    def attached(self) -> bool:
        return self._attached
