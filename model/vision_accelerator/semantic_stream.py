import torch
import torch.nn.functional as F


class SemanticStreamGate:
    def __init__(
        self,
        refresh_interval: int = 4,
        skip_patch_threshold: float = 0.01,
    ):
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be >= 1")
        if skip_patch_threshold < 0:
            raise ValueError("skip_patch_threshold must be >= 0")
        self.refresh_interval = refresh_interval
        self.skip_patch_threshold = skip_patch_threshold
        self.anchor_feature = None
        self.frame_idx = 0
        self.stats = {
            "input_frames": 0,
            "kept_frames": 0,
            "skipped_frames": 0,
            "input_tokens": 0,
            "written_tokens": 0,
        }
        self.recent_decisions = []

    def reset(self):
        self.anchor_feature = None
        self.frame_idx = 0
        for key in self.stats:
            self.stats[key] = 0
        self.recent_decisions.clear()

    @staticmethod
    def _frame_signature(selected_video_feature: torch.Tensor) -> torch.Tensor:
        signature = selected_video_feature.mean(dim=1)
        return F.normalize(signature.float(), dim=-1)

    def _should_keep(self, signature: torch.Tensor):
        if self.anchor_feature is None:
            return True, 0.0, "reference"
        similarity = F.cosine_similarity(signature, self.anchor_feature, dim=-1)
        drift = float((1.0 - similarity).mean().item())
        forced_refresh = (self.frame_idx % self.refresh_interval) == 0
        if forced_refresh:
            return True, drift, "refresh"
        if drift <= self.skip_patch_threshold:
            return False, drift, "skip"
        return True, drift, "drift_keep"

    def select_indices_from_signatures(self, signatures: torch.Tensor, token_count: int) -> torch.Tensor:
        keep_indices = []
        for local_idx in range(signatures.shape[0]):
            keep, drift, decision = self._should_keep(signatures[local_idx : local_idx + 1])
            self.stats["input_frames"] += 1
            self.stats["input_tokens"] += token_count
            if keep:
                keep_indices.append(local_idx)
                self.anchor_feature = signatures[local_idx : local_idx + 1].detach()
                self.stats["kept_frames"] += 1
                self.stats["written_tokens"] += token_count
            else:
                self.stats["skipped_frames"] += 1
            self.recent_decisions.append(
                {
                    "frame_idx": self.frame_idx,
                    "decision": decision,
                    "drift": drift,
                    "written_tokens": token_count if keep else 0,
                }
            )
            self.frame_idx += 1

        return torch.tensor(keep_indices, device=signatures.device, dtype=torch.long)

    @torch.no_grad()
    def __call__(
        self,
        video_features: torch.Tensor,
        selected_video_feature: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = int(kwargs.get("batch_size", 1))
        if batch_size != 1:
            return video_features

        signatures = self._frame_signature(selected_video_feature)
        token_count = int(video_features.shape[1]) if video_features.ndim == 3 else 0
        keep_tensor = self.select_indices_from_signatures(signatures, token_count)
        if keep_tensor.numel() == 0:
            return video_features[:0]
        return video_features.index_select(0, keep_tensor)
