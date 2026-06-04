import torch
import torch.nn.functional as F


class SemanticStreamGate:
    def __init__(
        self,
        refresh_interval: int = 4,
        skip_patch_threshold: float = 0.01,
        recency_keep_frames: int = 0,
        recency_updates_anchor: bool = False,
        coverage_interval: int = 0,
        coverage_updates_anchor: bool = False,
    ):
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be >= 1")
        if skip_patch_threshold < 0:
            raise ValueError("skip_patch_threshold must be >= 0")
        if recency_keep_frames < 0:
            raise ValueError("recency_keep_frames must be >= 0")
        if coverage_interval < 0:
            raise ValueError("coverage_interval must be >= 0")
        self.refresh_interval = refresh_interval
        self.skip_patch_threshold = skip_patch_threshold
        self.recency_keep_frames = recency_keep_frames
        self.recency_updates_anchor = recency_updates_anchor
        self.coverage_interval = coverage_interval
        self.coverage_updates_anchor = coverage_updates_anchor
        self.anchor_feature = None
        self.frame_idx = 0
        self._recency_start_idx = None
        self._recency_end_idx = None
        self.stats = {
            "input_frames": 0,
            "kept_frames": 0,
            "skipped_frames": 0,
            "input_tokens": 0,
            "written_tokens": 0,
            "recency_kept_frames": 0,
            "coverage_kept_frames": 0,
        }
        self.recent_decisions = []

    def reset(self):
        self.anchor_feature = None
        self.frame_idx = 0
        self._recency_start_idx = None
        self._recency_end_idx = None
        for key in self.stats:
            self.stats[key] = 0
        self.recent_decisions.clear()

    def set_recency_window(self, start_frame_idx: int, end_frame_idx: int):
        if self.recency_keep_frames <= 0:
            self._recency_start_idx = None
            self._recency_end_idx = None
            return
        self._recency_start_idx = max(int(start_frame_idx), int(end_frame_idx) - self.recency_keep_frames)
        self._recency_end_idx = int(end_frame_idx)

    @staticmethod
    def _frame_signature(selected_video_feature: torch.Tensor) -> torch.Tensor:
        signature = selected_video_feature.mean(dim=1)
        return F.normalize(signature.float(), dim=-1)

    def _should_keep(self, signature: torch.Tensor):
        if self.anchor_feature is None:
            return True, 0.0, "reference", True
        similarity = F.cosine_similarity(signature, self.anchor_feature, dim=-1)
        drift = float((1.0 - similarity).mean().item())
        forced_refresh = (self.frame_idx % self.refresh_interval) == 0
        if forced_refresh:
            return True, drift, "refresh", True
        if drift > self.skip_patch_threshold:
            return True, drift, "drift_keep", True
        if self.coverage_interval > 0 and (self.frame_idx % self.coverage_interval) == 0:
            return True, drift, "coverage_keep", self.coverage_updates_anchor
        if (
            self._recency_start_idx is not None
            and self._recency_start_idx <= self.frame_idx < self._recency_end_idx
        ):
            return True, drift, "recency_keep", self.recency_updates_anchor
        if drift <= self.skip_patch_threshold:
            return False, drift, "skip", False
        return True, drift, "drift_keep", True

    def select_indices_from_signatures(self, signatures: torch.Tensor, token_count: int) -> torch.Tensor:
        keep_indices = []
        for local_idx in range(signatures.shape[0]):
            keep, drift, decision, update_anchor = self._should_keep(signatures[local_idx : local_idx + 1])
            self.stats["input_frames"] += 1
            self.stats["input_tokens"] += token_count
            if keep:
                keep_indices.append(local_idx)
                if update_anchor:
                    self.anchor_feature = signatures[local_idx : local_idx + 1].detach()
                self.stats["kept_frames"] += 1
                self.stats["written_tokens"] += token_count
                if decision == "recency_keep":
                    self.stats["recency_kept_frames"] += 1
                if decision == "coverage_keep":
                    self.stats["coverage_kept_frames"] += 1
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
