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
        selection_policy: str = "threshold",
        budget_window_size: int = 0,
        budget_keep_per_window: int = 1,
    ):
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be >= 1")
        if skip_patch_threshold < 0:
            raise ValueError("skip_patch_threshold must be >= 0")
        if recency_keep_frames < 0:
            raise ValueError("recency_keep_frames must be >= 0")
        if coverage_interval < 0:
            raise ValueError("coverage_interval must be >= 0")
        if selection_policy not in {"threshold", "budget_topk", "periodic"}:
            raise ValueError(
                "selection_policy must be 'threshold', 'budget_topk', or 'periodic'"
            )
        if budget_window_size < 0:
            raise ValueError("budget_window_size must be >= 0")
        if budget_keep_per_window < 1:
            raise ValueError("budget_keep_per_window must be >= 1")
        self.refresh_interval = refresh_interval
        self.skip_patch_threshold = skip_patch_threshold
        self.recency_keep_frames = recency_keep_frames
        self.recency_updates_anchor = recency_updates_anchor
        self.coverage_interval = coverage_interval
        self.coverage_updates_anchor = coverage_updates_anchor
        self.selection_policy = selection_policy
        self.budget_window_size = budget_window_size
        self.budget_keep_per_window = budget_keep_per_window
        self.anchor_feature = None
        self.frame_idx = 0
        self._recency_start_idx = None
        self._recency_end_idx = None
        self.stats = {
            "input_frames": 0,
            "kept_frames": 0,
            "skipped_frames": 0,
            "candidate_frames": 0,
            "preprocessed_frames": 0,
            "input_tokens": 0,
            "written_tokens": 0,
            "recency_kept_frames": 0,
            "coverage_kept_frames": 0,
            "budget_kept_frames": 0,
            "proposal_sec": 0.0,
            "preprocess_sec": 0.0,
            "embedding_sec": 0.0,
            "verification_sec": 0.0,
            "vit_encoder_sec": 0.0,
            "context_write_sec": 0.0,
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
        drift = max(0.0, float((1.0 - similarity).mean().item()))
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

    def _in_recency_window(self, frame_idx: int):
        return (
            self._recency_start_idx is not None
            and self._recency_start_idx <= frame_idx < self._recency_end_idx
        )

    def _record_decision(self, local_idx, keep, drift, decision, update_anchor, signatures, token_count):
        self.stats["input_frames"] += 1
        self.stats["input_tokens"] += token_count
        if keep:
            if update_anchor:
                self.anchor_feature = signatures[local_idx : local_idx + 1].detach()
            self.stats["kept_frames"] += 1
            self.stats["written_tokens"] += token_count
            if decision == "recency_keep":
                self.stats["recency_kept_frames"] += 1
            if decision == "coverage_keep":
                self.stats["coverage_kept_frames"] += 1
            if decision == "budget_keep":
                self.stats["budget_kept_frames"] += 1
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

    def _record_skip_without_signature(self, drift, decision, token_count):
        self.stats["input_frames"] += 1
        self.stats["input_tokens"] += token_count
        self.stats["skipped_frames"] += 1
        self.recent_decisions.append(
            {
                "frame_idx": self.frame_idx,
                "decision": decision,
                "drift": drift,
                "written_tokens": 0,
            }
        )
        self.frame_idx += 1

    def select_periodic_indices(
        self,
        total_frames: int,
        token_count: int,
        device=None,
    ) -> torch.Tensor:
        keep_indices = []
        for local_idx in range(total_frames):
            global_idx = self.frame_idx
            if global_idx % self.refresh_interval == 0:
                keep = True
                decision = "reference" if global_idx == 0 else "refresh"
            elif self._in_recency_window(global_idx):
                keep = True
                decision = "recency_keep"
            else:
                keep = False
                decision = "periodic_skip"

            self.stats["input_frames"] += 1
            self.stats["input_tokens"] += token_count
            if keep:
                keep_indices.append(local_idx)
                self.stats["kept_frames"] += 1
                self.stats["written_tokens"] += token_count
                if decision == "recency_keep":
                    self.stats["recency_kept_frames"] += 1
            else:
                self.stats["skipped_frames"] += 1
            self.recent_decisions.append(
                {
                    "frame_idx": global_idx,
                    "decision": decision,
                    "drift": 0.0,
                    "written_tokens": token_count if keep else 0,
                }
            )
            self.frame_idx += 1

        return torch.tensor(keep_indices, device=device, dtype=torch.long)

    def select_indices_from_signatures(self, signatures: torch.Tensor, token_count: int) -> torch.Tensor:
        if self.selection_policy == "periodic":
            return self.select_periodic_indices(
                total_frames=int(signatures.shape[0]),
                token_count=token_count,
                device=signatures.device,
            )
        if self.selection_policy == "budget_topk":
            return self.select_indices_from_window_signatures(signatures, token_count)

        keep_indices = []
        for local_idx in range(signatures.shape[0]):
            keep, drift, decision, update_anchor = self._should_keep(signatures[local_idx : local_idx + 1])
            if keep:
                keep_indices.append(local_idx)
            self._record_decision(local_idx, keep, drift, decision, update_anchor, signatures, token_count)

        return torch.tensor(keep_indices, device=signatures.device, dtype=torch.long)

    def select_indices_from_candidate_signatures(
        self,
        signatures: torch.Tensor,
        candidate_indices: torch.Tensor,
        total_frames: int,
        token_count: int,
    ) -> torch.Tensor:
        if self.selection_policy != "budget_topk":
            raise ValueError("candidate signature selection is only supported for selection_policy='budget_topk'")
        if self.budget_window_size <= 0:
            raise ValueError("budget_window_size must be > 0 when selection_policy='budget_topk'")

        candidate_indices = candidate_indices.to(device=signatures.device, dtype=torch.long)
        candidate_positions = {int(idx.item()): pos for pos, idx in enumerate(candidate_indices)}
        keep = {}
        drifts = {}
        compare_anchor = self.anchor_feature
        if compare_anchor is None and signatures.shape[0] > 0:
            compare_anchor = signatures[0:1].detach()

        for local_idx, sig_pos in candidate_positions.items():
            global_idx = self.frame_idx + local_idx
            if compare_anchor is None:
                drift = 0.0
            else:
                similarity = F.cosine_similarity(signatures[sig_pos : sig_pos + 1], compare_anchor, dim=-1)
                drift = max(0.0, float((1.0 - similarity).mean().item()))
            drifts[local_idx] = drift

            if self.anchor_feature is None and local_idx == 0:
                keep[local_idx] = ("reference", True)
            elif (global_idx % self.refresh_interval) == 0:
                keep[local_idx] = ("refresh", True)
            elif self.coverage_interval > 0 and (global_idx % self.coverage_interval) == 0:
                keep[local_idx] = ("coverage_keep", self.coverage_updates_anchor)
            elif self._in_recency_window(global_idx):
                keep[local_idx] = ("recency_keep", self.recency_updates_anchor)

        candidates_by_window = {}
        for local_idx, drift in drifts.items():
            if local_idx in keep or drift <= self.skip_patch_threshold:
                continue
            global_idx = self.frame_idx + local_idx
            window_id = global_idx // self.budget_window_size
            candidates_by_window.setdefault(window_id, []).append((drift, local_idx))

        reserved_by_window = {}
        for local_idx, (decision, _) in keep.items():
            if decision == "recency_keep":
                continue
            global_idx = self.frame_idx + local_idx
            window_id = global_idx // self.budget_window_size
            reserved_by_window[window_id] = reserved_by_window.get(window_id, 0) + 1

        for window_id, candidates in candidates_by_window.items():
            candidates.sort(reverse=True)
            remaining_budget = max(
                0,
                self.budget_keep_per_window - reserved_by_window.get(window_id, 0),
            )
            for _, local_idx in candidates[:remaining_budget]:
                keep[local_idx] = ("budget_keep", True)

        keep_positions = []
        for local_idx in range(total_frames):
            sig_pos = candidate_positions.get(local_idx)
            if sig_pos is None:
                self._record_skip_without_signature(0.0, "prefilter_skip", token_count)
                continue
            drift = drifts[local_idx]
            if local_idx in keep:
                decision, update_anchor = keep[local_idx]
                keep_positions.append(sig_pos)
                self._record_decision(sig_pos, True, drift, decision, update_anchor, signatures, token_count)
            else:
                self._record_decision(sig_pos, False, drift, "skip", False, signatures, token_count)

        return torch.tensor(keep_positions, device=signatures.device, dtype=torch.long)

    def select_indices_from_paired_candidate_signatures(
        self,
        signatures: torch.Tensor,
        candidate_indices: torch.Tensor,
        total_frames: int,
        token_count: int,
        similarity_threshold: float,
    ) -> torch.Tensor:
        if self.selection_policy != "budget_topk":
            raise ValueError(
                "paired candidate selection requires selection_policy='budget_topk'"
            )
        if self.budget_window_size <= 0:
            raise ValueError(
                "budget_window_size must be > 0 for paired candidate selection"
            )
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be within [-1, 1]")

        candidate_indices = candidate_indices.to(
            device=signatures.device,
            dtype=torch.long,
        )
        candidate_positions = {
            int(idx.item()): pos for pos, idx in enumerate(candidate_indices)
        }
        base_frame_idx = self.frame_idx
        keep = {}
        pair_similarities = {}

        candidates_by_window = {}
        for local_idx, sig_pos in candidate_positions.items():
            global_idx = base_frame_idx + local_idx
            if self._in_recency_window(global_idx):
                keep[local_idx] = ("recency_keep", self.recency_updates_anchor)
                continue
            window_id = global_idx // self.budget_window_size
            candidates_by_window.setdefault(window_id, []).append(
                (local_idx, sig_pos)
            )

        for window_id, candidates in candidates_by_window.items():
            periodic_local_idx = (
                window_id * self.budget_window_size - base_frame_idx
            )
            periodic_position = candidate_positions.get(periodic_local_idx)
            if periodic_position is None:
                continue

            selected_local_idx = periodic_local_idx
            selected_similarity = 1.0
            for local_idx, sig_pos in candidates:
                if local_idx == periodic_local_idx:
                    continue
                similarity = float(
                    F.cosine_similarity(
                        signatures[periodic_position : periodic_position + 1],
                        signatures[sig_pos : sig_pos + 1],
                        dim=-1,
                    ).item()
                )
                pair_similarities[local_idx] = similarity
                if similarity < selected_similarity:
                    selected_local_idx = local_idx
                    selected_similarity = similarity

            if (
                selected_local_idx != periodic_local_idx
                and selected_similarity <= similarity_threshold
            ):
                keep[selected_local_idx] = ("semantic_reallocate", True)
            else:
                keep[periodic_local_idx] = ("budget_keep", True)

        keep_positions = []
        for local_idx in range(total_frames):
            sig_pos = candidate_positions.get(local_idx)
            if sig_pos is None:
                self._record_skip_without_signature(
                    0.0,
                    "prefilter_skip",
                    token_count,
                )
                continue
            if local_idx in keep:
                decision, update_anchor = keep[local_idx]
                similarity = pair_similarities.get(local_idx, 1.0)
                drift = max(0.0, 1.0 - similarity)
                keep_positions.append(sig_pos)
                self._record_decision(
                    sig_pos,
                    True,
                    drift,
                    decision,
                    update_anchor,
                    signatures,
                    token_count,
                )
            else:
                similarity = pair_similarities.get(local_idx, 1.0)
                drift = max(0.0, 1.0 - similarity)
                self._record_decision(
                    sig_pos,
                    False,
                    drift,
                    "pair_reject",
                    False,
                    signatures,
                    token_count,
                )

        return torch.tensor(
            keep_positions,
            device=signatures.device,
            dtype=torch.long,
        )

    def select_indices_from_window_signatures(self, signatures: torch.Tensor, token_count: int) -> torch.Tensor:
        if self.budget_window_size <= 0:
            raise ValueError("budget_window_size must be > 0 when selection_policy='budget_topk'")

        keep = {}
        drifts = []
        compare_anchor = self.anchor_feature
        if compare_anchor is None and signatures.shape[0] > 0:
            compare_anchor = signatures[0:1].detach()

        for local_idx in range(signatures.shape[0]):
            global_idx = self.frame_idx + local_idx
            if compare_anchor is None:
                drift = 0.0
            else:
                similarity = F.cosine_similarity(signatures[local_idx : local_idx + 1], compare_anchor, dim=-1)
                drift = max(0.0, float((1.0 - similarity).mean().item()))
            drifts.append(drift)

            if self.anchor_feature is None and local_idx == 0:
                keep[local_idx] = ("reference", True)
            elif (global_idx % self.refresh_interval) == 0:
                keep[local_idx] = ("refresh", True)
            elif self.coverage_interval > 0 and (global_idx % self.coverage_interval) == 0:
                keep[local_idx] = ("coverage_keep", self.coverage_updates_anchor)
            elif self._in_recency_window(global_idx):
                keep[local_idx] = ("recency_keep", self.recency_updates_anchor)

        candidates_by_window = {}
        for local_idx, drift in enumerate(drifts):
            if local_idx in keep or drift <= self.skip_patch_threshold:
                continue
            global_idx = self.frame_idx + local_idx
            window_id = global_idx // self.budget_window_size
            candidates_by_window.setdefault(window_id, []).append((drift, local_idx))

        reserved_by_window = {}
        for local_idx, (decision, _) in keep.items():
            if decision == "recency_keep":
                continue
            global_idx = self.frame_idx + local_idx
            window_id = global_idx // self.budget_window_size
            reserved_by_window[window_id] = reserved_by_window.get(window_id, 0) + 1

        for window_id, candidates in candidates_by_window.items():
            candidates.sort(reverse=True)
            remaining_budget = max(
                0,
                self.budget_keep_per_window - reserved_by_window.get(window_id, 0),
            )
            for _, local_idx in candidates[:remaining_budget]:
                keep[local_idx] = ("budget_keep", True)

        keep_indices = []
        for local_idx, drift in enumerate(drifts):
            if local_idx in keep:
                decision, update_anchor = keep[local_idx]
                keep_indices.append(local_idx)
                self._record_decision(local_idx, True, drift, decision, update_anchor, signatures, token_count)
            else:
                self._record_decision(local_idx, False, drift, "skip", False, signatures, token_count)

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
