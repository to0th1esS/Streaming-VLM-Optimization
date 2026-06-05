import torch
import torch.nn.functional as F


class FixedBudgetTokenReducer:
    """以固定预算压缩每帧视觉 token，同时保持 ReKV 块长度稳定。"""

    def __init__(
        self,
        output_token_budget: int,
        coverage_tokens: int = 16,
        policy: str = "coverage_innovation",
        drift_feature_dims: int = 0,
    ):
        if output_token_budget < 1:
            raise ValueError("output_token_budget must be >= 1")
        if coverage_tokens < 0 or coverage_tokens > output_token_budget:
            raise ValueError(
                "coverage_tokens must satisfy 0 <= coverage_tokens <= output_token_budget"
            )
        if policy not in {"uniform", "coverage_innovation"}:
            raise ValueError(
                "policy must be 'uniform' or 'coverage_innovation'"
            )
        self.output_token_budget = output_token_budget
        self.coverage_tokens = coverage_tokens
        self.policy = policy
        self.drift_feature_dims = max(0, int(drift_feature_dims))
        self.previous_features = None
        self.stats = {}
        self.reset()

    def reset(self):
        self.previous_features = None
        self.stats = {
            "input_tokens": 0,
            "output_tokens": 0,
            "frames": 0,
            "coverage_tokens": 0,
            "innovation_tokens": 0,
        }

    @staticmethod
    def _uniform_indices(token_count: int, count: int, device):
        if count <= 0:
            return torch.empty(0, device=device, dtype=torch.long)
        if count >= token_count:
            return torch.arange(token_count, device=device, dtype=torch.long)
        return torch.linspace(
            0,
            token_count - 1,
            steps=count,
            device=device,
        ).round().long().unique(sorted=True)

    def _select_indices(self, frame_features):
        token_count = int(frame_features.shape[0])
        budget = min(self.output_token_budget, token_count)
        if self.policy == "uniform" or self.previous_features is None:
            return self._uniform_indices(
                token_count,
                budget,
                frame_features.device,
            ), budget, 0

        coverage_count = min(self.coverage_tokens, budget)
        coverage_indices = self._uniform_indices(
            token_count,
            coverage_count,
            frame_features.device,
        )
        innovation_count = max(0, budget - int(coverage_indices.numel()))
        if innovation_count == 0:
            return coverage_indices, int(coverage_indices.numel()), 0

        current_drift_features = self._drift_features(frame_features)
        previous_drift_features = self._drift_features(self.previous_features)
        similarities = F.cosine_similarity(
            current_drift_features.float(),
            previous_drift_features.float(),
            dim=-1,
        )
        drift = 1.0 - similarities
        if coverage_indices.numel():
            drift = drift.clone()
            drift[coverage_indices] = float("-inf")
        innovation_indices = torch.topk(
            drift,
            k=min(innovation_count, token_count - int(coverage_indices.numel())),
            largest=True,
        ).indices
        selected = torch.cat([coverage_indices, innovation_indices]).unique(
            sorted=True
        )

        if selected.numel() < budget:
            # 极小 token 网格或重复均匀索引时，用剩余空间位置补足固定预算。
            remaining_mask = torch.ones(
                token_count,
                device=frame_features.device,
                dtype=torch.bool,
            )
            remaining_mask[selected] = False
            fill = torch.arange(
                token_count,
                device=frame_features.device,
            )[remaining_mask][: budget - selected.numel()]
            selected = torch.cat([selected, fill]).sort().values
        return selected, int(coverage_indices.numel()), int(
            selected.numel() - coverage_indices.numel()
        )

    def _drift_features(self, features):
        feature_dim = int(features.shape[-1])
        if self.drift_feature_dims <= 0 or self.drift_feature_dims >= feature_dim:
            return features
        # 均匀抽取通道形成确定性 temporal sketch（时间变化草图），不引入训练参数。
        stride = max(1, feature_dim // self.drift_feature_dims)
        return features[..., ::stride][..., : self.drift_feature_dims]

    @torch.inference_mode()
    def __call__(
        self,
        video_features,
        batch_size=1,
        frames=1,
        selection_features=None,
        **kwargs,
    ):
        if batch_size != 1:
            raise ValueError("FixedBudgetTokenReducer currently requires batch_size=1")
        if video_features.ndim != 3:
            raise ValueError(
                "video_features must have shape [frames, tokens, hidden]"
            )
        if int(video_features.shape[0]) != int(frames):
            raise ValueError(
                "video_features frame dimension must match the frames argument"
            )
        if selection_features is None:
            selection_features = video_features
        if selection_features.shape[:2] != video_features.shape[:2]:
            raise ValueError(
                "selection_features must match video_features in frames and tokens"
            )

        reduced_frames = []
        for frame_features, frame_selection_features in zip(
            video_features,
            selection_features,
        ):
            selected, coverage_count, innovation_count = self._select_indices(
                frame_selection_features
            )
            reduced_frames.append(frame_features.index_select(0, selected))
            self.previous_features = frame_selection_features.detach()
            self.stats["input_tokens"] += int(frame_features.shape[0])
            self.stats["output_tokens"] += int(selected.numel())
            self.stats["frames"] += 1
            self.stats["coverage_tokens"] += coverage_count
            self.stats["innovation_tokens"] += innovation_count
        return torch.stack(reduced_frames, dim=0)
