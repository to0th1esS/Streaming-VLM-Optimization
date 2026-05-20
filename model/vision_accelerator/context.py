class InferenceContext:
    """
    用于在 Vision Tower 的不同层之间传递推理状态。
    包含：当前处理到了第几个 Chunk、是否需要全量更新等。
    """
    def __init__(
        self, 
        update_token_ratio: float = 0.25, 
        cache_interval: int = 2,
    ):
        """
        初始化上下文状态。
        
        Args:
            chunk_idx: 当前处理的视频块索引。
            update_token_ratio: 稀疏更新的比率，默认为 0.25。
            cache_interval: 全量更新的间隔,默认为 2。
            is_reference_chunk: 当前是否是参考帧。
        """
        self.chunk_idx :int = 0
        self.update_token_ratio :float = update_token_ratio
        self.cache_interval :int = cache_interval
        self.is_reference_chunk :bool = False

    def update(self, chunk_idx: int):
        """
        在处理每一个 Video Chunk 之前调用此方法更新状态
        """
        self.chunk_idx = chunk_idx
        self.is_reference_chunk = (chunk_idx % self.cache_interval == 0)
