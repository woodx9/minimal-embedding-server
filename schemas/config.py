class MESConfig:
    def __init__(
        self,
        attn_backend: str = "flash_attention",
        model_name: str | None = None,
        max_tokens_per_batch: int | None = None,
        enable_monitoring: bool | None = None,
        dtype: str = "auto",
        model_config = None,
    ):
        self.attn_backend = attn_backend
        self.model_name = model_name
        self.max_tokens_per_batch = max_tokens_per_batch
        self.enable_monitoring = enable_monitoring
        self.dtype = dtype
        self.model_config = model_config
