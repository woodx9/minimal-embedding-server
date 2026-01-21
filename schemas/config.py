class MSEConfig:
    def __init__(self, device: str = "cuda:0", attn_backend: str = "flashAttention"):
        self.device = device
        self.attn_backend = attn_backend
