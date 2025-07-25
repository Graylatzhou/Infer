from transformers import AutoConfig

class Config:
    model_name: str
    gpu_memory_utilization: float = 0.85 # GPU内存利用率
    transformer_config: AutoConfig

    def __post_init__(self):
        self.transformer_config = AutoConfig.from_pretrained(self.model_name)


