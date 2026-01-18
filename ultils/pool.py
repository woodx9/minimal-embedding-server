import torch


def last_token_pool(
    hidden_states: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    # 直接取序列的最后一个 token
    last_hidden_state = hidden_states[seq_len - 1]
    return last_hidden_state