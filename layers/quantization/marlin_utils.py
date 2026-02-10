"""Marlin量化工具函数"""
import torch
import numpy


# Marlin kernel常量
GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16


def get_pack_factor(num_bits: int) -> int:
    """计算打包因子：一个int32可以打包多少个量化值"""
    assert 32 % num_bits == 0, f"不支持的量化位数: {num_bits}"
    return 32 // num_bits


def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    """将量化权重按列打包到int32"""
    assert q_w.shape == (size_k, size_n)
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    
    orig_device = q_w.device
    q_w = q_w.cpu().numpy().astype(numpy.uint32)
    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)
    
    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i
    
    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    return q_res.contiguous()


def unpack_cols(
    packed_q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    """将打包的int32权重按列解包"""
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    assert packed_q_w.shape == (size_k, size_n // pack_factor)
    
    orig_device = packed_q_w.device
    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)
    
    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_cpu & mask
        packed_q_w_cpu >>= num_bits
        q_res[:, i::pack_factor] = vals
    
    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    return q_res.contiguous()


def get_scale_perms():
    """获取Marlin scale的排列模式"""
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:
    """将scales从AWQ格式转换为Marlin格式"""
    scale_perm, scale_perm_single = get_scale_perms()
    
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    
    s = s.reshape((-1, size_n)).contiguous()
    return s


def marlin_zero_points(
    zp: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """将zero-points转换为Marlin格式并打包"""
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]
    
    # 交错列维度并打包到int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise ValueError(f"num_bits必须是4或8，收到{num_bits}")
    
    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)
    
    return zp


def awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """将AWQ格式的zero-points转换为Marlin格式"""
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)
    
    # 撤销交错（使用argsort获取逆排列）
    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise ValueError(f"num_bits必须是4或8，收到{num_bits}")
    
    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()
    
    marlin_zp = marlin_zero_points(q_zp, size_k, size_n, num_bits)
    return marlin_zp


def verify_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
):
    """验证tensor shape是否与Marlin kernel兼容"""
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            f"output_size_per_partition={output_size_per_partition} "
            f"不能被{GPTQ_MARLIN_MIN_THREAD_N}整除，考虑减小tensor_parallel_size"
        )
    
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            f"input_size_per_partition={input_size_per_partition} "
            f"不能被{GPTQ_MARLIN_MIN_THREAD_K}整除，考虑减小tensor_parallel_size"
        )
    
    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            f"input_size_per_partition={input_size_per_partition} "
            f"不能被group_size={group_size}整除，考虑减小tensor_parallel_size"
        )


def marlin_make_workspace(device: torch.device) -> torch.Tensor:
    """为Marlin kernel创建workspace缓冲区"""
    return torch.zeros(
        GPTQ_MARLIN_MAX_PARALLEL * GPTQ_MARLIN_TILE * GPTQ_MARLIN_TILE,
        dtype=torch.int,
        device=device,
    )


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    """创建空的g_idx tensor（AWQ不用，但kernel需要）"""
    return torch.empty(0, dtype=torch.int, device=device)
