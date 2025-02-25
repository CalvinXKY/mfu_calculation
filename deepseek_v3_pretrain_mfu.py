from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        gbs (int): Global batch size
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_mtp_modules (int): Number of mtp modules.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        is_causal (bool): Attention calculation type.
    """
    # block
    gbs: int = 1024
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 129280
    dim: int = 7168  # embed
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    n_mtp_modules: int = 2
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 2.5
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    is_causal: bool = False


class DeepSeekV3Calculation:
    def __init__(self,
                 model_args: ModelArgs,
                 ):
        self.model_args = model_args

    def calcu_embedding_layer(self):
        args = self.model_args
        embedding_flops = 2 * args.gbs * args.seq_len * args.dim * args.vocab_size
        return embedding_flops

    def calcu_mla_flops(self):
        # attention flops:
        args = self.model_args
        gbs = args.gbs
        num_heads = args.n_heads
        hidden_size = args.dim
        qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        q_down_proj = 2 * args.gbs * args.seq_len * hidden_size * args.q_lora_rank
        q_up_proj = 2 * args.gbs * args.seq_len * args.q_lora_rank * num_heads * qk_head_dim
        q_linear = q_down_proj + q_up_proj

        kv_down_proj = 2 * gbs * args.seq_len * hidden_size * (args.kv_lora_rank + args.qk_rope_head_dim)
        kv_up_proj = 2 * gbs * args.seq_len * args.kv_lora_rank * num_heads * (qk_head_dim + args.v_head_dim)
        kv_linear = kv_down_proj + kv_up_proj

        kv_scores = 2 * gbs * args.seq_len ** 2 * num_heads * qk_head_dim
        qkv = 2 * gbs * args.seq_len ** 2 * num_heads * args.v_head_dim

        out_linear = 2 * gbs * args.seq_len * args.n_heads * args.v_head_dim * hidden_size
        if args.dtype == 'fp8':
            q_linear /= 2
            kv_linear /= 2
            out_linear /= 2
        attention_layer_flops = q_linear + kv_linear + kv_scores + qkv + out_linear
        return attention_layer_flops

    def calcu_moe_flops(self):
        args = self.model_args
        hidden_size = args.dim
        share = args.n_shared_experts
        top_k = args.n_activated_experts
        linear_layer_flops = 2 * 3 * args.gbs * args.seq_len * hidden_size * args.moe_inter_dim
        route_flops = 2 * args.gbs * args.seq_len * hidden_size * args.n_routed_experts
        if args.dtype == 'fp8':
            linear_layer_flops /= 2
        moe_layer_flops = linear_layer_flops * (share + top_k) + route_flops
        return moe_layer_flops

    def calcu_mlp_flops(self):
        args = self.model_args
        hidden_size = args.dim
        linear_layer_flops = 2 * 3 * args.gbs * args.seq_len * hidden_size * args.inter_dim
        if args.dtype == 'fp8':
            linear_layer_flops /= 2
        return linear_layer_flops

    def calcu_main_model(self):
        moe_layers = self.model_args.n_layers - self.model_args.n_dense_layers
        embedding_flops = self.calcu_embedding_layer()
        mla_layer_flops = self.calcu_mla_flops()
        moe_layer_flops = self.calcu_moe_flops()
        mlp_layer_flops = self.calcu_mlp_flops()

        main_model_flops = 3 * (embedding_flops +
                      moe_layers * (mla_layer_flops + moe_layer_flops) +
                      self.model_args.n_dense_layers * (mla_layer_flops + mlp_layer_flops))
        return main_model_flops

    def calcu_mtp_model(self):
        args = self.model_args
        gbs = args.gbs
        hidden_size = args.dim
        linear_proj = 2 * 3 * gbs * args.seq_len * hidden_size * (hidden_size * 2)
        if args.dtype == 'fp8':
            linear_proj /= 2

        embedding_flops = self.calcu_embedding_layer()
        mla_layer_flops = self.calcu_mla_flops()
        moe_layer_flops = self.calcu_moe_flops()
        mtp_flops = 3 * (embedding_flops + mla_layer_flops + moe_layer_flops + linear_proj)
        return mtp_flops

    def calculate(self, step_time, world_size, gpu_peak_bf16_flops):
        main_model_flops = self.calcu_main_model()
        mtp_flops = self.calcu_mtp_model()
        total_flops = main_model_flops + mtp_flops * self.model_args.n_mtp_modules
        mfu = total_flops / (world_size * step_time * (10 ** 12)) / gpu_peak_bf16_flops
        return mfu, total_flops


if __name__ == "__main__":
    args = ModelArgs()
    calculation = DeepSeekV3Calculation(args)
    mfu, total_flops_bf16 = calculation.calculate(21.00, 512, 354)
    print(f"MFU:{mfu}, Total flops：{total_flops_bf16}")
