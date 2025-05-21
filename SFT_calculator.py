import math

def get_optimizer_details(optimizer_type: str, bytes_optimizer_state_input: int):
    """
    Determines the number of states per parameter (N_os) and effective byte size
    for optimizer states.
    """
    n_os = 0
    effective_bytes_os = bytes_optimizer_state_input

    if optimizer_type.lower() == "adamw":
        n_os = 2  # m and v states
    elif optimizer_type.lower() == "adamw_8bit":
        n_os = 2  # m and v states
        effective_bytes_os = 1 # 8-bit Adam stores states in 1 byte
    elif optimizer_type.lower() == "sgd_momentum":
        n_os = 1  # momentum state
    elif optimizer_type.lower() == "sgd": # Assuming no momentum if not specified
        n_os = 0
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. "
                         f"Supported types are 'AdamW', 'AdamW_8bit', 'SGD_momentum', 'SGD'.")
    return n_os, effective_bytes_os

def calculate_sft_gpu_memory(
    # Model intrinsic parameters
    P_total_model_params: float,
    L_total_layers: int,
    h_hidden_size: int,
    a_num_attn_heads: int,
    # Training configuration
    s_seq_len: int,
    b_micro_batch_size: int,
    optimizer_type: str = "AdamW",
    # Precision settings (bytes per element)
    bytes_param_compute: int = 2,       # e.g., 2 for FP16/BF16, 4 for FP32
    bytes_param_master: int = 0,        # e.g., 4 for Mixed Precision AMP FP32 master, 0 otherwise
    bytes_gradient: int = 4,            # Typically 4 for FP32, or 2 if gradients are in FP16/BF16
    bytes_optimizer_state: int = 4,     # Typically 4 for FP32 states
    bytes_activation: int = 2,          # Typically 2 for FP16/BF16 in SFT
    # Optimization techniques
    use_activation_checkpointing: bool = False,
    use_flash_attention: bool = False,
    use_lora: bool = False,
    P_lora_params: float = 0,           # Number of LoRA parameters (e.g., 20e6)
    bytes_lora_param: int = 2,          # e.g., 2 for BF16 LoRA weights
    bytes_lora_gradient: int = 2,       # e.g., 2 for BF16, or 4 for FP32 LoRA gradients
    bytes_lora_optimizer_state: int = 4,# e.g., 4 for FP32 LoRA optimizer states
    use_qlora: bool = False,
    qlora_base_bits: int = 4,           # Bits for QLoRA base model quantization (e.g., 4 for NF4)
    # Distributed training
    dp_degree: int = 1,
    tp_degree: int = 1,
    pp_degree: int = 1,
    zero_stage: int = 0, # 0, 1, 2, 3
    # Overheads
    fixed_overhead_gb: float = 1.5,     # For CUDA context, framework, etc.
    percentage_overhead_factor: float = 0.05 # For fragmentation, minor non-captured items
):
    """
    Calculates the estimated GPU memory required for Supervised Fine-Tuning (SFT) of a language model.

    Returns:
        dict: A dictionary containing the breakdown of memory usage in GB for each component
              and the total estimated GPU memory in GB.
    """
    if tp_degree <= 0: tp_degree = 1
    if pp_degree <= 0: pp_degree = 1
    if dp_degree <= 0: dp_degree = 1

    # --- Initialize Memory Components (in bytes) ---
    mem_params_bytes = 0
    mem_gradients_bytes = 0
    mem_optimizer_bytes = 0
    mem_activations_bytes = 0

    # --- Determine effective parameters and precisions ---
    P_base_model = P_total_model_params
    P_trainable_effective = P_total_model_params
    
    # Precisions for the main model parameters or the full model if not LoRA
    eff_bytes_param_compute = bytes_param_compute
    eff_bytes_param_master = bytes_param_master

    if use_qlora:
        use_lora = True # QLoRA implies LoRA for trainable parameters
        eff_bytes_param_compute = qlora_base_bits / 8.0 # Base model quantized
        eff_bytes_param_master = 0 # No separate master weights for quantized base

    # Precisions for gradients and optimizer states (depend on whether LoRA is used)
    if use_lora:
        P_trainable_effective = P_lora_params
        eff_bytes_gradient = bytes_lora_gradient
        # Get LoRA optimizer details (N_os and effective bytes for LoRA states)
        # Assuming LoRA optimizer type matches main optimizer type for N_os calculation logic
        # but uses specific LoRA byte size for states
        n_os_trainable, eff_bytes_optimizer_state_trainable = get_optimizer_details(optimizer_type, bytes_lora_optimizer_state)
        if optimizer_type.lower() == "adamw_8bit": # If main opt is 8bit, LoRA opt also 8bit
             eff_bytes_optimizer_state_trainable = 1

    else: # Full fine-tuning
        P_trainable_effective = P_total_model_params
        eff_bytes_gradient = bytes_gradient
        n_os_trainable, eff_bytes_optimizer_state_trainable = get_optimizer_details(optimizer_type, bytes_optimizer_state)

    # --- 1. Model Parameter Memory ---
    # This includes frozen base model (if LoRA/QLoRA) and active trainable parameters
    
    # A. Memory for the base model parameters
    mem_base_model_params_bytes = 0
    B_p_base_eff = eff_bytes_param_compute + eff_bytes_param_master # eff_bytes_param_master is 0 for QLoRA
    
    if zero_stage == 3:
        mem_base_model_params_bytes = (P_base_model * B_p_base_eff) / dp_degree
    else:
        mem_base_model_params_bytes = (P_base_model / (tp_degree * pp_degree)) * B_p_base_eff

    # B. Memory for LoRA parameters (if LoRA is used, these are separate from base model calculation above)
    # If not LoRA, mem_lora_active_params_bytes remains 0, and full model params handled by B_p_base_eff.
    # If LoRA, B_p_base_eff handles the (potentially quantized) base, and this handles LoRA weights.
    mem_lora_active_params_bytes = 0
    if use_lora:
        B_p_lora_eff = bytes_lora_param # LoRA adapters usually don't have master weights
        if zero_stage == 3: # LoRA params are trainable, so sharded by ZeRO-3
            mem_lora_active_params_bytes = (P_lora_params * B_p_lora_eff) / dp_degree
        else: # Replicated across TP/PP slices, and across DP if not ZeRO-3
              # However, LoRA parameters are generally small and often loaded on all GPUs
              # within a DP group if not specifically sharded by ZeRO-3.
              # Let's assume they are replicated if not ZeRO-3.
            mem_lora_active_params_bytes = P_lora_params * B_p_lora_eff

    if use_lora :
        mem_params_bytes = mem_base_model_params_bytes + mem_lora_active_params_bytes
    else: # Full fine-tuning, base model *is* the active parameter set
        mem_params_bytes = mem_base_model_params_bytes


    # --- 2. Gradient Memory --- (Only for P_trainable_effective)
    if P_trainable_effective > 0:
        if zero_stage == 2 or zero_stage == 3:
            mem_gradients_bytes = (P_trainable_effective * eff_bytes_gradient) / dp_degree
        else: # ZeRO 0 or 1, or no ZeRO
            if use_lora: # LoRA gradients are for P_lora_params
                mem_gradients_bytes = P_trainable_effective * eff_bytes_gradient # Replicated if not ZeRO 2/3
            else: # Full FT gradients for the slice
                mem_gradients_bytes = (P_trainable_effective / (tp_degree * pp_degree)) * eff_bytes_gradient
    
    # --- 3. Optimizer State Memory --- (Only for P_trainable_effective)
    if P_trainable_effective > 0 and n_os_trainable > 0:
        if zero_stage >= 1 and zero_stage <= 3: # ZeRO 1, 2, or 3
            mem_optimizer_bytes = (P_trainable_effective * n_os_trainable * eff_bytes_optimizer_state_trainable) / dp_degree
        else: # ZeRO 0 or no ZeRO
            if use_lora: # LoRA optimizer states for P_lora_params
                mem_optimizer_bytes = P_trainable_effective * n_os_trainable * eff_bytes_optimizer_state_trainable # Replicated
            else: # Full FT optimizer states for the slice
                mem_optimizer_bytes = (P_trainable_effective / (tp_degree * pp_degree)) * n_os_trainable * eff_bytes_optimizer_state_trainable

    # --- 4. Activation Memory ---
    L_slice = L_total_layers / pp_degree
    B_act_eff = bytes_activation

    if L_slice > 0 and b_micro_batch_size > 0 and s_seq_len > 0 and h_hidden_size > 0 :
        if use_activation_checkpointing:
            mem_activations_bytes = L_slice * (2 * s_seq_len * b_micro_batch_size * h_hidden_size / tp_degree) * B_act_eff
        else:
            # Formula: (L/PP) * (s*b*h/TP) * (C_base + C_attn_score_elements * a * s / h) * (B_act / C_precision_norm_factor)
            # Constants C_base=34, C_attn_score=5 are for FP16 element counts.
            # C_precision_norm_factor=2 aligns B_act with these FP16-based constants.
            base_activation_chunk_per_layer_tp_scaled = (s_seq_len * b_micro_batch_size * h_hidden_size) / tp_degree
            
            bytes_per_fp16_equiv_element_count = B_act_eff / 2.0 # Normalizing factor
            
            linear_term_coeff = 34.0 # Corresponds to non-attention score related activations
            attn_score_term_coeff = 0.0
            if not use_flash_attention and a_num_attn_heads > 0 and h_hidden_size > 0 : # Avoid division by zero
                attn_score_term_coeff = 5.0 * a_num_attn_heads * s_seq_len / h_hidden_size
            
            total_coeff_factor = linear_term_coeff + attn_score_term_coeff
            mem_activations_bytes = L_slice * base_activation_chunk_per_layer_tp_scaled * total_coeff_factor * bytes_per_fp16_equiv_element_count

        # Add memory for backward all-gather in Tensor Parallelism (m_tp_back from LLMem)
        if tp_degree > 1:
            mem_tp_backward_bytes = L_slice * (s_seq_len * b_micro_batch_size * h_hidden_size) * \
                                    ((tp_degree - 1) / tp_degree) * B_act_eff
            mem_activations_bytes += mem_tp_backward_bytes
    
    # --- 5. Sum Components and Add Overheads ---
    total_model_compute_memory_bytes = mem_params_bytes + mem_gradients_bytes + mem_optimizer_bytes + mem_activations_bytes
    
    # Convert to GB
    bytes_to_gb = 1 / (1024**3)
    mem_params_gb = mem_params_bytes * bytes_to_gb
    mem_gradients_gb = mem_gradients_bytes * bytes_to_gb
    mem_optimizer_gb = mem_optimizer_bytes * bytes_to_gb
    mem_activations_gb = mem_activations_bytes * bytes_to_gb
    
    total_model_compute_memory_gb = total_model_compute_memory_bytes * bytes_to_gb
    
    # Add fixed and percentage overheads
    total_estimated_gpu_memory_gb = total_model_compute_memory_gb + fixed_overhead_gb
    total_estimated_gpu_memory_gb *= (1 + percentage_overhead_factor)

    return {
        "P_total_model_params": P_total_model_params,
        "P_lora_params": P_lora_params if use_lora else 0,
        "P_trainable_effective": P_trainable_effective,
        "L_total_layers": L_total_layers,
        "h_hidden_size": h_hidden_size,
        "a_num_attn_heads": a_num_attn_heads,
        "s_seq_len": s_seq_len,
        "b_micro_batch_size": b_micro_batch_size,
        "optimizer_type": optimizer_type,
        "N_os_optimizer_states": n_os_trainable,
        "bytes_param_compute": bytes_param_compute if not use_qlora else qlora_base_bits / 8.0,
        "bytes_param_master": bytes_param_master if not use_qlora else 0,
        "bytes_gradient": eff_bytes_gradient,
        "bytes_optimizer_state": eff_bytes_optimizer_state_trainable,
        "bytes_activation": bytes_activation,
        "use_activation_checkpointing": use_activation_checkpointing,
        "use_flash_attention": use_flash_attention,
        "use_lora": use_lora,
        "use_qlora": use_qlora,
        "qlora_base_bits": qlora_base_bits if use_qlora else None,
        "dp_degree": dp_degree,
        "tp_degree": tp_degree,
        "pp_degree": pp_degree,
        "zero_stage": zero_stage,
        "breakdown_gb": {
            "parameters": mem_params_gb,
            "gradients": mem_gradients_gb,
            "optimizer_states": mem_optimizer_gb,
            "activations": mem_activations_gb,
            "subtotal_compute_memory": total_model_compute_memory_gb,
            "fixed_overhead": fixed_overhead_gb,
            "total_with_fixed_overhead": total_model_compute_memory_gb + fixed_overhead_gb,
            "percentage_overhead_applied": (total_model_compute_memory_gb + fixed_overhead_gb) * percentage_overhead_factor,
        },
        "total_estimated_gpu_memory_gb": total_estimated_gpu_memory_gb,
    }

if __name__ == '__main__':
    # --- Example Usage: QLoRA Fine-tuning of a Llama-like 7B model ---
    print("--- Example 1: QLoRA Fine-tuning (Llama 7B like) ---")
    qlora_7b_config = {
        "P_total_model_params": 7e9,
        "L_total_layers": 32,
        "h_hidden_size": 4096,
        "a_num_attn_heads": 32,
        "s_seq_len": 2048,
        "b_micro_batch_size": 1,
        "optimizer_type": "AdamW_8bit", # Paged AdamW 8bit often used with QLoRA

        "bytes_param_compute": 2, # This will be overridden by qlora_base_bits internally for base model
        "bytes_param_master": 0,  # No master for base in QLoRA
        "bytes_gradient": 2,      # For LoRA grads, assuming BF16
        "bytes_optimizer_state": 4, # This will be overridden by AdamW_8bit logic for LoRA optimizer states
        "bytes_activation": 2,    # BF16/FP16 activations

        "use_activation_checkpointing": True, # Common with long sequences
        "use_flash_attention": True,         # Common for efficiency

        "use_lora": True, # Implied by QLoRA
        "P_lora_params": 50e6, # Example: ~50M LoRA parameters (can vary widely)
        "bytes_lora_param": 2,          # LoRA weights in BF16
        "bytes_lora_gradient": 2,       # LoRA gradients in BF16
        "bytes_lora_optimizer_state": 1,# LoRA AdamW 8bit states

        "use_qlora": True,
        "qlora_base_bits": 4,

        "dp_degree": 1,
        "tp_degree": 1,
        "pp_degree": 1,
        "zero_stage": 0, # ZeRO often not heavily used with QLoRA on single GPU, or minimal stage

        "fixed_overhead_gb": 1.0,
        "percentage_overhead_factor": 0.05
    }
    memory_estimate_qlora = calculate_sft_gpu_memory(**qlora_7b_config)
    for key, value in memory_estimate_qlora.items():
        if key == "breakdown_gb":
            print(f"  {key}:")
            for k_br, v_br in value.items():
                print(f"    {k_br}: {v_br:.2f} GB")
        else:
            print(f"  {key}: {value}")
    print(f"\n  --> Total Estimated GPU Memory for QLoRA 7B: {memory_estimate_qlora['total_estimated_gpu_memory_gb']:.2f} GB\n")

    # --- Example Usage: Full Fine-tuning of a 1.3B model with Mixed Precision & ZeRO-2 ---
    print("--- Example 2: Full Fine-tuning (1.3B model, Mixed Precision, ZeRO-2) ---")
    full_ft_1_3b_config = {
        "P_total_model_params": 1.3e9,
        "L_total_layers": 24,
        "h_hidden_size": 2048, # Typical for 1.3B
        "a_num_attn_heads": 16,
        "s_seq_len": 1024,
        "b_micro_batch_size": 4,
        "optimizer_type": "AdamW",

        # Mixed Precision AMP typical settings
        "bytes_param_compute": 2,       # FP16/BF16 for computation
        "bytes_param_master": 4,        # FP32 master weights
        "bytes_gradient": 4,            # FP32 gradients
        "bytes_optimizer_state": 4,     # FP32 optimizer states
        "bytes_activation": 2,          # FP16/BF16 activations

        "use_activation_checkpointing": False,
        "use_flash_attention": True,

        "use_lora": False, # Full fine-tuning
        # LoRA specific params are ignored if use_lora is False

        "use_qlora": False,

        "dp_degree": 2, # Using 2 GPUs for Data Parallelism
        "tp_degree": 1,
        "pp_degree": 1,
        "zero_stage": 2, # ZeRO Stage 2

        "fixed_overhead_gb": 1.5,
        "percentage_overhead_factor": 0.05
    }
    memory_estimate_full_ft = calculate_sft_gpu_memory(**full_ft_1_3b_config)
    for key, value in memory_estimate_full_ft.items():
        if key == "breakdown_gb":
            print(f"  {key}:")
            for k_br, v_br in value.items():
                print(f"    {k_br}: {v_br:.2f} GB")
        else:
            print(f"  {key}: {value}")
    print(f"\n  --> Total Estimated GPU Memory for Full FT 1.3B (per GPU): {memory_estimate_full_ft['total_estimated_gpu_memory_gb']:.2f} GB\n")


    # --- Example Usage: Full Fine-tuning of a 7B model with Mixed Precision, AC, FA, ZeRO-3, TP=2, PP=2 ---
    print("--- Example 3: Full Fine-tuning (7B model, Mixed Precision, AC, FA, ZeRO-3, TP=2, PP=2, DP=2) ---")
    # Effective DP = total_gpus / (tp*pp) = 8 / (2*2) = 2
    complex_ft_7b_config = {
        "P_total_model_params": 7e9,
        "L_total_layers": 32,
        "h_hidden_size": 4096,
        "a_num_attn_heads": 32,
        "s_seq_len": 2048,
        "b_micro_batch_size": 1, # Small micro batch due to large model slices and sequence length
        "optimizer_type": "AdamW",

        "bytes_param_compute": 2,
        "bytes_param_master": 4,
        "bytes_gradient": 4,
        "bytes_optimizer_state": 4,
        "bytes_activation": 2,

        "use_activation_checkpointing": True,
        "use_flash_attention": True,

        "use_lora": False,
        "use_qlora": False,

        "dp_degree": 2, # Data Parallel degree
        "tp_degree": 2, # Tensor Parallel degree
        "pp_degree": 2, # Pipeline Parallel degree
        "zero_stage": 3,

        "fixed_overhead_gb": 1.5,
        "percentage_overhead_factor": 0.05
    }
    memory_estimate_complex_ft = calculate_sft_gpu_memory(**complex_ft_7b_config)
    for key, value in memory_estimate_complex_ft.items():
        if key == "breakdown_gb":
            print(f"  {key}:")
            for k_br, v_br in value.items():
                print(f"    {k_br}: {v_br:.2f} GB")
        else:
            print(f"  {key}: {value}")
    print(f"\n  --> Total Estimated GPU Memory for Complex FT 7B (per GPU): {memory_estimate_complex_ft['total_estimated_gpu_memory_gb']:.2f} GB\n")

    # --- Example Usage: Full Fine-tuning of QwQ 32B model with Mixed Precision, AC, FA, ZeRO-3, TP=2, PP=2, DP=2 ---
    print("--- Example 4: Full SFT QwQ 32B (Mixed Precision, AC, FA, ZeRO-3, TP=2, PP=2, DP=2) ---")
    # Based on QwQ 32B specs: 32.5B params, 64 layers, 40 Q heads (8 KV heads), 131072 context length.
    # Assuming d_head = 128, so h_hidden_size = 40 * 128 = 5120.
    # Using 8 GPUs total: TP=2, PP=2 => DP = 8 / (2*2) = 2.
    qwq_32b_sft_config = {
        "P_total_model_params": 32.5e9,
        "L_total_layers": 64,
        "h_hidden_size": 5120,        # Derived: 40 Q_heads * 128 d_head
        "a_num_attn_heads": 40,         # Q heads for GQA
        "s_seq_len": 131072,          # Very long context length
        "b_micro_batch_size": 1,        # Micro batch size per GPU
        "optimizer_type": "AdamW",

        "bytes_param_compute": 2,       # Half-precision (e.g., BF16/FP16) compute
        "bytes_param_master": 4,        # FP32 master weights for AdamW
        "bytes_gradient": 2,            # Half-precision gradients (BF16/FP16)
        "bytes_optimizer_state": 4,     # FP32 optimizer states (sharded by ZeRO-3)
        "bytes_activation": 2,          # Half-precision activations

        "use_activation_checkpointing": True, # Essential for very long sequences
        "use_flash_attention": True,         # Recommended for efficiency

        "use_lora": False,              # Full fine-tuning
        "use_qlora": False,

        "dp_degree": 2,                 # Data Parallel degree
        "tp_degree": 2,                 # Tensor Parallel degree
        "pp_degree": 2,                 # Pipeline Parallel degree
        "zero_stage": 3,                # ZeRO Stage 3

        "fixed_overhead_gb": 1.5,
        "percentage_overhead_factor": 0.05
    }
    memory_estimate_qwq_32b = calculate_sft_gpu_memory(**qwq_32b_sft_config)
    for key, value in memory_estimate_qwq_32b.items():
        if key == "breakdown_gb":
            print(f"  {key}:")
            for k_br, v_br in value.items():
                print(f"    {k_br}: {v_br:.2f} GB")
        else:
            print(f"  {key}: {value}")
    print(f"\n  --> Total Estimated GPU Memory for QwQ 32B SFT (per GPU): {memory_estimate_qwq_32b['total_estimated_gpu_memory_gb']:.2f} GB\n")
