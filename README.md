# SFT_VRAM_Calculator
A very accurate calculator for calculating the required VRAM of SFT

**MADE BY GEMINI 2.5 PRO**

**result sample:**
```
--- Example 1: QLoRA Fine-tuning (Llama 7B like) ---
  P_total_model_params: 7000000000.0
  P_lora_params: 50000000.0
  P_trainable_effective: 50000000.0
  L_total_layers: 32
  h_hidden_size: 4096
  a_num_attn_heads: 32
  s_seq_len: 2048
  b_micro_batch_size: 1
  optimizer_type: AdamW_8bit
  N_os_optimizer_states: 2
  bytes_param_compute: 0.5
  bytes_param_master: 0
  bytes_gradient: 2
  bytes_optimizer_state: 1
  bytes_activation: 2
  use_activation_checkpointing: True
  use_flash_attention: True
  use_lora: True
  use_qlora: True
  qlora_base_bits: 4
  dp_degree: 1
  tp_degree: 1
  pp_degree: 1
  zero_stage: 0
  breakdown_gb:
    parameters: 3.35 GB
    gradients: 0.09 GB
    optimizer_states: 0.09 GB
    activations: 1.00 GB
    subtotal_compute_memory: 4.54 GB
    fixed_overhead: 1.00 GB
    total_with_fixed_overhead: 5.54 GB
    percentage_overhead_applied: 0.28 GB
  total_estimated_gpu_memory_gb: 5.81597707271576

  --> Total Estimated GPU Memory for QLoRA 7B: 5.82 GB

--- Example 2: Full Fine-tuning (1.3B model, Mixed Precision, ZeRO-2) ---
  P_total_model_params: 1300000000.0
  P_lora_params: 0
  P_trainable_effective: 1300000000.0
  L_total_layers: 24
  h_hidden_size: 2048
  a_num_attn_heads: 16
  s_seq_len: 1024
  b_micro_batch_size: 4
  optimizer_type: AdamW
  N_os_optimizer_states: 2
  bytes_param_compute: 2
  bytes_param_master: 4
  bytes_gradient: 4
  bytes_optimizer_state: 4
  bytes_activation: 2
  use_activation_checkpointing: False
  use_flash_attention: True
  use_lora: False
  use_qlora: False
  qlora_base_bits: None
  dp_degree: 2
  tp_degree: 1
  pp_degree: 1
  zero_stage: 2
  breakdown_gb:
    parameters: 7.26 GB
    gradients: 2.42 GB
    optimizer_states: 4.84 GB
    activations: 6.38 GB
    subtotal_compute_memory: 20.90 GB
    fixed_overhead: 1.50 GB
    total_with_fixed_overhead: 22.40 GB
    percentage_overhead_applied: 1.12 GB
  total_estimated_gpu_memory_gb: 23.52381377220154

  --> Total Estimated GPU Memory for Full FT 1.3B (per GPU): 23.52 GB

--- Example 3: Full Fine-tuning (7B model, Mixed Precision, AC, FA, ZeRO-3, TP=2, PP=2, DP=2) ---
  P_total_model_params: 7000000000.0
  P_lora_params: 0
  P_trainable_effective: 7000000000.0
  L_total_layers: 32
  h_hidden_size: 4096
  a_num_attn_heads: 32
  s_seq_len: 2048
  b_micro_batch_size: 1
  optimizer_type: AdamW
  N_os_optimizer_states: 2
  bytes_param_compute: 2
  bytes_param_master: 4
  bytes_gradient: 4
  bytes_optimizer_state: 4
  bytes_activation: 2
  use_activation_checkpointing: True
  use_flash_attention: True
  use_lora: False
  use_qlora: False
  qlora_base_bits: None
  dp_degree: 2
  tp_degree: 2
  pp_degree: 2
  zero_stage: 3
  breakdown_gb:
    parameters: 19.56 GB
    gradients: 13.04 GB
    optimizer_states: 26.08 GB
    activations: 0.38 GB
    subtotal_compute_memory: 59.05 GB
    fixed_overhead: 1.50 GB
    total_with_fixed_overhead: 60.55 GB
    percentage_overhead_applied: 3.03 GB
  total_estimated_gpu_memory_gb: 63.575738310813904

  --> Total Estimated GPU Memory for Complex FT 7B (per GPU): 63.58 GB

--- Example 4: Full SFT QwQ 32B (Mixed Precision, AC, FA, ZeRO-3, TP=2, PP=2, DP=2) ---
  P_total_model_params: 32500000000.0
  P_lora_params: 0
  P_trainable_effective: 32500000000.0
  L_total_layers: 64
  h_hidden_size: 5120
  a_num_attn_heads: 40
  s_seq_len: 131072
  b_micro_batch_size: 1
  optimizer_type: AdamW
  N_os_optimizer_states: 2
  bytes_param_compute: 2
  bytes_param_master: 4
  bytes_gradient: 2
  bytes_optimizer_state: 4
  bytes_activation: 2
  use_activation_checkpointing: True
  use_flash_attention: True
  use_lora: False
  use_qlora: False
  qlora_base_bits: None
  dp_degree: 2
  tp_degree: 2
  pp_degree: 2
  zero_stage: 3
  breakdown_gb:
    parameters: 90.80 GB
    gradients: 30.27 GB
    optimizer_states: 121.07 GB
    activations: 60.00 GB
    subtotal_compute_memory: 302.14 GB
    fixed_overhead: 1.50 GB
    total_with_fixed_overhead: 303.64 GB
    percentage_overhead_applied: 15.18 GB
  total_estimated_gpu_memory_gb: 318.8260628700256

  --> Total Estimated GPU Memory for QwQ 32B SFT (per GPU): 318.83 GB
```
