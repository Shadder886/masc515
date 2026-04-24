# microgpt Enhanced Implementation

**MASC 515 - Assignment 3: AI and microgpt**

This repository contains an enhanced implementation of [Andrej Karpathy's microgpt](https://karpathy.github.io/2026/02/12/microgpt/) with four advanced deep learning algorithms:

1. **GELUs** (Gaussian Error Linear Units)
2. **RoPE** (Rotary Position Embedding)
3. **LoRA** (Low Rank Adaptation)
4. **MoE** (Mixture of Experts)

---

## Table of Contents

- [Overview](#overview)
- [Algorithm 1: GELUs](#algorithm-1-gelus-gaussian-error-linear-units)
- [Algorithm 2: RoPE](#algorithm-2-rope-rotary-position-embedding)
- [Algorithm 3: LoRA](#algorithm-3-lora-low-rank-adaptation)
- [Algorithm 4: MoE](#algorithm-4-moe-mixture-of-experts)
- [Implementation Details](#implementation-details)
- [Usage](#usage)

---

## Overview

microgpt is a minimal, dependency-free implementation of a GPT (Generative Pre-trained Transformer) model in pure Python (~200 lines). It includes:

- Custom autograd engine (`Value` class)
- RMSNorm (Root Mean Square Layer Normalization)
- Multi-Head Self-Attention
- MLP (Multi-Layer Perceptron)
- Adam optimizer
- Text generation

This enhanced version replaces the original components with more sophisticated algorithms that are used in modern large language models.

---

## Algorithm 1: GELUs (Gaussian Error Linear Units)

### Paper
[Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) - Hendrycks and Gimpel, 2016

### Underlying Idea

GELUs are **smooth, probabilistic activation functions** that replace the simple thresholding used by ReLU.

**ReLU's Problem:** ReLU(x) = max(0, x) simply zeros out negative inputs. This creates a "hard" boundary that can cause:
- Dying ReLU problem (neurons that never activate)
- Loss of information from negative inputs

**GELUs Solution:** GELUs use the **cumulative distribution function (CDF)** of the standard normal distribution to weight inputs:

```
GELU(x) = x * Φ(x)
```

Where Φ(x) is the CDF of the standard normal distribution N(0, 1).

**Practical Approximation:**
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

### Why It Works

1. **Smooth gradient flow:** Unlike ReLU, GELUs can have non-zero outputs for negative inputs, allowing gradients to flow even when the neuron isn't "active"

2. **Probabilistic interpretation:** The CDF weighting naturally gates inputs based on how likely they are under a normal distribution

3. **BERT & GPT default:** GELU became the default activation in BERT, GPT-2, GPT-3, and most modern transformers due to consistent performance improvements

### Code Implementation

```python
def gelu(self):
    pi = math.pi
    sqrt_2_over_pi = math.sqrt(2 / pi)
    c1 = 0.044715
    x = self.data
    inner = sqrt_2_over_pi * (x + c1 * x**3)
    out = 0.5 * x * (1 + math.tanh(inner))
    # Gradient for backprop
    tanh_inner = math.tanh(inner)
    grad = 0.5 * (1 + tanh_inner) + 0.5 * x * (1 - tanh_inner**2) * sqrt_2_over_pi * (1 + 3 * c1 * x**2)
    return Value(out, (self,), (grad,))
```

---

## Algorithm 2: RoPE (Rotary Position Embedding)

### Paper
[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Su et al., 2021

### Underlying Idea

RoPE encodes **relative position** information directly into the attention mechanism through **rotations** of query and key vectors.

**Traditional Position Embedding Problem:** Models like GPT-2 use learnable absolute position embeddings added to token embeddings. This requires the model to learn position patterns from scratch and doesn't naturally generalize to longer sequences.

**RoPE Solution:** Instead of adding position information, RoPE **rotates** the Q and K vectors by an angle proportional to their position:

For dimension pair (x₀, x₁) at position m:
```
[x₀']   [cos(mθ)  -sin(mθ)] [x₀]
[x₁'] = [sin(mθ)   cos(mθ)] [x₁]
```

The rotation frequency θ depends on the dimension:
```
θ_i = 10000^(-2i/d)
```

### Why It Works

1. **Relative position encoding:** After applying RoPE, the dot product between Q(m) and K(n) depends only on their **relative distance** (m - n), not their absolute positions

2. **Decay with distance:** Rotations naturally decay with distance because cos(mθ) and sin(mθ) oscillate, making attention scores decrease for distant tokens

3. **Length extrapolation:** RoPE enables better generalization to longer sequences than the training length, as used in LLaMA, GLM-4, and Qwen

4. **No extra parameters:** Position information is encoded through mathematical operations, not learned weights

### Code Implementation

```python
def apply_rope_vector(x, pos_id, freqs):
    """
    Apply RoPE rotation to a single vector x at position pos_id.
    
    For dimension pair (2i, 2i+1), we rotate by angle (pos_id * theta_i):
    [cos(θ)  -sin(θ)] [x0]
    [sin(θ)   cos(θ)] [x1]
    
    This encodes RELATIVE position: after RoPE, dot(q_pos_m, k_pos_n)
    depends only on (m - n), not on m or n individually.
    """
    d = len(x)  # Should be n_embd
    result = list(x)  # Copy
    
    # freqs layout: [pos0_dim0, pos0_dim1, ..., pos1_dim0, pos1_dim1, ...]
    # So for position pos_id and dimension pair i:
    # index = pos_id * (d // 2) + i
    base_idx = pos_id * (d // 2)
    
    for i in range(d // 2):
        freq_idx = base_idx + i
        sin_i, cos_i = freqs[freq_idx]
        idx1, idx2 = 2 * i, 2 * i + 1
        x0, x1 = x[idx1], x[idx2]
        # Rotate the pair
        result[idx1] = x0 * cos_i - x1 * sin_i
        result[idx2] = x0 * sin_i + x1 * cos_i
    return result
```

---

## Algorithm 3: LoRA (Low Rank Adaptation)

### Paper
[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Hu et al., 2021

### Underlying Idea

LoRA enables **efficient fine-tuning** of large models by learning **low-rank updates** to the weight matrices.

**The Problem:** Fine-tuning a large model (e.g., 7B parameters) requires updating all parameters, which is:
- Computationally expensive
- Memory intensive (storing gradients and optimizer states)
- Risk of catastrophic forgetting

**LoRA Insight:** The weight update ΔW during fine-tuning often has low "intrinsic rank." Instead of learning ΔW directly, LoRA factorizes it:

```
ΔW = B × A
```

Where:
- A: (rank × d_in) - randomly initialized
- B: (d_out × rank) - initialized to zero
- rank << min(d_in, d_out)

The forward pass becomes:
```
h = Wx + (α/rank) × BAx
```

### Why It Works

1. **Parameter efficiency:** With rank=4 and hidden_dim=4096:
   - Standard update: 4096 × 4096 = 16M parameters
   - LoRA update: 4 × 4096 + 4096 × 4 = 32K parameters (500× reduction!)

2. **Frozen pretrained weights:** Only LoRA matrices are trained, preserving the model's knowledge

3. **Composability:** Multiple LoRA adapters can be trained for different tasks and swapped at inference

4. **Widespread adoption:** LoRA, QLoRA, and variants (AdaLoRA, DoRA) are the standard for efficient LLM fine-tuning

### Code Implementation

```python
def lora_linear(x, w_orig, lora_A, lora_B, alpha=1.0):
    # Standard linear
    out = linear(x, w_orig)

    # LoRA update: x @ B @ A
    h = linear(x, lora_A)      # (batch, rank)
    delta = linear(h, lora_B) # (batch, out_dim)
    scale = alpha / lora_rank

    return [o + d * scale for o, d in zip(out, delta)]
```

---

## Algorithm 4: MoE (Mixture of Experts)

### Paper
[Mixture of Experts](https://huggingface.co/blog/moe) - Various sources; landmark papers include ST-MoE, Mixtral

### Underlying Idea

MoE replaces a single feedforward network with **multiple expert networks** and a **router** that selects which experts handle each input token.

**Traditional Transformer MLP:**
```
output = FFN(input)  # Same computation for all tokens
```

**MoE Architecture:**
```
output = Σᵢ (weight_i × Expert_i(input))  # Only top-k experts active
```

The router computes a probability distribution over experts:
```
weights = softmax(Router(input))
```

Only the top-k experts (e.g., top-2) are activated per token.

### Why It Works

1. **Sparse activation:** Only k out of N experts are active per token, dramatically reducing compute while increasing model capacity

2. **Expert specialization:** Different experts can learn to handle different types of information (e.g., syntax, semantics, numbers, code)

3. **Parameter scaling:** An MoE model with E experts and 1 active has E× more parameters but only ~k/E× more compute

4. **Example - Mixtral 8×7B:** 8 experts total, 2 active per token → effectively a 46B parameter model at 12B compute cost

### Code Implementation

```python
def moe_forward(x, layer_idx):
    # Compute router weights
    router_logits = linear(x, state_dict[f'layer{layer_idx}.router'])
    router_probs = softmax(router_logits)

    # Select top-k experts
    top_k_indices = top_k(router_probs, k=top_k)

    # Normalize selected weights
    total_weight = sum(router_probs[i] for i in top_k_indices)
    top_k_weights = [p / total_weight for p in top_k_weights]

    # Compute expert outputs and combine
    expert_outputs = []
    for expert_idx in top_k_indices:
        expert_out = expert_forward(x, expert_idx)
        expert_outputs.append(expert_out)

    return weighted_sum(top_k_weights, expert_outputs)
```

---

## Implementation Details

### Architecture Comparison

| Component | Original microgpt | Enhanced microgpt |
|-----------|-------------------|-------------------|
| Activation | ReLU | GELUs |
| Position Encoding | Absolute (additive) | RoPE (rotary) |
| Fine-tuning | Full parameter update | LoRA (rank=4) |
| MLP | Single FFN | MoE (4 experts, top-2) |

### Configuration

```python
n_layer = 1           # Transformer layers
n_embd = 16           # Embedding dimension
n_head = 4            # Attention heads
head_dim = 4          # Per-head dimension
lora_rank = 4         # LoRA decomposition rank
num_experts = 4       # MoE expert count
top_k = 2             # Active experts per token
```

### Dependencies

- **None!** This implementation uses only Python standard library:
  - `os` - file operations
  - `math` - mathematical functions
  - `random` - randomization
  - `urllib` - data download

---

## Usage

### Training

```bash
# Run training (will download names.txt automatically)
python microgpt_enhanced.py

# Training outputs progress:
# step 1 / 1000 | loss 3.5824 | GELUs/RoPE/LoRA/MoE active
# ...
# step 1000 / 1000 | loss 1.2345 | GELUs/RoPE/LoRA/MoE active
```

### Inference

After training, the model generates names:
```
sample  1: ariah
sample  2: makya
sample  3: leelani
...
```

### Modifying Configuration

To experiment with different settings:

```python
# Adjust LoRA rank (higher = more parameters, potentially better)
lora_rank = 8

# Adjust MoE expert count (more experts = more parameters)
num_experts = 8
top_k = 2

# Disable algorithms for comparison
use_lora = False
use_moe = False
```

---

## References

1. Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv:1606.08415

2. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864

3. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685

4. Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017

5. Jiang, A., et al. (2024). Mixtral of Experts. arXiv:2401.04088

---

## License

Based on microgpt by Andrej Karpathy. See original project for licensing terms.
