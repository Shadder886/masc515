"""
microgpt_enhanced.py
Enhanced microgpt with 4 advanced algorithms:
1. GELUs - Gaussian Error Linear Units activation
2. RoPE - Rotary Position Embedding
3. LoRA - Low Rank Adaptation
4. MoE - Mixture of Experts

@author: MASC 515 Assignment 3
"""

import os
import math
import random

random.seed(42)

# ============================================================================
# DATASET
# ============================================================================
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ============================================================================
# TOKENIZER
# ============================================================================
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ============================================================================
# AUTOGRAD ENGINE
# ============================================================================
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children if children else ()
        self._local_grads = local_grads if local_grads else ()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0.0, self.data), (self,), (float(self.data > 0),))

    def gelu(self):
        """
        GELUs: Gaussian Error Linear Units activation.
        GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
        """
        x = self.data
        pi = math.pi
        sqrt_2_over_pi = math.sqrt(2.0 / pi)
        c = 0.044715
        inner = sqrt_2_over_pi * (x + c * x ** 3)
        out = 0.5 * x * (1.0 + math.tanh(inner))
        # Gradient
        tanh_inner = math.tanh(inner)
        sech_sq = 1.0 - tanh_inner ** 2
        grad = 0.5 * (1 + tanh_inner) + 0.5 * x * sech_sq * sqrt_2_over_pi * (1 + 3 * c * x ** 2)
        return Value(out, (self,), (grad,))

    def tanh(self):
        out = math.tanh(self.data)
        return Value(out, (self,), (1.0 - out ** 2,))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# ============================================================================
# MODEL CONFIG
# ============================================================================
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

# LoRA config
lora_rank = 4
lora_alpha = 2.0
use_lora = True

# MoE config
num_experts = 4
top_k = 2
use_moe = True

# ============================================================================
# WEIGHT INITIALIZATION
# ============================================================================
def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

def lora_matrix(nout, nin, rank, std=0.02):
    """LoRA A (init random) and B (init zeros) matrices."""
    A = [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(rank)]
    B = [[Value(0.0) for _ in range(rank)] for _ in range(nout)]
    return A, B

# Build state dict
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}

lora_state = {}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)

    if use_lora:
        lora_state[f'layer{i}.lora_q_A'], lora_state[f'layer{i}.lora_q_B'] = lora_matrix(n_embd, n_embd, lora_rank)
        lora_state[f'layer{i}.lora_k_A'], lora_state[f'layer{i}.lora_k_B'] = lora_matrix(n_embd, n_embd, lora_rank)

    if use_moe:
        for e in range(num_experts):
            state_dict[f'layer{i}.expert{e}.fc1'] = matrix(4 * n_embd, n_embd)
            state_dict[f'layer{i}.expert{e}.fc2'] = matrix(n_embd, 4 * n_embd)
        state_dict[f'layer{i}.router'] = matrix(num_experts, n_embd)
    else:
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
if use_lora:
    for mat_pair in lora_state.values():
        for row in mat_pair:
            for p in row:
                params.append(p)
print(f"num params: {len(params)}")


# ============================================================================
# RoPE PRECOMPUTATION
# ============================================================================
def precompute_rope_freqs(dim, max_len=2048):
    """
    Precompute sin/cos frequencies for RoPE.
    Returns a list where freqs[pos * (dim//2) + i] gives (sin, cos) for
    position `pos` and dimension pair `i`.
    
    The key insight: RoPE rotates each dimension pair by position-dependent angle.
    - theta_i = base^(-2i/dim) controls frequency per dimension
    - angle = pos * theta_i changes per position
    
    IMPORTANT: dim should be n_embd, NOT head_dim, because RoPE rotates
    dimension pairs across the FULL embedding, not just per head.
    """
    freqs = []
    for pos in range(max_len):
        for i in range(dim // 2):
            theta = 10000.0 ** (-2.0 * i / dim)
            angle = pos * theta
            freqs.append((math.sin(angle), math.cos(angle)))
    return freqs

# RoPE frequencies for full embedding dimension (n_embd=16)
# This ensures correct theta_i = 10000^(-2i/16) for all dimension pairs
rope_freqs = precompute_rope_freqs(n_embd)


# ============================================================================
# CORE MATH
# ============================================================================
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_v = max(v.data for v in logits)
    exps = [(v - max_v).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def lora_linear(x, w, A, B, alpha=lora_alpha):
    """LoRA-enabled linear: y = Wx + (alpha/rank) * BAx"""
    out = linear(x, w)
    if use_lora:
        h = linear(x, A)        # (rank, n_embd) @ x = (rank,)
        delta = linear(h, B)    # (n_embd, rank) @ (rank,) = (n_embd,)
        scale = alpha / lora_rank
        out = [o + d * scale for o, d in zip(out, delta)]
    return out


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


# ============================================================================
# MoE FORWARD
# ============================================================================
def moe_forward(x, layer_idx):
    """
    Mixture of Experts: select top-k experts via router.
    
    The router computes a probability distribution over experts using softmax,
    then selects the top-k most likely experts to process the input.
    Only the selected experts compute their outputs, reducing average compute.
    """
    router_w = state_dict[f'layer{layer_idx}.router']
    router_logits = linear(x, router_w)
    router_probs = softmax(router_logits)  # Convert logits to probabilities
    
    # Top-k selection based on router probabilities
    prob_with_idx = [(p.data, i) for i, p in enumerate(router_probs)]
    prob_with_idx.sort(reverse=True)
    top_k_idx = [idx for _, idx in prob_with_idx[:top_k]]

    # Compute each expert output using GELUs activation
    expert_outs = []
    for e_idx in top_k_idx:
        fc1 = state_dict[f'layer{layer_idx}.expert{e_idx}.fc1']
        fc2 = state_dict[f'layer{layer_idx}.expert{e_idx}.fc2']
        h = linear(x, fc1)
        h_g = [xi.gelu() for xi in h]  # GELUs in MoE experts
        expert_out = linear(h_g, fc2)
        expert_outs.append(expert_out)

    # Weighted sum of expert outputs (weighted by router probabilities)
    output = [sum(router_probs[top_k_idx[e]].data * expert_outs[e][i] 
                  for e in range(len(top_k_idx))) 
              for i in range(len(expert_outs[0]))]
    return output


# ============================================================================
# GPT MODEL
# ============================================================================
def gpt(token_id, pos_id, keys, values):
    # Token embedding
    x = list(state_dict['wte'][token_id])  # copy
    x = rmsnorm(x)

    for li in range(n_layer):
        # ----- Attention -----
        x_res = x
        x = rmsnorm(x)

        # Q, K, V projections
        if use_lora:
            q = lora_linear(x, state_dict[f'layer{li}.attn_wq'],
                          lora_state[f'layer{li}.lora_q_A'],
                          lora_state[f'layer{li}.lora_q_B'])
            k = lora_linear(x, state_dict[f'layer{li}.attn_wk'],
                          lora_state[f'layer{li}.lora_k_A'],
                          lora_state[f'layer{li}.lora_k_B'])
        else:
            q = linear(x, state_dict[f'layer{li}.attn_wq'])
            k = linear(x, state_dict[f'layer{li}.attn_wk'])

        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        # Apply RoPE to Q and K (RoPE must be applied before caching)
        # Note: RoPE encodes ABSOLUTE position, but the attention mechanism
        # naturally computes relative position dependencies
        q_rot = apply_rope_vector(q, pos_id, rope_freqs)
        k_rot = apply_rope_vector(k, pos_id, rope_freqs)

        # Cache rotated K and V
        keys[li].append(k_rot)
        values[li].append(v)

        # Multi-head attention
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q_rot[hs:hs + head_dim]
            k_h = [kv[hs:hs + head_dim] for kv in keys[li]]
            v_h = [vi[hs:hs + head_dim] for vi in values[li]]

            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / (head_dim ** 0.5)
                          for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[i] * v_h[i][j] for i in range(len(v_h)))
                       for j in range(head_dim)]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_res)]

        # ----- MoE / MLP -----
        x_res = x
        x = rmsnorm(x)

        if use_moe:
            x = moe_forward(x, li)
        else:
            x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.gelu() for xi in x]  # GELUs!
            x = linear(x, state_dict[f'layer{li}.mlp_fc2'])

        x = [a + b for a, b in zip(x, x_res)]

    logits = linear(x, state_dict['lm_head'])
    return logits


# ============================================================================
# OPTIMIZER
# ============================================================================
learning_rate, beta1, beta2, eps = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

# ============================================================================
# TRAINING LOOP
# ============================================================================
num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    loss = (1.0 / n) * sum(losses)
    loss.backward()

    lr_t = learning_rate * (1.0 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
        p.grad = 0.0

    print(f"step {step + 1:4d}/{num_steps:4d} | loss {loss.data:.4f} | [GELUs+RoPE+LoRA+MoE]", end='\r')

# ============================================================================
# INFERENCE
# ============================================================================
print()
print("--- inference (hallucinated names) ---")
temp = 0.5
for i in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    tok_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(tok_id, pos_id, keys, values)
        probs = softmax([l / temp for l in logits])
        tok_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if tok_id == BOS:
            break
        sample.append(uchars[tok_id])
    print(f"sample {i + 1:2d}: {''.join(sample)}")

print()
print("=" * 60)
print("MASC 515 Assignment 3 - microgpt Enhanced")
print("Implemented: GELUs + RoPE + LoRA + MoE")
print("=" * 60)
