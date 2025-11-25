import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from coral_viz_server import start_background_server, GLOBAL_STATE
from coral_curriculum import SequenceCurriculum

# --- CONFIGURATION ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hardware/Architecture Constraints
MAX_FF_DIM = 1024   # Max width of FFN (Restored for multimodal capacity)
MAX_HEADS = 8       # Max attention heads (Reduced as per user request)
D_MODEL = 256       # Embedding dimension (Increased)
SEQ_LEN = 256       # Context window (16x16 Grid)
VOCAB_SIZE = 256    # Size of token dictionary (Matches SEQ_LEN for pointers)
BATCH_SIZE = 64

# Training Schedule
TIERS = 35
STEPS_PER_TIER = 500

print("Initializing C.O.R.A.L. O-Former (Organic Transformer)...")
print(f"Substrate Capacity: {MAX_HEADS} Heads | {MAX_FF_DIM} FFN Neurons")

# ==========================================
# 1. THE CURRICULUM (The Environment)
# ==========================================


# Curriculum is now imported from coral_curriculum.py

# ==========================================
# 2. THE SLEEPER SUBSTRATE (GPU Friendly)
# ==========================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
            
        if self.cached_cos is None or self.cached_cos.shape[0] < seq_len:
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos()[None, :, None, :]
            self.cached_sin = emb.sin()[None, :, None, :]
        
        return self.cached_cos[:, :seq_len, :, :], self.cached_sin[:, :seq_len, :, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k shape: (B, S, H, D)
    # cos, sin shape: (1, S, 1, D)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SleeperLinear(nn.Module):
    """ Pre-allocated Linear layer that starts sparse. """

    def __init__(self, in_features, max_out_features, chunk_size=16):
        super().__init__()
        self.chunk_size = chunk_size
        self.linear = nn.Linear(in_features, max_out_features)

        # Mask logic
        num_chunks = max_out_features // chunk_size
        self.max_chunks = num_chunks
        self.register_buffer('active_chunks', torch.zeros(num_chunks))
        self.active_chunks[0] = 1.0  # Start with 1 chunk
        
        # Metadata for Visualization
        # -1 means "not created yet"
        self.register_buffer('chunk_tiers', torch.ones(num_chunks) * -1)
        self.chunk_tiers[0] = 1.0 # First chunk is Tier 1

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self._reset_chunk(0)
        self.register_buffer('activity', torch.zeros(max_out_features))
        self.register_buffer('chunk_activity', torch.zeros(num_chunks)) # Coarse activity for viz
        
        # CACHE: Avoid re-calculating indices every step (removes GPU sync)
        self.cached_indices = None
        self.cached_num_active = 0
        self.update_cache()

    def update_cache(self):
        """ Re-calculates active indices. Call this after modifying active_chunks. """
        chunk_indices = (self.active_chunks == 1.0).nonzero().squeeze(1) # (K,)
        self.cached_num_active = chunk_indices.size(0)
        offsets = torch.arange(self.chunk_size, device=self.active_chunks.device)
        self.cached_indices = (chunk_indices.unsqueeze(1) * self.chunk_size + offsets.unsqueeze(0)).view(-1)

    def _reset_chunk(self, idx):
        # print(f"    >> [RESET] Resetting Chunk {idx} weights...")
        start = idx * self.chunk_size
        end = start + self.chunk_size
        
        # FIX: Use explicit data assignment to ensure the slice is updated
        with torch.no_grad():
            # Create fresh random weights
            fresh_w = torch.randn(self.chunk_size, self.linear.in_features, device=self.linear.weight.device) * 0.02
            self.linear.weight.data[start:end, :] = fresh_w
            
            # Also reset bias if it exists
            if self.linear.bias is not None:
                self.linear.bias.data[start:end] = 0.0
        
        # Verify
        # s = self.linear.weight.data[start:end, :].abs().sum().item()
        # print(f"    >> [RESET] Chunk {idx} initialized. Sum: {s:.2f}")

    def forward(self, x):
        # NOTE: This method is NOT used by OrganicBlock, which implements its own
        # sliced execution for performance. This is kept for reference or standalone usage.

        # SELF-HEALING: Check for Dead Chunks (Active but Zero Weights)
        # This fixes corrupted checkpoints or initialization failures.
        # Removed self.training check to ensure this ALWAYS runs
        with torch.no_grad():
             # Check if any active chunk has zero weight sum
             w = self.linear.weight.view(self.max_chunks, self.chunk_size, -1)
             w_sum = w.abs().sum(dim=[1, 2]) # (K,)
             
             # Find chunks that are Active (1.0) but have Zero Weight (0.0)
             for i in range(self.max_chunks):
                 if self.active_chunks[i] > 0.5 and w_sum[i] < 1e-6:
                     # print(f"    >> [SELF-HEAL] Found Dead Chunk {i} (Active but Zero Weights). Re-initializing.")
                     self._reset_chunk(i)

        out = self.linear(x)
        mask = self.active_chunks.repeat_interleave(self.chunk_size)
        out = out * mask
        
        # OPTIMIZATION: Only update stats periodically (10% of steps)
        # This avoids expensive reductions (mean/max) on the GPU every step.
        if self.training and np.random.rand() < 0.1:
            with torch.no_grad():
                # Fine-grained activity (per neuron) - used for pruning logic
                curr_act = (out > 0).float().mean(dim=[0, 1])
                self.activity = 0.9 * self.activity + 0.1 * curr_act
                
                # Coarse-grained activity (per chunk) - used for visualization
                # We take the max activation in the chunk to show "peak firing"
                # Reshape to (B*S, NumChunks, ChunkSize)
                B, S, D = out.shape
                out_reshaped = out.view(B*S, self.max_chunks, self.chunk_size)
                # Max over batch/seq and chunk_dim
                chunk_max = out_reshaped.abs().max(dim=2)[0].mean(dim=0) # Mean over batch
                self.chunk_activity = 0.8 * self.chunk_activity + 0.2 * chunk_max

        return out

    def wake_chunk(self, current_tier=1):
        dead = (self.active_chunks == 0).nonzero()
        if len(dead) > 0:
            idx = dead[0].item()
            self.active_chunks[idx] = 1.0
            self.chunk_tiers[idx] = float(current_tier)
            self._reset_chunk(idx)
            self.update_cache() # Update cache
            return idx
        return None

    def get_active_indices(self):
        """ Returns a tensor of indices for all active neurons. """
        if self.cached_indices is None:
            self.update_cache()
        return self.cached_indices


class SleeperAttention(nn.Module):
    """ Multihead Attention that wakes up new heads. """

    def __init__(self, d_model, max_heads):
        super().__init__()
        self.d_model = d_model
        self.max_heads = max_heads
        self.head_dim = d_model // max_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim)
        
        # Adaptive Rotary Gating
        # Learnable parameter per head: 0 = Absolute, 1 = Relative (RoPE)
        # Initialized to -2.0 (Sigmoid(-2.0) ~= 0.12) -> Bias towards Absolute
        # This gives the model a "Standard Transformer" starting point, which is better for Pointers.
        self.rope_gate = nn.Parameter(torch.ones(max_heads) * -2.0)

        self.register_buffer('head_mask', torch.zeros(max_heads))
        self.head_mask[0] = 1.0
        self.register_buffer('head_entropy', torch.zeros(max_heads))
        
        # CACHE: Avoid re-calculating indices every step (removes GPU sync)
        self.cached_indices = None
        self.cached_num_active = 0
        self.cached_active_heads_idx = None
        self.update_cache()
        
        self._reset_head(0)

    def update_cache(self):
        """ Re-calculates active indices. Call this after modifying head_mask. """
        self.cached_active_heads_idx = (self.head_mask == 1.0).nonzero().squeeze(1) # (K,)
        self.cached_num_active = self.cached_active_heads_idx.size(0)
        
        if self.cached_num_active > 0:
            offsets = torch.arange(self.head_dim, device=self.head_mask.device)
            self.cached_indices = (self.cached_active_heads_idx.unsqueeze(1) * self.head_dim + offsets.unsqueeze(0)).view(-1)
        else:
            self.cached_indices = torch.tensor([], dtype=torch.long, device=self.head_mask.device)

    def _reset_head(self, idx, frustration=0.0):
        start = idx * self.head_dim
        end = start + self.head_dim
        nn.init.xavier_uniform_(self.w_q.weight[start:end, :])
        nn.init.xavier_uniform_(self.w_k.weight[start:end, :])
        nn.init.xavier_uniform_(self.w_v.weight[start:end, :])
        # Initialize Output projection
        # If frustrated, use LOUDER noise to break local minima
        std = 0.02 + (0.08 * frustration) # Up to 0.10 if fully frustrated
        nn.init.normal_(self.w_o.weight[:, start:end], std=std)

    def forward(self, x, mask=None):
        B, S, _ = x.size()
        
        # OPTIMIZATION: Sliced Execution (Structured Sparsity)
        # Only compute projections for active heads.
        
        # 1. Get Active Heads (Cached)
        if self.cached_indices is None:
            self.update_cache()
            
        active_heads_idx = self.cached_active_heads_idx
        num_active = self.cached_num_active
        active_indices = self.cached_indices
        
        if num_active == 0:
            # Should not happen, but handle gracefully
            return torch.zeros_like(x)
            
        # 2. Construct Sliced Weights (Already Cached)
        
        # Slice Q, K, V
        # w_q: (D, D) -> (Active_D, D)
        w_q_sliced = self.w_q.weight[active_indices]
        w_k_sliced = self.w_k.weight[active_indices]
        w_v_sliced = self.w_v.weight[active_indices]
        
        # 3. Compute Projections
        # x: (B, S, D) -> q: (B, S, Active_D)
        # We use F.linear (bias is False)
        q = torch.nn.functional.linear(x, w_q_sliced)
        k = torch.nn.functional.linear(x, w_k_sliced)
        v = torch.nn.functional.linear(x, w_v_sliced)
        
        # Reshape to (B, S, K, D_h)
        q = q.view(B, S, num_active, self.head_dim)
        k = k.view(B, S, num_active, self.head_dim)
        v = v.view(B, S, num_active, self.head_dim)
        
        # Apply RoPE
        # RoPE needs to know the *original* head indices to apply the correct gate?
        # No, RoPE is applied per head dimension, it doesn't care about head index.
        # BUT, the 'rope_gate' IS per head. We need to slice the gate too!
        
        cos, sin = self.rope(v, seq_len=S)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Slice Gate
        active_gate = self.rope_gate[active_heads_idx] # (K,)
        gate = torch.sigmoid(active_gate).view(1, 1, num_active, 1)
        
        q = q * (1 - gate) + q_rot * gate
        k = k * (1 - gate) + k_rot * gate

        # Transpose for Attention: (B, S, K, D) -> (B, K, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # No need to apply head_mask to v, because we only computed active heads!
        
        # OPTIMIZATION: Always use Flash Attention (SDPA) for the forward pass.
        # We calculate entropy separately on a sampled subset to avoid O(N^2) overhead.
        
        # Prepare mask for SDPA
        attn_mask = None
        if mask is not None:
            attn_mask = torch.zeros_like(mask, dtype=q.dtype)
            attn_mask.masked_fill_(mask, float('-inf'))
            attn_mask = attn_mask.view(B, 1, 1, S)
        
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )
        
        # Transpose back: (B, K, S, D) -> (B, S, K, D)
        out_final = out.transpose(1, 2).contiguous().view(B, S, -1)
        
        # Slice Output Projection
        # w_o: (D, D) -> (D, Active_D) (Slice columns)
        w_o_sliced = self.w_o.weight[:, active_indices]
        output = torch.nn.functional.linear(out_final, w_o_sliced)

        # --- Sampled Entropy Calculation (O(N) instead of O(N^2)) ---
        if self.training and np.random.rand() < 0.05:
            with torch.no_grad():
                # Sample K queries (e.g., 32) to estimate entropy
                num_samples = min(32, S)
                indices = torch.randperm(S, device=x.device)[:num_samples]
                
                # q_sample: (B, H, K, D)
                q_sample = q[:, :, indices, :]
                
                # Compute scores for sampled queries: (B, H, K, D) @ (B, H, D, S) -> (B, H, K, S)
                scores = torch.matmul(q_sample, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                
                if mask is not None:
                    m = mask.view(B, 1, 1, S)
                    scores = scores.masked_fill(m, float('-inf'))

                attn_weights = torch.softmax(scores, dim=-1)

                # Update Entropy Stats
                ent = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1) # (B, H, K)
                max_possible_ent = math.log(S)
                ent_norm = ent / (max_possible_ent + 1e-9)
                avg_ent = ent_norm.mean(dim=[0, 2]) # Mean over Batch and Samples
                
                # Update only active heads
                self.head_entropy[active_heads_idx] = 0.95 * self.head_entropy[active_heads_idx] + 0.05 * avg_ent

        return output

    def update_stats(self, x):
        # Not used anymore
        pass

    def wake_head(self, frustration=0.0):
        dead = (self.head_mask == 0).nonzero()
        if len(dead) > 0:
            idx = dead[0].item()
            self.head_mask[idx] = 1.0
            self._reset_head(idx, frustration)
            self.update_cache() # Update cache
            return idx
        return None

# ==========================================
# 3. THE ORGANIC BLOCK (The Cell)
# ==========================================


class OrganicBlock(nn.Module):
    def __init__(self, d_model, max_heads, max_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SleeperAttention(d_model, max_heads)

        self.norm2 = nn.LayerNorm(d_model)
        # Increased chunk_size to 32 to improve visualization fullness while keeping high capacity
        self.ffn_up = SleeperLinear(d_model, max_ff, chunk_size=32)
        self.relu = nn.ReLU()
        self.ffn_down = nn.Linear(max_ff, d_model)
        
        # Growth Cooldown to prevent runaway expansion
        self.growth_cooldown = 0
        
        # Homeostatic Stats (Adaptive Baselines)
        # We track the "Normal" range of stress for this block.
        # Growth is triggered only by statistical outliers (Z-Score > 3).
        self.stats_alpha = 0.05 # Moving average momentum (approx 20 samples window)
        
        self.ffn_mean = 0.5
        self.ffn_var = 0.05
        
        self.ent_mean = 0.5
        self.ent_var = 0.05

        # Initialize FFN Down to SMALL NOISE to allow gradient flow
        # Was Zero, which blocked gradients to FFN Up
        nn.init.normal_(self.ffn_down.weight, std=0.02)
        nn.init.zeros_(self.ffn_down.bias)
        # w_o is handled by SleeperAttention._reset_head now (zeros)

    def forward(self, x, mask=None, residual=True):
        if self.growth_cooldown > 0:
            self.growth_cooldown -= 1
            
        # SELF-HEALING: Check for Dead Chunks (Active but Zero Weights)
        # OPTIMIZATION: Only run this check periodically (1% chance)
        # Dead chunks are rare (initialization/pruning artifacts), no need to scan every step.
        if self.training and np.random.rand() < 0.01:
             with torch.no_grad():
                 w = self.ffn_up.linear.weight.view(self.ffn_up.max_chunks, self.ffn_up.chunk_size, -1)
                 w_sum = w.abs().sum(dim=[1, 2])
                 
                 # Vectorized check (No loop, No sync)
                 # Find chunks that are Active (1.0) but have Zero Weight (0.0)
                 dead_indices = (self.ffn_up.active_chunks > 0.5) & (w_sum < 1e-6)
                 
                 if dead_indices.any():
                     # Only sync if we actually found dead chunks (rare)
                     dead_idx_list = dead_indices.nonzero().squeeze(1).tolist()
                     for i in dead_idx_list:
                         print(f"    >> [SELF-HEAL] Found Dead Chunk {i} in Block. Re-initializing.")
                         self.ffn_up._reset_chunk(i)
                     # Update cache if we reset anything (though reset doesn't change active status, just weights)
                     # But good practice.
                     self.ffn_up.update_cache()

        # Attention Block
        attn_out = self.attn(self.norm1(x), mask=mask)
        x_attn = x + attn_out
        
        # FFN Block
        # OPTIMIZATION: Sliced Execution (Structured Sparsity)
        # Instead of computing the full matrix and masking, we only compute the active neurons.
        # This makes the layer computationally sparse, not just theoretically.
        
        # 1. Get Active Indices
        active_idx = self.ffn_up.get_active_indices()
        
        # 2. Slice Weights (Gather)
        # FFN Up: (Active, D)
        w_up = self.ffn_up.linear.weight[active_idx]
        b_up = self.ffn_up.linear.bias[active_idx]
        
        # 3. Compute Hidden (Reduced Dimensionality)
        # x: (B, S, D) -> h: (B, S, Active)
        # We use F.linear directly
        ffn_hidden = torch.nn.functional.linear(self.norm2(x_attn), w_up, b_up)
        ffn_hidden = self.relu(ffn_hidden)
        
        # Update Activity Stats (mapped back to chunks)
        # OPTIMIZATION: Only update stats periodically (10% of steps) to avoid overhead
        if self.training and np.random.rand() < 0.1:
            with torch.no_grad():
                 # 1. Per-Neuron Activity (for Pruning)
                 curr_act = (ffn_hidden > 0).float().mean(dim=[0, 1])
                 self.ffn_up.activity[active_idx] = 0.9 * self.ffn_up.activity[active_idx] + 0.1 * curr_act

                 # 2. Per-Chunk Activity (for Visualization)
                 # ffn_hidden: (B, S, Active) -> Reshape to (B*S, NumActiveChunks, ChunkSize)
                 # USE CACHED VALUE to avoid GPU sync (.item())
                 num_active_chunks = self.ffn_up.cached_num_active
                 
                 if num_active_chunks > 0:
                     B, S, _ = ffn_hidden.shape
                     # Reshape to (B*S, NumActiveChunks, ChunkSize)
                     hidden_reshaped = ffn_hidden.view(B*S, num_active_chunks, self.ffn_up.chunk_size)
                     # Max over chunk_size (dim 2) -> (B*S, NumActiveChunks)
                     # Mean over batch (dim 0) -> (NumActiveChunks,)
                     chunk_max = hidden_reshaped.abs().max(dim=2)[0].mean(dim=0)
                     
                     # Scatter back to self.ffn_up.chunk_activity
                     # We need the indices of active chunks
                     # Re-calculate indices locally or cache them? 
                     # We can infer them from active_chunks, but that requires .nonzero()
                     # Let's just use the fact that we are updating ALL active chunks.
                     # But self.ffn_up.chunk_activity is size (MaxChunks,).
                     # We need to know WHICH chunks are active to scatter 'chunk_max' (size NumActive) to them.
                     
                     # This .nonzero() is unavoidable if we want to map back, BUT we can cache it too!
                     # Let's add cached_chunk_indices to SleeperLinear.
                     # For now, let's just do the nonzero, it's small (MaxChunks is small, e.g. 64).
                     # But wait, .nonzero() returns a tensor. We use it as an index.
                     
                     active_chunk_indices = (self.ffn_up.active_chunks == 1.0).nonzero().squeeze(1)
                     self.ffn_up.chunk_activity[active_chunk_indices] = 0.8 * self.ffn_up.chunk_activity[active_chunk_indices] + 0.2 * chunk_max

        # 4. Slice FFN Down (Gather Columns)
        # FFN Down: (D, Active)
        w_down = self.ffn_down.weight[:, active_idx]
        b_down = self.ffn_down.bias # Full bias
        
        # 5. Compute Output (Project back to D)
        # h: (B, S, Active) -> out: (B, S, D)
        ffn_out = torch.nn.functional.linear(ffn_hidden, w_down, b_down)
        
        x_out = x_attn + ffn_out
        
        if residual:
            return x_out
        else:
            # Return delta (what this block added to the original x)
            # Delta = (Attn) + (FFN)
            # Note: FFN input is (x+Attn), so it's coupled.
            # But for parallel blocks, we want the total transformation.
            return x_out - x

    def check_micro_growth(self, frustration=0.0, sensitivity=1.0, allow_attn=True, current_tier=1):
        """
        Checks if Width needs to expand based on Homeostatic Stress (Z-Score).
        Instead of fixed thresholds, we check if the current stress is a statistical outlier
        relative to the block's recent history.
        """
        if self.growth_cooldown > 0:
            return None

        # 1. Measure Current State
        
        # FFN Activity
        active_mask = self.ffn_up.active_chunks.repeat_interleave(
            self.ffn_up.chunk_size)
        current_ffn = (
            self.ffn_up.activity * active_mask).sum() / (active_mask.sum() + 1e-9)
        current_ffn = current_ffn.item()

        # Attention Entropy (Mean across active heads)
        # FIX: Use Mean instead of Max. Max is too sensitive to a single confused head,
        # causing the model to spam new heads to "fix" the outlier.
        # Mean ensures we only grow if the *collective* attention is confused.
        active_count = self.attn.head_mask.sum()
        current_ent = (self.attn.head_entropy * self.attn.head_mask).sum() / max(1.0, active_count)
        current_ent = current_ent.item()
        if math.isnan(current_ent): current_ent = 0.0

        # 2. Update Homeostatic Stats (Welford's / EMA)
        # We update stats BEFORE checking for growth to incorporate the new data point
        # into the "normal" range.
        
        # FFN Stats
        diff_ffn = current_ffn - self.ffn_mean
        incr_ffn = self.stats_alpha * diff_ffn
        self.ffn_mean += incr_ffn
        self.ffn_var = (1 - self.stats_alpha) * (self.ffn_var + diff_ffn * incr_ffn)
        
        # Entropy Stats
        diff_ent = current_ent - self.ent_mean
        incr_ent = self.stats_alpha * diff_ent
        self.ent_mean += incr_ent
        self.ent_var = (1 - self.stats_alpha) * (self.ent_var + diff_ent * incr_ent)
        
        # 3. Calculate Z-Scores (Stress)
        # How many standard deviations above normal is the current stress?
        # FIX: Clamp sigma to a minimum value to prevent hypersensitivity when stable.
        # If variance is near zero, any tiny noise becomes a massive Z-Score.
        
        sigma_ffn = max(0.01, math.sqrt(self.ffn_var + 1e-9))
        z_ffn = (current_ffn - self.ffn_mean) / sigma_ffn
        
        sigma_ent = max(0.01, math.sqrt(self.ent_var + 1e-9))
        z_ent = (current_ent - self.ent_mean) / sigma_ent
        
        # 4. Determine Thresholds
        # Base Threshold: 3.0 Sigma (99.7% confidence it's an outlier)
        # Frustration lowers the bar: If frustrated, we accept 1.5 Sigma.
        # Sensitivity scales the threshold (Higher sensitivity = Lower threshold)
        base_thresh = 3.0 / sensitivity
        
        if frustration > 0.0:
            # Linearly interpolate: 0.0 -> 3.0, 1.0 -> 1.5
            thresh = base_thresh - (1.5 * frustration)
        else:
            thresh = base_thresh

        # DEBUG: Trace Growth Logic
        if frustration > 0.1 or z_ent > 2.0:
             print(f"    >> [MICRO] Frust: {frustration:.2f} | Ent: {current_ent:.2f} (u={self.ent_mean:.2f}, s={sigma_ent:.2f}) | Z-Score: {z_ent:.2f} | Thresh: {thresh:.2f}")

        # 5. Winner Takes All
        # We only grow the component with the HIGHEST Z-Score (if it exceeds threshold)
        
        if z_ffn > z_ent and z_ffn > thresh:
            # Grow FFN
            if self.ffn_up.active_chunks.sum() < self.ffn_up.max_chunks:
                print(f"    >> [MICRO] FFN Saturation (Z={z_ffn:.2f} > {thresh:.2f}). Waking Chunk.")
                self.growth_cooldown = 50 # Refractory period
                self.ffn_up.wake_chunk(current_tier=current_tier)
                return ('ffn', None)
            elif self.ffn_up.active_chunks.sum() >= self.ffn_up.max_chunks:
                 # FFN Full -> Trigger Block Split (Macro)
                 return ('block', None)

        elif z_ent > z_ffn and z_ent > thresh:
            # Grow Attention
            if allow_attn:
                if self.attn.head_mask.sum() < self.attn.max_heads:
                    print(f"    >> [MICRO] Head Confusion (Z={z_ent:.2f} > {thresh:.2f}). Waking Head.")
                    self.growth_cooldown = 150 # Increased from 50 to give new heads time to mature
                    idx = self.attn.wake_head(frustration)
                    return ('attn', idx)
                elif self.attn.head_mask.sum() >= self.attn.max_heads:
                    # Heads Full -> Trigger Block Split (Macro)
                    return ('block', None)
            else:
                 if z_ent > thresh + 1.0:
                     print(f"    >> [MICRO] Head Confusion (Z={z_ent:.2f}) suppressed by Load Balancing.")
                
        return None


class OrganicLayer(nn.Module):
    """
    A Layer that can grow horizontally (add parallel blocks).
    """
    def __init__(self, d_model, max_heads, max_ff):
        super().__init__()
        self.d_model = d_model
        self.max_heads = max_heads
        self.max_ff = max_ff
        self.blocks = nn.ModuleList([
            OrganicBlock(d_model, max_heads, max_ff)
        ])

    def forward(self, x, mask=None):
        # Parallel Execution: x_out = x + Block1(x) + Block2(x) ...
        # Since OrganicBlock(residual=False) returns delta, we sum deltas.
        delta_sum = 0
        for block in self.blocks:
            delta_sum = delta_sum + block(x, mask=mask, residual=False)
        
        # Normalize the sum by the number of blocks to prevent signal explosion
        # This acts like a mean-field approximation and stabilizes training as width grows
        return x + (delta_sum / len(self.blocks))

    def grow_width(self):
        print(f"  >>> [MACRO] Growing WIDTH (Adding Parallel Block to Layer).")
        new_block = OrganicBlock(self.d_model, self.max_heads, self.max_ff)
        # Move to device
        if len(self.blocks) > 0:
            device = self.blocks[0].norm1.weight.device
            new_block = new_block.to(device)
        self.blocks.append(new_block)
        return new_block

    def forward(self, x, mask=None):
        # Parallel Execution: x_out = x + Block1(x) + Block2(x) ...
        # Since OrganicBlock(residual=False) returns delta, we sum deltas.
        delta_sum = 0
        for block in self.blocks:
            delta_sum = delta_sum + block(x, mask=mask, residual=False)
        
        # Normalize the sum by the number of blocks to prevent signal explosion
        # This acts like a mean-field approximation and stabilizes training as width grows
        return x + (delta_sum / len(self.blocks))


class HybridInputAdapter(nn.Module):
    def __init__(self, d_model, vocab_size, patch_size=16):
        super().__init__()
        self.d_model = d_model
        # 1. For our current Curriculum (Discrete)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # FIX: Initialize embeddings with larger variance (Xavier-ish) to ensure signal flow
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.1)
        
        # 2. For Real Images (Continuous)
        # Projects 3 channels -> d_model
        # We assume input images are scaled such that patch_size divides dimensions
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Auto-detect mode based on input type
        if x.dtype == torch.long:
            # It's a curriculum task (Integers)
            # SCALE EMBEDDINGS: Multiply by sqrt(d_model) to balance with Positional Encodings
            return self.token_embed(x) * math.sqrt(self.d_model)
        else:
            # It's an image (Floats) [Batch, Channels, Height, Width]
            # Project patches and flatten to sequence
            x = self.patch_embed(x) 
            # x shape: [Batch, D_Model, H_grid, W_grid]
            # Flatten to [Batch, D_Model, Seq_Len] then transpose to [Batch, Seq_Len, D_Model]
            return x.flatten(2).transpose(1, 2)


class ArchitectController:
    """
    A Meta-Learning Agent (Contextual Bandit) that decides HOW to grow the network.
    It learns to map (State) -> (Action) by maximizing Reward (Loss Drop - Cost).
    """
    def __init__(self, learning_rate=0.01, epsilon=0.05):
        self.lr = learning_rate
        self.epsilon = epsilon
        
        # Actions: 0=Wait, 1=Wake Head, 2=Wake Chunk, 3=Add Block, 4=Add Layer
        self.actions = ['wait', 'head', 'chunk', 'block', 'layer']
        self.n_actions = len(self.actions)
        
        # State Dimension: 
        # 1. Frustration (0-1)
        # 2. Loss Derivative (Slope)
        # 3. Entropy Mean (0-1)
        # 4. FFN Saturation (0-1)
        # 5. Current Depth (Normalized)
        # 6. Current Width (Normalized)
        # 7. Current Heads (Normalized)
        self.state_dim = 7
        
        # Q-Function Weights: [n_actions, state_dim]
        # Initialize with heuristics
        self.weights = torch.zeros(self.n_actions, self.state_dim)
        
        # Bias:
        # Wait: Default preference when frustration is low
        self.weights[0, 0] = -1.0 # Frustration -> Wait (High Frustration = Low Wait Score)
        self.weights[0, 1] = -1.0 # Slope -> Wait (Negative Slope (improving) = High Wait Score)
        
        # Head: High base preference
        self.weights[1, 0] = 0.8 # Frustration -> Head (Increased to favor heads)
        
        # Chunk: Medium preference (FFN Width)
        self.weights[2, 0] = 0.4 # Frustration -> Chunk
        self.weights[2, 3] = 0.1 # Saturation -> Chunk (Reduced from 0.5 to prevent runaway chunk growth)
        
        # Block: High preference (Encourage Width)
        # We want the model to try parallel processing before resorting to depth.
        # Since blocks are pruned if unused, the risk is low.
        self.weights[3, 0] = 0.8 # Frustration -> Block (Higher than Head to encourage width)
        self.weights[3, 5] = -1.0 # Width -> Block (Penalty if already very wide, but we only allow 2 anyway)
        
        # Layer: Negative base preference (Costly)
        self.weights[4, 0] = -0.5 # Frustration -> Layer (Don't panic add layers)
        self.weights[4, 4] = -0.5 # Depth -> Layer (Moderate penalty for depth)
        self.weights[4, 3] = 0.5 # Saturation -> Layer (If FFN is full, we need a new layer)
        
        # Dynamic Head Weight: Fixed to 0.8
        # We use a fixed weight because 'norm_heads' (state[6]) is already normalized (0.0-1.0).
        # Whether MAX_HEADS is 8 or 32, '1.0' means full.
        # 0.8 ensures we prioritize filling heads, but once full (and FFN partially full), we add a layer.
        self.weights[4, 6] = 0.8
        
        self.pending_action = None # (state, action_idx, step, start_loss)

    def get_state(self, brain, frustration, loss_window):
        # Construct State Vector
        if len(loss_window) > 10:
            slope = (loss_window[-1] - loss_window[-10]) / 10.0
        else:
            slope = 0.0
            
        # Entropy & Saturation
        ent_sum = 0
        ffn_sum = 0
        n_blocks = 0
        for layer in brain.layers:
            for block in layer.blocks:
                ent_sum += block.ent_mean
                ffn_sum += (block.ffn_up.active_chunks.sum() / block.ffn_up.max_chunks)
                n_blocks += 1
        
        avg_ent = ent_sum / max(1, n_blocks)
        avg_ffn = ffn_sum / max(1, n_blocks)
        
        # Architecture Stats
        norm_depth = len(brain.layers) / 12.0 # Assume max 12 layers for normalization
        norm_width = n_blocks / (len(brain.layers) * 2.0) # Max 2 blocks per layer
        
        # Calculate Normalized Heads (Saturation)
        # FIX: Use MINIMUM layer saturation instead of average.
        # This prevents "Empty Skyscraper" syndrome where the model adds layers
        # on top of empty ones. It forces the model to fill existing capacity first.
        layer_saturations = []
        for layer in brain.layers:
            l_heads = 0
            l_max = 0
            for block in layer.blocks:
                l_heads += block.attn.head_mask.sum().item()
                l_max += block.attn.max_heads
            layer_saturations.append(l_heads / max(1, l_max))
            
        norm_heads = min(layer_saturations) if layer_saturations else 0.0
        
        state = torch.tensor([
            frustration,
            slope,
            avg_ent,
            avg_ffn,
            norm_depth,
            norm_width,
            norm_heads
        ], dtype=torch.float32)
        
        return state

    def decide(self, state, valid_mask):
        """
        Selects an action using Epsilon-Greedy policy.
        valid_mask: Boolean array [n_actions] indicating which actions are possible.
        """
        # 1. Calculate Q-Values: Q = W * s
        q_values = torch.mv(self.weights, state)
        
        # Mask invalid actions (set Q to -infinity)
        q_values[~valid_mask] = -float('inf')
        
        # 2. Epsilon-Greedy
        if np.random.rand() < self.epsilon:
            # Explore: Random valid action
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) == 0: return None
            action_idx = valid_indices[np.random.choice(len(valid_indices))].item()
        else:
            # Exploit: Argmax Q
            action_idx = torch.argmax(q_values).item()
            
        return action_idx

    def store_action(self, state, action_idx, step, current_loss):
        self.pending_action = (state, action_idx, step, current_loss)

    def update(self, step, current_loss):
        """
        Updates Q-Function based on reward.
        Reward = (Start_Loss - Current_Loss) - Cost
        """
        if self.pending_action is None:
            return
            
        state, action_idx, start_step, start_loss = self.pending_action
        
        # Evaluate after 100 steps
        if step - start_step >= 100:
            # Calculate Reward
            delta_loss = start_loss - current_loss
            
            # Metabolic Costs (Heuristic)
            # FIX: Balanced costs.
            # Layer cost 1.0 means we need a solid loss drop, but not impossible.
            # Block cost reduced to 0.3 to encourage width exploration.
            # Chunk cost 0.25 (Slightly increased to prevent spam)
            # Head cost 0.05 (Very cheap to encourage fine-grained attention)
            costs = [-0.1, 0.05, 0.25, 0.3, 1.0] # Wait (Bonus), Head, Chunk, Block, Layer
            cost = costs[action_idx]
            
            reward = delta_loss - cost
            
            # Update Weights (Gradient Descent on MSE)
            # Target = Reward (We want Q(s,a) to predict the Reward)
            # Error = Reward - Q(s,a)
            # Delta_W = lr * Error * s
            
            q_pred = torch.dot(self.weights[action_idx], state)
            error = reward - q_pred
            
            self.weights[action_idx] += self.lr * error * state
            
            print(f"    >> [ARCHITECT] Feedback: Action {self.actions[action_idx]} | Reward: {reward:.4f} (dL: {delta_loss:.4f} - C: {cost}) | Q-Pred: {q_pred:.4f} -> {q_pred + error:.4f}")
            
            self.pending_action = None

class OFormer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # Replaced simple embedding with Hybrid Adapter
        self.input_adapter = HybridInputAdapter(d_model, vocab_size)
        
        # Re-introduced Absolute Positional Encodings to help with "Jump to Index" tasks (Tier 13)
        # RoPE handles relative attention, but Absolute PosEnc helps map "Value P" to "Position P".
        self.pos_enc = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        # INDUCTIVE BIAS: Sinusoidal Initialization
        # Instead of random noise, we give the model a "Ruler" to measure distance immediately.
        # This is critical for Pointer tasks (Tier 13+).
        with torch.no_grad():
            pe = torch.zeros(SEQ_LEN, d_model)
            position = torch.arange(0, SEQ_LEN, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pos_enc.data[0] = pe
        
        self.layers = nn.ModuleList([
            OrganicLayer(d_model, MAX_HEADS, MAX_FF_DIM)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Metabolic Feedback (Option 2)
        # Tracks the success of previous growth events to regulate future growth.
        self.growth_sensitivity = 1.0 # Multiplier for growth thresholds (1.0 = Normal, >1.0 = Cautious)
        self.pending_feedback = [] # List of (step_count, loss_at_growth)
        
        # NEW: Architect Controller
        self.architect = ArchitectController()

    def forward(self, x, mask=None, lengths=None):
        # 1. Embed Input (Discrete Tokens or Visual Patches)
        x = self.input_adapter(x)
        
        # 2. Add Positional Encoding
        seq_len = x.size(1)
        if seq_len > self.pos_enc.size(1):
             raise ValueError(f"Input sequence length {seq_len} exceeds maximum {self.pos_enc.size(1)}")
            
        x = x + self.pos_enc[:, :seq_len, :]
        
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        # Select the last valid token for each sequence
        if lengths is not None:
            # lengths: (B,)
            # We want x[b, lengths[b]-1, :]
            # Ensure lengths are at least 1 and do not exceed current sequence length
            safe_lengths = torch.clamp(lengths, min=1, max=x.size(1))
            batch_indices = torch.arange(x.size(0), device=x.device)
            last_tokens = x[batch_indices, safe_lengths - 1, :]
            return self.fc_out(last_tokens)
        else:
            # Fallback to last token (assuming full sequence)
            return self.fc_out(x[:, -1, :])

    def check_micro_growth(self, x, y, optimizer, frustration=0.0, current_loss=None, step=0, current_tier=1):
        # Check last layer for saturation/confusion
        if len(self.layers) == 0:
            return optimizer, False
            
        # 1. Process Metabolic Feedback
        # Check if previous growth paid off
        if current_loss is not None:
            remaining = []
            for grow_step, old_loss in self.pending_feedback:
                if step - grow_step > 100: # Evaluate after 100 steps
                    if current_loss < old_loss * 0.9:
                        # Success! Growth helped. We are on the right track. Maintain or slightly increase plasticity.
                        self.growth_sensitivity = min(2.0, self.growth_sensitivity * 1.05)
                        print(f"    >> [METABOLISM] Growth Successful (Loss {old_loss:.2f} -> {current_loss:.2f}). Sensitivity: {self.growth_sensitivity:.2f}")
                    else:
                        # Failure! Growth didn't help. Reduce sensitivity (raise threshold) to prevent panic.
                        self.growth_sensitivity = max(0.5, self.growth_sensitivity * 0.8)
                        print(f"    >> [METABOLISM] Growth Failed (Loss {old_loss:.2f} -> {current_loss:.2f}). Sensitivity: {self.growth_sensitivity:.2f}")
                else:
                    remaining.append((grow_step, old_loss))
            self.pending_feedback = remaining

        # Iterate through ALL layers, starting from the top (most plastic)
        # This ensures older layers can still grow if they become bottlenecks.
        for layer_idx, layer in enumerate(reversed(self.layers)):
            real_layer_idx = len(self.layers) - 1 - layer_idx
            
            # Load Balancing: Calculate min heads across blocks IN THIS LAYER
            block_heads = [b.attn.head_mask.sum().item() for b in layer.blocks]
            min_heads = min(block_heads) if block_heads else 0
            
            for block_idx, block in enumerate(layer.blocks):
                # Only allow attention growth if we are the smallest block (or equal)
                allow_attn = (block_heads[block_idx] == min_heads)
                
                # Pass sensitivity to block
                res = block.check_micro_growth(frustration, sensitivity=self.growth_sensitivity, allow_attn=allow_attn, current_tier=current_tier)
                if res:
                    type, val = res
                    
                    # Record Growth Event for Feedback
                    if current_loss is not None:
                        self.pending_feedback.append((step, current_loss))
                    
                    # LOGGING TO VIZ
                    desc = ""
                    if type == 'ffn':
                        desc = f"Micro: Added FFN Chunk (Layer {real_layer_idx})"
                    elif type == 'attn':
                        desc = f"Micro: Added Attention Head (Layer {real_layer_idx})"
                    elif type == 'block':
                        desc = f"Macro: Added Parallel Block (Layer {real_layer_idx})"
                    
                    if desc:
                        GLOBAL_STATE['events'].append({
                            "step": step,
                            "type": "growth",
                            "desc": desc
                        })

                    if type == 'block':
                        # FIX: Enforce Width Limit (Max 2 blocks per layer)
                        # This forces the model to grow DEPTH (Macro Growth) when width is saturated.
                        if len(layer.blocks) < 2:
                            print(f"    >> [DECISION] Growing WIDTH (New Parallel Block in Layer {real_layer_idx}).")
                            layer.grow_width()
                            # Update Optimizer to include new block parameters
                            # NOTE: We do NOT trigger full neurogenesis for micro-block growth, just update the optimizer.
                            return get_opt(self, new_layer_idx=None, old_optimizer=optimizer, neurogenesis=False), True
                        else:
                            # print(f"    >> [DECISION] Width Limit Reached (Max 2). Denying Block Growth.")
                            return optimizer, False
                    
                    # For 'ffn' and 'attn', weights were modified in-place.
                    # We return the existing optimizer.
                    return optimizer, True
        return optimizer, False

    def grow_width(self):
        if len(self.layers) > 0:
            self.layers[-1].grow_width()

    def grow_depth(self):
        print(
            f"  >>> [MACRO] Abstraction Wall Hit. Growing DEPTH (Layer {len(self.layers)+1}).")
        new_layer = OrganicLayer(D_MODEL, MAX_HEADS, MAX_FF_DIM)
        # Move to the same device as the rest of the model
        device = self.fc_out.weight.device
        new_layer = new_layer.to(device)
        self.layers.append(new_layer)

    def calculate_metabolic_cost(self):
        """
        Calculates the L1 norm of all active weights.
        This acts as a 'Metabolic Cost' for the organism.
        Minimizing this encourages sparsity and pruning of useless connections.
        """
        cost = 0.0
        for layer in self.layers:
            for block in layer.blocks:
                # FFN Cost
                cost += torch.abs(block.ffn_up.linear.weight).sum()
                cost += torch.abs(block.ffn_down.weight).sum()
                
                # Attention Cost
                cost += torch.abs(block.attn.w_q.weight).sum()
                cost += torch.abs(block.attn.w_k.weight).sum()
                cost += torch.abs(block.attn.w_v.weight).sum()
                cost += torch.abs(block.attn.w_o.weight).sum()
        return cost

    def sleep(self, current_tier=1):
        """
        Sleep Phase: Prune weak connections using Adaptive Thresholding.
        We calculate the 1st percentile of weight magnitudes per layer.
        We prune weights below this percentile, but capped at a safe maximum (0.001).
        This ensures we prune 'relatively' weak connections, but never strong ones.
        """
        print(f"  >>> [SLEEP] Pruning weak connections (Adaptive)...")
        with torch.no_grad():
            for layer in self.layers:
                for block in layer.blocks:
                    # 1. Determine Adaptive Threshold for this Block
                    # Collect sample of weights to estimate distribution
                    # We use FFN Down as a proxy for the block's scale
                    w_sample = block.ffn_down.weight.detach().abs()
                    if w_sample.numel() > 0:
                        # 1st percentile
                        quantile_thresh = torch.quantile(w_sample, 0.01).item()
                    else:
                        quantile_thresh = 0.0
                    
                    # Cap threshold: Never prune anything > 0.001
                    # Also ensure we don't prune everything if weights are tiny (min 1e-6)
                    threshold = max(1e-6, min(0.001, quantile_thresh))
                    
                    # PROTECT NEW CHUNKS: Do not prune chunks that are very new or have very small weights
                    # If a chunk is active but has small weights, it might be new.
                    # Pruning it to zero kills it before it can grow.
                    # We only prune if the weight is small AND it's not the only chunk.
                    
                    # FFN Pruning
                    active_mask = block.ffn_up.active_chunks.repeat_interleave(block.ffn_up.chunk_size).view(-1, 1)
                    
                    # FFN Up
                    w = block.ffn_up.linear.weight
                    should_prune = (active_mask == 1) & (torch.abs(w) < threshold)
                    
                    # FIX: Don't prune if the chunk is new (check if all weights in chunk are small)
                    # Actually, just be less aggressive. If a weight is < threshold, it's noise.
                    # But if the WHOLE chunk is < threshold, it's a new chunk!
                    # Let's skip pruning for chunks that have very low total weight (newly initialized)
                    # unless they are REALLY dead.
                    
                    w.data.masked_fill_(should_prune, 0.0)
                    
                    # FFN Down
                    active_mask_t = active_mask.view(1, -1)
                    w = block.ffn_down.weight
                    should_prune = (active_mask_t == 1) & (torch.abs(w) < threshold)
                    w.data.masked_fill_(should_prune, 0.0)

                    # Deactivate FFN Chunks
                    for c in range(len(block.ffn_up.active_chunks)):
                        if block.ffn_up.active_chunks[c] == 1.0:
                            start = c * block.ffn_up.chunk_size
                            end = start + block.ffn_up.chunk_size
                            w_down_slice = block.ffn_down.weight[:, start:end]
                            w_up_slice = block.ffn_up.linear.weight[start:end, :]
                            
                            # Check if effectively dead (using a slightly higher threshold for chunk death)
                            # FIX: Increased threshold to 1e-3 to catch "zombie" chunks that are barely active.
                            # BUT: If it's a NEW chunk, it might be small.
                            # We should check if it was RECENTLY added.
                            # For now, let's just be careful not to kill it if it has ANY life.
                            
                            # NEW: Check Activity (ReLU Death)
                            chunk_act = block.ffn_up.activity[start:end].mean().item()
                            
                            # Check Weights (Mass)
                            w_sum = torch.abs(w_down_slice).sum() + torch.abs(w_up_slice).sum()
                            
                            # Check Age (Protection)
                            chunk_tier = block.ffn_up.chunk_tiers[c].item()
                            is_new = (chunk_tier == current_tier)
                            
                            # Criteria
                            # 1. Dead ReLU: Activity is near zero
                            is_dead_relu = (chunk_act < 1e-5)
                            
                            # 2. Weak Weights: Total mass is insignificant (< 1.0, approx 1% of init)
                            is_weak_weights = (w_sum < 1.0) 
                            
                            should_kill = False
                            reason = ""
                            
                            if not is_new:
                                if is_dead_relu:
                                    should_kill = True
                                    reason = f"Dead ReLU (Act {chunk_act:.5f})"
                                elif is_weak_weights:
                                    should_kill = True
                                    reason = f"Weak Weights (Sum {w_sum:.4f})"
                            
                            if should_kill:
                                if block.ffn_up.active_chunks.sum() > 2:
                                     print(f"    >> [SLEEP] Deactivating FFN Chunk {c} ({reason}).")
                                     block.ffn_up.active_chunks[c] = 0.0
                                     block.ffn_up.chunk_tiers[c] = -1.0 # Reset Tier to indicate removal
                                     block.ffn_up.activity[start:end] = 0.0
                                     block.ffn_up.chunk_activity[c] = 0.0 # Reset Viz Activity
                    
                    # Update Cache after pruning
                    block.ffn_up.update_cache()

                    # Prune Attention
                    head_dim = block.attn.head_dim
                    h_mask = block.attn.head_mask.repeat_interleave(head_dim)
                    h_mask_rows = h_mask.view(-1, 1)
                    h_mask_cols = h_mask.view(1, -1)
                    
                    for param in [block.attn.w_q.weight, block.attn.w_k.weight, block.attn.w_v.weight]:
                        should_prune = (h_mask_rows == 1) & (torch.abs(param) < threshold)
                        param.data.masked_fill_(should_prune, 0.0)
                    
                    param = block.attn.w_o.weight
                    should_prune = (h_mask_cols == 1) & (torch.abs(param) < threshold)
                    param.data.masked_fill_(should_prune, 0.0)

                    # Deactivate Heads
                    active_heads_count = block.attn.head_mask.sum().item()
                    for h in range(block.attn.max_heads):
                        if block.attn.head_mask[h] == 1.0:
                            start = h * block.attn.head_dim
                            end = start + block.attn.head_dim
                            w_o_slice = block.attn.w_o.weight[:, start:end]
                            
                            if torch.abs(w_o_slice).sum() < 1e-4:
                                if active_heads_count > 1:
                                    print(f"    >> [SLEEP] Deactivating Head {h} in Layer due to inactivity.")
                                    block.attn.head_mask[h] = 0.0
                                    block.attn.head_entropy[h] = 0.0
                                    active_heads_count -= 1
                                else:
                                    print(f"    >> [SLEEP] Keeping Head {h} alive (Last Survivor).")
                    
                    # Update Cache after pruning
                    block.attn.update_cache()

# ==========================================
# 5. MANAGERS (Sentry & Optimizer)
# ==========================================


class InputSentry(nn.Module):
    """Detects Context Shifts (Data Distribution Changes)"""

    def __init__(self, d_model):
        super().__init__()
        self.ae = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.ReLU(),
            nn.Linear(8, d_model))
        self.opt = optim.Adam(self.ae.parameters(), lr=0.01)
        self.baseline_mean = 0.0
        self.baseline_var = 0.0
        self.alpha = 0.05 # Momentum

    def check(self, x_emb):
        flat = x_emb.detach().view(-1, D_MODEL)
        with torch.no_grad():
            recon = self.ae(flat)
            err = ((recon - flat)**2).mean().item()

        if self.baseline_mean == 0:
            self.baseline_mean = err
            self.baseline_var = err * 0.1 # Initial guess
            return False
        
        # Z-Score Anomaly Detection
        # We use standard deviation to determine if the error is statistically significant
        std = math.sqrt(self.baseline_var + 1e-9)
        z_score = (err - self.baseline_mean) / std
        
        # Train
        self.opt.zero_grad()
        out = self.ae(flat)
        loss = ((out - flat)**2).mean()
        loss.backward()
        self.opt.step()
        
        # Update Stats (Welford's algorithm-ish momentum)
        diff = loss.item() - self.baseline_mean
        incr = self.alpha * diff
        self.baseline_mean += incr
        self.baseline_var = (1 - self.alpha) * (self.baseline_var + diff * incr)
        
        # Enforce Minimum Variance to prevent overfitting to a stable state
        # If variance is too low, any small noise triggers a Z-score spike.
        self.baseline_var = max(self.baseline_var, 1e-4)

        # Threshold: 8 Sigma (Extremely high confidence interval)
        # Increased from 5.0 to prevent false positives on noisy tiers (like Tier 7)
        return z_score > 8.0


class ReplayMemory:
    """
    The Hippocampus: Stores examples from previous experiences to prevent catastrophic forgetting.
    Uses Stratified Eviction to ensure balanced representation across Tiers.
    Now optimized with pre-allocated GPU tensors to avoid CPU-GPU sync.
    """
    def __init__(self, capacity=2000, seq_len=256, device='cpu'):
        self.capacity = capacity
        self.seq_len = seq_len
        self.device = device
        self.size = 0
        self.pos = 0
        
        # Pre-allocate memory on device
        self.mem_x = torch.zeros((capacity, seq_len), dtype=torch.long, device=device)
        self.mem_y = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.mem_l = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.mem_t = torch.zeros((capacity,), dtype=torch.long, device=device) # Tier
        
        self.tier_counts = {} # tier -> count

    def push(self, x, y, tier):
        # x: (B, S), y: (B,)
        # Store a few samples from the current batch
        # OPTIMIZATION: Keep on GPU, write directly to pre-allocated tensors
        
        B = x.size(0)
        orig_len = x.size(1)
        
        # Pad x to SEQ_LEN if necessary
        if x.size(1) < self.seq_len:
            padding = torch.zeros((B, self.seq_len - x.size(1)), dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        elif x.size(1) > self.seq_len:
            x_padded = x[:, :self.seq_len]
            orig_len = self.seq_len
        else:
            x_padded = x
            
        # OPTIMIZATION: Only store a subset (e.g., 4 samples)
        n_samples = min(4, B)
        indices = torch.randperm(B, device=x.device)[:n_samples]
        
        for i in indices:
            # Stratified Eviction Logic
            # If full, we need to pick a slot to overwrite
            if self.size < self.capacity:
                idx = self.size
                self.size += 1
                self.tier_counts[tier] = self.tier_counts.get(tier, 0) + 1
            else:
                # Find most over-represented tier
                max_tier = max(self.tier_counts, key=self.tier_counts.get)
                
                if max_tier == tier and len(self.tier_counts) == 1:
                     # Simple ring buffer if only one tier
                     idx = self.pos
                     self.pos = (self.pos + 1) % self.capacity
                else:
                    # GPU-based search for candidate
                    candidates = (self.mem_t[:self.size] == max_tier).nonzero().squeeze(1)
                    if candidates.numel() > 0:
                        # Pick random candidate
                        rand_idx = torch.randint(0, candidates.numel(), (1,), device=self.device).item()
                        idx = candidates[rand_idx].item()
                        self.tier_counts[max_tier] -= 1
                        self.tier_counts[tier] = self.tier_counts.get(tier, 0) + 1
                    else:
                        # Fallback
                        idx = self.pos
                        self.pos = (self.pos + 1) % self.capacity
                        # Update counts for whatever we overwrote
                        old_tier = self.mem_t[idx].item()
                        if old_tier in self.tier_counts:
                            self.tier_counts[old_tier] -= 1
                        self.tier_counts[tier] = self.tier_counts.get(tier, 0) + 1

            # Write to memory
            self.mem_x[idx] = x_padded[i]
            self.mem_y[idx] = y[i]
            self.mem_l[idx] = orig_len
            self.mem_t[idx] = tier

    def sample(self, batch_size, match_dtype=None):
        if self.size == 0:
            return None, None, None
        
        k = min(self.size, batch_size)
        indices = torch.randperm(self.size, device=self.device)[:k]
        
        batch_x = self.mem_x[indices]
        batch_y = self.mem_y[indices]
        batch_l = self.mem_l[indices]
        
        return batch_x, batch_y, batch_l


def get_opt(model, new_layer_idx=None, old_optimizer=None, neurogenesis=False):
    """
    Dual-Optimizer Strategy:
    - Mature Layers (Old): Low LR (Stiff)
    - New Layers (Plastic): High LR (Plastic)
    - Neurogenesis (Burst): High LR for EVERYONE (Rapid Rewiring)
    
    Preserves momentum state from old_optimizer if provided.
    """
    # If no specific new layer is identified, treat all as "Stiff" (fine-tuning)
    # unless it's the very start.
    
    params_stiff = []
    params_plastic = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        # If we just added a layer, it will be at the end of the list
        # We can identify it by checking if it belongs to the last layer
        if new_layer_idx is not None and f"layers.{new_layer_idx}" in name:
            params_plastic.append(p)
        # Always keep pos_enc plastic to allow rapid adaptation to new spatial tasks
        elif "pos_enc" in name:
            params_plastic.append(p)
        else:
            params_stiff.append(p)

    # If everything is new (start of training), make it all plastic
    if len(params_stiff) == 0:
        params_plastic = params_stiff + params_plastic
        params_stiff = []

    param_groups = []
    
    if neurogenesis:
        # NEUROGENESIS BURST: Everyone gets high plasticity
        # This allows old layers to realign to the new reality
        if params_stiff:
            param_groups.append({'params': params_stiff, 'lr': 0.004}) # Boosted Stiff
        if params_plastic:
            param_groups.append({'params': params_plastic, 'lr': 0.008}) # Boosted Plastic
    else:
        # Normal Operation
        if params_stiff:
            param_groups.append({'params': params_stiff, 'lr': 0.002}) # Stiff
        if params_plastic:
            param_groups.append({'params': params_plastic, 'lr': 0.01}) # Plastic

    new_opt = optim.Adam(param_groups)
    
    # Transfer State
    if old_optimizer is not None:
        # Map id(param) -> state
        old_state_map = {}
        for group in old_optimizer.param_groups:
            for p in group['params']:
                if p in old_optimizer.state:
                    old_state_map[id(p)] = old_optimizer.state[p]
        
        # Apply to new optimizer
        for group in new_opt.param_groups:
            for p in group['params']:
                if id(p) in old_state_map:
                    new_opt.state[p] = old_state_map[id(p)]
                    
    return new_opt

def run_dream_cycle(brain, replay, optimizer, criterion, steps=50):
    """
    Offline Consolidation Phase ("Dreaming").
    The model disconnects from the sensory stream (Curriculum) and
    trains exclusively on Hippocampal Replay to consolidate memories
    and align internal weights.
    """
    print(f"  >>> [DREAM] Entering REM Sleep for {steps} steps (Consolidation)...")
    brain.train()
    losses = []
    
    for _ in range(steps):
        optimizer.zero_grad()
        # Sample only from Replay
        rx, ry, rl = replay.sample(batch_size=32)
        
        if rx is None:
            break
            
        rx, ry, rl = rx.to(device), ry.to(device), rl.to(device)
        
        # Forward
        # Pass lengths for correct token selection
        logits = brain(rx, lengths=rl)
        loss = criterion(logits, ry)
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    avg_loss = sum(losses) / (len(losses) + 1e-9)
    print(f"  >>> [DREAM] Woke up. Avg Dream Loss: {avg_loss:.4f}")

def extract_weight_grid(tensor, size=(64, 64)):
    """
    Downsamples a weight matrix for visualization.
    Returns a 2D list of values between 0 and 1.
    """
    with torch.no_grad():
        if tensor.numel() == 0:
            return []
        
        # FIX: Normalize PER ROW based on Absolute Maximum.
        # 1. Per-Row Normalization: Ensures new (small) chunks are visible alongside mature (large) chunks.
        # 2. Abs-Max: Ensures 0.0 stays 0.0 (Black), preserving masks.
        
        # tensor: (Rows, Cols)
        row_max = tensor.abs().max(dim=1, keepdim=True)[0]
        norm = tensor.abs() / (row_max + 1e-9)
        
        # VISUAL SEPARATOR: Inject a black line between chunks (every 32 rows)
        # This helps distinguish adjacent chunks in the visualization.
        # We do this BEFORE resizing to ensure the line is captured.
        # However, resizing might blur it. Better to do it AFTER resizing?
        # No, let's do it on the raw data if possible, or just rely on the grid.
        # Actually, let's just ensure the data is correct first.
        
        # Resize
        # Input: (H, W) -> (1, 1, H, W)
        img = norm.unsqueeze(0).unsqueeze(0)
        
        # FIX: Use Max Pooling if downsampling to preserve sparse features
        # If we are downsampling significantly, 'nearest' might skip active rows.
        # 'area' might blur them too much.
        # Since we want to visualize "activity", Max Pooling is semantically correct.
        # But torch doesn't have adaptive_max_pool2d for arbitrary output size easily without calculation.
        # Let's stick to interpolate but ensure we don't miss chunks.
        
        # If we have 1024 rows and output 512, we are merging 2 rows into 1.
        # 'nearest' picks one. 'area' averages them.
        # Let's use 'area' but boost the contrast?
        # Or just stick to 'nearest' because 32 rows is huge.
        
        # DEBUG: Print stats of the image before resizing
        # if img.sum() > 0:
        #    print(f"    >> [VIZ DEBUG] Pre-Resize Active Rows: {(img.sum(dim=3) > 0).sum().item()}")

        resized = torch.nn.functional.interpolate(img, size=size, mode='nearest')
        
        # DEBUG: Print stats after resizing
        # if resized.sum() > 0:
        #    print(f"    >> [VIZ DEBUG] Post-Resize Active Rows: {(resized.sum(dim=3) > 0).sum().item()}")
        
        return resized.squeeze().tolist()

def get_weight_snapshot(brain):
    snapshots = []
    for i, layer in enumerate(brain.layers):
        if len(layer.blocks) > 0:
            block = layer.blocks[0]
            snapshot = {'id': i}
            
            # FFN Up Weights (Features)
            # We visualize the raw weight matrix. 
            # Rows = Neurons, Cols = Input Dim
            # FIX: Explicitly apply mask to ensure dead chunks are black
            # FIX: Increase resolution (512 height) to ensure chunks are visible
            chunk_size = block.ffn_up.chunk_size
            mask = block.ffn_up.active_chunks # (NumChunks,)
            
            mask_rows = mask.repeat_interleave(chunk_size).unsqueeze(1)
            w_ffn_masked = block.ffn_up.linear.weight * mask_rows

            # VISUAL SEPARATOR: Set the last row of every active chunk to 0 (Black Line)
            # FIX: Make the separator THICKER (2 rows) to ensure it survives resizing
            with torch.no_grad():
                for c_idx in range(len(mask)):
                    if mask[c_idx] > 0:
                        sep_row = (c_idx + 1) * chunk_size - 1
                        if sep_row < w_ffn_masked.shape[0]:
                            w_ffn_masked[sep_row, :] = 0.0
                            if sep_row - 1 >= 0:
                                w_ffn_masked[sep_row - 1, :] = 0.0

            snapshot['ffn_up'] = extract_weight_grid(w_ffn_masked, (512, 64))
            
            # Attention Query Weights (Attention Patterns)
            # Mask inactive heads to show them as black
            head_dim = block.attn.head_dim
            mask = block.attn.head_mask # (MaxHeads,)
            
            # Expand mask for Query (Rows are heads)
            # mask: (H,) -> (H, 1) -> (H, HeadDim) -> (D_Model,) -> (D_Model, 1)
            mask_rows = mask.repeat_interleave(head_dim).unsqueeze(1)
            w_q_masked = block.attn.w_q.weight * mask_rows
            snapshot['attn_q'] = extract_weight_grid(w_q_masked, (64, 64))
            
            # Attention Output Weights (Projection)
            # Expand mask for Output (Cols are heads)
            # mask: (H,) -> (H, 1) -> (H, HeadDim) -> (D_Model,) -> (1, D_Model)
            mask_cols = mask.repeat_interleave(head_dim).unsqueeze(0)
            w_o_masked = block.attn.w_o.weight * mask_cols
            snapshot['attn_o'] = extract_weight_grid(w_o_masked, (64, 64))
            
            snapshots.append(snapshot)
            
    return snapshots

# ==========================================
# 6. MAIN TRAINING LOOP
# ==========================================


if __name__ == "__main__":
    start_background_server()
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize
    curriculum = SequenceCurriculum(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    brain = OFormer(VOCAB_SIZE, D_MODEL).to(device)
    sentry = InputSentry(D_MODEL).to(device)
    replay = ReplayMemory(capacity=2000) # Initialize Hippocampus
    criterion = nn.CrossEntropyLoss()
    optimizer = get_opt(brain, new_layer_idx=0) # Start with layer 0 as plastic

    history = []
    tier_loss_window = [] # Track loss within a tier for frustration detection

    print("--- C.O.R.A.L. O-FORMER TRAINING START ---")
    print(f"Schedule: {TIERS} Tiers, Dynamic Duration (Min 300, Max 3000 steps).")

    current_tier = 1
    step = 0
    step_in_tier = 0
    tier_start_steps = [0]
    
    # Neurogenesis State
    neurogenesis_active = False
    neurogenesis_timer = 0
    neurogenesis_start_loss = 10.0 # Track loss at start of burst for dynamic exit
    growth_cooldown = 0
    frustration = 0.0 # Initialize frustration state

    while current_tier <= TIERS:
        step += 1
        step_in_tier += 1
        
        if growth_cooldown > 0:
            growth_cooldown -= 1
        
        # Handle Neurogenesis Decay (Annealing)
        if neurogenesis_active:
            neurogenesis_timer -= 1
            
            # OPTIMIZATION: Anneal Learning Rate
            # Decay LR by 0.5% every step during burst to stabilize
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(0.002, param_group['lr'] * 0.995)

            # DYNAMIC EXIT: Check if we have adapted successfully
            # If loss drops significantly (50%) or stabilizes at a low value, exit early.
            should_exit = False
            if len(tier_loss_window) > 20:
                recent_loss = sum(tier_loss_window[-20:]) / 20
                
                # Condition 1: Significant Drop
                if recent_loss < neurogenesis_start_loss * 0.5:
                    should_exit = True
                    # print(f"    >> [NEUROGENESIS] Early Exit: Loss Dropped ({neurogenesis_start_loss:.2f} -> {recent_loss:.2f})")
                
                # Condition 2: Stability at Low Loss
                if recent_loss < 0.1:
                    should_exit = True
                    # print(f"    >> [NEUROGENESIS] Early Exit: Stable Low Loss ({recent_loss:.2f})")

            if neurogenesis_timer <= 0 or should_exit:
                print("  >>> [NEUROGENESIS] Burst Complete. Stabilizing Plasticity.")
                neurogenesis_active = False
                GLOBAL_STATE['events'].append({
                    "step": step,
                    "type": "neurogenesis_end",
                    "desc": "Neurogenesis Burst Ended"
                })
                # Revert to normal optimizer (Stiff/Plastic split)
                # We pass new_layer_idx=None to treat everyone as Stiff (unless we just grew, but this is a decay step)
                # Actually, we should respect the last known new layer.
                # Simplified: Just reset to standard stiff/plastic based on architecture.
                # The last layer is usually the plastic one.
                optimizer = get_opt(brain, new_layer_idx=len(brain.layers)-1, old_optimizer=optimizer, neurogenesis=False)

        # OPTIMIZATION: Curriculum Mixing (Anticipation)
        # If we are mastering the current tier (loss < 0.15), peek at the next one.
        target_tier = current_tier
        if len(tier_loss_window) > 50:
             smooth_loss = sum(tier_loss_window[-50:]) / 50
             if smooth_loss < 0.15 and current_tier < TIERS:
                 if np.random.rand() < 0.2: # 20% chance to peek
                     target_tier = current_tier + 1

        x, y, desc = curriculum.get_batch(target_tier)
        x, y = x.to(device), y.to(device)

        # 2. Sentry Check (Context Switch -> Macro Growth)
        with torch.no_grad():
            seq_len = x.size(1)
            emb = brain.input_adapter(x) + brain.pos_enc[:, :seq_len, :]

        # We force a check at the start of a new tier to simulate "Noticing" the change
        force_check = (step_in_tier == 10)

        # OPTIMIZATION: Only check sentry every 20 steps to save compute (it runs a backward pass)
        # Also, don't check if we are already in a neurogenesis burst (plasticity is high, shifts are expected)
        # AND don't check if we are peeking at a future tier (target_tier != current_tier), as that is INTENDED to be different.
        # NEW: Respect growth_cooldown
        # FIX: DISABLE SENTRY GROWTH. The Architect is now smart enough to handle growth.
        # The Sentry was a heuristic from the pre-RL era. It is now causing "Zombie Growth"
        # by forcing layers when the Architect is locked out.
        # We keep the Sentry for logging/alerting, but we REMOVE the `brain.grow_depth()` call.
        if step > 50 and (force_check or step % 20 == 0) and not neurogenesis_active and growth_cooldown == 0 and target_tier == current_tier and sentry.check(emb):
            print(f"\n[!] SENTRY ALERT: Context Shift Detected ({desc}).")
            # brain.grow_depth() <--- DISABLED
            # growth_cooldown = 300
            
            # Trigger Neurogenesis Burst WITHOUT Growth
            # This just boosts plasticity to help adapt to the new tier, without forcing a new layer.
            print("  >>> [NEUROGENESIS] Burst Activated! Boosting Global Plasticity for 200 steps (No Growth).")
            neurogenesis_active = True
            neurogenesis_timer = 200 # Shorter burst for pure plasticity
            neurogenesis_start_loss = loss_main.item() # Record start loss
            GLOBAL_STATE['events'].append({
                "step": step,
                "type": "neurogenesis_start",
                "desc": "Neurogenesis Burst Started (Sentry - Plasticity Only)"
            })
            
            # The new layer is the last one
            new_layer_idx = len(brain.layers) - 1
            optimizer = get_opt(brain, new_layer_idx=new_layer_idx, old_optimizer=optimizer, neurogenesis=True)
            
            sentry.baseline_mean = 0.0 # Reset sentry stats
            sentry.baseline_var = 0.0
            tier_loss_window = [] # Reset frustration after growth

        # 3. Forward & Learn
        optimizer.zero_grad()
        
        # B. Replay (Hippocampus)
        # OPTIMIZATION: Intermittent Replay (Every 5 steps)
        # We don't need to refresh long-term memory every single step.
        # This saves significant compute by avoiding the second forward pass 80% of the time.
        run_replay = (step % 5 == 0)
        
        if run_replay:
            rx, ry, rl = replay.sample(batch_size=16)
        else:
            rx = None
        
        # OPTIMIZATION: Active Encoding (Short-Term Memory)
        # Intelligent Gating: Use Z-Score to detect "Surprise" (Anomalies) instead of raw loss.
        # Also keep a small background rate for general distribution coverage.
        # CRITICAL: Do NOT push "peeked" batches (future tiers) to replay. 
        if target_tier == current_tier:
            should_save = False
            
            # 1. Background Rate (0.5%) - Capture "Normal"
            if np.random.rand() < 0.005:
                should_save = True
            
            # 2. Surprise (Z-Score > 3.0) - Capture "Anomalies"
            # Only calculate if we have enough history for stats
            elif len(tier_loss_window) > 20:
                mean_loss = sum(tier_loss_window) / len(tier_loss_window)
                # Calculate std dev (approximate)
                variance = sum([(l - mean_loss)**2 for l in tier_loss_window]) / len(tier_loss_window)
                # FIX: Clamp std_dev to avoid hypersensitivity to noise when loss is stable
                std_dev = max(0.01, math.sqrt(variance) + 1e-6)
                
                z_score = (loss_main.item() - mean_loss) / std_dev
                
                # If Z-Score > 3 (3 Sigma event), it's a surprise.
                # Also ensure it's a negative surprise (loss is HIGHER than mean)
                if z_score > 3.0 and loss_main.item() > mean_loss:
                    should_save = True
                    # print(f"    >> [MEMORY] Surprise Detected (Z={z_score:.1f}). Encoding.")

            if should_save:
                 replay.push(x, y, current_tier)
        
        if rx is not None:
            # rx, ry, rl are already on device if buffer is on device
            # But if we just switched logic, buffer might be mixed.
            # Safe to call .to(device) anyway (no-op if already there)
            rx, ry, rl = rx.to(device), ry.to(device), rl.to(device)
            
            # OPTIMIZATION: Trim Replay Batch to Max Effective Length
            # The buffer stores everything padded to SEQ_LEN, but we only need max(rl)
            max_r_len = int(rl.max().item())
            if max_r_len < rx.size(1):
                rx = rx[:, :max_r_len]

            # --- OPTIMIZATION: SEPARATE FORWARD PASSES ---
            # We run x and rx separately to avoid padding x to rx's length,
            # which causes massive slowdowns in early tiers (16 vs 256 seq len).
            
            # 1. Forward Pass for Main Task (x)
            # Pass length of x
            x_len = x.size(1)
            x_lengths = torch.full((x.size(0),), x_len, dtype=torch.long, device=device)
            logits = brain(x, lengths=x_lengths)
            loss_main = criterion(logits, y)
            
            # OPTIMIZATION: Dampen loss for peeked tiers to prevent spikes/instability
            # We want to "preview" the future, not get destroyed by it.
            if target_tier > current_tier:
                loss_main *= 0.1
            
            # 2. Forward Pass for Replay (rx)
            # Pass lengths for rx
            r_logits = brain(rx, lengths=rl)
            r_loss = criterion(r_logits, ry)
            
            if neurogenesis_active:
                loss = 0.6 * loss_main + 0.4 * r_loss
            else:
                loss = 0.8 * loss_main + 0.2 * r_loss
        else:
            # No Replay available yet (or skipped this step)
            # Pass length of x
            x_len = x.size(1)
            lengths = torch.full((x.size(0),), x_len, dtype=torch.long, device=device)
            logits = brain(x, lengths=lengths)
            loss_main = criterion(logits, y) # Calculate main loss for logging
            loss = loss_main
            
            # OPTIMIZATION: Dampen loss for peeked tiers
            if target_tier > current_tier:
                loss *= 0.1
                loss_main *= 0.1 # Keep consistent
        
        # C. Metabolic Cost (L1 Regularization)
        # Encourages sparsity and drives useless weights to zero
        # OPTIMIZATION: Only calculate every 50 steps to save compute
        if step % 50 == 0:
            metabolic_cost = brain.calculate_metabolic_cost()
            # Reduced metabolic cost weight to prevent fighting against learning
            loss += 1e-8 * metabolic_cost

        loss.backward()
        
        # OPTIMIZATION: Clip Gradients to prevent spikes
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        
        optimizer.step()

        # 4. Micro Growth (Width) & Frustration Check (Depth)
        # Periodically check if the active layer is stressed
        if step % 25 == 0:
            # Helper to reset state
            def reset_state(param, indices, dim):
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key in ['exp_avg', 'exp_avg_sq']:
                        if key in state:
                            if dim == 0:
                                state[key][indices] = 0.0
                            else:
                                state[key][:, indices] = 0.0

            # Calculate Frustration Level (0.0 to 1.0)
            # Based on recent loss improvement
            # Increased window to 200 steps to smooth out batch noise and prevent oscillation
            instant_frustration = 0.0
            if len(tier_loss_window) > 200:
                recent = sum(tier_loss_window[-100:]) / 100
                
                # FIX: If loss is already low (converged), we are not frustrated!
                # We use 0.5 as a safe threshold for "good enough" to stop panic growing.
                if recent < 0.5:
                    instant_frustration = 0.0
                else:
                    older = sum(tier_loss_window[-200:-100]) / 100
                    improvement = (older - recent) / (older + 1e-9)
                    # If improvement < 1%, frustration rises
                    # 1.0 - (imp / 0.01) -> if imp=0, frust=1. if imp=0.01, frust=0.
                    instant_frustration = max(0.0, min(1.0, 1.0 - (improvement / 0.01)))
            
            # Smooth Frustration (Sticky)
            # If instant is high, we jump up SLOWER (less panic). If instant is low, we decay slowly (relief).
            if instant_frustration > frustration:
                frustration = 0.9 * frustration + 0.1 * instant_frustration # Was 0.5/0.5
            else:
                frustration = 0.98 * frustration + 0.02 * instant_frustration # Was 0.95/0.05

            # A. Frustration Check (Abstraction Wall)
            # If loss has been high for a while, we are stuck.
            # We scale the patience window by growth_sensitivity.
            # If sensitivity is low (previous growth failed), we wait longer (e.g. 300 steps).
            patience_window = max(300, int(300 / brain.growth_sensitivity)) # Was 150

            if len(tier_loss_window) > patience_window and growth_cooldown == 0:
                avg_loss = sum(tier_loss_window[-patience_window:]) / patience_window
                
                # AI ARCHITECT DECISION
                # We call the Architect periodically (every patience_window steps).
                # It decides whether to Wait, Wake Head, Add Block, or Add Layer.
                
                # 1. Get State
                state = brain.architect.get_state(brain, frustration, tier_loss_window)
                
                # 2. Determine Valid Actions
                # 0=Wait, 1=Head, 2=Chunk, 3=Block, 4=Layer
                valid_mask = torch.tensor([True, False, False, False, False], dtype=torch.bool) # Wait always valid
                
                # Check Heads (Load Balancing)
                # Find the block with the FEWEST heads to ensure symmetrical growth.
                head_arg = None
                min_heads = float('inf')
                
                for block_idx, block in enumerate(top_layer.blocks):
                    n_heads = block.attn.head_mask.sum().item()
                    if n_heads < block.attn.max_heads:
                        valid_mask[1] = True
                        if n_heads < min_heads:
                            min_heads = n_heads
                            head_arg = block_idx
                
                # Check Chunks (FFN Expansion)
                # Find block with room for chunks AND balance them (pick one with fewest chunks)
                chunk_arg = None
                min_chunks = float('inf')
                
                for block_idx, block in enumerate(top_layer.blocks):
                    n_chunks = block.ffn_up.active_chunks.sum().item()
                    if n_chunks < block.ffn_up.max_chunks:
                        valid_mask[2] = True
                        if n_chunks < min_chunks:
                            min_chunks = n_chunks
                            chunk_arg = block_idx 
                        
                # Check Width
                if len(top_layer.blocks) < 2:
                    valid_mask[3] = True

                # Check Layer (Only if frustrated)
                # We gate this to prevent random exploration from adding layers when happy.
                # NOTE: We removed the hardcoded "Pancake" check (total_heads > 12) because
                # the Architect now learns this via the "Heads -> Layer" weight.
                if frustration > 0.1:
                    valid_mask[4] = True
                
                # FIX: Re-introduce a softer "Pancake" heuristic.
                # If the top layer is extremely wide (e.g. > 75% capacity), we force the Architect
                # to consider adding a layer even if frustration is low.
                # This prevents the "Super-Layer" trap where it solves everything with one massive layer.
                total_heads = sum([b.attn.head_mask.sum().item() for b in top_layer.blocks])
                max_possible_heads = sum([b.attn.max_heads for b in top_layer.blocks])
                
                if total_heads > (max_possible_heads * 0.75):
                     valid_mask[4] = True
                    
                # 3. Decide
                action_idx = brain.architect.decide(state, valid_mask)
                
                if action_idx is not None:
                    # Store decision for reinforcement learning
                    brain.architect.store_action(state, action_idx, step, avg_loss)
                
                if action_idx == 0: # Wait
                    # Do nothing. Just reset the window.
                    # We might want a small cooldown or just let it run another window.
                    # print(f"    >> [ARCHITECT] Decision: Wait (Patience).")
                    pass

                elif action_idx == 1: # Wake Head
                    block = top_layer.blocks[head_arg]
                    print(f"    >> [ARCHITECT] Decision: Waking Head in Block {head_arg}.")
                    idx = block.attn.wake_head(frustration)
                    # Reset optimizer for new head
                    start = idx * block.attn.head_dim
                    end = start + block.attn.head_dim
                    indices = slice(start, end)
                    for p in [block.attn.w_q.weight, block.attn.w_k.weight, block.attn.w_v.weight]:
                        reset_state(p, indices, 0)
                    reset_state(block.attn.w_o.weight, indices, 1)
                    growth_cooldown = 100 
                    GLOBAL_STATE['events'].append({
                        "step": step,
                        "type": "growth",
                        "desc": f"Macro: Woke Head (Layer {len(brain.layers)})"
                    })
                
                elif action_idx == 2: # Wake Chunk
                    block = top_layer.blocks[chunk_arg]
                    print(f"    >> [ARCHITECT] Decision: Waking Chunk (Lobe) in Block {chunk_arg}.")
                    block.ffn_up.wake_chunk(current_tier=current_tier)
                    growth_cooldown = 100
                    GLOBAL_STATE['events'].append({
                        "step": step,
                        "type": "growth",
                        "desc": f"Macro: Woke Chunk (Layer {len(brain.layers)})"
                    })
                    
                elif action_idx == 3: # Add Block
                    print(f"    >> [ARCHITECT] Decision: Growing WIDTH (New Parallel Block).")
                    print("  >>> [MACRO] Growing WIDTH (Adding Parallel Block to Layer).")
                    brain.grow_width()
                    growth_cooldown = 300
                    GLOBAL_STATE['events'].append({
                        "step": step,
                        "type": "growth",
                        "desc": f"Macro: Added Parallel Block (Layer {len(brain.layers)})"
                    })
                    
                    # Trigger Neurogenesis
                    neurogenesis_active = True
                    neurogenesis_timer = 500
                    neurogenesis_start_loss = avg_loss # Record start loss
                    GLOBAL_STATE['events'].append({
                        "step": step,
                        "type": "neurogenesis_start",
                        "desc": "Neurogenesis Burst Started (Macro Block)"
                    })
                    optimizer = get_opt(brain, new_layer_idx=None, old_optimizer=optimizer, neurogenesis=True)
                    
                elif action_idx == 4: # Add Layer
                    print(f"    >> [ARCHITECT] Decision: Growing DEPTH (New Layer).")
                    print(f"  >>> [MACRO] Abstraction Wall Hit. Growing DEPTH (Layer {len(brain.layers)+1}).")
                    brain.grow_depth()
                    growth_cooldown = 300
                    GLOBAL_STATE['events'].append({
                        "step": step,
                        "type": "growth",
                        "desc": f"Macro: Added Layer {len(brain.layers)}"
                    })
                    
                    # Trigger Neurogenesis
                    neurogenesis_active = True
                    neurogenesis_timer = 500
                    neurogenesis_start_loss = avg_loss # Record start loss
                    GLOBAL_STATE['events'].append({
                        "step": step,
                        "type": "neurogenesis_start",
                        "desc": "Neurogenesis Burst Started (Macro Layer)"
                    })
                    new_layer_idx = len(brain.layers) - 1
                    optimizer = get_opt(brain, new_layer_idx=new_layer_idx, old_optimizer=optimizer, neurogenesis=True)
                    
                tier_loss_window = [] # Reset frustration

            # B. Micro Growth (Head Saturation)
            # Check if individual heads are confused (High Entropy + High Loss contribution)
            # This is "Micro" because it happens within existing layers.
            # FIX: Inhibit Micro Growth if Neurogenesis (Macro Growth) is active.
            # We want the new layer to settle before we start adding heads to it.
            
            # UPDATE ARCHITECT
            brain.architect.update(step, loss.item())
            
            # OPTIMIZATION: Check Micro Growth less frequently (every 25 steps)
            # This reduces the overhead of iterating through all layers and blocks.
            if growth_cooldown == 0 and not neurogenesis_active and step % 25 == 0:
                optimizer, grew = brain.check_micro_growth(x, y, optimizer, frustration, current_loss=loss.item(), step=step, current_tier=current_tier)
                if grew:
                    # FIX: Do NOT reset frustration window on micro-growth.
                    # If micro-growth fixes the issue, loss will drop naturally.
                    # If not, we need the window to fill up to trigger Macro Growth (New Layer).
                    # tier_loss_window = [] 
                    growth_cooldown = 100 # Cooldown for micro growth

        # 5. Sleep Phase
        if step % 500 == 0 and step > 0:
            brain.sleep(current_tier=current_tier)

        # 6. Logging
        # OPTIMIZATION: Only log loss from the current tier to avoid noise from peeking
        # FIX: Log loss_main (Current Task) instead of mixed loss (Task + Replay) to reduce graph noise.
        if target_tier == current_tier:
            history.append(loss_main.item())
            tier_loss_window.append(loss_main.item()) # Track for frustration
        else:
            history.append(np.nan) # Keep history aligned with steps, but show gaps for peeking

        if step % 100 == 0:
            depth = len(brain.layers)
            # Get stats from top layer
            top_layer = brain.layers[-1]
            heads = sum([int(b.attn.head_mask.sum().item()) for b in top_layer.blocks])
            ffn_chunks = sum([int(b.ffn_up.active_chunks.sum().item()) for b in top_layer.blocks])
            width = len(top_layer.blocks)
            
            # Calculate Gradient Norms for Diagnostics
            total_grad_norm = 0.0
            for p in brain.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            peek_str = " (PEEK)" if target_tier > current_tier else ""
            print(
                f"Step {step} [{desc}]{peek_str}: Loss {loss_main.item():.4f} | Depth {depth} | TopL Width {width} | TopL Heads {heads} | TopL FFN-Chunks {ffn_chunks} | Win {len(tier_loss_window)} | GradNorm {total_grad_norm:.2f}")

            # Update Graph periodically
            # OPTIMIZATION: Reduce plotting frequency from 100 to 500 to save I/O time
            if step % 500 == 0:
                plt.figure(figsize=(12, 6))
                plt.plot(history, label='Loss', linewidth=1, alpha=0.7)
                
                # Plot Tier Boundaries
                for t_step in tier_start_steps[1:]:
                    plt.axvline(t_step, color='r', linestyle='--', alpha=0.5)
                
                # Plot Growth Events
                if 'events' in GLOBAL_STATE:
                    # Separate events for batch plotting
                    events_by_type = {
                        'layer': {'steps': [], 'vals': [], 'color': 'red', 'marker': '*', 'size': 200, 'label': 'New Layer'},
                        'block': {'steps': [], 'vals': [], 'color': 'orange', 'marker': 's', 'size': 100, 'label': 'New Block'},
                        'head': {'steps': [], 'vals': [], 'color': 'cyan', 'marker': '^', 'size': 60, 'label': 'New Head'},
                        'ffn': {'steps': [], 'vals': [], 'color': 'green', 'marker': 'o', 'size': 40, 'label': 'New FFN Chunk'}
                    }
                    
                    # Neurogenesis Tracking
                    neuro_starts = []
                    neuro_ends = []
                    
                    for evt in GLOBAL_STATE['events']:
                        estep = evt['step']
                        etype = evt['type']
                        edesc = evt['desc']
                        
                        if etype == 'neurogenesis_start':
                            neuro_starts.append(estep)
                        elif etype == 'neurogenesis_end':
                            neuro_ends.append(estep)
                        
                        # Ensure step is within history bounds
                        # FIX: history is 0-indexed, step is 1-based.
                        # history[step-1] corresponds to step.
                        if estep <= len(history):
                            val = history[estep-1]
                            if np.isnan(val): val = 0 # Fallback
                            
                            if etype == 'growth':
                                if 'Added Layer' in edesc:
                                    events_by_type['layer']['steps'].append(estep)
                                    events_by_type['layer']['vals'].append(val)
                                elif 'Block' in edesc:
                                    events_by_type['block']['steps'].append(estep)
                                    events_by_type['block']['vals'].append(val)
                                elif 'Head' in edesc:
                                    events_by_type['head']['steps'].append(estep)
                                    events_by_type['head']['vals'].append(val)
                                elif 'FFN' in edesc:
                                    events_by_type['ffn']['steps'].append(estep)
                                    events_by_type['ffn']['vals'].append(val)
                    
                    # Plot Neurogenesis Regions
                    # Match starts with ends. If currently active, end is current step.
                    for i, start in enumerate(neuro_starts):
                        end = neuro_ends[i] if i < len(neuro_ends) else step
                        plt.axvspan(start, end, color='purple', alpha=0.1, label='Neurogenesis' if i == 0 else "")

                    for key, data in events_by_type.items():
                        if data['steps']:
                            plt.scatter(data['steps'], data['vals'], 
                                      color=data['color'], 
                                      marker=data['marker'], 
                                      s=data['size'], 
                                      label=data['label'], 
                                      zorder=10,
                                      edgecolors='black',
                                      linewidths=0.5)

                plt.title(f"C.O.R.A.L. Training (Step {step})")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig("training_curve.png")
                plt.close()

        # 7. Update Visualization State (Every 50 steps)
        # OPTIMIZATION: Reduced frequency from 10 to 50 to reduce CPU/IO bottleneck
        if step % 50 == 0:
            layer_data = []
            for i, layer in enumerate(brain.layers):
                # Aggregate data from all blocks in the layer
                total_heads = 0
                total_ffn = 0
                merged_head_mask = []
                merged_head_entropy = []
                merged_ffn_mask = []
                merged_ffn_tiers = []
                merged_ffn_activity = []
                
                for block in layer.blocks:
                    total_heads += int(block.attn.head_mask.sum().item())
                    total_ffn += int(block.ffn_up.active_chunks.sum().item())
                    merged_head_mask.extend(block.attn.head_mask.tolist())
                    merged_head_entropy.extend(block.attn.head_entropy.tolist())
                    
                    # FIX: Filter out dead chunks from the blueprint data
                    # If active_chunks is 0, we should probably set tier to -1 or handle it in frontend
                    # But wait, the frontend likely iterates over this list.
                    # If we want to "remove" them, we can't just delete them from the list because indices matter?
                    # No, the frontend likely draws them based on index.
                    # If we want them to disappear, we should ensure their 'mask' is 0.
                    # The frontend should use 'ffn_mask' to decide whether to draw.
                    
                    merged_ffn_mask.extend(block.ffn_up.active_chunks.tolist())
                    merged_ffn_tiers.extend(block.ffn_up.chunk_tiers.tolist())
                    merged_ffn_activity.extend(block.ffn_up.chunk_activity.tolist())

                layer_data.append({
                    "id": i,
                    "heads": total_heads,
                    "ffn_chunks": total_ffn,
                    "head_mask": merged_head_mask,
                    "head_entropy": merged_head_entropy,
                    "ffn_mask": merged_ffn_mask,
                    "ffn_tiers": merged_ffn_tiers,
                    "ffn_activity": merged_ffn_activity
                })
            
            GLOBAL_STATE["step"] = step
            GLOBAL_STATE["loss"] = float(loss_main.item())
            GLOBAL_STATE["tier"] = current_tier
            GLOBAL_STATE["desc"] = desc
            GLOBAL_STATE["layers"] = layer_data
            GLOBAL_STATE["hippocampus_size"] = replay.size
            GLOBAL_STATE["hippocampus_capacity"] = replay.capacity
            GLOBAL_STATE["hippocampus_tier_counts"] = replay.tier_counts
            
            # Update Weights Snapshot (Every 100 steps)
            # OPTIMIZATION: Reduced frequency from 50 to 100
            if step % 100 == 0:
                GLOBAL_STATE["weights"] = get_weight_snapshot(brain)
            
            # Update Loss History (Keep last 100 points)
            if "loss_history" not in GLOBAL_STATE: GLOBAL_STATE["loss_history"] = []
            GLOBAL_STATE["loss_history"].append(float(loss_main.item()))
            if len(GLOBAL_STATE["loss_history"]) > 100:
                GLOBAL_STATE["loss_history"].pop(0)
            
            # Determine Status
            status = "HOMEOSTASIS"
            if step % 500 == 0 and step > 0: status = "SLEEPING"
            elif (sentry.check(emb) or force_check) and step > 50: status = "SENTRY ALERT" # Note: This check is redundant as it happens earlier, but good for viz
            # We can't easily detect growth here as it happened earlier in the loop
            # But we can check if we just grew
            
            GLOBAL_STATE["status"] = status

        # Check for Tier Completion
        # Calculate smooth loss
        if len(tier_loss_window) > 50:
            smooth_loss = sum(tier_loss_window[-50:]) / 50
        else:
            smooth_loss = 10.0
            
        # Condition: Min steps passed AND (Loss is low OR Max steps passed)
        # We use 0.15 as convergence threshold (reasonable for this vocab size)
        # FIX: Increased max steps to 3000 to match schedule and allow for growth.
        # FIX: Increased convergence threshold from 0.05 to 0.15 to avoid chasing noise.
        is_converged = (smooth_loss < 0.15)
        is_maxed_out = (step_in_tier >= 3000)
        is_min_steps = (step_in_tier >= 300)
        
        # STAGNATION CHECK: If we are stuck for a long time with no improvement
        # FIX: Removed Stagnation from progression trigger. 
        # Stagnation should trigger GROWTH (via Architect), not skipping the tier.
        # If we skip, we reset frustration and the model never learns to grow.
        
        if is_min_steps and (is_converged or is_maxed_out):
             if current_tier < TIERS:
                 # Before switching, memorize some of the current tier
                print(f"  >>> [MEMORY] Consolidating Tier {current_tier} into Hippocampus...")
                # We push the last batch of the old tier (Already handled by Active Encoding, but good to be sure)
                replay.push(x, y, current_tier)
                
                # Dream to consolidate memories before switching context
                run_dream_cycle(brain, replay, optimizer, criterion, steps=50)
                
                # Save Checkpoint
                torch.save(brain.state_dict(), f"checkpoints/coral_tier_{current_tier}.pt")
                print(f"  >>> [CHECKPOINT] Saved model to checkpoints/coral_tier_{current_tier}.pt")

                current_tier += 1
                step_in_tier = 0
                tier_start_steps.append(step)
                tier_loss_window = [] # Reset frustration tracker
                growth_cooldown = 200 # Give model time to adapt to new tier before growing
                print(f"\n*** ENTERING TIER {current_tier} ***")
                GLOBAL_STATE['events'].append({
                    "step": step,
                    "type": "tier",
                    "desc": f"Entered Tier {current_tier}"
                })
                
                # OPTIMIZATION: Reset Optimizer Momentum to prevent overshoot
                # When task changes abruptly (Identity -> Count), old momentum is harmful.
                print("  >>> [OPTIMIZER] Resetting momentum for Tier Transition.")
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p in optimizer.state:
                            state = optimizer.state[p]
                            if 'exp_avg' in state:
                                state['exp_avg'].zero_()
             else:
                 # Finished last tier
                 break

    print("\n--- EVOLUTION COMPLETE ---")
    torch.save(brain.state_dict(), "checkpoints/coral_final.pt")
    print("  >>> [CHECKPOINT] Saved final model to checkpoints/coral_final.pt")

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Loss')
    # Draw Tier Lines
    for i, t_step in enumerate(tier_start_steps[1:]):
        plt.axvline(t_step, color='r', linestyle='--',
                    alpha=0.5, label=f'Tier {i+2} Start' if i == 0 else "")
    plt.title("C.O.R.A.L. O-Former Adaptation Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_curve.png")
    print("  >>> [GRAPH] Saved training curve to training_curve.png")

