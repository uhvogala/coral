C.O.R.A.L.
Constructive Organic Residual Architecture for Learning
1. Executive Summary
C.O.R.A.L. is a bio-mimetic neural network architecture designed for Continual Learning and Dynamic Scalability. Unlike traditional Deep Learning models which have fixed topologies (static width/depth), C.O.R.A.L. begins as a minimal seed and grows physically—adding neurons (width) and layers (depth)—in response to environmental stress and data complexity.
The architecture solves the "Catastrophic Forgetting" and "Static Capacity" problems while maintaining GPU efficiency through novel masking and memory management techniques.
2. Core Philosophy
The system operates on a biological lifecycle rather than a static training loop:
Birth: Initialize as a minimal "Seed" network.
Homeostasis: Attempt to learn tasks with current capacity.
Stress: Detect capacity limits (Saturation) or logic limits (Abstraction).
Growth: Physically expand topology (Width or Depth).
Sleep: Prune inefficient connections to conserve energy.
Defrag: Re-organize memory to optimize hardware utilization.
3. Anatomy: The "SuperLobe" Strategy
To overcome the inefficiency of dynamic graph resizing on GPUs, C.O.R.A.L. utilizes the SuperLobe.
3.1. The Pre-Allocated Substrate
Instead of creating thousands of tiny, separate nn.Linear layers (which causes GPU kernel fragmentation), a SuperLobe is a single, massive pre-allocated matrix (e.g., 4096 x 4096).
Virtual Lobes: The matrix is mathematically divided into "Chunks" (e.g., blocks of 32 neurons).
The Mask: A binary buffer controls which Chunks are "Awake" (1) or "Asleep" (0).
Zero-Cost Growth: "Growing" the network does not require memory reallocation; it simply requires flipping a bit in the mask from 0 to 1, instantly enabling a new block of neurons.
3.2. Hierarchy (DeepCoral)
Multiple SuperLobes are stacked sequentially to form Regions.
Width: Represents parallel processing capacity (e.g., more features). Managed by unmasking chunks within a SuperLobe.
Depth: Represents abstraction capability (e.g., logic/reasoning). Managed by inserting new SuperLobe layers into the stack.
4. Physiology: The Growth Triggers
The system is governed by a GrowthController that acts as the brain's executive function. It decides when and how to grow based on two signals.
4.1. Micro-Growth (Width)
Analogy: Hiring more workers for the factory line.
Trigger: Saturation. High neuron activity (>90%), high gradient norms, but loss is plateauing. The current neurons are overworked.
Action: Unmask a dormant Chunk within the active SuperLobe.
Mechanism: The new chunk is initialized using Net2Net theory (often with low/zero weights) to prevent "Shock" to the existing gradients.
4.2. Macro-Growth (Depth)
Analogy: Adding a management floor to the factory.
Trigger: Abstraction Wall. Loss is stuck, but gradients are low/stable. The network lacks the non-linearity to solve the problem.
Alternative Trigger: Sentry Alert. The InputSentry (Autoencoder) detects a fundamental shift in data distribution (Context Switch).
Action: Insert a completely new SuperLobe layer into the stack.
Mechanism: Identity Injection. The new layer is initialized as an identity function ($y = x + f(x)$ where $f(x) \approx 0$). This allows gradients to flow through it immediately, preserving old skills while the new layer slowly comes online.
5. Maintenance: Sleep & Housecleaning
To prevent the network from becoming a dense, slow monolith, C.O.R.A.L. implements an active maintenance cycle.
5.1. Sleep Phase (Pruning)
Occurs periodically or after major growth events.
Magnitude Pruning: Neurons or weights with low contribution are set to zero.
Sparsity: This creates a "Swiss Cheese" effect in the SuperLobe matrices.
Goal: Increases generalization and simulates biological synaptic pruning.
5.2. Housecleaning (Defrag)
Occurs rarely (e.g., every 5000 steps). Solves the GPU inefficiency of sparse matrices.
Scan: Identify active vs. dead neurons in the SuperLobe.
Sort: Permute the weight matrices to push all Active neurons to the left (indices $0..N$) and all Dead neurons to the right.
Truncate: (Optional) Physically slice the tensor to free VRAM, effectively "compacting" the knowledge.
6. Memory: The Hippocampus
To solve the Plasticity-Stability Dilemma (learning new things without overwriting old things):
Dual-Optimizer: Old/Mature lobes are assigned a "Stiff" optimizer (Low LR). New/Growing lobes are assigned a "Plastic" optimizer (High LR).
Replay Buffer: A memory bank stores examples from previous Tiers. During training, these samples are interleaved with new data to enforce constraints on the "Stiff" weights, ensuring backward compatibility.
7. Scalability & Future Proofing
7.1. Vision (VisualCoral)
The SuperLobe concept applies to Convolutional Neural Networks. Instead of masking Neurons, we mask Filters/Kernels. Growth implies adding more feature maps to a layer.
7.2. Transformers (The O-Former)
Macro-Growth: Stacking new Transformer Blocks (Depth).
Micro-Growth:
FFN: Widening the Feed-Forward Network via chunk masking.
Attention: Using "Head Masking" to dynamically wake up new Attention Heads when Entropy (Confusion) is high.
8. Summary of Key Components
Component
Role
Biological Analogy
SuperLobe
The physical container for weights.
Cortical Tissue
Chunks
The unit of growth.
Neural Clusters
Sentry
Detects distribution shifts.
Amygdala / Surprise
Gardener
Monitors loss & manages optimizers.
Homeostasis
SleepManager
Prunes weak connections.
Synaptic Pruning
Housecleaner
Defragments memory.
Glial Cells / Waste Removal
ReplayMemory
Prevents forgetting.
Hippocampus

Generated by Gemini & User Research Session - 2024
