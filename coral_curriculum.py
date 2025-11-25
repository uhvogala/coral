import torch

class SequenceCurriculum:
    """
    Manages the complexity of tasks presented to the AI.
    """

    def __init__(self, batch_size, max_seq_len, vocab_size):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        # self.side and self.seq_len are now dynamic per batch

    def get_batch(self, tier):
        # VISUAL TIERS (37+)
        if tier >= 37:
            return self.get_visual_batch(tier)

        # Dynamic Input Size Schedule
        if tier <= 5:
            side = 4
        elif tier <= 10:
            side = 8
        elif tier <= 15:
            side = 12
        else:
            side = 16
            
        seq_len = side * side
        B = self.batch_size
        V = self.vocab_size
        
        # --- VECTORIZED TIERS (1-21) ---
        if tier <= 21:
            x = torch.zeros((B, seq_len), dtype=torch.long)
            y = torch.zeros(B, dtype=torch.long)
            desc = "Unknown"

            # --- TIER 1: ECHO (Identity) ---
            if tier == 1:
                seed = torch.randint(0, 10, (B, 1))
                x = seed.repeat(1, seq_len)
                y = seed.squeeze()
                desc = f"Tier 1: Echo (Identity) [{side}x{side}]"

            # --- TIER 2: COUNT (Linear) ---
            elif tier == 2:
                seed = torch.randint(0, 10, (B, 1))
                k = torch.arange(seq_len + 1).unsqueeze(0) # (1, L+1)
                seq = (seed + k) % V
                x = seq[:, :-1]
                y = seq[:, -1]
                desc = f"Tier 2: Count (Linear) [{side}x{side}]"

            # --- TIER 3: PATTERN (Context) ---
            elif tier == 3:
                seed = torch.randint(0, 10, (B, 1))
                k = torch.arange(seq_len + 1).unsqueeze(0)
                mask = (k % 2 == 0)
                seq = torch.zeros((B, seq_len + 1), dtype=torch.long)
                seq = torch.where(mask, seed, torch.tensor(0))
                x = seq[:, :-1]
                y = seq[:, -1]
                desc = f"Tier 3: Pattern (Alternating) [{side}x{side}]"

            # --- TIER 4: VERTICAL ECHO (Spatial Stride) ---
            elif tier == 4:
                row0 = torch.randint(0, V, (B, side))
                x = row0.unsqueeze(1).repeat(1, side, 1).view(B, -1)
                y = x[:, -1].clone()
                x[:, -1] = 0
                desc = f"Tier 4: Vertical Echo (2D Stride) [{side}x{side}]"

            # --- TIER 5: CHECKERBOARD (2D Parity) ---
            elif tier == 5:
                color_a = torch.randint(0, V, (B, 1, 1))
                color_b = torch.randint(0, V, (B, 1, 1))
                r = torch.arange(side).view(-1, 1)
                c = torch.arange(side).view(1, -1)
                mask = ((r + c) % 2 == 0).unsqueeze(0)
                grid = torch.where(mask, color_a, color_b)
                x = grid.view(B, -1)
                y = x[:, -1].clone()
                x[:, -1] = 0
                desc = f"Tier 5: Checkerboard (2D Parity) [{side}x{side}]"

            # --- TIER 6: OBJECT COMPLETION (2D Shape) ---
            elif tier == 6:
                x = torch.randint(0, 5, (B, seq_len))
                color_obj = torch.randint(10, V, (B,))
                idx1 = seq_len - 1
                idx2 = seq_len - 2
                idx3 = seq_len - 1 - side
                idx4 = seq_len - 2 - side
                x[:, idx1] = color_obj
                x[:, idx2] = color_obj
                x[:, idx3] = color_obj
                x[:, idx4] = color_obj
                y = x[:, -1].clone()
                x[:, -1] = 0
                desc = f"Tier 6: Object Completion (2D Shape) [{side}x{side}]"

            # --- TIER 7: DIAGONAL CONTINUITY (2D Line) ---
            elif tier == 7:
                x = torch.randint(0, 5, (B, seq_len))
                color = torch.randint(10, V, (B,))
                diag_indices = torch.arange(side) * side + torch.arange(side)
                x[:, diag_indices] = color.unsqueeze(1)
                y = x[:, -1].clone()
                x[:, -1] = 0
                desc = f"Tier 7: Diagonal Continuity (2D Line) [{side}x{side}]"

            # --- TIER 8: VERTICAL SYMMETRY (Mirror) ---
            elif tier == 8:
                left_half = torch.randint(0, V, (B, side, side // 2))
                right_half = left_half.flip(2)
                grid = torch.cat([left_half, right_half], dim=2)
                x = grid.view(B, -1)
                y = x[:, -1].clone()
                x[:, -1] = 0
                desc = f"Tier 8: Vertical Symmetry (Mirror) [{side}x{side}]"

            # --- TIER 9: MEMORY RETRIEVAL (Key-Value) ---
            elif tier == 9:
                key = torch.randint(0, V, (B,))
                val = torch.randint(0, V, (B,))
                x = torch.randint(0, V, (B, seq_len))
                x[:, 0] = key
                x[:, 1] = val
                x[:, -1] = key
                y = val
                desc = f"Tier 9: Memory Retrieval (Long-Range) [{side}x{side}]"

            # --- TIER 10: CONDITIONAL LOGIC (Branching) ---
            elif tier == 10:
                seed = torch.randint(0, 10, (B,))
                is_even = (seed % 2 == 0)
                x = torch.zeros((B, seq_len + 1), dtype=torch.long)
                x[:, 0] = seed
                for k in range(1, seq_len + 1):
                    prev = x[:, k-1]
                    up = (prev + 1) % V
                    down = (prev - 1) % V
                    x[:, k] = torch.where(is_even, up, down)
                y = x[:, -1].clone()
                x = x[:, :-1]
                desc = f"Tier 10: Conditional Logic (Branching) [{side}x{side}]"

            # --- TIER 11: SYMBOLIC INCREMENT (Next Value) ---
            elif tier == 11:
                seed = torch.randint(0, V-1, (B,))
                x = torch.zeros((B, seq_len), dtype=torch.long)
                x[:, 0] = seed
                y = seed + 1
                desc = f"Tier 11: Symbolic Increment (Next Value) [{side}x{side}]"

            # --- TIER 12: SYMBOLIC DECREMENT (Prev Value) ---
            elif tier == 12:
                seed = torch.randint(1, V, (B,))
                x = torch.zeros((B, seq_len), dtype=torch.long)
                x[:, 0] = seed
                y = seed - 1
                desc = f"Tier 12: Symbolic Decrement (Prev Value) [{side}x{side}]"

            # --- TIER 13: MAGNITUDE COMPARISON (Max) ---
            elif tier == 13:
                limit = 100 # Reduced from V
                a = torch.randint(0, limit, (B,))
                b = torch.randint(0, limit, (B,))
                x = torch.zeros((B, seq_len), dtype=torch.long)
                x[:, 0] = a
                x[:, 1] = b
                y = torch.max(a, b)
                desc = f"Tier 13: Magnitude Comparison (Max) [{side}x{side}]"

            # --- TIER 14: ARITHMETIC (Addition) ---
            elif tier == 14:
                limit = 20
                a = torch.randint(0, limit, (B,))
                b = torch.randint(0, limit, (B,))
                target = a + b
                x = torch.zeros((B, seq_len), dtype=torch.long)
                x[:, 0] = a
                x[:, 1] = b
                y = target
                desc = f"Tier 14: Arithmetic (Addition, Small) [{side}x{side}]"

            # --- TIER 15: FIBONACCI (Recursive) ---
            elif tier == 15:
                seed = torch.randint(0, 10, (B,))
                x = torch.zeros((B, seq_len + 1), dtype=torch.long)
                x[:, 0] = seed
                x[:, 1] = (seed + 1) % V
                for k in range(2, seq_len + 1):
                    x[:, k] = (x[:, k-1] + x[:, k-2]) % V
                y = x[:, -1].clone()
                x = x[:, :-1]
                desc = f"Tier 15: Fibonacci (Recursive) [{side}x{side}]"

            # --- TIER 16: BOOLEAN LOGIC (AND/OR/XOR) ---
            elif tier == 16:
                op = torch.randint(0, 3, (B,)) # 0=AND, 1=OR, 2=XOR
                a = torch.randint(0, 2, (B, seq_len))
                b = torch.randint(0, 2, (B, seq_len))
                # Interleave a and b
                x = torch.zeros((B, seq_len), dtype=torch.long)
                x[:, 0::2] = a[:, :seq_len//2]
                x[:, 1::2] = b[:, :seq_len//2]
                
                # Target is operation on last pair
                last_a = x[:, -2]
                last_b = x[:, -1]
                
                res_and = last_a & last_b
                res_or = last_a | last_b
                res_xor = last_a ^ last_b
                
                y = torch.where(op == 0, res_and, torch.where(op == 1, res_or, res_xor))
                x[:, -1] = 0 # Mask last input
                desc = f"Tier 16: Boolean Logic (AND/OR/XOR) [{side}x{side}]"

            # --- TIER 17: EXISTENCE CHECK (Search) ---
            elif tier == 17:
                query = torch.randint(0, V, (B,))
                x = torch.randint(0, V, (B, seq_len))
                # Ensure 50% chance of existence
                exists = torch.rand(B) < 0.5
                
                # For 'exists', force query into sequence
                for i in range(B):
                    if exists[i]:
                        pos = torch.randint(1, seq_len-1, (1,)).item()
                        x[i, pos] = query[i]
                    else:
                        # Ensure query is NOT in sequence
                        mask = (x[i] == query[i])
                        x[i][mask] = (query[i] + 1) % V
                
                x[:, 0] = query # First token is query
                y = exists.long()
                x[:, -1] = 0
                desc = f"Tier 17: Existence Check (Is X in Seq?) [{side}x{side}]"

            # --- TIER 18: LINE FOLLOWING (Connectivity) ---
            elif tier == 18:
                # Simple contiguous path on grid
                x = torch.zeros((B, seq_len), dtype=torch.long)
                y = torch.zeros(B, dtype=torch.long)
                for i in range(B):
                    grid = torch.zeros((side, side), dtype=torch.long)
                    r, c = 0, 0
                    grid[r, c] = 1
                    path_len = 0
                    # Random walk right/down
                    while r < side-1 or c < side-1:
                        moves = []
                        if r < side-1: moves.append((1, 0))
                        if c < side-1: moves.append((0, 1))
                        dr, dc = moves[torch.randint(0, len(moves), (1,)).item()]
                        r, c = r+dr, c+dc
                        grid[r, c] = 1
                        path_len += 1
                    
                    # Task: Predict path length (or endpoint, but endpoint is always corner here)
                    # Let's make it harder: Random start, random length
                    # New Task: Predict if (r, c) is connected to (0,0) via 1s?
                    # Let's stick to "Path Continuation": Given path, predict next step?
                    # Let's do: "Is the path valid?" (No breaks).
                    # Let's do: "Path Length" (Count 1s in the main path).
                    y[i] = path_len % V
                    x[i] = grid.view(-1)
                
                x[:, -1] = 0
                desc = f"Tier 18: Line Following (Path Length) [{side}x{side}]"

            # --- TIER 19: 2D GRADIENT (Neighbor Sum) ---
            elif tier == 19:
                grid = torch.zeros((B, side, side), dtype=torch.long)
                grid[:, 0, :] = torch.randint(0, V, (B, side))
                grid[:, :, 0] = torch.randint(0, V, (B, side))
                for r in range(1, side):
                    for c in range(1, side):
                        grid[:, r, c] = (grid[:, r-1, c] + grid[:, r, c-1]) % V
                x = grid.view(B, -1)
                y = x[:, -1].clone()
                x[:, -1] = 0
                desc = f"Tier 19: 2D Gradient (Neighbor Sum) [{side}x{side}]"

            # --- TIER 20: POINTER (Indirection) ---
            elif tier == 20:
                x = torch.randint(0, V, (B, seq_len))
                valid_range = seq_len - 1
                p_val = torch.randint(0, valid_range, (B,))
                x[:, 0] = p_val
                y = x.gather(1, p_val.unsqueeze(1)).squeeze()
                x[:, -1] = 0
                desc = f"Tier 20: Pointer (Indirection) [{side}x{side}]"

            # --- TIER 21: MAJORITY VOTE (Histogram) ---
            elif tier == 21:
                a = torch.randint(0, V, (B,))
                offset = torch.randint(1, V, (B,))
                b = (a + offset) % V
                vals = torch.randint(0, 2, (B, seq_len))
                x = torch.where(vals == 0, a.unsqueeze(1), b.unsqueeze(1))
                count_a = (x == a.unsqueeze(1)).sum(dim=1)
                count_b = (x == b.unsqueeze(1)).sum(dim=1)
                y = torch.where(count_a > count_b, a, b)
                x[:, -1] = 0
                desc = f"Tier 21: Majority Vote (Histogram) [{side}x{side}]"

            return x, y, desc

        # --- LOOP-BASED TIERS (22-36) ---
        if tier <= 36:
            x = torch.zeros((B, seq_len), dtype=torch.long)
            y = torch.zeros(B, dtype=torch.long)
            desc = "Unknown"

            # --- TIER 22: ARITHMETIC (Multiplication) ---
            if tier == 22:
                limit = 12
                a = torch.randint(0, limit, (B,))
                b = torch.randint(0, limit, (B,))
                target = a * b
                x = torch.randint(0, V, (B, seq_len))
                x[:, 0] = a
                x[:, 1] = b
                x[:, -1] = 0
                y = target
                desc = f"Tier 22: Arithmetic (Multiplication, Small) [{side}x{side}]"

            # --- TIER 23: NON-LINEAR 2D (Interaction) ---
            elif tier == 23:
                x = torch.randint(0, V, (B, seq_len))
                n1 = x[:, seq_len - 2 - side]
                n2 = x[:, seq_len - 1 - side]
                n3 = x[:, seq_len - 2]
                y = (n1 * n2 + n3) % V
                x[:, -1] = 0
                desc = f"Tier 23: Non-Linear 2D (n1*n2+n3) [{side}x{side}]"

            # --- TIER 24: MULTI-HOP POINTER (Chained) ---
            elif tier == 24:
                x = torch.randint(0, V, (B, seq_len))
                p1 = torch.randint(0, seq_len - 1, (B,))
                x[:, 0] = p1
                p2 = torch.randint(0, seq_len - 1, (B,))
                x.scatter_(1, p1.unsqueeze(1), p2.unsqueeze(1))
                y = x.gather(1, p2.unsqueeze(1)).squeeze()
                x[:, -1] = 0
                desc = f"Tier 24: Multi-hop Pointer (Chained) [{side}x{side}]"

            # --- TIER 25: ROTATIONAL SYMMETRY (90 deg) ---
            elif tier == 25:
                quad = torch.randint(0, V, (B, (side + 1) // 2, side // 2))
                grid = torch.zeros((B, side, side), dtype=torch.long)
                for r in range((side + 1) // 2):
                    for c in range(side // 2):
                        color = quad[:, r, c]
                        grid[:, r, c] = color
                        grid[:, c, side-1-r] = color
                        grid[:, side-1-r, side-1-c] = color
                        grid[:, side-1-c, r] = color
                x = grid.view(B, -1)
                y = x[:, -1].clone()
                x[:, -1] = 0
                desc = f"Tier 25: Rotational Symmetry (90 deg) [{side}x{side}]"

            # --- TIER 26: GAME OF LIFE (Simulation) ---
            elif tier == 26:
                x = torch.randint(0, 2, (B, seq_len))
                grid = x.view(B, side, side)
                r, c = side - 1, side - 1
                neighbors_sum = torch.zeros(B, dtype=torch.long)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = (r + dr) % side, (c + dc) % side
                        neighbors_sum += grid[:, nr, nc]
                curr = grid[:, r, c]
                cond1 = (curr == 1) & ((neighbors_sum == 2) | (neighbors_sum == 3))
                cond2 = (curr == 0) & (neighbors_sum == 3)
                next_val = torch.where(cond1 | cond2, torch.tensor(1), torch.tensor(0))
                y = next_val
                x[:, -1] = 0
                desc = f"Tier 26: Game of Life (1-Step) [{side}x{side}]"

            # --- TIER 27: MAZE PATHFINDING (BFS) ---
            elif tier == 27:
                x = torch.randint(0, 2, (B, seq_len))
                x[:, 0] = 0
                x[:, -1] = 0
                y = torch.zeros(B, dtype=torch.long)
                for i in range(B):
                    grid = x[i].view(side, side)
                    q = [(0,0)]
                    seen = {(0,0)}
                    found = 0
                    target_r, target_c = side - 1, side - 1
                    while q:
                        r, c = q.pop(0)
                        if r == target_r and c == target_c:
                            found = 1
                            break
                        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < side and 0 <= nc < side:
                                if grid[nr, nc] == 0 and (nr, nc) not in seen:
                                    seen.add((nr, nc))
                                    q.append((nr, nc))
                    y[i] = found
                x[:, -1] = 0
                desc = f"Tier 27: Maze Pathfinding (Reachability) [{side}x{side}]"

            # --- TIER 28: OBJECT COUNTING (Connectivity) ---
            elif tier == 28:
                x = torch.randint(0, 2, (B, seq_len))
                x[:, -1] = 0
                y = torch.zeros(B, dtype=torch.long)
                for i in range(B):
                    grid = x[i].view(side, side)
                    visited = torch.zeros((side, side))
                    count = 0
                    for r in range(side):
                        for c in range(side):
                            if grid[r, c] == 1 and visited[r, c] == 0:
                                count += 1
                                q = [(r, c)]
                                visited[r, c] = 1
                                while q:
                                    curr_r, curr_c = q.pop(0)
                                    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                                        nr, nc = curr_r+dr, curr_c+dc
                                        if 0 <= nr < side and 0 <= nc < side:
                                            if grid[nr, nc] == 1 and visited[nr, nc] == 0:
                                                visited[nr, nc] = 1
                                                q.append((nr, nc))
                    y[i] = count % V
                desc = f"Tier 28: Object Counting (Connectivity) [{side}x{side}]"

            # --- TIER 29: CARTPOLE PHYSICS (Control) ---
            elif tier == 29:
                state_x = (torch.rand(B) * 4.8) - 2.4
                state_x_dot = (torch.rand(B) * 6.0) - 3.0
                state_theta = (torch.rand(B) * 0.4) - 0.2
                state_theta_dot = (torch.rand(B) * 6.0) - 3.0
                score = (1.0 * state_theta) + (0.1 * state_theta_dot) + (0.05 * state_x) + (0.1 * state_x_dot)
                action = torch.where(score > 0, torch.tensor(1), torch.tensor(0))
                def discretize_vec(val, min_val, max_val):
                    norm = (val - min_val) / (max_val - min_val)
                    norm = torch.clamp(norm, 0.0, 1.0)
                    return (norm * (V - 1)).long()
                t_x = discretize_vec(state_x, -2.4, 2.4)
                t_xd = discretize_vec(state_x_dot, -3.0, 3.0)
                t_t = discretize_vec(state_theta, -0.2, 0.2)
                t_td = discretize_vec(state_theta_dot, -3.0, 3.0)
                x = torch.zeros((B, seq_len), dtype=torch.long)
                x[:, 0] = t_x
                x[:, 1] = t_xd
                x[:, 2] = t_t
                x[:, 3] = t_td
                x[:, 4:10] = torch.randint(0, V, (B, 6))
                y = action
                x[:, -1] = 0
                desc = f"Tier 29: CartPole Physics (Control) [{side}x{side}]"

            # --- TIER 30: DENOISING (Visual Repair) ---
            elif tier == 30:
                x = torch.randint(0, 2, (B, seq_len))
                shape_type = torch.randint(0, 2, (B,))
                for i in range(B):
                    if shape_type[i] == 0:
                        row = torch.randint(0, side, (1,)).item()
                        x[i, row*side : (row+1)*side] = 2
                    else:
                        col = torch.randint(0, side, (1,)).item()
                        x[i, col::side] = 2
                noise_mask = torch.rand((B, seq_len)) < 0.1
                x[noise_mask] = torch.randint(0, 2, (noise_mask.sum(),)).long()
                y = shape_type
                x[:, -1] = 0
                desc = f"Tier 30: Denoising (Shape Classification) [{side}x{side}]"

            # --- TIER 31: EDGE DETECTION (Convolution) ---
            elif tier == 31:
                x = torch.zeros((B, seq_len), dtype=torch.long)
                r = torch.randint(0, side-3, (B,))
                c = torch.randint(0, side-3, (B,))
                base = r * side + c
                offsets = []
                for dr in range(3):
                    for dc in range(3):
                        offsets.append(dr*side + dc)
                for off in offsets:
                    indices = base + off
                    x.scatter_(1, indices.unsqueeze(1), 1)
                y = r
                x[:, -1] = 0
                desc = f"Tier 31: Edge Detection (Localization) [{side}x{side}]"

            # --- TIER 32: OCR (Optical Character Recognition) ---
            elif tier == 32:
                x = torch.zeros((B, seq_len), dtype=torch.long)
                char_idx = torch.randint(0, 3, (B,))
                y = 65 + char_idx
                bitmaps = torch.tensor([
                    [0,1,1,1,0, 1,0,0,0,1, 1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1],
                    [1,1,1,1,0, 1,0,0,0,1, 1,1,1,1,0, 1,0,0,0,1, 1,1,1,1,0],
                    [0,1,1,1,1, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 0,1,1,1,1]
                ], dtype=torch.long)
                r = torch.randint(0, side-5, (B,))
                c = torch.randint(0, side-5, (B,))
                base = r * side + c
                for i in range(25):
                    pixel_active = bitmaps[char_idx, i]
                    br = i // 5
                    bc = i % 5
                    pos = base + (br * side + bc)
                    mask = (pixel_active == 1)
                    if mask.any():
                        x.scatter_(1, pos.unsqueeze(1), pixel_active.unsqueeze(1))
                x[:, -1] = 0
                desc = f"Tier 32: OCR (Simple A/B/C) [{side}x{side}]"

            # --- TIER 33: STRING REVERSAL (Working Memory) ---
            elif tier == 33:
                length = torch.randint(3, 8, (B,))
                x = torch.zeros((B, seq_len), dtype=torch.long)
                rand_chars = torch.randint(65, 91, (B, 8))
                col_idx = torch.arange(8).unsqueeze(0)
                len_mask = col_idx < length.unsqueeze(1)
                x[:, :8] = torch.where(len_mask, rand_chars, torch.tensor(0))
                query_idx = (torch.rand(B) * length.float()).long()
                x[:, seq_len-2] = query_idx
                rev_idx = (length - 1) - query_idx
                y = x.gather(1, rev_idx.unsqueeze(1)).squeeze()
                x[:, -1] = 0
                desc = f"Tier 33: String Reversal (Indexed) [{side}x{side}]"

            # --- TIER 34: CAESAR CIPHER (Symbolic Arithmetic) ---
            elif tier == 34:
                char = torch.randint(65, 91, (B,))
                shift = torch.randint(1, 5, (B,))
                target = char + shift
                target = torch.where(target > 90, target - 26, target)
                x = torch.randint(0, V, (B, seq_len))
                x[:, 0] = char
                x[:, 1] = shift
                x[:, -1] = 0
                y = target
                desc = f"Tier 34: Caesar Cipher (Symbolic Shift) [{side}x{side}]"

            # --- TIER 35: ARITHMETIC (Addition, Large) ---
            elif tier == 35:
                limit = 100
                a = torch.randint(0, limit, (B,))
                b = torch.randint(0, limit, (B,))
                target = (a + b) % V
                x = torch.randint(0, V, (B, seq_len))
                x[:, 0] = a
                x[:, 1] = b
                x[:, -1] = 0
                y = target
                desc = f"Tier 35: Arithmetic (Addition, Large) [{side}x{side}]"

            # --- TIER 36: ARITHMETIC (Multiplication, Large) ---
            elif tier == 36:
                limit = 15
                a = torch.randint(0, limit, (B,))
                b = torch.randint(0, limit, (B,))
                target = (a * b) % V
                x = torch.randint(0, V, (B, seq_len))
                x[:, 0] = a
                x[:, 1] = b
                x[:, -1] = 0
                y = target
                desc = f"Tier 36: Arithmetic (Multiplication, Large) [{side}x{side}]"

            return x, y, desc

    def get_visual_batch(self, tier):
        """
        Generates synthetic visual data (Images).
        Returns: x (Float, [B, 3, H, W]), y (Long, [B]), desc
        """
        # Base image size
        # Tier 37: 16x16 (1 patch)
        # Tier 38+: 32x32 (2x2 patches)
        if tier == 37:
            H, W = 16, 16
        else:
            H, W = 32, 32
            
        B = self.batch_size
        x = torch.zeros((B, 3, H, W), dtype=torch.float32)
        y = torch.zeros(B, dtype=torch.long)
        desc = "Unknown Visual Task"
        
        # OPTIMIZATION: Vectorized Visual Tiers
        
        # --- TIER 37: DOMINANT COLOR (Global Avg) ---
        if tier == 37:
            means = torch.rand(B, 3)
            noise = torch.randn(B, 3, H, W) * 0.1
            img = noise + means.view(B, 3, 1, 1)
            x = torch.clamp(img, 0, 1)
            y = torch.argmax(means, dim=1)
            desc = f"Tier 37: Visual - Dominant Color [16x16]"

        # --- TIER 38: ORIENTATION (Texture) ---
        elif tier == 38:
            orientation = torch.randint(0, 2, (B,)) # 0=Horz, 1=Vert
            color = torch.rand(B, 3, 1, 1)
            r_idx = torch.arange(H).view(-1, 1)
            c_idx = torch.arange(W).view(1, -1)
            mask_h = ((r_idx % 4) < 2).expand(H, W).unsqueeze(0)
            mask_v = ((c_idx % 4) < 2).expand(H, W).unsqueeze(0)
            mask = torch.where(orientation.view(B, 1, 1) == 0, mask_h, mask_v)
            mask = mask.unsqueeze(1)
            img = torch.where(mask, color, torch.zeros_like(color))
            img += torch.randn(B, 3, H, W) * 0.05
            x = torch.clamp(img, 0, 1)
            y = orientation
            desc = f"Tier 38: Visual - Orientation (Stripes) [32x32]"

        # --- TIER 39: QUADRANT LOCALIZATION (Spatial) ---
        elif tier == 39:
            quad = torch.randint(0, 4, (B,))
            img = torch.ones(B, 3, H, W) * 0.1
            color = torch.rand(B, 3, 1, 1) * 0.8 + 0.2
            h2, w2 = H//2, W//2
            r_idx = torch.arange(H).view(1, -1, 1)
            c_idx = torch.arange(W).view(1, 1, -1)
            q0_mask = (r_idx < h2) & (c_idx < w2)
            q1_mask = (r_idx < h2) & (c_idx >= w2)
            q2_mask = (r_idx >= h2) & (c_idx < w2)
            q3_mask = (r_idx >= h2) & (c_idx >= w2)
            masks = torch.stack([q0_mask, q1_mask, q2_mask, q3_mask], dim=0).squeeze(1)
            batch_masks = masks[quad]
            batch_masks = batch_masks.unsqueeze(1)
            img = torch.where(batch_masks, color, img)
            x = img
            y = quad
            desc = f"Tier 39: Visual - Quadrant Localization [32x32]"

        # --- TIER 40: VISUAL COMPARISON (Siamese-lite) ---
        elif tier == 40:
            left_bright = torch.rand(B)
            right_bright = torch.rand(B)
            diff = (left_bright - right_bright).abs()
            to_close = diff < 0.1
            right_bright = torch.where(to_close, (right_bright + 0.2) % 1.0, right_bright)
            img = torch.zeros(B, 3, H, W)
            img[:, :, :, :W//2] = left_bright.view(B, 1, 1, 1)
            img[:, :, :, W//2:] = right_bright.view(B, 1, 1, 1)
            img += torch.randn(B, 3, H, W) * 0.05
            x = torch.clamp(img, 0, 1)
            y = (left_bright > right_bright).long()
            desc = f"Tier 40: Visual - Brightness Comparison [32x32]"

        # --- TIER 41: SHAPE CLASSIFICATION (Geometry) ---
        elif tier == 41:
            shape = torch.randint(0, 2, (B,)) # 0=Square, 1=Diamond
            img = torch.zeros(B, 3, H, W)
            color = torch.rand(B, 3, 1, 1)
            center_r, center_c = H//2, W//2
            radius = torch.randint(6, 14, (B, 1, 1))
            r_idx = torch.arange(H).view(1, H, 1).expand(B, H, W)
            c_idx = torch.arange(W).view(1, 1, W).expand(B, H, W)
            sq_mask = (r_idx - center_r).abs() <= radius
            sq_mask &= (c_idx - center_c).abs() <= radius
            di_mask = (r_idx - center_r).abs() + (c_idx - center_c).abs() <= radius
            mask = torch.where(shape.view(B, 1, 1) == 0, sq_mask, di_mask)
            img = torch.where(mask.unsqueeze(1), color, img)
            x = img
            y = shape
            desc = f"Tier 41: Visual - Shape (Square vs Diamond) [32x32]"

        return x, y, desc
