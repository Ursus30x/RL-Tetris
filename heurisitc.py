def simple_heuristic(env):
    best_score = -float('inf')
    best_action_sequence = []

    original_piece = env.current_piece.copy()
    original_pos = tuple(env.current_pos)

    for rot in range(4):  # max 4 rotations
        piece = original_piece
        for _ in range(rot):
            piece = env.rotate(piece)

        min_x = -piece.shape[1] + 1
        max_x = env.grid.shape[1]

        for x in range(min_x, max_x):
            # Simulate dropping piece at position (0, x)
            px, py = 0, x
            if env.collision(piece, (px, py)):
                continue

            # Drop down
            while not env.collision(piece, (px + 1, py)):
                px += 1

            temp_grid = env.grid.copy()
            for dy in range(piece.shape[0]):
                for dx in range(piece.shape[1]):
                    if piece[dy][dx]:
                        gx, gy = px + dy, py + dx
                        if 0 <= gx < temp_grid.shape[0] and 0 <= gy < temp_grid.shape[1]:
                            temp_grid[gx][gy] = 1  # mark filled

            cleared = simulate_clear_lines(temp_grid)
            holes = count_holes(temp_grid)
            max_height = get_max_height(temp_grid)
            bumpiness = get_bumpiness(temp_grid)

            score = (
                1.0 * cleared -
                0.5 * holes -
                0.5 * max_height -
                0.2 * bumpiness
            )

            if score > best_score:
                best_score = score
                best_action_sequence = [rot, x]

    # Decide what move to do NOW to reach best_action_sequence
    # For now: always rotate then hard drop (simplified)
    if not best_action_sequence:
        return 3  # drop by default

    rot_needed, target_x = best_action_sequence
    curr_piece = original_piece
    curr_pos = list(original_pos)

    if rot_needed > 0:
        return 2  # rotate
    elif curr_pos[1] < target_x:
        return 1  # move right
    elif curr_pos[1] > target_x:
        return 0  # move left
    else:
        return 3  # drop

# === Supporting helpers ===

def simulate_clear_lines(grid):
    full_rows = [i for i in range(grid.shape[0]) if all(grid[i])]
    return len(full_rows)

def count_holes(grid):
    holes = 0
    for col in range(grid.shape[1]):
        block_found = False
        for row in range(grid.shape[0]):
            if grid[row][col]:
                block_found = True
            elif block_found and not grid[row][col]:
                holes += 1
    return holes

def get_max_height(grid):
    for row in range(grid.shape[0]):
        if any(grid[row]):
            return grid.shape[0] - row
    return 0

def get_bumpiness(grid):
    heights = []
    for col in range(grid.shape[1]):
        for row in range(grid.shape[0]):
            if grid[row][col]:
                heights.append(grid.shape[0] - row)
                break
        else:
            heights.append(0)
    return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
