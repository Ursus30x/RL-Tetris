import numpy as np
import pygame
import random
import json
import os
import sys

# Game constants
GRID_WIDTH = 10
GRID_HEIGHT = 20
BLOCK_SIZE = 30
FPS = 100

# Colors
COLORS = [
    (0, 0, 0),  # Empty
    (0, 255, 255),  # I
    (0, 0, 255),  # J
    (255, 165, 0),  # L
    (255, 255, 0),  # O
    (0, 255, 0),  # S
    (128, 0, 128),  # T
    (255, 0, 0),  # Z
]

# Tetrimino shapes
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[2, 0, 0], [2, 2, 2]],  # J
    [[0, 0, 3], [3, 3, 3]],  # L
    [[4, 4], [4, 4]],  # O
    [[0, 5, 5], [5, 5, 0]],  # S
    [[0, 6, 0], [6, 6, 6]],  # T
    [[7, 7, 0], [0, 7, 7]],  # Z
]


class TetrisEnv:
    def __init__(self, render_mode=False, config_path="config.json"):
        # Load config
        self.reward_config = self.load_config(config_path)

        # Example:
        self.step_penalty = self.reward_config.get("step_penalty", -0.1)
        self.drop_bonus = self.reward_config.get("drop_bonus", 0.1)
        self.death_penalty = self.reward_config.get("death_penalty", -1.0)
        self.line_rewards = self.reward_config.get("line_rewards", {"1": 40, "2": 100, "3": 300, "4": 1200})

        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.score = 0
        self.game_over = False
        self.render_mode = render_mode

        self.current_piece = None
        self.current_pos = None
        self.spawn_piece()

        if self.render_mode:
            # Windows-specific pygame initialization
            if sys.platform.startswith('win'):
                os.environ['SDL_VIDEODRIVER'] = 'windib'

            pygame.init()
            pygame.display.init()

            self.screen = pygame.display.set_mode((GRID_WIDTH * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE))
            pygame.display.set_caption("Tetris RL")

            try:
                self.clock = pygame.time.Clock()
            except pygame.error as e:
                print(f"Warning: Clock initialization failed: {e}")
                self.clock = None

    def load_config(self, path):
        try:
            # Używaj ścieżki względnej do pliku
            if not os.path.isabs(path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(script_dir, path)

            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"Config file not found at {path}. Using defaults.")
                return {}
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return {}

    def spawn_piece(self):
        shape_id = random.randint(0, len(SHAPES) - 1)
        self.current_piece = np.array(SHAPES[shape_id])
        self.current_pos = [0, GRID_WIDTH // 2 - len(self.current_piece[0]) // 2]
        if self.collision(self.current_piece, self.current_pos):
            self.game_over = True

    def collision(self, piece, pos):
        px, py = pos
        for y in range(piece.shape[0]):
            for x in range(piece.shape[1]):
                if piece[y][x] and (
                        y + px >= GRID_HEIGHT or
                        x + py < 0 or
                        x + py >= GRID_WIDTH or
                        self.grid[y + px][x + py]
                ):
                    return True
        return False

    def freeze(self):
        """Fix current piece into the grid and check for line clears"""
        px, py = self.current_pos
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    self.grid[px + i][py + j] = 1
        lines_cleared = self.clear_lines()
        self.spawn_piece()

        return lines_cleared

    def clear_lines(self):
        # Keep only rows that are NOT fully filled (not all True/non-zero)
        new_grid = [row for row in self.grid if not np.all(row)]

        lines_cleared = GRID_HEIGHT - len(new_grid)

        if lines_cleared > 0:
            self.score += lines_cleared

            # Create empty rows at the top for cleared lines
            empty_rows = np.zeros((lines_cleared, GRID_WIDTH), dtype=int)

            # Stack empty rows above the remaining grid rows
            self.grid = np.vstack((empty_rows, np.array(new_grid)))

        return lines_cleared

    def rotate(self, piece):
        return np.rot90(piece, -1)

    def line_clear_reward(self, lines_cleared):
        return self.line_rewards.get(str(lines_cleared), 0)

    def get_max_height(self):
        # Assuming self.grid is 2D numpy array with 0 = empty, >0 = block
        for row in range(self.grid.shape[0]):
            if np.any(self.grid[row, :] != 0):
                return self.grid.shape[0] - row  # Height counted from bottom
        return 0

    def get_hole_count(self):
        holes = 0
        grid = self.grid
        for col in range(grid.shape[1]):
            column = grid[:, col]
            filled_found = False
            for cell in column:
                if cell != 0:
                    filled_found = True
                elif filled_found and cell == 0:
                    holes += 1
        return holes

    def step(self, action):
        if self.game_over:
            return self.grid.copy(), self.death_penalty, True

        px, py = self.current_pos
        reward = -0.1  # small step penalty to encourage faster play

        frozen_this_step = False
        cleared = 0

        if action == 0 and not self.collision(self.current_piece, (px, py - 1)):
            self.current_pos[1] -= 1
        elif action == 1 and not self.collision(self.current_piece, (px, py + 1)):
            self.current_pos[1] += 1
        elif action == 2:
            rotated = self.rotate(self.current_piece)
            if not self.collision(rotated, self.current_pos):
                self.current_piece = rotated
        elif action == 3:
            while not self.collision(self.current_piece, (self.current_pos[0] + 1, self.current_pos[1])):
                self.current_pos[0] += 1
            frozen_this_step = True
            cleared = self.freeze()

        if not frozen_this_step:
            if not self.collision(self.current_piece, (self.current_pos[0] + 1, self.current_pos[1])):
                self.current_pos[0] += 1
            else:
                frozen_this_step = True
                cleared = self.freeze()

        if frozen_this_step:
            max_height = self.get_max_height()
            holes = self.get_hole_count()
            reward += cleared * 10  # reward clearing lines
            reward -= max_height * 0.5  # penalty for tall towers
            reward -= holes * 1.0  # penalty for holes

        return self.grid.copy(), reward, self.game_over

    def render(self):
        if not self.render_mode:
            return

        try:
            self.screen.fill((0, 0, 0))

            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    val = self.grid[y][x]
                    pygame.draw.rect(self.screen, COLORS[val], (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
                                     0)

            # Draw current piece
            px, py = self.current_pos
            for y in range(self.current_piece.shape[0]):
                for x in range(self.current_piece.shape[1]):
                    if self.current_piece[y][x]:
                        pygame.draw.rect(
                            self.screen,
                            COLORS[self.current_piece[y][x]],
                            ((py + x) * BLOCK_SIZE, (px + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
                            0
                        )

            pygame.display.flip()

            if self.clock:
                self.clock.tick(FPS)

        except pygame.error as e:
            print(f"Render error: {e}")

    def reset(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.score = 0
        self.game_over = False
        self.spawn_piece()
        return self.grid.copy()

    def close(self):
        if self.render_mode:
            try:
                pygame.quit()
            except:
                pass


def get_user_action():
    """
    Maps arrow key presses to action codes:
    0 - left
    1 - right
    2 - rotate (up)
    3 - drop (down)
    4 - do nothing
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return 'quit'
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                return 0
            elif event.key == pygame.K_RIGHT:
                return 1
            elif event.key == pygame.K_UP:
                return 2
            elif event.key == pygame.K_DOWN:
                return 3
    return 4  # No action


if __name__ == "__main__":
    env = TetrisEnv(render_mode=True)
    obs = env.reset()
    done = False

    print("Sterowanie: Strzałki - ruch, ESC - wyjście")

    while not done:
        action = get_user_action()
        if action == 'quit':
            break
        obs, reward, done = env.step(action)
        env.render()

    env.close()