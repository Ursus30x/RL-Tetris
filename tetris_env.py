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
    (0, 0, 0),      # Empty
    (0, 255, 255),  # I
    (0, 0, 255),    # J
    (255, 165, 0),  # L
    (255, 255, 0),  # O
    (0, 255, 0),    # S
    (128, 0, 128),  # T
    (255, 0, 0),    # Z
]

# Tetrimino shapes
SHAPES = [
    [[1, 1, 1, 1]],              # I
    [[2, 0, 0], [2, 2, 2]],      # J
    [[0, 0, 3], [3, 3, 3]],      # L
    [[4, 4], [4, 4]],            # O
    [[0, 5, 5], [5, 5, 0]],      # S
    [[0, 6, 0], [6, 6, 6]],      # T
    [[7, 7, 0], [0, 7, 7]],      # Z
]


class TetrisEnv:
    def __init__(self, render_mode=False, config_path="config.json"):
        # Load config
        self.reward_config = self.load_config(config_path)
        self.death_penalty = self.reward_config.get("death_penalty", -100.0)
        self.line_rewards = self.reward_config.get("line_rewards", {"1": 40, "2": 100, "3": 300, "4": 1200})

        # Game state
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.score = 0
        self.game_over = False
        self.render_mode = render_mode

        self.current_piece = None
        self.current_pos = None
        self.next_piece = np.array(SHAPES[random.randint(0, len(SHAPES) - 1)])  # Add this line
        self.spawn_piece()

        # Pygame initialization for rendering
        if self.render_mode:
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

    def get_next_piece(self):
        """Get the next piece that will appear after the current one"""
        return self.next_piece.copy() if self.next_piece is not None else np.array(SHAPES[random.randint(0, len(SHAPES) - 1)])

    def simulate_placement(self, piece, rotation, column):
        """
        Simulate placing a piece at a specific rotation and column
        Returns the resulting grid and lines cleared
        """
        # Create a copy of the current grid
        grid_copy = self.grid.copy()
        
        # Rotate the piece
        rotated_piece = np.rot90(piece, -rotation)
        
        # Find drop position
        drop_row = 0
        while not self._collision_at_pos(rotated_piece, (drop_row + 1, column), grid_copy):
            drop_row += 1
        
        # Place the piece
        for y in range(rotated_piece.shape[0]):
            for x in range(rotated_piece.shape[1]):
                if rotated_piece[y][x]:
                    grid_copy[y + drop_row][x + column] = rotated_piece[y][x]
        
        # Clear lines
        lines_cleared = 0
        new_grid = []
        for row in grid_copy:
            if not np.all(row):
                new_grid.append(row.copy())
            else:
                lines_cleared += 1
        
        # Add empty rows at the top
        while len(new_grid) < GRID_HEIGHT:
            new_grid.insert(0, np.zeros(GRID_WIDTH, dtype=int))
        
        return np.array(new_grid), lines_cleared

    def _collision_at_pos(self, piece, pos, grid):
        """Check collision at specific position on a grid"""
        px, py = pos
        for y in range(piece.shape[0]):
            for x in range(piece.shape[1]):
                if piece[y][x] and (
                    y + px >= GRID_HEIGHT or
                    x + py < 0 or
                    x + py >= GRID_WIDTH or
                    (y + px >= 0 and grid[y + px][x + py])
                ):
                    return True
        return False

    def load_config(self, path):
        """Load configuration from JSON file"""
        try:
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
        """Spawn a new random piece at the top"""
        self.current_piece = self.next_piece  # Use the next piece
        self.current_pos = [0, GRID_WIDTH // 2 - len(self.current_piece[0]) // 2]
        self.next_piece = np.array(SHAPES[random.randint(0, len(SHAPES) - 1)])  # Prepare the next piece

        if self.collision(self.current_piece, self.current_pos):
            self.game_over = True

    def collision(self, piece, pos):
        """Check if piece collides with grid boundaries or existing blocks"""
        px, py = pos
        for y in range(piece.shape[0]):
            for x in range(piece.shape[1]):
                if piece[y][x] and (
                        y + px >= GRID_HEIGHT or
                        x + py < 0 or
                        x + py >= GRID_WIDTH or
                        (y + px >= 0 and self.grid[y + px][x + py])
                ):
                    return True
        return False

    def freeze(self):
        """Fix current piece into the grid and clear completed lines"""
        px, py = self.current_pos
        
        # Place piece in grid
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    if px + i >= 0:  # Only place if within grid
                        self.grid[px + i][py + j] = 1
        
        # Clear completed lines
        lines_cleared = self.clear_lines()
        
        # Spawn next piece
        self.spawn_piece()
        
        return lines_cleared

    def clear_lines(self):
        """Clear completed lines and return number of lines cleared"""
        # Find incomplete rows
        new_grid = []
        lines_cleared = 0
        
        for row in self.grid:
            if not np.all(row):  # Row is not completely filled
                new_grid.append(row.copy())
            else:
                lines_cleared += 1
        
        # Add empty rows at the top
        while len(new_grid) < GRID_HEIGHT:
            new_grid.insert(0, np.zeros(GRID_WIDTH, dtype=int))
        
        # Update grid and score
        self.grid = np.array(new_grid)
        self.score += lines_cleared
        
        return lines_cleared

    def rotate(self, piece):
        """Rotate piece 90 degrees clockwise"""
        return np.rot90(piece, -1)

    def get_observation(self):
        """Get current game state observation"""
        return {
            "grid": self.grid.copy(),
            "current_piece": self.current_piece.copy(),
            "current_pos": tuple(self.current_pos)
        }

    def step(self, action):
        """Execute one game step with given action"""
        if self.game_over:
            return self.get_observation(), self.death_penalty, True

        reward = 0.0
        frozen_this_step = False
        cleared = 0

        # Execute action
        px, py = self.current_pos
        if action == 0 and not self.collision(self.current_piece, (px, py - 1)):  # Left
            self.current_pos[1] -= 1
        elif action == 1 and not self.collision(self.current_piece, (px, py + 1)):  # Right
            self.current_pos[1] += 1
        elif action == 2:  # Rotate
            rotated = self.rotate(self.current_piece)
            if not self.collision(rotated, self.current_pos):
                self.current_piece = rotated
        elif action == 3:  # Hard drop
            while not self.collision(self.current_piece, (self.current_pos[0] + 1, self.current_pos[1])):
                self.current_pos[0] += 1
            frozen_this_step = True
            cleared = self.freeze()

        # Natural falling (gravity)
        if not frozen_this_step:
            if not self.collision(self.current_piece, (self.current_pos[0] + 1, self.current_pos[1])):
                self.current_pos[0] += 1
            else:
                frozen_this_step = True
                cleared = self.freeze()

        # Calculate reward for line clears
        if cleared > 0:
            reward += int(self.line_rewards.get(str(cleared), 0))

        return self.get_observation(), reward, self.game_over

    def render(self):
        """Render the game using pygame"""
        if not self.render_mode:
            return

        try:
            self.screen.fill((0, 0, 0))

            # Draw grid
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    val = self.grid[y][x]
                    color = COLORS[val] if val < len(COLORS) else COLORS[0]
                    pygame.draw.rect(
                        self.screen, 
                        color, 
                        (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 
                        0
                    )

            # Draw current piece
            if self.current_piece is not None:
                px, py = self.current_pos
                for y in range(self.current_piece.shape[0]):
                    for x in range(self.current_piece.shape[1]):
                        if self.current_piece[y][x]:
                            screen_x = (py + x) * BLOCK_SIZE
                            screen_y = (px + y) * BLOCK_SIZE
                            if screen_y >= 0:  # Only draw if visible
                                color_idx = self.current_piece[y][x]
                                color = COLORS[color_idx] if color_idx < len(COLORS) else COLORS[1]
                                pygame.draw.rect(
                                    self.screen,
                                    color,
                                    (screen_x, screen_y, BLOCK_SIZE, BLOCK_SIZE),
                                    0
                                )

            pygame.display.flip()

            if self.clock:
                self.clock.tick(FPS)

        except pygame.error as e:
            print(f"Render error: {e}")

    def reset(self):
        """Reset the game to initial state"""
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.score = 0
        self.game_over = False
        self.spawn_piece()
        return self.get_observation()

    def close(self):
        """Clean up pygame resources"""
        if self.render_mode:
            try:
                pygame.quit()
            except:
                pass


def get_user_action():
    """Get user input for manual play"""
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
    # Manual play mode
    env = TetrisEnv(render_mode=True)
    obs = env.reset()
    done = False

    print("Controls: Arrow keys for movement, ESC to quit")

    while not done:
        action = get_user_action()
        if action == 'quit':
            break
        obs, reward, done = env.step(action)
        env.render()
        
        if reward > 0:
            print(f"Lines cleared! Reward: {reward}")

    env.close()