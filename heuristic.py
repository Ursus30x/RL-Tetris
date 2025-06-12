import numpy as np
from copy import deepcopy
from tetris_env import TetrisEnv, GRID_WIDTH, GRID_HEIGHT

class TetrisHeuristicAgent:
    """
    A standalone Tetris agent that uses heuristic evaluation to make decisions.
    This agent doesn't use any machine learning - just rule-based evaluation.
    """
    
    def __init__(self, weights=None):
        """
        Initialize the heuristic agent with evaluation weights.
        
        Args:
            weights (dict): Dictionary of weights for heuristic components.
                           If None, uses optimized default weights.
        """
        # Optimized default weights based on Tetris strategy research
        self.weights = weights or {
            'aggregate_height': -0.510066,
            'complete_lines': 0.760666,
            'holes': -0.35663,
            'bumpiness': -0.184483,
            'wells': 0.25,  # Bonus for potential well structures
            'blockades': -0.5,  # Penalty for blocks covering holes
            'row_transitions': -0.35,  # Penalty for row changes
            'col_transitions': -0.35,  # Penalty for column changes
            'pit_count': -0.8,  # Penalty for deep pits
            'landing_height': -0.5  # Penalty for high placements
        }
        
        # Cached values for performance
        self._last_evaluations = []
        self._last_action = None
    
    def evaluate_position(self, grid, lines_cleared=0):
        """
        Evaluate a Tetris grid state using multiple heuristics.
        
        Args:
            grid (np.array): 2D numpy array representing the Tetris board
            lines_cleared (int): Number of lines just cleared
            
        Returns:
            float: Combined heuristic score
        """
        score = 0
        
        # Calculate each heuristic component
        metrics = {
            'aggregate_height': self._calculate_aggregate_height(grid),
            'complete_lines': lines_cleared,
            'holes': self._calculate_holes(grid),
            'bumpiness': self._calculate_bumpiness(grid),
            'wells': self._calculate_wells(grid),
            'blockades': self._calculate_blockades(grid),
            'row_transitions': self._calculate_row_transitions(grid),
            'col_transitions': self._calculate_col_transitions(grid),
            'pit_count': self._calculate_pits(grid),
            'landing_height': self._calculate_landing_height(grid),
        }
        
        # Apply weights to each component
        for key, value in metrics.items():
            score += self.weights.get(key, 0) * value
            
        return score
    
    def get_action(self, env):
        """
        Determine the best action based on heuristic evaluation of all possible moves.
        
        Args:
            env (TetrisEnv): The Tetris environment instance
            
        Returns:
            tuple: (rotation, column) placement action
        """
        current_piece = env.current_piece.copy()
        best_score = float('-inf')
        best_action = (0, GRID_WIDTH // 2)  # Default to center placement
        
        # Try all possible rotations (0-3)
        for rotation in range(4):
            rotated_piece = current_piece.copy()
            for _ in range(rotation):
                rotated_piece = np.rot90(rotated_piece, -1)
            
            # Try all possible columns where piece fits
            for col in range(GRID_WIDTH - rotated_piece.shape[1] + 1):
                # Simulate the move
                temp_env = self._simulate_placement(env, rotated_piece, col)
                
                # Evaluate the resulting position
                lines_cleared = temp_env.clear_lines()
                score = self.evaluate_position(temp_env.grid, lines_cleared)
                
                # Track best action
                if score > best_score or (score == best_score and col == GRID_WIDTH // 2):
                    best_score = score
                    best_action = (rotation, col)
        
        self._last_action = best_action
        return best_action
    
    def _simulate_placement(self, env, piece, col):
        """
        Simulate placing a piece in a column and return the resulting game state.
        
        Args:
            env (TetrisEnv): Current game environment
            piece (np.array): Piece to place (already rotated)
            col (int): Column to place the piece
            
        Returns:
            TetrisEnv: New environment with piece placed
        """
        # Create a new environment without copying pygame elements
        temp_env = TetrisEnv(render_mode=False)
        
        # Copy the essential game state
        temp_env.grid = env.grid.copy()
        temp_env.score = env.score
        temp_env.game_over = env.game_over
        temp_env.current_piece = piece.copy()
        temp_env.current_pos = [0, col]  # Start at top
        
        # Hard drop the piece
        while not temp_env.collision(temp_env.current_piece, 
                                (temp_env.current_pos[0] + 1, temp_env.current_pos[1])):
            temp_env.current_pos[0] += 1
        
        # Freeze the piece in place
        temp_env.freeze()
        return temp_env
    
    # Heuristic calculation methods
    def _calculate_aggregate_height(self, grid):
        """Sum of heights of each column"""
        heights = self._get_column_heights(grid)
        return sum(heights)
    
    def _get_column_heights(self, grid):
        """Height of each column (distance from top to highest block)"""
        heights = []
        for col in range(grid.shape[1]):
            col_data = grid[:, col]
            # Find the highest occupied cell in the column
            for row in range(grid.shape[0]):
                if col_data[row] > 0:
                    heights.append(grid.shape[0] - row)
                    break
            else:
                heights.append(0)
        return heights
    
    def _calculate_holes(self, grid):
        """Count of empty cells below blocks in each column"""
        heights = self._get_column_heights(grid)
        holes = 0
        
        for col in range(grid.shape[1]):
            if heights[col] == 0:
                continue
                
            col_data = grid[:, col]
            highest_block = grid.shape[0] - heights[col]
            
            # Count empty cells below the highest block
            for row in range(highest_block, grid.shape[0]):
                if col_data[row] == 0:
                    holes += 1
                    
        return holes
    
    def _calculate_bumpiness(self, grid):
        """Sum of absolute differences between adjacent column heights"""
        heights = self._get_column_heights(grid)
        bumpiness = 0
        
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
            
        return bumpiness
    
    def _calculate_wells(self, grid):
        """Sum of well depths (columns lower than both neighbors)"""
        heights = self._get_column_heights(grid)
        wells = 0
        
        for i in range(len(heights)):
            left = heights[i - 1] if i > 0 else float('inf')
            right = heights[i + 1] if i < len(heights) - 1 else float('inf')
            
            if heights[i] < left and heights[i] < right:
                well_depth = min(left, right) - heights[i]
                wells += well_depth
                
        return wells
    
    def _calculate_blockades(self, grid):
        """Count of blocks covering holes"""
        heights = self._get_column_heights(grid)
        blockades = 0
        
        for col in range(grid.shape[1]):
            if heights[col] == 0:
                continue
                
            col_data = grid[:, col]
            highest_block = grid.shape[0] - heights[col]
            
            # Check for blocks covering holes
            hole_found = False
            for row in range(highest_block, grid.shape[0]):
                if col_data[row] == 0:
                    hole_found = True
                elif hole_found:
                    blockades += 1
                    
        return blockades
    
    def _calculate_row_transitions(self, grid):
        """Count of transitions between filled and empty cells in rows"""
        transitions = 0
        
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1] - 1):
                if (grid[row, col] > 0) != (grid[row, col + 1] > 0):
                    transitions += 1
                    
        return transitions
    
    def _calculate_col_transitions(self, grid):
        """Count of transitions between filled and empty cells in columns"""
        transitions = 0
        
        for col in range(grid.shape[1]):
            for row in range(grid.shape[0] - 1):
                if (grid[row, col] > 0) != (grid[row + 1, col] > 0):
                    transitions += 1
                    
        return transitions
    
    def _calculate_pits(self, grid):
        """Sum of pit depths (columns much lower than neighbors)"""
        heights = self._get_column_heights(grid)
        pits = 0
        
        for i in range(len(heights)):
            left = heights[i - 1] if i > 0 else heights[i]
            right = heights[i + 1] if i < len(heights) - 1 else heights[i]
            avg_neighbor = (left + right) / 2
            
            if heights[i] < avg_neighbor - 2:  # At least 2 units deeper
                pits += (avg_neighbor - heights[i])
                
        return pits
    
    def _calculate_landing_height(self, grid):
        """Average height of placed pieces (higher placements are worse)"""
        heights = self._get_column_heights(grid)
        return sum(heights) / len(heights) if heights else 0