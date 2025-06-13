import numpy as np
import pygame
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

# Define constants for Tetris game
BLOCK_SIZE = 30

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

class ScreenshotProcessor:
    def __init__(self, target_size=(84, 84), grayscale=True):
        """
        Initialize screenshot processor for Tetris game
        
        Args:
            target_size: Tuple of (width, height) for processed images
            grayscale: Whether to convert to grayscale
        """
        self.target_size = target_size
        self.grayscale = grayscale
        
        # Define transforms for preprocessing
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(target_size),
        ]
        
        if grayscale:
            transform_list.append(transforms.Grayscale())
            
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]) if grayscale 
            else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = transforms.Compose(transform_list)
    
    def render_game_state(self, game_state):
        """Render game state to an image without using pygame"""
        grid = game_state['grid']
        current_piece = game_state['current_piece']
        current_pos = game_state['current_pos']
        
        # Create blank image
        height, width = grid.shape
        img = np.zeros((height * BLOCK_SIZE, width * BLOCK_SIZE, 3), dtype=np.uint8)
        
        # Draw grid
        for y in range(height):
            for x in range(width):
                color_idx = grid[y][x]
                color = COLORS[color_idx] if color_idx < len(COLORS) else (0, 0, 0)
                cv2.rectangle(
                    img, 
                    (x * BLOCK_SIZE, y * BLOCK_SIZE),
                    ((x + 1) * BLOCK_SIZE, (y + 1) * BLOCK_SIZE),
                    color, 
                    -1
                )
        
        # Draw current piece
        px, py = current_pos
        for y in range(current_piece.shape[0]):
            for x in range(current_piece.shape[1]):
                if current_piece[y][x]:
                    color_idx = current_piece[y][x]
                    color = COLORS[color_idx] if color_idx < len(COLORS) else (0, 255, 255)
                    cv2.rectangle(
                        img,
                        ((py + x) * BLOCK_SIZE, (px + y) * BLOCK_SIZE),
                        ((py + x + 1) * BLOCK_SIZE, (px + y + 1) * BLOCK_SIZE),
                        color,
                        -1
                    )
        
        # Resize to target size
        if self.target_size:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        
        return img

    def capture_game_screen(self, env):
        """
        Capture screenshot from pygame surface
        
        Args:
            env: TetrisEnv instance with pygame surface
            
        Returns:
            numpy array of raw screenshot
        """
        if not env.render_mode or env.screen is None:
            # If not in render mode, create a temporary surface
            temp_surface = pygame.Surface((env.grid.shape[1] * 30, env.grid.shape[0] * 30))
            self._draw_game_state(temp_surface, env)
            screenshot = pygame.surfarray.array3d(temp_surface)
        else:
            # Capture from existing surface
            screenshot = pygame.surfarray.array3d(env.screen)
        
        # Convert from pygame format (width, height, channels) to standard (height, width, channels)
        screenshot = np.transpose(screenshot, (1, 0, 2))
        
        return screenshot
    
    def _draw_game_state(self, surface, env):
        """
        Draw the current game state on a pygame surface
        
        Args:
            surface: pygame surface to draw on
            env: TetrisEnv instance
        """
        # Colors for Tetris pieces
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
        
        BLOCK_SIZE = 30
        
        # Clear surface
        surface.fill((0, 0, 0))
        
        # Draw grid
        for y in range(env.grid.shape[0]):
            for x in range(env.grid.shape[1]):
                val = env.grid[y][x]
                color = COLORS[val] if val < len(COLORS) else COLORS[0]
                pygame.draw.rect(
                    surface, 
                    color, 
                    (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 
                    0
                )
        
        # Draw current piece
        if env.current_piece is not None:
            px, py = env.current_pos
            for y in range(env.current_piece.shape[0]):
                for x in range(env.current_piece.shape[1]):
                    if env.current_piece[y][x]:
                        screen_x = (py + x) * BLOCK_SIZE
                        screen_y = (px + y) * BLOCK_SIZE
                        if screen_y >= 0:  # Only draw if visible
                            color_idx = env.current_piece[y][x]
                            color = COLORS[color_idx] if color_idx < len(COLORS) else COLORS[1]
                            pygame.draw.rect(
                                surface,
                                color,
                                (screen_x, screen_y, BLOCK_SIZE, BLOCK_SIZE),
                                0
                            )
    
    def preprocess_screenshot(self, screenshot):
        """
        Preprocess raw screenshot for neural network input
        
        Args:
            screenshot: Raw screenshot as numpy array
            
        Returns:
            Preprocessed tensor ready for neural network
        """
        # Apply transforms
        processed = self.transform(screenshot)
        return processed
    
    def create_frame_stack(self, frames, stack_size=4):
        """
        Stack multiple frames together for temporal information
        
        Args:
            frames: List of preprocessed frames
            stack_size: Number of frames to stack
            
        Returns:
            Stacked frames tensor
        """
        if len(frames) < stack_size:
            # Pad with duplicates of first frame if not enough frames
            while len(frames) < stack_size:
                frames.insert(0, frames[0])
        
        # Take last stack_size frames
        stacked = torch.stack(frames[-stack_size:], dim=0)
        return stacked
    
    def get_game_features(self, screenshot):
        """
        Extract simple game features from screenshot for debugging/analysis
        
        Args:
            screenshot: Raw screenshot
            
        Returns:
            Dictionary of extracted features
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        
        # Simple feature extraction
        features = {
            'mean_brightness': np.mean(gray),
            'std_brightness': np.std(gray),
            'filled_pixels': np.sum(gray > 10),  # Non-black pixels
            'edge_density': np.sum(cv2.Canny(gray, 50, 150)) / gray.size
        }
        
        return features


class FrameBuffer:
    """Buffer to store recent frames for frame stacking"""
    
    def __init__(self, capacity=4):
        self.capacity = capacity
        self.frames = []
    
    def add_frame(self, frame):
        """Add a new frame to the buffer"""
        self.frames.append(frame)
        if len(self.frames) > self.capacity:
            self.frames.pop(0)
    
    def get_stacked_frames(self):
        """Get current stacked frames"""
        if len(self.frames) == 0:
            return None

        # Pad with last frame if buffer not full
        padded_frames = self.frames.copy()
        while len(padded_frames) < self.capacity:
            padded_frames.insert(0, padded_frames[0] if padded_frames else None)

        # Stack as channels: [channels, H, W]
        stacked = torch.cat([f for f in padded_frames], dim=0)  # Each f: [1, H, W], result: [channels, H, W]
        return stacked
    
    def reset(self):
        """Clear the frame buffer"""
        self.frames.clear()


def test_screenshot_processing():
    """Test function to verify screenshot processing works correctly"""
    from tetris_env import TetrisEnv
    
    # Create environment
    env = TetrisEnv(render_mode=True)
    env.reset()
    
    # Create processor
    processor = ScreenshotProcessor(target_size=(84, 84), grayscale=True)
    
    # Capture and process screenshot
    screenshot = processor.capture_game_screen(env)
    processed = processor.preprocess_screenshot(screenshot)
    
    print(f"Raw screenshot shape: {screenshot.shape}")
    print(f"Processed screenshot shape: {processed.shape}")
    print(f"Processed screenshot range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Test frame stacking
    frame_buffer = FrameBuffer(capacity=4)
    frame_buffer.add_frame(processed)
    frame_buffer.add_frame(processed)
    
    stacked = frame_buffer.get_stacked_frames()
    print(f"Stacked frames shape: {stacked.shape}")
    
    # Extract features
    features = processor.get_game_features(screenshot)
    print(f"Extracted features: {features}")
    
    env.close()
    print("Screenshot processing test completed successfully!")


if __name__ == "__main__":
    test_screenshot_processing()