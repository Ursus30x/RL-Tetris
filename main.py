import numpy as np
import torch
from collections import deque
from tetris_env import TetrisEnv  # Your env module
from agent import ImprovedTetrisAgent  # Your agent module
import time

def normalize_state(obs):
    grid = obs["grid"].copy()
    piece = obs["current_piece"]
    pos = obs["current_pos"]
    
    # Oznacz aktualny klocek specjalną wartością (2.0)
    px, py = pos
    for y in range(piece.shape[0]):
        for x in range(piece.shape[1]):
            if piece[y][x]:
                gx, gy = px + y, py + x
                if 0 <= gx < grid.shape[0] and 0 <= gy < grid.shape[1]:
                    grid[gx][gy] = 2.0  # Specjalna wartość dla aktywnego klocka
                    
    # Normalizuj i dodaj kanał wysokości
    normalized = grid.astype(np.float32) / 7.0
    
    # Dodaj kanał z wysokością kolumn
    height_map = np.zeros_like(grid, dtype=np.float32)
    for col in range(grid.shape[1]):
        for row in range(grid.shape[0]):
            if grid[row][col] > 0:
                height = grid.shape[0] - row
                height_map[:, col] = height / grid.shape[0]
                break
    
    # Połącz obie mapy w wielokanałową reprezentację
    return np.stack([normalized, height_map], axis=0)


# Main training loop
def train_tetris_agent(episodes=2000, render_every=500, save_every=100, eval_every=100):
    env = TetrisEnv(render_mode=False)  # Wyłącz rendering dla szybszego treningu
    n_frames = 4
    state_shape = (n_frames * 2, 20, 10)  # 2 kanały * 4 klatki
    n_actions = 5

    agent = ImprovedTetrisAgent(state_shape=state_shape, n_actions=n_actions, n_frames=n_frames)
    
    # Tracking najlepszych wyników
    best_reward = float('-inf')
    plateau_counter = 0
    
    print("Rozpoczynanie treningu z ulepszonymi nagrodami...")
    print(f"Epsilon start: {agent.epsilon}, min: {agent.epsilon_min}, decay: {agent.epsilon_decay}")

    for episode in range(episodes):
        raw_state = env.reset()
        agent.reset_frame_stack()
        state = normalize_state(raw_state)
        stacked_state = agent.preprocess_state(state)

        total_reward = 0
        done = False
        steps_in_episode = 0
        lines_cleared_total = 0

        while not done:
            action = agent.act(stacked_state, training=True)
            raw_next_state, reward, done = env.step(action)
            next_state = normalize_state(raw_next_state)
            stacked_next_state = agent.preprocess_state(next_state)

            agent.remember(stacked_state, action, reward, stacked_next_state, done)
            
            # Trenuj częściej przy dużym buforze
            if len(agent.memory) > agent.warmup_steps:
                loss = agent.replay(episode)
                if loss is not None:
                    agent.training_stats['losses'].append(loss)

            stacked_state = stacked_next_state
            total_reward += reward
            steps_in_episode += 1
            
            # Zlicz usunięte linie
            if hasattr(env, 'score'):
                lines_cleared_total = env.score

            # Renderuj co 50 epizodów
            if episode % render_every == 0:
                if not env.render_mode:
                    env.render_mode = True
                    env.__init__(render_mode=True)
                env.render()
                time.sleep(0.02)  # Mniejsze opóźnienie dla płynności

        # Aktualizuj statystyki
        avg_reward = agent.update_stats(episode, total_reward)
        
        # Logging co 50 epizodów
        if episode % 50 == 0 or episode < 50:
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Reward: {total_reward:.2f}")
            print(f"  Avg (50): {avg_reward:.2f}" if len(agent.recent_rewards) >= 50 else "")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Lines: {lines_cleared_total}")
            print(f"  Steps: {steps_in_episode}")
            print(f"  Memory: {len(agent.memory)}")
            if agent.training_stats['losses']:
                print(f"  Loss: {agent.training_stats['losses'][-1]:.4f}")
            print("-" * 40)
        
        # Zapisuj model co jakiś czas
        if episode % save_every == 0 and episode > 0:
            filename = f"tetris_model_ep_{episode}.pth"
            agent.save(filename)
            print(f"Model saved: {filename}")
        
        # Sprawdź czy model się poprawia
        if len(agent.recent_rewards) >= 100:
            current_avg = np.mean(list(agent.recent_rewards)[-100:])
            if current_avg > best_reward:
                best_reward = current_avg
                plateau_counter = 0
                agent.save("best_tetris_model.pth")
            else:
                plateau_counter += 1
                
            # Jeśli długo brak postępu, zwiększ eksplorację
            if plateau_counter > 200:
                agent.epsilon = min(0.3, agent.epsilon * 1.2)
                plateau_counter = 0
                print("Plateau detected, increasing exploration")
        
        # Wyłącz rendering po demonstracji
        if episode % render_every == 0 and episode > 0:
            env.render_mode = False

    # Zapisz finalny model
    agent.save("final_tetris_model.pth")
    env.close()
    
    # Pokaż statystyki końcowe
    print("\n" + "="*50)
    print("TRENING ZAKOŃCZONY")
    print(f"Najlepszy średni wynik: {best_reward:.2f}")
    print(f"Finalny epsilon: {agent.epsilon:.4f}")
    print(f"Całkowite kroki: {agent.steps}")
    if agent.training_stats['losses']:
        print(f"Średnia strata: {np.mean(agent.training_stats['losses'][-100:]):.4f}")
    print("="*50)

if __name__ == "__main__":
    train_tetris_agent(episodes=5000)
