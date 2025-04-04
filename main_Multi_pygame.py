import pygame
import sys
import time
from learning_Multi import train_multi_agents
from Environment_Multi import MultiAgentEnvironment

CELL_SIZE = 40
GRID_COLOR = (200, 200, 200)
AGENT_COLORS = [(255, 0, 0), (0, 0, 255)]
GOAL_COLOR = (0, 255, 0)
OBSTACLE_COLOR = (0, 0, 0)
FPS = 10

def draw_grid(screen, env):
    screen.fill((255, 255, 255))
    width, height = env.width, env.height

    # Draw obstacles
    for (x, y) in env.obstacles:
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, OBSTACLE_COLOR, rect)

    # Draw goals
    for (x, y) in env.active_goals:
        center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(screen, GOAL_COLOR, center, CELL_SIZE // 3)

    # Draw agents
    for i, (x, y) in enumerate(env.agent_positions):
        center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(screen, AGENT_COLORS[i], center, CELL_SIZE // 2)

    # Draw grid lines
    for x in range(0, width * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, height * CELL_SIZE))
    for y in range(0, height * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (width * CELL_SIZE, y))

def run_pygame_simulation(episodes=1, width=10, height=10):
    pygame.init()
    screen = pygame.display.set_mode((width * CELL_SIZE, height * CELL_SIZE))
    pygame.display.set_caption("Multi-Agent Gridworld (Pygame)")

    clock = pygame.time.Clock()
    env = MultiAgentEnvironment(width, height)
    agent1, agent2, _ = train_multi_agents(episodes=0, width=width, height=height)

    for ep in range(episodes):
        state = env.reset()
        step = 0
        running = True
        while running and step < 100:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            draw_grid(screen, env)
            pygame.display.flip()
            clock.tick(FPS)

            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)
            next_state, _ = env.step([action1, action2])
            state = next_state
            step += 1

    pygame.quit()

if __name__ == "__main__":
    run_pygame_simulation(episodes=5)