import matplotlib.pyplot as plt
import matplotlib.patches as patches
from learning_Multi import train_multi_agents
from Environment_Multi import MultiAgentEnvironment
import time

def run_and_visualize_static(episodes=3, width=10, height=10):
    env = MultiAgentEnvironment(width, height)
    agent1, agent2, _ = train_multi_agents(episodes=0, width=width, height=height)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True)

    agent_patches = [
        patches.Circle((0.5, 0.5), 0.35, color='red'),
        patches.Circle((width - 0.5, height - 0.5), 0.35, color='blue')
    ]
    goal_patches = [patches.Circle((0.5, 0.5), 0.3, color='green') for _ in range(2)]
    obstacle_patches = []

    for patch in agent_patches + goal_patches:
        ax.add_patch(patch)

    for (x, y) in env.obstacles:
        rect = patches.Rectangle((x, height - y - 1), 1, 1, color='black')
        ax.add_patch(rect)
        obstacle_patches.append(rect)

    plt.ion()
    fig.show()

    for ep in range(1, episodes + 1):
        state = env.reset()
        step = 0
        while step < 100:
            ax.set_title(f"Episode {ep}, Step {step}")

            for i, (x, y) in enumerate(env.agent_positions):
                agent_patches[i].center = (x + 0.5, height - y - 0.5)

            for i, (x, y) in enumerate(env.active_goals):
                goal_patches[i].center = (x + 0.5, height - y - 0.5)

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.2)

            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)
            next_state, _ = env.step([action1, action2])
            state = next_state
            step += 1

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_and_visualize_static(episodes=5)