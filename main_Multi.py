import matplotlib.pyplot as plt
import matplotlib.patches as patches
from learning_Multi import train_multi_agents
from Environment_Multi import MultiAgentEnvironment
import time

def run_and_visualize_fast(episodes=3, width=10, height=10):
    env = MultiAgentEnvironment(width, height, goal_count=6)
    agent1, agent2, _ = train_multi_agents(episodes=0, width=width, height=height)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    agent_patches = [
        patches.Circle((0.5, 0.5), 0.35, color='red'),
        patches.Circle((width - 0.5, height - 0.5), 0.35, color='blue')
    ]
    goal_patches = [
        patches.Circle((0.5, 0.5), 0.3, color='green'),
        patches.Circle((0.5, 0.5), 0.3, color='green')
    ]

    for patch in agent_patches + goal_patches:
        ax.add_patch(patch)

    for (x, y) in env.obstacles:
        rect = patches.Rectangle((x, height - y - 1), 1, 1, color='black')
        ax.add_patch(rect)

    fig.canvas.draw()
    fig.show()

    for ep in range(1, episodes + 1):
        state = env.reset()
        goal_counter = 0

        for step in range(2000):
            ax.set_title(f"Episode {ep}, Step {step}, Goals Collected: {goal_counter}")

            for i, (x, y) in enumerate(env.agent_positions):
                agent_patches[i].center = (x + 0.5, height - y - 0.5)

            for i in range(2):
                if i < len(env.active_goals):
                    gx, gy = env.active_goals[i]
                    goal_patches[i].set_visible(True)
                    goal_patches[i].center = (gx + 0.5, height - gy - 0.5)
                else:
                    goal_patches[i].set_visible(False)

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)
            state, info = env.step([action1, action2])

            goal_counter += info.get("goals_collected", 0)
            if goal_counter >= 6:
                break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_and_visualize_fast(episodes=100)
