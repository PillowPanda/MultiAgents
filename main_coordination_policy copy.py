import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Environment_Multi import MultiAgentEnvironment
import time
from collections import deque

# BFS pathfinder to get first move toward goal avoiding obstacles
def bfs_path(start, goal, obstacles, width, height):
    queue = deque([(start, [])])
    visited = set([start])
    directions = {
        (0, -1): 0,  # up
        (0, 1): 1,   # down
        (-1, 0): 2,  # left
        (1, 0): 3    # right
    }

    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path[0] if path else 0  # Already on goal

        for (dx, dy), action in directions.items():
            nx, ny = x + dx, y + dy
            next_pos = (nx, ny)
            if 0 <= nx < width and 0 <= ny < height and next_pos not in visited and next_pos not in obstacles:
                queue.append(((nx, ny), path + [action]))
                visited.add(next_pos)

    return 0  # fallback: move up if goal unreachable

# Manhattan distance helper
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Smart coordinated goal assignment and movement
def choose_coordinated_actions(agent_positions, active_goals, env):
    assignments = [None, None]
    distances = [[], []]

    for agent_idx in range(2):
        for goal in active_goals:
            dist = manhattan_distance(agent_positions[agent_idx], goal)
            distances[agent_idx].append((dist, goal))
        distances[agent_idx].sort()

    claimed_goals = set()

    # First pass: assign goal to closest agent
    for i in range(2):
        for dist, goal in distances[i]:
            other = 1 - i
            other_dist = manhattan_distance(agent_positions[other], goal)
            if goal not in claimed_goals and dist <= other_dist:
                assignments[i] = goal
                claimed_goals.add(goal)
                break

    # Second pass: assign any remaining goal to idle agent
    for i in range(2):
        if assignments[i] is None:
            for dist, goal in distances[i]:
                if goal not in claimed_goals:
                    assignments[i] = goal
                    claimed_goals.add(goal)
                    break

    # Compute actual path direction using BFS
    actions = []
    for i in range(2):
        agent_pos = agent_positions[i]
        target = assignments[i]

        if target is None:
            actions.append(0)  # default to up
        else:
            action = bfs_path(agent_pos, target, env.obstacles, env.width, env.height)
            actions.append(action)

    return actions

# Main visualization + coordination run
def run_coordinated_policy(episodes=50, width=35, height=35):
    env = MultiAgentEnvironment(width, height, goal_count=6)

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
            time.sleep(0.1)

            actions = choose_coordinated_actions(env.agent_positions, env.active_goals, env)
            state, info = env.step(actions)

            goal_counter += info.get("goals_collected", 0)
            if goal_counter >= 6:
                break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_coordinated_policy()
