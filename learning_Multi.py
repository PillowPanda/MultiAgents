from Agent_Multi import QLearningAgent
from Environment_Multi import MultiAgentEnvironment

def train_multi_agents(episodes=500, width=10, height=10):
    env = MultiAgentEnvironment(width=width, height=height)
    agent1 = QLearningAgent(name="Agent1")
    agent2 = QLearningAgent(name="Agent2")

    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = [0, 0]

        done = False
        steps = 0
        while steps < 100:  # limit steps per episode
            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)
            next_state, rewards = env.step([action1, action2])

            agent1.learn(state, action1, rewards[0], next_state)
            agent2.learn(state, action2, rewards[1], next_state)

            total_reward[0] += rewards[0]
            total_reward[1] += rewards[1]

            state = next_state
            steps += 1

        rewards_per_episode.append((total_reward[0], total_reward[1]))
        print(f"Episode {episode + 1}: Agent1 Reward = {total_reward[0]}, Agent2 Reward = {total_reward[1]}")

    return agent1, agent2, rewards_per_episode
