import numpy as np
import random

class MultiAgentEnvironment:
    def __init__(self, width, height, goal_count=6):
        self.width = width
        self.height = height
        self.goal_count = goal_count
        self.obstacles = self.generate_obstacles()
        self.reset()

    def generate_obstacles(self):
        obstacles = set()
        for row in range(2, self.height, 3):  # every 3rd row
            row_cells = list(range(self.width))
            random.shuffle(row_cells)
            count = int(0.8 * self.width)
            for col in row_cells[:count]:
                obstacles.add((col, row))
        return obstacles

    def generate_goal_pool(self):
        pool = set()
        while len(pool) < self.goal_count:
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if pos not in self.obstacles and pos not in self.agent_positions:
                pool.add(pos)
        return list(pool)

    def reset(self):
        self.agent_positions = [(0, 0), (self.width - 1, self.height - 1)]
        self.total_goals = self.generate_goal_pool()
        self.active_goals = []
        while len(self.active_goals) < 2 and self.total_goals:
            self.active_goals.append(self.total_goals.pop())
        return self.get_state()

    def get_state(self):
        return tuple(self.agent_positions + self.active_goals)

    def is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1

    def step(self, actions):
        rewards = [0, 0]
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        goals_collected = 0
        goals_reached = set()

        for i, action in enumerate(actions):
            dx, dy = directions[action]
            new_x = self.agent_positions[i][0] + dx
            new_y = self.agent_positions[i][1] + dy
            new_pos = (max(0, min(new_x, self.width - 1)), max(0, min(new_y, self.height - 1)))
            if new_pos not in self.obstacles:
                self.agent_positions[i] = new_pos

        agent_pos_set = set(self.agent_positions)
        for goal in list(self.active_goals):
            if goal in agent_pos_set:
                goals_reached.add(goal)
                goals_collected += 1
                for i in range(2):
                    if self.agent_positions[i] == goal:
                        rewards[i] += 10

        for goal in goals_reached:
            self.active_goals.remove(goal)

        # Add goals only if total_goals remain
        while len(self.active_goals) < 2 and self.total_goals:
            self.active_goals.append(self.total_goals.pop())

        if self.is_adjacent(*self.agent_positions):
            rewards = [-5, -5]

        info = {
            "goals_collected": goals_collected
        }

        return self.get_state(), info
