# myAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import Agent
from game import Directions
from searchProblems import PositionSearchProblem

import util
import time
import search
from collections import deque

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

class MyAgent(Agent):
    """
    Competitive multi-pacman agent using Voronoi food partitioning
    and cached BFS pathfinding for minimal compute overhead.
    """

    _shared = {}

    def initialize(self):
        if self.index == 0:
            MyAgent._shared = {}

    def registerInitialState(self, state):
        s = MyAgent._shared
        if self.index == 0:
            walls = state.getWalls()
            width = walls.width
            height = walls.height

            # Build adjacency list
            adj = {}
            for x in range(width):
                for y in range(height):
                    if not walls[x][y]:
                        neighbors = []
                        for action, dx, dy in [
                            (Directions.NORTH, 0, 1),
                            (Directions.SOUTH, 0, -1),
                            (Directions.EAST, 1, 0),
                            (Directions.WEST, -1, 0),
                        ]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny]:
                                neighbors.append(((nx, ny), action))
                        adj[(x, y)] = neighbors
            s['adj'] = adj
            s['width'] = width
            s['height'] = height

            num_agents = state.getNumPacmanAgents()
            s['num_agents'] = num_agents

            # Collect initial food positions
            food_grid = state.getFood()
            food_set = set()
            for x in range(width):
                for y in range(height):
                    if food_grid[x][y]:
                        food_set.add((x, y))
            s['remaining_food'] = food_set

            # Agent starting positions
            positions = []
            for i in range(num_agents):
                p = state.getPacmanPosition(i)
                positions.append((int(p[0]), int(p[1])))
            s['positions'] = positions

            # Initialize per-agent tracking
            s['paths'] = [deque() for _ in range(num_agents)]
            s['targets'] = [None for _ in range(num_agents)]
            s['claimed'] = set()

            # Voronoi food assignment via multi-source BFS
            s['assignments'] = [set() for _ in range(num_agents)]
            self._voronoi_assign(s, positions, food_set)

            # Pre-compute initial paths for all agents
            for i in range(num_agents):
                self._compute_path_for(s, i, positions[i])

    def _voronoi_assign(self, s, positions, food_set):
        adj = s['adj']
        num_agents = s['num_agents']
        assignments = s['assignments']

        visited = {}
        queue = deque()
        for i, pos in enumerate(positions):
            if pos not in visited:
                visited[pos] = i
                queue.append((pos, i))

        while queue:
            pos, agent_id = queue.popleft()
            if pos in food_set:
                assignments[agent_id].add(pos)
            for neighbor, _ in adj.get(pos, []):
                if neighbor not in visited:
                    visited[neighbor] = agent_id
                    queue.append((neighbor, agent_id))

    def _compute_path_for(self, s, agent_idx, start_pos):
        adj = s['adj']
        my_food = s['assignments'][agent_idx]
        remaining = s['remaining_food']
        claimed = s['claimed']

        # Try assigned food first
        target, path = self._bfs_to_food(adj, start_pos, my_food & remaining)
        if target is None:
            # Try unclaimed remaining food
            target, path = self._bfs_to_food(adj, start_pos, remaining - claimed)
        if target is None:
            # Try any remaining food
            target, path = self._bfs_to_food(adj, start_pos, remaining)

        if target is not None:
            s['paths'][agent_idx] = deque(path)
            s['targets'][agent_idx] = target
            s['claimed'].add(target)
        else:
            s['paths'][agent_idx] = deque()
            s['targets'][agent_idx] = None

    def _bfs_to_food(self, adj, start, food_targets):
        if not food_targets:
            return None, []
        start = (int(start[0]), int(start[1]))
        if start in food_targets:
            return start, []

        parent = {start: None}
        queue = deque([start])

        while queue:
            pos = queue.popleft()
            for neighbor, action in adj.get(pos, []):
                if neighbor in parent:
                    continue
                parent[neighbor] = (pos, action)
                if neighbor in food_targets:
                    path = []
                    cur = neighbor
                    while parent[cur] is not None:
                        prev_pos, act = parent[cur]
                        path.append(act)
                        cur = prev_pos
                    path.reverse()
                    return neighbor, path
                queue.append(neighbor)

        return None, []

    def getAction(self, state):
        s = MyAgent._shared
        idx = self.index

        # Fast path: follow cached path if target is still valid
        if s['paths'][idx]:
            target = s['targets'][idx]
            if target and state.hasFood(target[0], target[1]):
                return s['paths'][idx].popleft()

        # Slow path: need to re-plan
        pos = state.getPacmanPosition(self.index)
        pos = (int(pos[0]), int(pos[1]))

        # Update remaining food
        food_grid = state.getFood()
        new_remaining = set()
        for x in range(s['width']):
            for y in range(s['height']):
                if food_grid[x][y]:
                    new_remaining.add((x, y))
        s['remaining_food'] = new_remaining

        # Release stale target
        old_target = s['targets'][idx]
        if old_target and old_target not in new_remaining:
            s['claimed'].discard(old_target)
            s['targets'][idx] = None

        # Also clean up stale claims from other agents
        s['claimed'] = s['claimed'] & new_remaining

        # Compute new path
        self._compute_path_for(s, idx, pos)

        if s['paths'][idx]:
            return s['paths'][idx].popleft()

        # Fallback: return any legal action
        legal = state.getLegalActions(self.index)
        for a in legal:
            if a != Directions.STOP:
                return a
        return Directions.STOP


"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        problem = AnyFoodSearchProblem(gameState, self.index)
        return search.bfs(problem)

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x, y = state
        return self.food[x][y]
