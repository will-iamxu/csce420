# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        # Incentivize moving toward the nearest food pellet
        foodList = newFood.asList()
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 10.0 / (minFoodDist + 1)

        # Ghost proximity features
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            if newScaredTimes[i] > 0:
                # Scared ghost: reward being close (can eat it)
                score += 200.0 / (dist + 1)
            else:
                # Active ghost: heavy penalty for being close
                if dist <= 1:
                    score -= 1000
                elif dist <= 3:
                    score -= 200.0 / dist

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        numAgents = gameState.getNumAgents()

        def minimax(state, depth, agentIndex):
            # Terminal conditions: win/lose or depth exhausted (at Pacman's turn)
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % numAgents
            # Depth decrements when the last ghost finishes and we return to Pacman
            nextDepth = depth - 1 if nextAgent == 0 else depth

            if agentIndex == 0:  # Pacman: maximize
                return max(
                    minimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                    for action in legalActions
                )
            else:  # Ghost: minimize
                return min(
                    minimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                    for action in legalActions
                )

        # Root: choose the Pacman action that maximizes the minimax value
        legalActions = gameState.getLegalActions(0)
        bestAction = max(
            legalActions,
            key=lambda action: minimax(
                gameState.generateSuccessor(0, action), self.depth, 1
            )
        )
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()

        def alphaBeta(state, depth, agentIndex, alpha, beta):
            # Terminal conditions
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth

            if agentIndex == 0:  # Pacman: maximize
                v = float('-inf')
                for action in legalActions:
                    v = max(v, alphaBeta(
                        state.generateSuccessor(agentIndex, action),
                        nextDepth, nextAgent, alpha, beta
                    ))
                    if v > beta:      # Prune (strict inequality, not on equality)
                        return v
                    alpha = max(alpha, v)
                return v
            else:  # Ghost: minimize
                v = float('inf')
                for action in legalActions:
                    v = min(v, alphaBeta(
                        state.generateSuccessor(agentIndex, action),
                        nextDepth, nextAgent, alpha, beta
                    ))
                    if v < alpha:     # Prune (strict inequality, not on equality)
                        return v
                    beta = min(beta, v)
                return v

        # Root: Pacman's first move, track best action while updating alpha
        alpha = float('-inf')
        beta = float('inf')
        bestAction = None
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0):
            score = alphaBeta(
                gameState.generateSuccessor(0, action),
                self.depth, 1, alpha, beta
            )
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        numAgents = gameState.getNumAgents()

        def expectimax(state, depth, agentIndex):
            # Terminal conditions
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth

            if agentIndex == 0:  # Pacman: maximize
                return max(
                    expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                    for action in legalActions
                )
            else:  # Ghost: uniform random, expected value
                return sum(
                    expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                    for action in legalActions
                ) / len(legalActions)

        # Root: choose the Pacman action with the highest expectimax value
        legalActions = gameState.getLegalActions(0)
        bestAction = max(
            legalActions,
            key=lambda action: expectimax(
                gameState.generateSuccessor(0, action), self.depth, 1
            )
        )
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
      Combines the current game score with several heuristic features:
        1. Reciprocal distance to the nearest food pellet: rewards proximity to food.
        2. Penalty proportional to remaining food count: encourages clearing the board.
        3. Penalty for remaining capsules: encourages using power pellets.
        4. Ghost proximity:
             - Scared ghosts: large reward for being close (they are worth 200 pts).
             - Active ghosts: steep penalty for being within striking distance.
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # Feature 1: reciprocal distance to nearest food
    if foodList:
        minFoodDist = min(manhattanDistance(pos, f) for f in foodList)
        score += 10.0 / (minFoodDist + 1)

    # Feature 2: penalty for remaining food
    score -= 4 * len(foodList)

    # Feature 3: penalty for remaining capsules
    score -= 20 * len(capsules)

    # Feature 4: ghost proximity
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        dist = manhattanDistance(pos, ghostPos)
        if ghostState.scaredTimer > 0:
            # Scared ghost: reward being close so we can eat it
            score += 200.0 / (dist + 1)
        else:
            # Active ghost: penalize proximity
            if dist <= 1:
                score -= 500
            elif dist <= 3:
                score -= 100.0 / dist

    return score

# Abbreviation
better = betterEvaluationFunction
