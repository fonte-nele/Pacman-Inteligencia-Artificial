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

"""
  Feito por:
    Felipe Fontenele de Avila Magalhaes 15.1.4331
    Isadora Fonseca Alves               15.1.5951
    Jeferson Afonso do Patrocinio       13.2.4568
"""

from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distancia = []  # Distancia de Manhattan
        listaComida = currentGameState.getFood().asList() # Lista de comida
        posicaoPacman = list(successorGameState.getPacmanPosition())  # Posicao do Pacman

        if action == 'Stop':
          return -float("inf")

        for ghostState in newGhostStates: # Estado dos Fantasmas
          if ghostState.getPosition() == tuple(posicaoPacman) and ghostState.scaredTimer is 0:
            return -float("inf")

        for comida in listaComida:
          x = -1 * abs(comida[0] - posicaoPacman[0])
          y = -1 * abs(comida[1] - posicaoPacman[1])
          distancia.append(x + y)

        return max(distancia)
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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
    pacmanIndex = 0
    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        return self.computeMiniMaxAction(gameState)

        #util.raiseNotDefined()

    def computeMiniMaxAction(self, gameState):
        numAgents = gameState.getNumAgents()
        if numAgents > 1:
            #get all the first actions of pacman agent
            nextActions = gameState.getLegalActions(0)
            if nextActions:
                optimalAction, optimalScore = max([(action, self.min_value(gameState.generateSuccessor(0, action), 1, 0)) for action in nextActions], key = lambda x: x[1])

                return optimalAction

            else:
                return None

        else:
            nextActions = gameState.getLegalActions(0)
            if nextActions:
                optimalAction, optimalScore = max([(action, self.max_value(gameState.generateSuccessor(0, action), 0, 0)) for action in nextActions], key = lambda x: x[1])

                return optimalAction

            else:
                return None


    def checkExitRecursing(self, state, depth):
        return depth == self.depth or state.isWin() or state.isLose()


    def value(self, state, agentIndex, depth):
        if self.checkExitRecursing(state, depth):
            return self.evaluationFunction(state)

        if agentIndex == 0:
            return self.minimax_value(state, agentIndex, depth, max)

        else:
            return self.minimax_value(state, agentIndex, depth, min)


    def minimax_value(self, state, agentIndex, depth, minmax):
        if self.checkExitRecursing(state, depth):
            return self.evaluationFunction(state)

        else:
            v = (-9999999 if minmax(2, 1) == 2 else +9999999)
            nextActions = state.getLegalActions(agentIndex)
            successorStates = [state.generateSuccessor(agentIndex, action) for action in nextActions]

            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            for successor in successorStates:
                if nextAgentIndex != 0:
                    v = minmax(v, self.value(successor, nextAgentIndex, depth))

                else:
                    v = minmax(v, self.value(successor, nextAgentIndex, depth + 1))

            return v

    def max_value(self, state, agentIndex, depth):
        return self.minimax_value(state, agentIndex, depth, max)

    def min_value(self, state, agentIndex, depth):
        return self.minimax_value(state, agentIndex, depth, min)
    "*** ENDS HERE ***"

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    pacmanIndex = 0
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxplayer(gameState, 0, -sys.maxint - 1, sys.maxint)[1]

    def maxplayer(self,gameState, depth, alpha, beta):
      if depth == self.depth:
        return self.evaluationFunction(gameState), None

      actionList = gameState.getLegalActions(0)
      bestScore = -sys.maxint - 1
      bestAction = None

      if len(actionList) == 0:
        return self.evaluationFunction(gameState), None

      for action in actionList:
        if alpha > beta:
          return bestScore, bestAction
        newState = gameState.generateSuccessor(0, action)
        newScore = self.minplayer(newState, 1, depth, alpha, beta)[0]
        if newScore > bestScore:
          bestScore, bestAction = newScore, action
        if newScore > alpha:
          alpha = newScore
      return bestScore, bestAction

    def minplayer(self, gameState, ID, depth, alpha, beta):
      actionList = gameState.getLegalActions(ID)
      bestScore = sys.maxint
      bestAction = None

      if len(actionList) == 0:
        return self.evaluationFunction(gameState), None

      for action in actionList:
        if alpha > beta:
          return bestScore, bestAction

        newState = gameState.generateSuccessor(ID, action)
        if ID == gameState.getNumAgents() - 1:
          newScore = self.maxplayer(newState, depth + 1, alpha, beta)[0]
        else:
          newScore = self.minplayer(newState, ID + 1, depth, alpha, beta)[0]

        if newScore < bestScore:
          bestScore, bestAction = newScore, action
        if newScore < beta:
          beta = newScore
      return bestScore, bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    pacmanIndex = 0
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        profundidadeAtual = 0
        indexAgentAtual = 0
        val = self.valor(gameState, indexAgentAtual, profundidadeAtual)
        return val[0]

    def valor(self, gameState, indexAgentAtual, profundidadeAtual):
        if indexAgentAtual >= gameState.getNumAgents():
          indexAgentAtual = 0
          profundidadeAtual += 1

        if profundidadeAtual == self.depth:
          return self.evaluationFunction(gameState)

        if indexAgentAtual == self.pacmanIndex:
          return self.maxValor(gameState, indexAgentAtual, profundidadeAtual)
        else:
          return self.expValor(gameState, indexAgentAtual, profundidadeAtual)

    def expValor(self, gameState, indexAgentAtual, profundidadeAtual):
        v = ["unknown", 0]

        if not gameState.getLegalActions(indexAgentAtual):
          return self.evaluationFunction(gameState)

        prob = 1.0 / len(gameState.getLegalActions(indexAgentAtual))

        for action in gameState.getLegalActions(indexAgentAtual):
          if action == "Stop":
            continue

          retVal = self.valor(gameState.generateSuccessor(indexAgentAtual, action), indexAgentAtual + 1, profundidadeAtual)
          if type(retVal) is tuple:
            retVal = retVal[1]

          v[1] += retVal * prob
          v[0] = action

        return tuple(v)

    def maxValor(self, gameState, indexAgentAtual, profundidadeAtual):
        v = ("unknown", -1 * float("inf"))

        if not gameState.getLegalActions(indexAgentAtual):
          return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(indexAgentAtual):
          if action == "Stop":
            continue

          retVal = self.valor(gameState.generateSuccessor(indexAgentAtual, action), indexAgentAtual + 1, profundidadeAtual)
          if type(retVal) is tuple:
            retVal = retVal[1]

          vNew = max(v[1], retVal)

          if vNew is not v[1]:
            v = (action, vNew)
        return v
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRICAO: Distancia(negativa) para a comida mais proxima + Distancia(negativa) para o fantasma mais perto - 80 pontos para cada capsula restante - 30 pontos para cada fantasma nao "scared"

      Distancia para comida = negativo distancia de manhattan
      Distancia para fantasmas = 0 para fantasmas assustados, inverso negativo da distancia de manhattan em demais fantasmas
    """
    "*** YOUR CODE HERE ***"
    distanceToFood = [] # Distancia de Manhattan
    distanceToNearestGhost = []

    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsuleList = currentGameState.getCapsules()
    numberOfScaredGhosts = 0

    pacmanPos = list(currentGameState.getPacmanPosition())

    for ghostState in ghostStates:
      if ghostState.scaredTimer is 0:
        numberOfScaredGhosts += 1
        distanceToNearestGhost.append(0)
        continue

      ghostCoord = ghostState.getPosition()
      x = abs(ghostCoord[0] - pacmanPos[0])
      y = abs(ghostCoord[1] - pacmanPos[1])
      if (x + y) == 0:
        distanceToNearestGhost.append(0)
      else:
        distanceToNearestGhost.append(-1.0 / (x + y))

    for food in foodList:
      x = abs(food[0] - pacmanPos[0])
      y = abs(food[1] - pacmanPos[1])
      distanceToFood.append(-1 * (x + y))

    if not distanceToFood:
      distanceToFood.append(0)

    return max(distanceToFood) + min(distanceToNearestGhost) + currentGameState.getScore() - 80*len(capsuleList) - 30 *(len(ghostStates) - numberOfScaredGhosts)
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
