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


        print('Legal moves: ', legalMoves)
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        currentFood = currentGameState.getFood()
        CurrentPos = currentGameState.getPacmanPosition()

        "*** YOUR CODE HERE ***"

        for index in range(len(newScaredTimes)):
            #Checks if the ghost is in scared state ( if = 0 ghost is not scared)
            #Checks the manhatten distance between pacmans position and ghost position)
            #if statement make sure pacman is a distance of 1 from the ghost
            if newScaredTimes[index] == 0 and manhattanDistance(newPos, newGhostPositions[index]) <= 1:
                return successorGameState.getScore() - 15*successorGameState.getNumFood()

            MD = []
            for items in newFood.asList():
                MD.append(manhattanDistance(newPos,items))

            #If the MD list is empty the evaluation funcitons reverts to the original evaluation function
            if not MD:
                return successorGameState.getScore() - 10 * successorGameState.getNumFood()
            else:
             Min_Manhatten = min(MD)

            # print("List of Manhatten distance",MD, "Minimum MD ",Min_Manhatten)

        #The closer Pacman is to food the larger the value.
        return successorGameState.getScore() - 10*successorGameState.getNumFood() - Min_Manhatten




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

              gameState.isWin():
                Returns whether or not the game state is a winning state

              gameState.isLose():
                Returns whether or not the game state is a losing state
            """
            "*** YOUR CODE HERE ***"
            legalactions = gameState.getLegalActions(self.index)
            Successor_State = [gameState.generateSuccessor(self.index, actions) for actions in legalactions]
            numAgents = gameState.getNumAgents()
            WinState = gameState.isWin()
            LoseState = gameState.isLose()

            #Gets the initial Successor states for pacman
            decision = []
            for s in Successor_State:
                x = self.value(s,0,0)
                decision.append(x)
            mx = max(decision)
            index = [i for i, j in enumerate(decision) if j == mx]
            index_action = index[0]

            return legalactions[index_action]


        def value(self, state, index, currentdepth):
            #Terminal check
            if (currentdepth == self.depth or state.isWin() or state.isLose()):
                return self.evaluationFunction(state)

            index_increase = index +1
            #if the max index is reached. Depth is increased by 1 and index is returned to 0 to call pacman
            if index_increase == state.getNumAgents():
                currentdepth +=1
                index_increase = 0
                if currentdepth==self.depth:
                    return self.evaluationFunction(state)

            #If index is Pacman calls max function
            if index_increase==0:
                return self.max_value(state,index_increase,currentdepth)
            #if index is any of the ghost calls min function
            if index_increase > 0:
                return self.min_value(state,index_increase,currentdepth)


        def max_value(self,state,index_increase,depth):
            #Gets the actions
            actions = state.getLegalActions(index_increase)

            max_v = -float("inf")

            successors = [state.generateSuccessor(index_increase,action) for action in actions]

            for states in successors:
                v = self.value(states,index_increase,depth)
                max_v = max(max_v,v)
            return max_v

        def min_value(self,state,index_increase,depth):
            actions = state.getLegalActions(index_increase)

            min_v = float("inf")
            successors = [state.generateSuccessor(index_increase,action) for action in actions]
            for states in successors:
                v = self.value(states, index_increase,depth)
                min_v = min(min_v,v)
            return min_v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalactions = gameState.getLegalActions(self.index)
        Successor_State = [gameState.generateSuccessor(self.index, actions) for actions in legalactions]
        numAgents = gameState.getNumAgents()
        WinState = gameState.isWin()
        LoseState = gameState.isLose()

        # Gets the initial Successor states for pacman
        decision = []
        alpha = -float("inf")
        beta = float("inf")
        for s in Successor_State:
            x = self.value(s, 0, 0,alpha,beta)
            alpha = max(alpha, x)
            decision.append(x)
        mx = max(decision)
        index = [i for i, j in enumerate(decision) if j == mx]
        index_action = index[0]

        return legalactions[index_action]

    def value(self, state, index, currentdepth, alpha, beta):
        # Terminal check
        if (currentdepth == self.depth or state.isWin() or state.isLose()):
            return self.evaluationFunction(state)

        index_increase = index + 1
        # if the max index is reached. Depth is increased by 1 and index is returned to 0 to call pacman
        if index_increase == state.getNumAgents():
            currentdepth += 1
            index_increase = 0
            if currentdepth == self.depth:
                return self.evaluationFunction(state)

        # If index is Pacman calls max function
        if index_increase == 0:
            return self.max_value(state, index_increase, currentdepth, alpha, beta)
        # if index is any of the ghost calls min function
        if index_increase > 0:
            return self.min_value(state, index_increase, currentdepth, alpha, beta)

    def max_value(self, state, index_increase, depth,alpha,beta):
        # Gets the actions
        actions = state.getLegalActions(index_increase)
        max_v = -float("inf")
        for action in actions:
            states = state.generateSuccessor(index_increase, action)
            v = self.value(states, index_increase, depth,alpha,beta)
            max_v = max(max_v, v)
            if max_v > beta:
                return max_v
            alpha = max(alpha, max_v)
        return max_v

    def min_value(self, state, index_increase, depth, alpha, beta):
        actions = state.getLegalActions(index_increase)
        min_v = float("inf")
        for action in actions:
            states = state.generateSuccessor(index_increase, action)
            v = self.value(states, index_increase, depth, alpha, beta)
            min_v = min(min_v, v)
            if min_v < alpha:
                return min_v
            beta = min(beta, min_v)
        return min_v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalactions = gameState.getLegalActions(self.index)
        Successor_State = [gameState.generateSuccessor(self.index, actions) for actions in legalactions]
        numAgents = gameState.getNumAgents()
        WinState = gameState.isWin()
        LoseState = gameState.isLose()

        # Gets the initial Successor states for pacman
        decision = []
        for s in Successor_State:
            x = self.value(s, 0, 0)
            decision.append(x)
        mx = max(decision)
        index = [i for i, j in enumerate(decision) if j == mx]
        index_action = index[0]

        return legalactions[index_action]

    def value(self, state, index, currentdepth):
        # Terminal check
        if (currentdepth == self.depth or state.isWin() or state.isLose()):
            return self.evaluationFunction(state)

        index_increase = index + 1
        # if the max index is reached. Depth is increased by 1 and index is returned to 0 to call pacman
        if index_increase == state.getNumAgents():
            currentdepth += 1
            index_increase = 0
            if currentdepth == self.depth:
                return self.evaluationFunction(state)

        # If index is Pacman calls max function
        if index_increase == 0:
            return self.max_value(state, index_increase, currentdepth)
        # if index is any of the ghost calls min function
        if index_increase > 0:
            return self.min_value(state, index_increase, currentdepth)

    def max_value(self, state, index_increase, depth):
        # Gets the actions
        actions = state.getLegalActions(index_increase)

        max_v = -float("inf")

        successors = [state.generateSuccessor(index_increase, action) for action in actions]

        for states in successors:
            v = self.value(states, index_increase, depth)
            max_v = max(max_v, v)
        return max_v

    def min_value(self, state, index_increase, depth):
        actions = state.getLegalActions(index_increase)

        score = 0
        successors = [state.generateSuccessor(index_increase, action) for action in actions]
        for states in successors:
            v = self.value(states, index_increase, depth)
            score = score + v
        exp_min = float(score/len(successors))
        return exp_min


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
    currentFood = currentGameState.getFood()
    CurrentPos = currentGameState.getPacmanPosition()

    "*** YOUR CODE HERE ***"

    for index in range(len(newScaredTimes)):
        #Checks if the ghost is in scared state ( if = 0 ghost is not scared)
        #Checks the manhatten distance between pacmans position and ghost position)
        #if statement make sure pacman is a distance of 1 from the ghost
        if newScaredTimes[index] == 0 and manhattanDistance(newPos, newGhostPositions[index]) <= 1:
            return currentGameState.getScore() - 15*currentGameState.getNumFood()

        MD = []
        for items in newFood.asList():
            MD.append(manhattanDistance(newPos,items))

        #If the MD list is empty the evaluation funcitons reverts to the original evaluation function
        if not MD:
            return currentGameState.getScore() - 5 * currentGameState.getNumFood()
        else:
         Min_Manhatten = min(MD)
         return currentGameState.getScore() - 5 * currentGameState.getNumFood() - Min_Manhatten

         # print("List of Manhatten distance",MD, "Minimum MD ",Min_Manhatten)

    #The closer Pacman is to food the larger the value.
    # return currentGameState.getScore() - 5*currentGameState.getNumFood() - Min_Manhatten



# Abbreviation
better = betterEvaluationFunction

