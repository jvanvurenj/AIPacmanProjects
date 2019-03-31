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

        "*** YOUR CODE HERE ***"

        #note: Michael Hagel and i talked about a general way to do this. Didn't get into coding specifics, it basically just ammounted to: go for food, but if youre gonna hit a ghost, give it an infinitely negative value
        x = float("inf")
        for i in newGhostStates:
            if i.getPosition() == newPos:
                toret = float("-inf")
                return toret #-inf isn't working. fix it later
        for i in currentGameState.getFood().asList():
            x = min(x, manhattanDistance(i, newPos))
            if Directions.STOP in action:
                toret= float("-inf")
                return toret   #same problem as above
        return 1.0/(1.0+x)





        return successorGameState.getScore()

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
        return self.myminimax(gameState, 0, 0)
        

    def myminimax(self, state, index, nodex):
        if index>=state.getNumAgents():
            index = 0
            nodex = nodex+1
        if nodex == self.depth:
            return self.evaluationFunction(state)
        if index == self.index:
            tag = 1
        else:
            tag = 0
        if tag == 1:
            toret = float("-inf")
        else:
            toret = float("inf")
        toAct = "None"
        if state.isWin():
            return self.evaluationFunction(state)
        if state.isLose():
            return self.evaluationFunction(state)
        for i in state.getLegalActions(index):
            if i == Directions.STOP:
                continue
            mynext = state.generateSuccessor(index, i)
            toCheck = self.myminimax(mynext, index+1, nodex)
            if tag == 1:
                if toCheck > toret:
                    toret = toCheck
                    toAct = i
            else:
                if toCheck < toret:
                    toret = toCheck
                    toAct = i
        if tag == 1:            
            if nodex:
                return toret
            return toAct

        else:
            return toret
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.myalphabeta(gameState, 0, 0, float("-inf"), float("inf"))
        
    def myalphabeta(self, state, index, nodex, a, b):
        if index>=state.getNumAgents():
            index = 0
            nodex = nodex+1
        if nodex == self.depth:
            return self.evaluationFunction(state)
        if index == self.index:
            tag = 1
        else:
            tag = 0
        if tag == 1:
            toret = float("-inf")
        else:
            toret = float("inf")
        toAct = "None"
        if state.isWin():
            return self.evaluationFunction(state)
        if state.isLose():
            return self.evaluationFunction(state)
        for i in state.getLegalActions(index):
            if i == Directions.STOP:
                continue
            mynext = state.generateSuccessor(index, i)
            toCheck = self.myalphabeta(mynext, index+1, nodex, a, b)
            if tag == 1:
                if toCheck > toret:
                    toret = toCheck
                    toAct = i
                if toret > b:
                    return toret
                if toret > a:
                    a = toret
            else:
                if toCheck < toret:
                    toret = toCheck
                    toAct = i
                if toret < a:
                    return toret
                if toret < b:
                    b = toret
        if tag == 1:            
            if nodex:
                return toret
            return toAct

        else:
            return toret




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.myexpectimax(gameState, 0, 0)


    def myexpectimax(self, state, index, nodex):
        if index>=state.getNumAgents():
            index = 0
            nodex = nodex+1
        if nodex == self.depth:
            return self.evaluationFunction(state)
        if index == self.index:
            tag = 1
        else:
            tag = 0
        if tag == 1:
            toret = float("-inf")
            toAct = "Stop"
        else:
            toret = 0
            if len(state.getLegalActions(index)) == 0:
                probability = 0
            else: 
                probability = 1.0/len(state.getLegalActions(index))
        
        if state.isWin():
            return self.evaluationFunction(state)
        if state.isLose():
            return self.evaluationFunction(state)
        for i in state.getLegalActions(index):
            if i == Directions.STOP:
                continue
            mynext = state.generateSuccessor(index, i)
            toCheck = self.myexpectimax(mynext, index+1, nodex)
            if tag == 1:
                if toCheck > toret:
                    toret = toCheck
                    toAct = i
            else:
                toret = toret + (probability * toCheck)
                toAct = i
        if tag == 1:            
            if nodex:
                return toret
            return toAct

        else:
            return toret

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Added points for eating ghosts when they are scared. If the scaredtime is more than 1, the score goes up depending on the manhattan distance to the ghost, so it determines whether or not its worth going for it point wise.
    Would like to add ccapsules at some point, but not quite sure how to evaluate it and i ended up not needing it here to pass the autograder. 
    """

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    "*** YOUR CODE HERE ***"
    
    score = currentGameState.getScore()
    x = float("inf")
    for i in range(len(newGhostStates)):
        ghost = newGhostStates[i].getPosition()
        if newScaredTimes[i] < 1:
            if newPos == ghost:
                return float("-inf")
            else:
                x = min(x, manhattanDistance(currentGameState.getPacmanPosition(), ghost))
        else:
            score+=100*( 1/min(x, manhattanDistance(currentGameState.getPacmanPosition(), ghost)))

    for i in currentGameState.getFood().asList():
        x = min(x, manhattanDistance(i, newPos))
    score-=x
    return score
# Abbreviation
better = betterEvaluationFunction
