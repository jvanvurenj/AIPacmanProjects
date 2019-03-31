# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
          counter = util.Counter()
          for state in self.mdp.getStates():
            ceiling = float("-inf")
            temp = 0
            if self.mdp.isTerminal(state):
              counter[state] = 0
              continue
            else:
              if not self.mdp.getPossibleActions(state):
                counter[state] = 0
              for x in self.mdp.getPossibleActions(state):
                temp = self.getQValue(state,x)
                if temp > ceiling:
                  counter[state] = temp
                  ceiling = temp
          self.values = counter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        toret = 0
        ##wasnt working with i and then using i[0], i[1]. try to figure out why later
        for i0, i1 in self.mdp.getTransitionStatesAndProbs(state,action):
          toret += i1 * (self.mdp.getReward(state, action, i0) + self.discount*self.getValue(i0))
        return toret

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        toAct = None
        ceiling = float("-inf")
        if not self.mdp.getPossibleActions(state):
          return None
        else:
          for x in self.mdp.getPossibleActions(state):
            temp = self.getQValue(state,x)
            if temp > ceiling:
              toAct = x
              ceiling = temp 
        return toAct



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        ##copying in the first one didnt work out in this one. still unsure why. look into it.
        counter = util.Counter()
        states = self.mdp.getStates()
        for i in range(self.iterations):
          now = states[i%len(states)]
          counter = self.values.copy() #have to copy over, recalling util.Counter causes problems
          ceiling = float("-inf")
          if self.mdp.isTerminal(now):
            counter[now] = 0
            continue
          else:
            for x in self.mdp.getPossibleActions(now):
              temp = 0
              for i0, i1 in self.mdp.getTransitionStatesAndProbs(now, x):
                temp += i1 * (self.mdp.getReward(now, x, i0) + self.discount*self.getValue(i0))
              if temp > ceiling:
                counter[now] = temp
                ceiling = temp
          self.values = counter




class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        ##copying also didnt work the same here, but doing it the same way as the 2nd did..
        pq = util.PriorityQueue()
        myDict = {}
        ##array is causing complications. maybe use a map? forget what python's version of it is - look it up later

        #first need to get priorityqueue populated with acceptable
        for i in self.mdp.getStates():
          myDict[i] = []
        for state in self.mdp.getStates():
          if not self.mdp.isTerminal(state):
            ceiling = float("-inf")
            temp = 0
            for x in self.mdp.getPossibleActions(state):
              for i0, i1 in self.mdp.getTransitionStatesAndProbs(state, x):
                if i1 > 0:
                  myDict[i0].append(state)
              temp = self.getQValue(state, x)
              if temp > ceiling:
                ceiling = temp
            dif = abs(self.values[state] - ceiling)
            pq.push(state, -dif)

        #now we iterate through pq unless pq is empty
        for i in range(self.iterations):
          if pq.isEmpty():
            continue
          now = pq.pop()
          ceiling = float("-inf")
          temp = 0
          for x in self.mdp.getPossibleActions(now):
            temp = self.getQValue(now, x)
            if temp > ceiling:
              ceiling = temp
          self.values[now] = ceiling

          for z in myDict[now]:
            ceiling = float("-inf")
            for x in self.mdp.getPossibleActions(z):
              temp = self.getQValue(z, x)
              if temp>ceiling:
                ceiling = temp
            dif = abs(self.values[z] - ceiling)
            if dif > self.theta:
              pq.update(z, -dif)
