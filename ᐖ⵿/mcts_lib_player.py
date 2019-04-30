from random import choice
from mcts import mcts

from .board import Board

NEXT_PLAYER = {
    "red": "green",
    "green": "blue",
    "blue": "red"
}


class MCTSState(Board):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = "red"


    def getPossibleActions(self):
        return [(i, self.color) for i in self.possible_actions(self.color)]
    
    def takeAction(self, action):
        ac, co = action
        b = self.move(co, ac)
        b.color = NEXT_PLAYER[co]
        return b

    def isTerminal(self):
        return self.get_exited_pieces("red") >= 4 or \
                self.get_exited_pieces("green") >= 4 or \
                self.get_exited_pieces("blue") >= 4
    
    def getReward(self):
        return 1 if self.get_exited_pieces(self.color) >= 4 else -1


class Player:
    def __init__(self, color):
        self.color = color
        self.board = MCTSState()
        self.mcts = mcts(timeLimit=10000)
        # self.mcts = mcts(iterationLimit=4)

    def action(self):
        return self.mcts.search(initialState=self.board)[0]

    def update(self, colour, action):
        self.board = self.board.takeAction((action, colour))
