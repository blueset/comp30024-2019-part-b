from random import choice

from .board import Board


class Player:
    def __init__(self, color):
        self.color = color
        self.board = Board()

    def action(self):
        return choice(list(self.board.possible_actions(self.color)))

    def update(self, colour, action):
        self.board = self.board.move(colour, action)
