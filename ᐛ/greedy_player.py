from scipy.spatial.distance import cityblock

from .board import Board, DESTINATIONS


class Player:
    def __init__(self, color):
        self.color = color
        self.board = Board()

    def action(self):
        dests = DESTINATIONS[self.color]

        best_dist = float('inf')
        best_action = None

        for verb, args in self.board.possible_actions(self.color):
            if verb == "EXIT":
                return verb, args
            elif verb == "PASS":
                return verb, args
            else:
                # MOVE or JUMP
                target = args[1]
                dist = min(cityblock(target, d) for d in dests)
                if dist < best_dist:
                    best_dist = dist
                    best_action = (verb, args)

        return best_action

    def update(self, colour, action):
        self.board = self.board.move(colour, action)
