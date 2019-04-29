from typing import Tuple, Dict, Optional
from collections import Counter

from .typing import Action, Color
from .board import Board, DESTINATIONS, DIRECTIONS, BOARD_DICT
from .utilities import exit_distance

CUT_OFF_DEPTH = 3
"""Cut of depth for maxⁿ search."""

EXIT_PIECES_TO_WIN = 4
"""Number of pieces to 'exit' to win a game."""

NEXT_PLAYER = {
    "red": "green",
    "green": "blue",
    "blue": "red"
}
"""The mapping of consecutive player."""

MAX_STATE_REPETITION = 2
"""Maximum number of repetitions of a board configuration."""


class MaxⁿPlayer:
    """
    Implementation of maxⁿ algorithm as a player.
    """

    __version__ = "4"

    class Score:
        """Recording the score on a certain node, with lazy evaluation."""

        _memory: Dict[Board, 'MaxⁿPlayer.Score'] = dict()
        """Known board configurations and scores"""

        MAX_TOTAL_SCORE = float('inf')
        """Maximum total evaluation score of 3 players."""
        # TODO: Fill this with actual value

        @classmethod
        def get(cls, board: Board):
            """
            Get the score of an board.
            This function retrieves score from memory if available.
            """
            if board not in cls._memory:
                cls._memory[board] = cls(board)
            return cls._memory[board]

        def __init__(self, board: Board):
            self.board: Board = board
            self._red: Optional[float] = None
            self._green: Optional[float] = None
            self._blue: Optional[float] = None

        def evaluation_function(self, player: Color) -> float:

            if self.board.get_exited_pieces(player) >= EXIT_PIECES_TO_WIN:
                # Player wins the game
                return float('inf')

            pieces = self.board.get_pieces(player)
            n_pieces_needed = EXIT_PIECES_TO_WIN - self.board.get_exited_pieces(player)

            if len(pieces) == 0:
                # No action is possible due to lack of pieces
                return 0

            # Sum of min (manhattan) distance to destinations
            dist = 14 * EXIT_PIECES_TO_WIN - sum(
                sorted(
                    exit_distance(p, player) for p in pieces
                )[:n_pieces_needed]
            )

            # Number of jump actions possible
            jumps = sum(1
                        for i in self.board.possible_actions(player)
                        if i[0] == "JUMP")

            # Number of actions that other players can take to convert
            # my pieces
            conv = 0
            for p in pieces:
                for d in DIRECTIONS:
                    pos = (p[0] + d[0], p[1] + d[1])
                    neg = (p[0] - d[0], p[1] - d[1])
                    if pos in BOARD_DICT and neg in BOARD_DICT and \
                            self.board.get_player(pos) != player and \
                            self.board.get_player(neg) is None:
                        conv += 1

            # Factor: Favor the case when
            # (number of pieces need to exit to win) is equal to or more than
            # (number of available pieces).
            factor = len(pieces) / n_pieces_needed

            # TODO: Train this weights (currently arbitrary)
            # TODO: Make the attributes zero-sum to apply shallow pruning
            return factor * (4 * dist + jumps - conv)

        @property
        def red(self) -> float:
            """Evaluation score of player red on this board"""
            if self._red is None:
                self._red = self.evaluation_function("red")
            return self._red

        @property
        def green(self) -> float:
            """Evaluation score of player green on this board"""
            if self._green is None:
                self._green = self.evaluation_function("green")
            return self._green

        @property
        def blue(self) -> float:
            """Evaluation score of player blue on this board"""
            if self._blue is None:
                self._blue = self.evaluation_function("blue")
            return self._blue

    def __init__(self, color: Color):
        """
        This method is called once at the beginning of the game to initialise
        your player. You should use this opportunity to set up your own internal
        representation of the game state, and any other information about the
        game state you would like to maintain for the duration of the game.

        The parameter colour will be a string representing the player your
        program will play as (Red, Green or Blue). The value will be one of the
        strings "red", "green", or "blue" correspondingly.
        """
        self.color = color
        self.board = Board()
        self.counter = Counter()
        self.counter[self.board] = 1

    def action(self) -> Action:
        """
        This method is called at the beginning of each of your turns to request
        a choice of action from your program.

        Based on the current state of the game, your player should select and
        return an allowed action to play on this turn. If there are no allowed
        actions, your player must return a pass instead. The action (or pass)
        must be represented based on the above instructions for representing
        actions.
        """
        action, _ = self.maxⁿ_search(self.board, player=self.color, depth=0)
        return action

    def update(self, color: Color, action: Action) -> None:
        """
        This method is called at the end of every turn (including your player’s
        turns) to inform your player about the most recent action. You should
        use this opportunity to maintain your internal representation of the
        game state and any other information about the game you are storing.

        The parameter colour will be a string representing the player whose turn
        it is (Red, Green or Blue). The value will be one of the strings "red",
        "green", or "blue" correspondingly.

        The parameter action is a representation of the most recent action (or
        pass) conforming to the above instructions for representing actions.

        You may assume that action will always correspond to an allowed action
        (or pass) for the player colour (your method does not need to validate
        the action/pass against the game rules).
        """
        self.board = self.board.move(color, action)
        self.counter[self.board] += 1

    def maxⁿ_search(self, board: Board,
                    player: Color,
                    depth: int,
                    prev_best: float = float('-inf')) -> Tuple[Action, Score]:
        """Search for the best move recursively using maxⁿ algorithm."""
        # Use evaluation function
        best_score = float('-inf')
        best_score_set = None
        best_action = None
        for mv in board.possible_actions(player):
            n_board = board.move(player, mv)
            if self.counter[n_board] + 1 >= MAX_STATE_REPETITION:
                # Actively avoid repetitive states
                continue
            if depth >= CUT_OFF_DEPTH:
                score = self.Score.get(n_board)
            else:
                next_player = NEXT_PLAYER[player]
                _, score = self.maxⁿ_search(n_board, next_player, depth + 1, best_score)
            if getattr(score, player) > best_score:
                best_score = getattr(score, player)
                best_score_set = score
                best_action = mv
            # TODO: Uncomment this when MAX_TOTAL_SCORE is decided
            # if self.Score.MAX_TOTAL_SCORE - getattr(score, player) < prev_best:
            #     # Shallow pruning: Stop searching when this branch
            #     # yields a smaller value than what is seen in a
            #     # previous branch.
            #     break
            if best_score == float('inf'):
                # Immediate pruning: Stop searching when the
                # maximum possible value is found -- the player
                # wins on this move
                break
        return best_action, best_score_set
