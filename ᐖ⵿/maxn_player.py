import pickle
import atexit
from math import tanh
from pathlib import Path
from typing import Tuple, Dict, Optional, FrozenSet, Set
from collections import Counter

from .typing import Action, Color, Coordinate
from .board import Board, DESTINATIONS, DIRECTIONS, BOARD_DICT
from .utilities import exit_distance, cw120, ccw120

TD_LEAF_LAMBDA_TRAIN_MODE = False
"""Flag to toggle TDLeaf(λ) train mode."""

DELTA = 1e-4
"""Delta for calculating partial derivative"""

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

BEST_DISTANCE: Dict[FrozenSet[Coordinate], int] = pickle.load(
    open(Path(__file__).parent / "min_steps.pkl", "rb"))
"""
Precomputed best number of steps of 0-4 pieces to exit the board
assuming no enemy presents.
"""

MAX_BEST_DISTANCE = 19
"""Maximum value of best distance."""

class MaxⁿPlayer:
    """
    Implementation of maxⁿ algorithm as a player.
    """

    __version__ = "5"

    class Score:
        """Recording the score on a certain node, with lazy evaluation."""

        _memory: Dict[Board, 'MaxⁿPlayer.Score'] = dict()
        """Known board configurations and scores"""

        MAX_TOTAL_SCORE = float('inf')
        """Maximum total evaluation score of 3 players."""
        # TODO: Fill this with actual value

        WEIGHTS = [
            10,  # dist
            0,   # n_pieces_to_exit
            -10 * MAX_BEST_DISTANCE,  # n_pieces_missing
            0,   # conv
            30,  # jumps
            0.5,  # coherence
            -1,  # conv * n_pieces_missing
        ]

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
            self._red_vector: Optional[List[Float]] = None
            self._green: Optional[float] = None
            self._green_vector: Optional[List[Float]] = None
            self._blue: Optional[float] = None
            self._blue_vector: Optional[List[Float]] = None

        @staticmethod
        def get_best_distance(pieces: FrozenSet, player: Color) -> int:
            """
            Get the minimum number of steps for the set of pieces to
            exit the board ignoring comtestants.
            """
            # convert blue and green coordinates to red coordinates
            if player == "blue":
                pieces = frozenset(cw120(i) for i in pieces)
            elif player == "green":
                pieces = frozenset(ccw120(i) for i in pieces)
    
            return BEST_DISTANCE.get(pieces, MAX_BEST_DISTANCE)

        def evaluation_function(self, player: Color) -> Tuple[float, List[float]]:

            if self.board.get_exited_pieces(player) >= EXIT_PIECES_TO_WIN:
                # Player wins the game
                return float('inf')

            pieces = self.board.get_pieces(player)
            n_pieces_to_exit = EXIT_PIECES_TO_WIN - self.board.get_exited_pieces(player)

            if len(pieces) == 0:
                # No action is possible due to lack of pieces
                return 0

            # Take the nearest needed pieces to the destination
            if len(pieces) > n_pieces_to_exit:
                nearest_pieces = sorted(
                    (exit_distance(p, player), p) for p in pieces
                )[:n_pieces_to_exit]
                nearest_pieces = frozenset(i[1] for i in nearest_pieces)
            else:
                nearest_pieces = frozenset(pieces)

            # Minimum steps for "nearest-pieces" to exit board
            # assuming no other pieces are on the board.
            dist = MAX_BEST_DISTANCE - \
                self.get_best_distance(nearest_pieces, player)

            # TODO: must classify jumps as jumping over friend pieces, 
            # otherwise, staying in that state may get eaten.
            # QUESTION: what does that mean?

            # Number of jump actions possible
            jumps = 0
            for p in pieces:
                for d in DIRECTIONS:
                    mv = (p[0] + d[0], p[1] + d[1])
                    jmp = (p[0] + d[0] + d[0], p[1] + d[1] + d[1])

                    if mv in BOARD_DICT and jmp in BOARD_DICT and \
                            self.board.get_player(mv) is not None and \
                            self.board.get_player(jmp) is None:
                        jumps += 1

            # Number of pieces of other players can can convert
            # our "nearest-pieces"
            conv = 0

            # 2 x Number of pieces that are next to each other
            coherence = 0

            for p in nearest_pieces:
                for d in DIRECTIONS:
                    pos = (p[0] + d[0], p[1] + d[1])
                    neg = (p[0] - d[0], p[1] - d[1])
                    if pos in nearest_pieces:
                        coherence += 1
                    if pos in BOARD_DICT and neg in BOARD_DICT and \
                            self.board.get_player(pos) != player and \
                            self.board.get_player(neg) is None:
                        conv += 1

            # Number of pieces that we are missing but needed to win the game
            n_pieces_missing = min(n_pieces_to_exit - len(pieces), 0)

            # TODO: Train this weights (currently arbitrary)
            # TODO: Make the attributes zero-sum to apply shallow pruning
            vector = [dist, n_pieces_to_exit, n_pieces_missing, conv,
                      jumps, coherence, conv * n_pieces_missing]
            
            return sum(i * j for i, j in zip(vector, self.WEIGHTS)), vector

        @property
        def red(self) -> float:
            """Evaluation score of player red on this board"""
            if self._red is None:
                self._red, self._red_vector = self.evaluation_function("red")
            return self._red

        @property
        def green(self) -> float:
            """Evaluation score of player green on this board"""
            if self._green is None:
                self._green, self._green_vector = self.evaluation_function("green")
            return self._green

        @property
        def blue(self) -> float:
            """Evaluation score of player blue on this board"""
            if self._blue is None:
                self._blue, self._blue_vector = self.evaluation_function("blue")
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
        self.color: Color = color
        self.board = Board()
        self.counter = Counter()
        self.counter[self.board] = 1

        if TD_LEAF_LAMBDA_TRAIN_MODE:
            self.score_history: List['MaxⁿPlayer.Score'] = []
            atexit.register(self.td_leaf)

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
        action, score = self.maxⁿ_search(self.board, player=self.color, depth=0)
        if TD_LEAF_LAMBDA_TRAIN_MODE:
            self.score_history.append(score)
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
                    prev_best: float = float('-inf')) -> Tuple[Action, 'MaxⁿPlayer.Score']:
        """Search for the best move recursively using maxⁿ algorithm."""
        # Use evaluation function
        best_score = float('-inf')
        best_score_set = None
        best_action = ("PASS", None)
        for mv in board.possible_actions(player):
            n_board = board.move(player, mv)
            if self.counter[n_board] + 1 >= MAX_STATE_REPETITION:
                # Actively avoid repetitive states
                continue
            if depth >= CUT_OFF_DEPTH:
                score = self.Score(n_board)
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

    def td_leaf(self, λ=0.8, η=1.0):
        """
        Update weight with TDLeaf(λ) using equation:

          / l  \         /     / l  \ \ 
        r| s , w| = tanh| eval| s , w| |
          \ i  /         \     \ i  / /

               / l    \      / l  \ 
        d  = r| s   , w| - r| s , w|
         i     \ i+1  /      \ i  /

                          /     l      _                _ \ 
                          | ∂r(s , w) |                  | |
                      N-1 |     i     |  N-1    m-i      | |
        w  <- w  + η   ∑  | --------- |   ∑    λ     d   | |
         j     j      i=1 |    ∂w     |  m=1          m  | |
                          |      j    |_                _| |
                           \                              /
        """
        # eval(s_i^l, w)
        eval_s_w = [getattr(i, self.color) for i in self.score_history]

        r = [tanh(i) for i in eval_s_w]

        d = [r[i + 1] - r[i]
                for i in range(len(self.score_history) - 1)]

        # Size of d, i.e. N-1
        N_1 = len(d)

        new_weights = []
        
        for j in range(len(self.Score.WEIGHTS)):
            old_weight = self.Score.WEIGHTS[j]
            ∑i = 0
            for i in range(len(differences)):
                ∑m = sum(λ ** (m-i) * d[m] for m in range(N_1))
                features: List[float] = \
                    getattr(self.score_history[i], f"_{self.color}_vector")

                # TODO: confirm if ∂r/∂w_j is really this.
                ∂r∂wj = (
                    (features[j] * (old_weight + DELTA)) -
                    (featuers[j] * old_weight)
                ) / DELTA
                ∑i += ∂r∂wj * ∑m
            new_weights.append(old_weight + η * ∑i)
        
        print("NEW_WEIGHTS")
        print(new_weights)

        # TODO: make weights persistent and rewritable between sessions


