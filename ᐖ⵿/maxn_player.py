#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import sys
import os
from math import tanh
from pathlib import Path
from typing import Tuple, Dict, Optional, FrozenSet, List
from collections import Counter

from .typing import Action, Color, Coordinate
from .board import Board, DIRECTIONS, BOARD_DICT, EXIT_PIECES_TO_WIN
from .utilities import exit_distance, cw120, ccw120, dot

DELTA = 1e-6
"""Delta for calculating partial derivative"""

NEXT_PLAYER = {"red": "green", "green": "blue", "blue": "red"}
"""The mapping of consecutive player."""

MAX_STATE_REPETITION = 2
"""Maximum number of repetitions of a board configuration."""

THIS_PATH = Path(__file__).parent
"""Path to the folder containing this file"""

WEIGHTS_PATH = THIS_PATH / "weights.pkl"
"""Path for the persistent trained weight for the evaluation function."""

BEST_DISTANCE: Dict[FrozenSet[Coordinate], int] = pickle.load(
    (THIS_PATH / "min_steps.pkl").open("rb")
)
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

    __version__ = "6"

    TD_LEAF_LAMBDA_TRAIN_MODE = True
    """Flag to toggle TDLeaf(λ) train mode."""

    CUT_OFF_DEPTH = 2
    """Cut of depth for maxⁿ search."""

    class Score:
        """Recording the score on a certain node, with lazy evaluation."""

        _memory: Dict[Board, "MaxⁿPlayer.Score"] = dict()
        """Known board configurations and scores"""

        MAX_PARAMS = [MAX_BEST_DISTANCE, 4, 4, 9, 10, 51, 4, 4 * 9]
        """Maximum value for each feature of a player."""

        WEIGHTS = pickle.load(WEIGHTS_PATH.open("rb"))
        """
        Weights for the evaluation function vector, which is from left to 
        right,
            0. dist
            1. n_pieces_to_exit
            2. n_pieces_missing
            3. conv
            4. jumps
            5. coherence
            6. n_pieces_exited
            7. conv * n_pieces_missing
        """

        MAX_TOTAL_SCORE = 3 * dot(WEIGHTS, MAX_PARAMS)
        """Maximum total evaluation score of 3 players."""

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
            self._red_vector: Optional[List[float]] = None
            self._green: Optional[float] = None
            self._green_vector: Optional[List[float]] = None
            self._blue: Optional[float] = None
            self._blue_vector: Optional[List[float]] = None

        @staticmethod
        def get_best_distance(pieces: FrozenSet, player: Color) -> int:
            """
            Get the minimum number of steps for the set of pieces to
            exit the board ignoring opponents.
            """
            # convert blue and green coordinates to red coordinates
            if player == "blue":
                pieces = frozenset(cw120(i) for i in pieces)
            elif player == "green":
                pieces = frozenset(ccw120(i) for i in pieces)

            return BEST_DISTANCE.get(pieces, MAX_BEST_DISTANCE)

        def mark_as_lose(self, player: Color):
            """
            Mark this score as "Lose" on a certain player.
            """
            setattr(self, f"_{player}", float("-inf"))
            setattr(self, f"_{player}_vector", [])

        def evaluation_function(
            self, player: Color
        ) -> Tuple[float, List[float]]:

            for i in NEXT_PLAYER:
                if self.board.get_exited_pieces(i) >= EXIT_PIECES_TO_WIN:
                    if i == player:
                        # Player wins the game
                        return float("inf"), []
                    else:
                        # Player lost the game
                        return float("-inf"), []

            pieces = self.board.get_pieces(player)
            n_piece_exited = self.board.get_exited_pieces(player)
            n_pieces_to_exit = EXIT_PIECES_TO_WIN - n_piece_exited

            if len(pieces) == 0:
                # No action is possible due to lack of pieces
                # Player loses the game at this stage
                return float("-inf"), []

            # Take the nearest needed pieces to the destination
            if len(pieces) > n_pieces_to_exit:
                np = sorted((exit_distance(p, player), p) for p in pieces)[
                    :n_pieces_to_exit
                ]
                nearest_pieces = frozenset(i[1] for i in np)
            else:
                nearest_pieces = frozenset(pieces)

            # Minimum steps for "nearest-pieces" to exit board
            # assuming no other pieces are on the board.
            dist = MAX_BEST_DISTANCE - self.get_best_distance(
                nearest_pieces, player
            )

            # TODO: must classify jumps as jumping over friend pieces,
            # otherwise, staying in that state may get eaten.
            # QUESTION: what does that mean?

            # Number of jump actions possible
            jumps = 0
            for p in pieces:
                for d in DIRECTIONS:
                    mv = (p[0] + d[0], p[1] + d[1])
                    jmp = (p[0] + d[0] + d[0], p[1] + d[1] + d[1])

                    if (
                        mv in BOARD_DICT
                        and jmp in BOARD_DICT
                        and self.board.get_player(mv) is not None
                        and self.board.get_player(jmp) is None
                    ):
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
                    if (
                        pos in BOARD_DICT
                        and neg in BOARD_DICT
                        and self.board.get_player(pos) != player
                        and self.board.get_player(neg) is None
                    ):
                        conv += 1

            # Number of pieces that we are missing but needed to win the game
            n_pieces_missing = min(n_pieces_to_exit - len(pieces), 0)

            vector = [
                dist,
                n_pieces_to_exit,
                n_pieces_missing,
                conv,
                jumps,
                coherence,
                n_piece_exited,
                conv * n_pieces_missing,
            ]

            return dot(vector, self.WEIGHTS), vector

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
                self._green, self._green_vector = self.evaluation_function(
                    "green"
                )
            return self._green

        @property
        def blue(self) -> float:
            """Evaluation score of player blue on this board"""
            if self._blue is None:
                self._blue, self._blue_vector = self.evaluation_function(
                    "blue"
                )
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
        self.counter: Counter = Counter()
        self.counter[self.board] = 1

        if self.TD_LEAF_LAMBDA_TRAIN_MODE:
            self.steps_in_game = 0
            self.score_history: List["MaxⁿPlayer.Score"] = []

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
        action, score = self.maxⁿ_search(
            self.board, player=self.color, depth=0
        )
        if self.TD_LEAF_LAMBDA_TRAIN_MODE:
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

        if self.TD_LEAF_LAMBDA_TRAIN_MODE:
            self.steps_in_game += 1
            if not self.board.get_pieces(self.color):
                self.td_leaf()
                return
            if self.board.winner:
                self.score_history.append(self.Score(self.board))
                self.td_leaf()
                return
            if self.steps_in_game > 750:
                score = self.Score(self.board)
                score.mark_as_lose()
                self.score_history.append(score)
                self.td_leaf()
                return

    def maxⁿ_search(
        self,
        board: Board,
        player: Color,
        depth: int,
        prev_best: float = float("-inf"),
    ) -> Tuple[Action, "MaxⁿPlayer.Score"]:
        """Search for the best move recursively using maxⁿ algorithm."""
        # Use evaluation function
        best_score = float("-inf")
        best_score_set = self.Score(board)
        best_action = ("PASS", None)

        aym_dancin = 0

        for mv in board.possible_actions(player):
            n_board = board.move(player, mv)
            if self.counter[n_board] + 1 >= MAX_STATE_REPETITION:
                # Aggressively avoid repetitive states
                aym_dancin += 1
                continue
            if depth >= self.CUT_OFF_DEPTH:
                score = self.Score(n_board)
            else:
                next_player = NEXT_PLAYER[player]
                _, score = self.maxⁿ_search(
                    n_board, next_player, depth + 1, best_score
                )
            if score and getattr(score, player) > best_score:
                best_score = getattr(score, player)
                best_score_set = score
                best_action = mv
            if self.Score.MAX_TOTAL_SCORE - getattr(score, player) < prev_best:
                # Shallow pruning: Stop searching when this branch
                # yields a smaller value than what is seen in a
                # previous branch.
                break
            if best_score == float("inf"):
                # Immediate pruning: Stop searching when the
                # maximum possible value is found -- the player
                # wins on this move
                break
        # This may happen due to the aggressive prevention of repeated states.
        # In this case, the game will try to end itself, mark this play as lost
        # And train the weights with TDLeaf.
        if (
            depth == 0
            and best_action[0] == "PASS"
            and best_action != next(board.possible_actions(player))
        ):
            if self.TD_LEAF_LAMBDA_TRAIN_MODE:
                best_score_set.mark_as_lose(self.color)
                self.score_history.append(best_score_set)
                self.td_leaf()
            best_action = next(board.possible_actions(player))
        return best_action, best_score_set

    def td_leaf(self, λ=0.7, η=1.0):
        r"""
        Update weight with TDLeaf(λ) using equations:

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

        d = [r[i + 1] - r[i] for i in range(len(self.score_history) - 1)]

        # Size of d, i.e. N-1
        N_1 = len(d)

        new_weights = []

        for j in range(len(self.Score.WEIGHTS)):
            old_weights = self.Score.WEIGHTS
            Σi = 0
            for i in range(N_1):
                Σm = sum(λ ** (m - i) * d[m] for m in range(N_1))
                features: List[float] = getattr(
                    self.score_history[i], f"_{self.color}_vector"
                )

                if not features:
                    # Skip steps when the player is confirmed to win/lose
                    continue

                old_weights_d = old_weights.copy()
                old_weights_d[j] += DELTA
                δrδwj = (
                    dot(features, old_weights_d) - dot(features, old_weights)
                ) / DELTA
                Σi += δrδwj * Σm
            new_weights.append(old_weights[j] + η * Σi)

        if self.Score.WEIGHTS == new_weights:
            print(
                "TRAIN PLAYER",
                self.color,
                "did not contribute anything. **converged**",
            )
            return

        for i, j in zip(self.Score.WEIGHTS, new_weights):
            if abs(i - j) > 100:
                print("TRAIN PLAYER", self.color, "is rocketing high...")
                return

        print("PLAYER     ", self.color)
        print("OLD_WEIGHTS", self.Score.WEIGHTS)
        print("NEW_WEIGHTS", new_weights)

        # Make weights persistent and rewritable between sessions
        with WEIGHTS_PATH.open("wb") as f:
            pickle.dump(new_weights, f)

        self.TD_LEAF_LAMBDA_TRAIN_MODE = False
