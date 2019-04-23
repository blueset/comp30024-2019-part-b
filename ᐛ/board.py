from collections import namedtuple
from typing import Set, Dict, Optional, List, Iterator, Tuple

from .typing import Coordinate, Color, Action

BOARD: List[Coordinate] = \
    [(-3, 0), (-3, 1), (-3, 2), (-3, 3),
     (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3),
     (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3),
     (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
     (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
     (2, -3), (2, -2), (2, -1), (2, 0), (2, 1),
     (3, -3), (3, -2), (3, -1), (3, 0)]
"""All legal hex coordinates as a list."""

BOARD_SET: Set[Coordinate] = set(BOARD)
"""All legal hex coordinates as a set."""

DESTINATIONS: Dict[Color, Set[Coordinate]] = {
    "red": {(3, -3), (3, -2), (3, -1), (3, 0)},
    "green": {(-3, 3), (-2, 3), (-1, 3), (0, 3)},
    "blue": {(-3, 0), (-2, -1), (-1, -2), (0, -3)}
}
"""All destinations of each player"""

DIRECTIONS: Set[Coordinate] = {(0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1)}
"""All possible directions to move"""


class Board:
    """
    Object representing the current and past status of a board.

    This object is immutable.
    """

    ExitedPieces = namedtuple("ExitedPieces", ["red", "green", "blue"])

    def __init__(self,
                 by_player: Optional[Dict[Color, Set[Coordinate]]] = None,
                 by_hex: Optional[Dict[Coordinate, Optional[Color]]] = None,
                 exited_pieces: Optional[ExitedPieces] = None):
        """
        Create a status object.
        :param by_player: Player-pieces mapping
        :param by_hex: Hex-player mapping
        :param exited_pieces: Number of pieces exited for players
            in the order of red, green, then blue.
        """

        # duplicate object
        if by_player is not None and \
                by_hex is not None and \
                exited_pieces is not None:
            self.__by_hex = by_hex
            self.__by_player = by_player
            self.__exited_pieces = exited_pieces
        else:
            # Index by player
            self.__by_player: Dict[Color, Set[Coordinate]] = {
                'red': {(-3, 3), (-3, 2), (-3, 1), (-3, 0)},
                'green': {(0, -3), (1, -3), (2, -3), (3, -3)},
                'blue': {(3, 0), (2, 1), (1, 2), (0, 3)},
            }

            # Index by hex
            self.__by_hex: Dict[Coordinate, Optional[Color]] = \
                {i: None for i in BOARD}
            for c, xs in self.__by_player.items():
                for x in xs:
                    self.__by_hex[x] = c

            self.__exited_pieces = self.ExitedPieces(red=0, green=0, blue=0)

        # Tuple representation of the object for hashing
        # noinspection PyTypeChecker
        self.__tuple = \
            tuple(self.__by_hex[i] for i in BOARD) + self.__exited_pieces

    def get_player(self, coord: Coordinate) -> Optional[Color]:
        """Get the player on a hex (if possible)"""
        return self.__by_hex[coord]

    def get_pieces(self, player: Color) -> Set[Coordinate]:
        """Get the set of location of pieces of a player."""
        return self.__by_player[player]

    def move(self, color: Color, action: Action) -> 'Board':
        """
        Produce a new board when a new action is made
        :param color: Color of the player
        :param action: The action
        """
        by_player = {k: v.copy() for k, v in self.__by_player.items()}
        by_hex = self.__by_hex.copy()

        verb, args = action
        if verb == "EXIT":
            by_player[color].remove(args)
            by_hex[args] = None
            r, g, b = self.__exited_pieces
            if color == "red":
                r += 1
            elif color == "green":
                g += 1
            elif color == "blue":
                b += 1
            return Board(by_player=by_player, by_hex=by_hex,
                         exited_pieces=self.ExitedPieces(r, g, b))
        elif verb in ("JUMP", "MOVE"):
            orig, dest = args
            by_player[color].remove(orig)
            by_player[color].add(dest)
            by_hex[orig] = None
            by_hex[dest] = color
            # Consider conversion of pieces
            if verb == "JUMP":
                mid = ((orig[0] + dest[0]) // 2, (orig[1] + dest[1]) // 2)
                ic = by_hex[mid]
                if ic != color:
                    by_player[ic].remove(mid)
                    by_player[color].add(mid)
                    by_hex[mid] = color
            return Board(by_player=by_player, by_hex=by_hex,
                         exited_pieces=self.__exited_pieces)
        else:
            return self

    def possible_actions(self, player: Color) -> Iterator[Action]:
        """
        Iterate all possible actions of a player on the board,
        one type of action at a time.
        :param player: Color of the player
        """
        pieces = self.__by_player[player]

        has_action = False

        # Check exit actions
        for p in pieces:
            if p in DESTINATIONS[player]:
                has_action = True
                yield ("EXIT", p)

        # Check jump actions
        for p in pieces:
            for d in DIRECTIONS:
                jump = (p[0] + 2 * d[0], p[1] + 2 * d[1])
                if jump in BOARD_SET and self.__by_hex[jump] is None:
                    has_action = True
                    yield ("JUMP", (p, jump))

        # Check move actions
        for p in pieces:
            for d in DIRECTIONS:
                move = (p[0] + d[0], p[1] + d[1])
                if move in BOARD_SET and self.__by_hex[move] is None:
                    has_action = True
                    yield ("MOVE", (p, move))

        # Yield pass if no action is available.
        if not has_action:
            yield ("PASS", None)

    def get_exited_pieces(self, player: Color) -> int:
        """Get the number of pieces exited by the player."""
        return getattr(self.__exited_pieces, player)

    def __eq__(self, other):
        if isinstance(other, Board):
            return self.__tuple == other.__tuple
        return False

    def __hash__(self):
        return hash(self.__tuple)
