from collections import namedtuple
from typing import Set, Dict, Optional, List, Iterator, Tuple

from .typing import Coordinate, Color, Action

# fmt: off
BOARD = [(-3, 0), (-3, 1), (-3, 2), (-3, 3),
         (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3),
         (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3),
         (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
         (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
         (2, -3), (2, -2), (2, -1), (2, 0), (2, 1),
         (3, -3), (3, -2), (3, -1), (3, 0)]
"""All legal hex coordinates as a list."""
# fmt: on

BOARD_DICT: Dict[Coordinate, int] = {i: idx for idx, i in enumerate(BOARD)}
"""All legal hex coordinates as a coordinate-index mapping."""

DESTINATIONS: Dict[Color, Set[Coordinate]] = {
    "red": {(3, -3), (3, -2), (3, -1), (3, 0)},
    "green": {(-3, 3), (-2, 3), (-1, 3), (0, 3)},
    "blue": {(-3, 0), (-2, -1), (-1, -2), (0, -3)},
}
"""All destinations of each player."""

DIRECTIONS: Set[Coordinate] = {
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
}
"""All possible directions to move."""

EXIT_PIECES_TO_WIN = 4
"""Number of pieces to 'exit' to win a game."""


class Board:
    """
    Object representing the current and past status of a board.

    This object is immutable.
    """

    ExitedPieces = namedtuple("ExitedPieces", ["red", "green", "blue"])

    def __init__(
        self,
        by_player: Optional[Dict[Color, Set[Coordinate]]] = None,
        by_hex: Optional[Tuple[Optional[Color]]] = None,
        exited_pieces: Optional[ExitedPieces] = None,
    ):
        """
        Create a status object.
        :param by_player: Player-pieces mapping
        :param by_hex: Hex-player mapping
        :param exited_pieces: Number of pieces exited for players
            in the order of red, green, then blue.
        """

        # duplicate object
        if (
            by_player is not None
            and by_hex is not None
            and exited_pieces is not None
        ):
            self.__by_hex = by_hex
            self.__by_player = by_player
            self.__exited_pieces = exited_pieces
        else:
            # Index by player
            self.__by_player: Dict[Color, Set[Coordinate]] = {
                "red": {(-3, 3), (-3, 2), (-3, 1), (-3, 0)},
                "green": {(0, -3), (1, -3), (2, -3), (3, -3)},
                "blue": {(3, 0), (2, 1), (1, 2), (0, 3)},
            }

            # Index by hex
            by_hex: List[Optional[Color]] = [None] * len(BOARD)
            for c, xs in self.__by_player.items():
                for x in xs:
                    by_hex[BOARD_DICT[x]] = c
            self.__by_hex = tuple(by_hex)

            self.__exited_pieces = self.ExitedPieces(red=0, green=0, blue=0)

        # Tuple representation of the object for hashing
        # noinspection PyTypeChecker
        self.__tuple = self.__by_hex + self.__exited_pieces

    def get_player(self, coord: Coordinate) -> Optional[Color]:
        """Get the player on a hex (if possible)"""
        return self.__by_hex[BOARD_DICT[coord]]

    def get_pieces(self, player: Color) -> Set[Coordinate]:
        """Get the set of location of pieces of a player."""
        return self.__by_player[player]

    def move(self, color: Color, action: Action) -> "Board":
        """
        Produce a new board when a new action is made
        :param color: Color of the player
        :param action: The action
        """
        by_player = {k: v.copy() for k, v in self.__by_player.items()}
        by_hex = list(self.__by_hex)

        verb, args = action
        if verb == "EXIT":
            by_player[color].remove(args)
            by_hex[BOARD_DICT[args]] = None
            r, g, b = self.__exited_pieces
            if color == "red":
                r += 1
            elif color == "green":
                g += 1
            elif color == "blue":
                b += 1
            return self.__class__(
                by_player=by_player,
                by_hex=tuple(by_hex),
                exited_pieces=self.ExitedPieces(r, g, b),
            )
        elif verb in ("JUMP", "MOVE"):
            orig, dest = args
            by_player[color].remove(orig)
            by_player[color].add(dest)
            by_hex[BOARD_DICT[orig]] = None
            by_hex[BOARD_DICT[dest]] = color
            # Consider conversion of pieces
            if verb == "JUMP":
                mid = ((orig[0] + dest[0]) // 2, (orig[1] + dest[1]) // 2)
                ic = by_hex[BOARD_DICT[mid]]
                if ic is not None and ic != color:
                    by_player[ic].remove(mid)
                    by_player[color].add(mid)
                    by_hex[BOARD_DICT[mid]] = color
            return self.__class__(
                by_player=by_player,
                by_hex=tuple(by_hex),
                exited_pieces=self.__exited_pieces,
            )
        else:
            return self

    def possible_actions(self, player: Color) -> Iterator[Action]:
        """
        Iterate all possible actions of a player on the board,
        one type of action at a time.
        :param player: Color of the player
        """
        pieces = self.__by_player[player]

        exits = []

        # Check exit actions
        for p in pieces:
            if p in DESTINATIONS[player]:
                exits.append(("EXIT", p))

        # Check move/jump actions
        moves = []
        jumps = []

        for p in pieces:
            for d in DIRECTIONS:
                move = (p[0] + d[0], p[1] + d[1])
                jump = (p[0] + 2 * d[0], p[1] + 2 * d[1])
                if move in BOARD_DICT:
                    if self.__by_hex[BOARD_DICT[move]] is None:
                        moves.append(("MOVE", (p, move)))
                    elif (
                        jump in BOARD_DICT
                        and self.__by_hex[BOARD_DICT[jump]] is None
                    ):
                        jumps.append(("JUMP", (p, jump)))

        if not exits and not moves and not jumps:
            yield ("PASS", None)
        else:
            for i in exits:
                yield i
            for i in moves:
                yield i
            for i in jumps:
                yield i

    def get_exited_pieces(self, player: Color) -> int:
        """Get the number of pieces exited by the player."""
        return getattr(self.__exited_pieces, player)

    @property
    def winner(self) -> Optional[Color]:
        """Winner of the current board, if available"""
        if self.__exited_pieces.red >= EXIT_PIECES_TO_WIN:
            return "red"
        if self.__exited_pieces.green >= EXIT_PIECES_TO_WIN:
            return "green"
        if self.__exited_pieces.blue >= EXIT_PIECES_TO_WIN:
            return "blue"

        return None

    def __eq__(self, other):
        if isinstance(other, Board):
            return self.__tuple == other.__tuple
        return False

    def __hash__(self):
        return hash(self.__tuple)

    def __str__(self):
        return f"Board({self.__by_player}, {self.__by_hex}, Board.{self.__exited_pieces})"

