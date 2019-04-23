"""Definitions of types used in the project"""

from typing import Tuple, Union


Coordinate = Tuple[int, int]
"""Type definition of a coordinate on the board"""

Move = Tuple[Coordinate, Coordinate]
"""Type definition of a move from one hex to another"""

Action = Tuple[str, Union[None, Coordinate, Move]]
"""
Type definition of action
Tuple of
    str, -- Action type
    Union of
        None       -- When action is "PASS"
        Coordinate -- When action is "EXIT"
        Move       -- When action is "MOVE" or "JUMP"
"""

Color = str
"""Type definition of a player's color"""
