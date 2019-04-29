from .typing import Coordinate, Color


def axial_distance(a: Coordinate, b: Coordinate) -> int:
    """
    Calculate simple manhattan distance between 2 points
    
    Code adapted from Axial Coordinates by Red Blob Games
    <https://www.redblobgames.com/grids/hexagons/>.
    """

    return (abs(aq - bq)
            + abs(aq + ar - bq - br)
            + abs(ar - br)) // 2

def exit_distance(coord: Coordinate, color: Color) -> int:
    """
    Calculate the distance of a coordinate to the exit hexes of a color.

    Code adapted from Sample Solution to Project A, 
    by Matt Farrugia (matt.farrugia@unimelb.edu.au)
    """
    q, r = coord
    if color == 'red':
        return 3 - q
    elif color == 'green':
        return 3 - r
    elif color == 'blue':
        return 3 + q + r
    return -1


def cw120(p: Coordinate) -> Coordinate:
    """Rotate the piece 120° clockwise"""
    return - p[0] - p[1], p[0]


def ccw120(p: Coordinate) -> Coordinate:
    """Rotate the piece 120° counter-clockwise"""
    return p[1], - p[0] - p[1]
