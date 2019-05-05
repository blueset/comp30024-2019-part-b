import json
import tkinter
import threading
import atexit
import time
import sys
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from xmlrpc.client import ServerProxy
from typing import Tuple, Dict

"""
GUI player for COMP30024 AI 2019 Sem 1 Part B.

Adapted from `texgen.py` by Matt Farrugia <matt.farrugia@unimelb.edu.au>.
Modified by Jin Han <jinh4@student.unimelb.edu.au>.
"""

# window size (fixed, but you can adjust here)
H, W = 360, 640

SERVER_LISTEN = "0.0.0.0"
CLIENT_ACCESS = "127.0.0.1"
PORTS = {
    "red": 8901,
    "green": 8902,
    "blue": 8903
}


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


def rpc_server():
    color = sys.argv[1]
    root = tkinter.Tk()
    root.title('Tkinter Player: ' + color)
    root.wm_resizable(0, 0)
    canvas = BoardViewCanvas(root, color)

    root.after(500, root.focus_force())

    port = PORTS.get(color, 8900)
    svr = SimpleXMLRPCServer((SERVER_LISTEN, port), requestHandler=RequestHandler, allow_none=True)
    svr.register_introspection_functions()

    svr.register_function(canvas.activate, 'activate')
    svr.register_function(canvas.update_status, 'update_status')

    @svr.register_function
    def get_last_move():
        return json.dumps(canvas.last_move)

    @svr.register_function
    def render_board(board):
        data = json.loads(board)
        root.focus_force()
        canvas.render_board(data)
        canvas.update()
        return 0

    @svr.register_function
    def destroy():
        print("Killing the server as the game is ended.")
        root.destroy()
        svr.server_close()
        svr.shutdown()
        exit(1)
        return 0

    print(f"Running RPC server for player {color} on port {port}")
    svr_t = threading.Thread(target=svr.serve_forever)
    svr_t.start()

    root.mainloop()


class Player:
    def __init__(self, colour):
        """
        This method is called once at the beginning of the game to initialise
        your player. You should use this opportunity to set up your own internal
        representation of the game state, and any other information about the 
        game state you would like to maintain for the duration of the game.

        The parameter colour will be a string representing the player your 
        program will play as (Red, Green or Blue). The value will be one of the 
        strings "red", "green", or "blue" correspondingly.
        """

        self.status = {
            'red': {(-3, 3), (-3, 2), (-3, 1), (-3, 0)},
            'green': {(0, -3), (1, -3), (2, -3), (3, -3)},
            'blue': {(3, 0), (2, 1), (1, 2), (0, 3)},
        }

        self.rpc = ServerProxy(f'http://{CLIENT_ACCESS}:{PORTS.get(colour, 8900)}', allow_none=True)

        # begin!
        atexit.register(self.rpc.destroy)
        self.render_board(self.status)

    def tup(self, val):
        if isinstance(val, list):
            return tuple(map(self.tup, val))
        return val

    def action(self):
        """
        This method is called at the beginning of each of your turns to request 
        a choice of action from your program.

        Based on the current state of the game, your player should select and 
        return an allowed action to play on this turn. If there are no allowed 
        actions, your player must return a pass instead. The action (or pass) 
        must be represented based on the above instructions for representing 
        actions.
        """
        self.rpc.activate()
        move = json.loads(self.rpc.get_last_move())
        while not move:
            time.sleep(0.1)
            move = json.loads(self.rpc.get_last_move())

        return self.tup(move)

    def update(self, color, action):
        """
        This method is called at the end of every turn (including your player’s 
        turns) to inform your player about the most recent action. You should 
        use this opportunity to maintain your internal representation of the 
        game state and any other information about the game you are storing.

        The parameter color will be a string representing the player whose turn
        it is (Red, Green or Blue). The value will be one of the strings "red", 
        "green", or "blue" correspondingly.

        The parameter action is a representation of the most recent action (or 
        pass) conforming to the above instructions for representing actions.

        You may assume that action will always correspond to an allowed action 
        (or pass) for the player color (your method does not need to validate 
        the action/pass against the game rules).
        """
        verb, args = action
        if verb == "PASS":
            desc = f"{color}: PASS."
        elif verb == "EXIT":
            self.status[color].remove(args)
            desc = f"{color}: EXIT from {args}."
            self.render_board(self.status)
        elif verb in ("JUMP", "MOVE"):
            orig, dest = args
            self.status[color].remove(orig)
            self.status[color].add(dest)
            if verb == "JUMP":
                mid = ((orig[0] + dest[0]) // 2, (orig[1]+dest[1]) // 2)
                for ic in ("red", "green", "blue"):
                    if mid in self.status[ic]:
                        self.status[ic].remove(mid)
                        self.status[color].add(mid)
                        break
            desc = f"{color}: {verb} from {orig} to {dest}."
            self.render_board(self.status)
        else:
            desc = f"{color}: Unknown move: {action}."
        self.rpc.update_status(desc)

    def render_board(self, board):
        json_str = json.dumps({i: list(board[i]) for i in board})
        self.rpc.render_board(json_str)


class Color:
    BG = "#663300"
    FG = "#ffd9b3"
    HL = "#ffe6cc"
    LINES = "#663300"
    STONE = "#fff2e6"
    RED = "#cc3300"
    RED_S = "#E57373"
    GREEN = "#339966"
    GREEN_S = "#9CCC65"
    BLUE = "#003399"
    BLUE_S = "#64B5F6"
    BLACK = "#000000"
    WHITE = "#ffffff"
    NONE = ""


# hex shape flat
FH, FW = 0.866, 1
HEX_F = [(-FW / 2, 0), (-FW / 4, -FH / 2), (FW / 4, -FH / 2),
         (FW / 2, 0), (FW / 4, FH / 2), (-FW / 4, FH / 2)]
# hex shape pointy
PH, PW = 1, 0.866
HEX_P = [(-PW / 2, -PH / 4), (0, -PH / 2), (PW / 2, -PH / 4),
         (PW / 2, PH / 4), (0, PH / 2), (-PW / 2, PH / 4)]


class BoardViewCanvas(tkinter.Canvas):

    def __init__(self, root, color):
        bg = Color.BG
        self.dests = {}

        if color == "red":
            self.dests = {(3, -3), (3, -2), (3, -1), (3, 0)}
            bg = Color.RED
        elif color == "green":
            self.dests = {(-3, 3), (-2, 3), (-1, 3), (0, 3)}
            bg = Color.GREEN
        elif color == "blue":
            self.dests = {(-3, 0), (-2, -1), (-1, -2), (0, -3)}
            bg = Color.BLUE

        super().__init__(root, width=W, height=H, background=bg,
                         highlightthickness=0)
        self.root = root
        self.color = color
        self.status = ""


        self.in_turn = False
        self.last_move = None
        self.evt = threading.Event()

        self.active_piece = None
        # set up state
        self.initialise()
        # # bind keypresses
        self.bind("<Button-1>", self.clicked)

        # pack into interface
        self.pack()

    # called in the beginning
    def initialise(self):
        # build game board
        self.board = Board(self)
        self.board.set_dests(self.color, self.dests)
        self.button = PassButton(self, self.board, self.root)

        self.label0 = self.create_text(
            (5, 5), anchor="nw",
            fill=Color.WHITE,
            text=f"Starting game as player {self.color}")
        self.label1 = self.create_text(
            (W - 5, H - 5), anchor="se",
            fill=Color.WHITE,
            text="Stand-by."
        )

    def render_board(self, status):
        """Re-render the board"""
        self.board.update_board(status)

    def update_status(self, status):
        self.status = (self.status + " " + status)[-100:]
        self.itemconfig(self.label1, text=self.status)

    def update_message(self, message):
        self.itemconfig(self.label0, text=message)
        return 0

    def activate(self):
        self.in_turn = True
        self.last_move = None
        self.active_piece = None
        self.evt.clear()
        self.update_message(f"{self.color}: Your turn now.")
        return 0

    def deactivate(self, move):
        self.in_turn = False
        if self.active_piece:
            self.board.pieces[self.active_piece].deactivate()
            self.active_piece = None
        self.last_move = move
        self.update_message(f"{self.color}: Stand-by.")
        self.evt.set()

    def clicked(self, event):
        click = self.find_withtag(tkinter.CURRENT)
        if len(click):
            if not self.in_turn:
                self.update_message("Not your turn now.")
                return
            (c_id,) = click
            if c_id in self.board.hexid:
                c_hex = self.board.hexid[c_id]
                c_coord = (c_hex.q, c_hex.r)
                print(f'clicked hex (q={c_hex.q}, r={c_hex.r}).')
                if self.active_piece is None:
                    # Activate the current piece
                    c_piece = self.board.pieces[c_coord]
                    if c_piece.col != self.color[0].upper():
                        self.update_message(f"No. You are {self.color}, and this is {c_piece.col}.")
                        return
                    c_piece.activate()
                    if c_coord in self.dests:
                        self.button.toggle_exit(True)
                    else:
                        self.button.toggle_exit(False)
                    self.active_piece = c_coord
                else:
                    c_piece = self.board.pieces[c_coord]
                    if c_coord == self.active_piece:
                        c_piece = self.board.pieces[c_coord]
                        c_piece.deactivate()
                        self.button.toggle_exit(False)
                        self.active_piece = None
                    elif c_coord in self.move_coords(self.active_piece):
                        if c_piece.col:
                            self.update_message(f"{c_coord} is occupied by {c_piece.col}.")
                            return
                        self.deactivate(("MOVE", (self.active_piece, c_coord)))
                    elif c_coord in self.jump_coords(self.active_piece):
                        if c_piece.col:
                            self.update_message(f"{c_coord} is occupied by {c_piece.col}.")
                            return
                        self.deactivate(("JUMP", (self.active_piece, c_coord)))
                    else:
                        self.update_message(f"No. You're at {self.active_piece}. {c_coord} is too far away.")
                return
            elif c_id == self.button.id:
                print(f'clicked the {self.button.name} button!')
                if self.button.is_exit:
                    self.deactivate(("EXIT", self.active_piece))
                else:
                    self.deactivate(("PASS", None))
                return
        print('clicked nothing.')

    @staticmethod
    def move_coords(c):
        return {(c[0] + i[0], c[1] + i[1]) for i in ((0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1))}

    @staticmethod
    def jump_coords(c):
        return {(c[0] + 2 * i[0], c[1] + 2 * i[1]) for i in ((0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1))}


class PassButton:
    def __init__(self, canvas, board, root):
        self.name = 'Pass'
        self.root = root
        self.board = board
        self.is_exit = False
        self.canvas = canvas

        # render the button (a hexagon):
        D = min(H, W)
        coordinates = transform(HEX_F, (0.8 * W, 0.8 * H), (0.2 * D, 0.2 * D))
        self.id = canvas.create_polygon(coordinates, fill=Color.FG)
        self.labelid = canvas.create_text((0.8 * W, 0.8 * H), text="Pass",
                                          fill=Color.BG, state=tkinter.DISABLED)

    def toggle_exit(self, value):
        self.is_exit = value
        if value:
            self.canvas.itemconfig(self.labelid, text="Exit")
        else:
            self.canvas.itemconfig(self.labelid, text="Pass")


class Board:
    def __init__(self, canvas):
        # create board outline:
        D = min(H, W)
        board_coordinates = transform(HEX_F, (W / 2, H / 2), (0.9 * D, 0.9 * D))
        self.id = canvas.create_polygon(board_coordinates, fill=Color.FG)

        # create hexagons and piece shadows on board:
        self.hexes = {}  # type: Dict[Tuple[int, int], Hex]
        self.hexid = {}
        self.pieces = {}  # type: Dict[Tuple[int, int], Piece]
        size = 0.12 * D
        for q in range(-3, 4):
            for r in range(-3, 4):
                s = -q - r
                if s in range(-3, 4):
                    new_hex = Hex(q, r, size, canvas)
                    self.hexes[q, r] = new_hex
                    self.hexid[new_hex.id] = new_hex

                    new_piece = Piece(q, r, size, canvas)
                    self.pieces[q, r] = new_piece

    def cycle(self, q, r):
        view_piece = self.pieces[q, r]
        view_piece.cycle()

    def set_dests(self, color, dests):
        c = Color.NONE
        if color == 'red':
            c = Color.RED_S
        elif color == 'blue':
            c = Color.BLUE_S
        elif color == 'green':
            c = Color.GREEN_S
        for i in dests:
            self.hexes[tuple(i)].set_color(c)

    def update_board(self, board):
        rev_board = {}
        for i in board['red']:
            rev_board[tuple(i)] = 'R'
        for i in board['green']:
            rev_board[tuple(i)] = 'G'
        for i in board['blue']:
            rev_board[tuple(i)] = 'B'
        for q in range(-3, 4):
            for r in range(-3, 4):
                s = -q - r
                if s in range(-3, 4):
                    col = rev_board.get((q, r), None)
                    self.pieces[q, r].paint(col=col)


def transform(coordinates, translate_xy=(0, 0), dilate_xy=(1, 1)):
    a, b = translate_xy
    A, B = dilate_xy
    return [(a + x * A, b + y * B) for (x, y) in coordinates]


class Hex:
    def __init__(self, q, r, d, canvas):
        # set me up:
        self.q = q
        self.r = r
        self.s = -q - r
        self.canvas = canvas  # type: BoardViewCanvas

        # and draw me on the canvas:
        # place in the center of the board
        coords = transform(HEX_P, (W / 2, H / 2), (d, d))
        # offset into position based on coordinates
        coords = transform(coords, (r * PW / 2 * d + q * PW * d, r * 3 * PH / 4 * d))
        # create a hexagon there!
        self.id = canvas.create_polygon(coords, tag="hex",
                                        outline=Color.BG, fill=Color.NONE, activefill=Color.HL)

        # remember my coordinates too!?
        self.coords = (W / 2 + r * PW / 2 * d + q * PW * d, H / 2 + r * 3 * PH / 4 * d)

    def set_color(self, color):
        self.canvas.itemconfig(self.id, fill=color)


class Piece:
    def __init__(self, q, r, d, canvas):
        self.canvas = canvas

        self.col = None

        # create generic stone, invisible, in position
        x, y = (W / 2 + r * PW / 2 * d + q * PW * d, H / 2 + r * 3 * PH / 4 * d)
        rad1 = 0.35 * d  # outer circle (stone)
        rad2 = 0.25 * d  # inner circle (paint)

        self.active = False

        self.stone_id = canvas.create_oval(x - rad1, y - rad1, x + rad1, y + rad1,
                                           tag="stone", outline=Color.BLACK, fill=Color.STONE,
                                           state=tkinter.HIDDEN)
        self.paint_id = canvas.create_oval(x - rad2, y - rad2, x + rad2, y + rad2,
                                           tag="paint", outline=Color.NONE, fill=Color.BLACK,
                                           state=tkinter.HIDDEN)
        self.letter_id = canvas.create_text(x, y,
                                            tag="letter", text="?", fill=Color.STONE,
                                            state=tkinter.HIDDEN)

    def paint(self, col=None):
        self.col = col
        if col is None:
            self.canvas.itemconfig(self.letter_id, text="?")
            self.canvas.itemconfig(self.paint_id, fill=Color.BLACK)
            self.hide()
        else:
            letter = None
            colour = None
            if col == 'R':
                colour = Color.RED
                letter = 'R'
            elif col == 'G':
                colour = Color.GREEN
                letter = 'G'
            elif col == 'B':
                colour = Color.BLUE
                letter = 'B'
            elif col == 'BLOCK':
                colour = Color.BLACK
                letter = ''
            self.canvas.itemconfig(self.letter_id, text=letter)
            self.canvas.itemconfig(self.paint_id, fill=colour)
            self.show()

    def activate(self):
        self.active = True
        if self.col == 'R':
            colour = Color.BLACK
        elif self.col == 'G':
            colour = Color.BLACK
        elif self.col == 'B':
            colour = Color.BLACK
        else:
            colour = None
        self.canvas.itemconfig(self.paint_id, fill=colour)

    def deactivate(self):
        self.active = False
        if self.col == 'R':
            colour = Color.RED
        elif self.col == 'G':
            colour = Color.GREEN
        elif self.col == 'B':
            colour = Color.BLUE
        else:
            colour = None
        self.canvas.itemconfig(self.paint_id, fill=colour)

    def cycle(self):
        if self.col == 'R':
            self.paint('G')
        elif self.col == 'G':
            self.paint('B')
        elif self.col == 'B':
            self.paint('BLOCK')
        elif self.col == 'BLOCK':
            self.paint(None)
        elif self.col is None:
            self.paint('R')

    def hide(self):
        self.canvas.itemconfig(self.letter_id, state=tkinter.HIDDEN)
        self.canvas.itemconfig(self.paint_id, state=tkinter.HIDDEN)
        self.canvas.itemconfig(self.stone_id, state=tkinter.HIDDEN)

    def show(self):
        self.canvas.itemconfig(self.letter_id, state=tkinter.DISABLED)
        self.canvas.itemconfig(self.paint_id, state=tkinter.DISABLED)
        self.canvas.itemconfig(self.stone_id, state=tkinter.DISABLED)


if __name__ == '__main__':
    rpc_server()
