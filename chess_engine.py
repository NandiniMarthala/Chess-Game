"""
Chess AI Engine using Minimax with Alpha-Beta Pruning
"""
import copy
import random

# Piece constants
EMPTY = 0
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
WHITE, BLACK = 1, -1

# Piece values for evaluation
PIECE_VALUES = {
    PAWN: 100,
    KNIGHT: 320,
    BISHOP: 330,
    ROOK: 500,
    QUEEN: 900,
    KING: 20000
}

# Position bonus tables (from white's perspective)
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_MIDDLE_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

POSITION_TABLES = {
    PAWN: PAWN_TABLE,
    KNIGHT: KNIGHT_TABLE,
    BISHOP: BISHOP_TABLE,
    ROOK: ROOK_TABLE,
    QUEEN: QUEEN_TABLE,
    KING: KING_MIDDLE_TABLE,
}


class ChessBoard:
    def __init__(self):
        self.board = self._create_initial_board()
        self.current_turn = WHITE
        self.white_castle_kingside = True
        self.white_castle_queenside = True
        self.black_castle_kingside = True
        self.black_castle_queenside = True
        self.en_passant_square = None
        self.move_history = []
        self.game_over = False
        self.winner = None

    def _create_initial_board(self):
        board = [[0] * 8 for _ in range(8)]
        # Black pieces (row 0-1)
        back_row = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]
        for col, piece in enumerate(back_row):
            board[0][col] = (piece, BLACK)
            board[7][col] = (piece, WHITE)
        for col in range(8):
            board[1][col] = (PAWN, BLACK)
            board[6][col] = (PAWN, WHITE)
        # Empty squares
        for row in range(2, 6):
            for col in range(8):
                board[row][col] = None
        return board

    def get_piece(self, row, col):
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None

    def is_valid_pos(self, row, col):
        return 0 <= row < 8 and 0 <= col < 8

    def get_legal_moves(self, color=None):
        if color is None:
            color = self.current_turn
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece[1] == color:
                    piece_moves = self._get_piece_moves(row, col, piece[0], color)
                    moves.extend(piece_moves)
        # Filter moves that leave king in check
        legal = []
        for move in moves:
            if not self._move_leaves_king_in_check(move, color):
                legal.append(move)
        return legal

    def _get_piece_moves(self, row, col, piece_type, color):
        moves = []
        if piece_type == PAWN:
            moves = self._pawn_moves(row, col, color)
        elif piece_type == KNIGHT:
            moves = self._knight_moves(row, col, color)
        elif piece_type == BISHOP:
            moves = self._sliding_moves(row, col, color, [(1,1),(1,-1),(-1,1),(-1,-1)])
        elif piece_type == ROOK:
            moves = self._sliding_moves(row, col, color, [(1,0),(-1,0),(0,1),(0,-1)])
        elif piece_type == QUEEN:
            moves = self._sliding_moves(row, col, color, [(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)])
        elif piece_type == KING:
            moves = self._king_moves(row, col, color)
        return moves

    def _pawn_moves(self, row, col, color):
        moves = []
        direction = -1 if color == WHITE else 1
        start_row = 6 if color == WHITE else 1
        promo_row = 0 if color == WHITE else 7

        # Forward move
        nr = row + direction
        if self.is_valid_pos(nr, col) and not self.board[nr][col]:
            if nr == promo_row:
                for pt in [QUEEN, ROOK, BISHOP, KNIGHT]:
                    moves.append({'from': (row, col), 'to': (nr, col), 'promotion': pt})
            else:
                moves.append({'from': (row, col), 'to': (nr, col)})
            # Double push
            if row == start_row and not self.board[row + 2 * direction][col]:
                moves.append({'from': (row, col), 'to': (row + 2 * direction, col), 'en_passant_set': (row + direction, col)})

        # Captures
        for dc in [-1, 1]:
            nc = col + dc
            if self.is_valid_pos(nr, nc):
                target = self.board[nr][nc]
                if target and target[1] != color:
                    if nr == promo_row:
                        for pt in [QUEEN, ROOK, BISHOP, KNIGHT]:
                            moves.append({'from': (row, col), 'to': (nr, nc), 'promotion': pt})
                    else:
                        moves.append({'from': (row, col), 'to': (nr, nc)})
                # En passant
                elif self.en_passant_square == (nr, nc):
                    moves.append({'from': (row, col), 'to': (nr, nc), 'en_passant_capture': (row, nc)})
        return moves

    def _knight_moves(self, row, col, color):
        moves = []
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr, nc = row + dr, col + dc
            if self.is_valid_pos(nr, nc):
                target = self.board[nr][nc]
                if not target or target[1] != color:
                    moves.append({'from': (row, col), 'to': (nr, nc)})
        return moves

    def _sliding_moves(self, row, col, color, directions):
        moves = []
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            while self.is_valid_pos(nr, nc):
                target = self.board[nr][nc]
                if not target:
                    moves.append({'from': (row, col), 'to': (nr, nc)})
                else:
                    if target[1] != color:
                        moves.append({'from': (row, col), 'to': (nr, nc)})
                    break
                nr += dr
                nc += dc
        return moves

    def _king_moves(self, row, col, color):
        moves = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if self.is_valid_pos(nr, nc):
                    target = self.board[nr][nc]
                    if not target or target[1] != color:
                        moves.append({'from': (row, col), 'to': (nr, nc)})

        # Castling
        if color == WHITE and row == 7 and col == 4:
            if self.white_castle_kingside and not self.board[7][5] and not self.board[7][6]:
                rook = self.board[7][7]
                if rook and rook == (ROOK, WHITE):
                    moves.append({'from': (7,4), 'to': (7,6), 'castle': 'kingside'})
            if self.white_castle_queenside and not self.board[7][3] and not self.board[7][2] and not self.board[7][1]:
                rook = self.board[7][0]
                if rook and rook == (ROOK, WHITE):
                    moves.append({'from': (7,4), 'to': (7,2), 'castle': 'queenside'})
        if color == BLACK and row == 0 and col == 4:
            if self.black_castle_kingside and not self.board[0][5] and not self.board[0][6]:
                rook = self.board[0][7]
                if rook and rook == (ROOK, BLACK):
                    moves.append({'from': (0,4), 'to': (0,6), 'castle': 'kingside'})
            if self.black_castle_queenside and not self.board[0][3] and not self.board[0][2] and not self.board[0][1]:
                rook = self.board[0][0]
                if rook and rook == (ROOK, BLACK):
                    moves.append({'from': (0,4), 'to': (0,2), 'castle': 'queenside'})
        return moves

    def _is_square_attacked(self, row, col, by_color):
        """Check if a square is attacked by given color"""
        # Check by enemy pieces
        enemy_moves = []
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece[1] == by_color:
                    if piece[0] == PAWN:
                        direction = -1 if by_color == WHITE else 1
                        for dc in [-1, 1]:
                            if (r + direction, c + dc) == (row, col):
                                return True
                    elif piece[0] == KNIGHT:
                        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                            if (r+dr, c+dc) == (row, col):
                                return True
                    elif piece[0] == KING:
                        for dr in [-1,0,1]:
                            for dc in [-1,0,1]:
                                if (r+dr, c+dc) == (row, col):
                                    return True
                    elif piece[0] in [BISHOP, QUEEN]:
                        for dr, dc in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                            nr, nc = r+dr, c+dc
                            while self.is_valid_pos(nr, nc):
                                if (nr, nc) == (row, col):
                                    return True
                                if self.board[nr][nc]:
                                    break
                                nr += dr
                                nc += dc
                    elif piece[0] in [ROOK, QUEEN]:
                        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                            nr, nc = r+dr, c+dc
                            while self.is_valid_pos(nr, nc):
                                if (nr, nc) == (row, col):
                                    return True
                                if self.board[nr][nc]:
                                    break
                                nr += dr
                                nc += dc
                    if piece[0] == QUEEN:
                        # Already handled above in bishop/rook sections
                        pass
        return False

    def _find_king(self, color):
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece[0] == KING and piece[1] == color:
                    return (r, c)
        return None

    def is_in_check(self, color):
        king_pos = self._find_king(color)
        if not king_pos:
            return False
        return self._is_square_attacked(king_pos[0], king_pos[1], -color)

    def _move_leaves_king_in_check(self, move, color):
        """Apply move temporarily and check if own king is in check"""
        saved_board = [row[:] for row in self.board]
        saved_ep = self.en_passant_square
        self._apply_move_internal(move)
        in_check = self.is_in_check(color)
        self.board = saved_board
        self.en_passant_square = saved_ep
        return in_check

    def _apply_move_internal(self, move):
        fr, fc = move['from']
        tr, tc = move['to']
        piece = self.board[fr][fc]
        self.board[tr][tc] = piece
        self.board[fr][fc] = None
        self.en_passant_square = None

        if 'en_passant_set' in move:
            self.en_passant_square = move['en_passant_set']
        if 'en_passant_capture' in move:
            cr, cc = move['en_passant_capture']
            self.board[cr][cc] = None
        if 'promotion' in move:
            self.board[tr][tc] = (move['promotion'], piece[1])
        if 'castle' in move:
            if move['castle'] == 'kingside':
                rook_row = fr
                self.board[rook_row][5] = self.board[rook_row][7]
                self.board[rook_row][7] = None
            else:
                rook_row = fr
                self.board[rook_row][3] = self.board[rook_row][0]
                self.board[rook_row][0] = None

    def apply_move(self, move):
        fr, fc = move['from']
        tr, tc = move['to']
        piece = self.board[fr][fc]
        self._apply_move_internal(move)

        # Update castling rights
        if piece and piece[0] == KING:
            if piece[1] == WHITE:
                self.white_castle_kingside = False
                self.white_castle_queenside = False
            else:
                self.black_castle_kingside = False
                self.black_castle_queenside = False
        if piece and piece[0] == ROOK:
            if (fr, fc) == (7, 0): self.white_castle_queenside = False
            if (fr, fc) == (7, 7): self.white_castle_kingside = False
            if (fr, fc) == (0, 0): self.black_castle_queenside = False
            if (fr, fc) == (0, 7): self.black_castle_kingside = False

        self.move_history.append(move)
        self.current_turn = -self.current_turn

        # Check game over
        opponent_moves = self.get_legal_moves(self.current_turn)
        if not opponent_moves:
            if self.is_in_check(self.current_turn):
                self.game_over = True
                self.winner = -self.current_turn
            else:
                self.game_over = True
                self.winner = 0  # Stalemate

    def evaluate(self):
        score = 0
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    pt, color = piece
                    value = PIECE_VALUES[pt]
                    # Position bonus
                    table = POSITION_TABLES.get(pt)
                    if table:
                        if color == WHITE:
                            pos_bonus = table[row * 8 + col]
                        else:
                            pos_bonus = table[(7 - row) * 8 + col]
                        value += pos_bonus
                    score += value * color
        return score

    def to_dict(self):
        board_data = []
        for row in range(8):
            row_data = []
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    row_data.append({'type': piece[0], 'color': piece[1]})
                else:
                    row_data.append(None)
            board_data.append(row_data)
        return {
            'board': board_data,
            'current_turn': self.current_turn,
            'game_over': self.game_over,
            'winner': self.winner,
            'in_check': self.is_in_check(self.current_turn),
        }


class ChessAI:
    def __init__(self, depth=3):
        self.depth = depth

    def get_best_move(self, board):
        color = board.current_turn
        legal_moves = board.get_legal_moves(color)
        if not legal_moves:
            return None

        best_move = None
        best_score = float('-inf') if color == BLACK else float('inf')
        alpha = float('-inf')
        beta = float('inf')

        # Shuffle for variety
        random.shuffle(legal_moves)

        for move in legal_moves:
            new_board = copy.deepcopy(board)
            new_board.apply_move(move)
            score = self._minimax(new_board, self.depth - 1, alpha, beta, color == WHITE)

            if color == BLACK:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)

        return best_move

    def _minimax(self, board, depth, alpha, beta, maximizing):
        if depth == 0 or board.game_over:
            return board.evaluate()

        legal_moves = board.get_legal_moves(board.current_turn)
        if not legal_moves:
            if board.is_in_check(board.current_turn):
                return -20000 if maximizing else 20000
            return 0

        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                new_board = copy.deepcopy(board)
                new_board.apply_move(move)
                eval_score = self._minimax(new_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                new_board = copy.deepcopy(board)
                new_board.apply_move(move)
                eval_score = self._minimax(new_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
