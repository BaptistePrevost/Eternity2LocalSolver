from solver_heuristic import *
from typing import List, Tuple
import random
from tqdm import tqdm
from itertools import combinations, permutations
import math

from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt

seed = random.randint(0, 999999999999999999)
# seed = 138238757792518736
random.seed(seed)


# Timer decorator
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Function {} took {} seconds".format(func.__name__, end - start))
        return result
    return wrapper

class ElitePool:
    def __init__(self, size: int):
        self.size = size
        self.pool = []
        self.scores = []

    def add(self, element, score):
        if len(self.pool) < self.size:
            self.pool.append(element)
            self.scores.append(score)
        else:
            differences = 0
            maxDistance = 0
            maxIndex = 0
            maxScore = 0
            scoreIndex = 0
            for i in range(len(self.pool)):
                distance = sum(self.pool[i][j] != element[j] for j in range(len(self.pool[i])))
                if distance > maxDistance:
                    maxDistance = distance
                    maxIndex = i
                if self.scores[i] > maxScore:
                    maxScore = self.scores[i]
                    scoreIndex = i

                differences += (distance > 0)

            if differences == len(self.pool):
                # It replaces the element with the maximum score
                self.pool[scoreIndex] = element
                self.scores[scoreIndex] = score
            elif differences and maxDistance > 0:
                # It replaces the element with the maximum distance
                self.pool[maxIndex] = element
                self.scores[maxIndex] = score

def is_corner(piece: Tuple[int, int, int, int]):
    return piece.count(0) == 2

def is_side(piece: Tuple[int, int, int, int]):
    return piece.count(0) == 1

def solve_local_search(eternity_puzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    def is_left(position: int):
        return position % eternity_puzzle.board_size == 0
    
    def is_right(position: int):
        return (position+1) % eternity_puzzle.board_size == 0
    
    def is_top(position: int):
        return position >= eternity_puzzle.board_size * (eternity_puzzle.board_size - 1)
    
    def is_bottom(position: int):
        return position < eternity_puzzle.board_size
    

    def edge_rotation(edge: Tuple[int, int, int, int], position: int):
        for rotation in eternity_puzzle.generate_rotation(edge):
            if position % eternity_puzzle.board_size == 0:
                if rotation[2] == 0:
                    return rotation
            elif position % eternity_puzzle.board_size == eternity_puzzle.board_size - 1:
                if rotation[3] == 0:
                    return rotation
            elif position // eternity_puzzle.board_size == 0:
                if rotation[1] == 0:
                    return rotation
            elif position // eternity_puzzle.board_size == eternity_puzzle.board_size - 1:
                if rotation[0] == 0:
                    return rotation
                
        raise Exception("No rotation found for edge")
                
    def corner_rotation(corner : Tuple[int, int, int, int], position: int):
        for rotation in eternity_puzzle.generate_rotation(corner):
            if position == 0:
                if rotation[1] == 0 and rotation[2] == 0:
                    return rotation
            elif position == eternity_puzzle.board_size - 1:
                if rotation[1] == 0 and rotation[3] == 0:
                    return rotation
            elif position == eternity_puzzle.board_size*(eternity_puzzle.board_size - 1):
                if rotation[0] == 0 and rotation[2] == 0:
                    return rotation
            elif position == eternity_puzzle.board_size*eternity_puzzle.board_size - 1:
                if rotation[0] == 0 and rotation[3] == 0:
                    return rotation
                
        raise Exception("No rotation found for corner")
                
    def border_rotation(piece: Tuple[int, int, int, int], position: int):
        if is_corner(piece):
            return corner_rotation(piece, position)
        elif is_side(piece):
            return edge_rotation(piece, position)
        else:
            raise Exception("Piece is not a border piece")
    
    """
        INITIALISATION
    """
    internal_positions = [pos for pos in range(eternity_puzzle.board_size + 1, eternity_puzzle.n_piece - eternity_puzzle.board_size - 1) if pos % eternity_puzzle.board_size != 0 and pos % eternity_puzzle.board_size != eternity_puzzle.board_size - 1 and pos // eternity_puzzle.board_size != 0 and pos // eternity_puzzle.board_size != eternity_puzzle.board_size - 1]
    edge_positions = [pos for pos in range(1, eternity_puzzle.board_size-1)] + [pos*eternity_puzzle.board_size for pos in range(1, eternity_puzzle.board_size-1)] + [pos + eternity_puzzle.board_size*(eternity_puzzle.board_size-1) for pos in range(1, eternity_puzzle.board_size-1)] + [pos*eternity_puzzle.board_size + eternity_puzzle.board_size-1 for pos in range(1, eternity_puzzle.board_size-1)]
    corner_positions = [eternity_puzzle.board_size - 1, eternity_puzzle.n_piece - eternity_puzzle.board_size, eternity_puzzle.n_piece - 1]

    all_positions = internal_positions + edge_positions + corner_positions

    remaining_pieces = []
    fixedCorner = None
    for piece in eternity_puzzle.piece_list:
        if not fixedCorner and piece.count(0) == 2:
            fixedCorner = piece
        else:
            remaining_pieces.append(piece)
    
    # Search among the external pieces

    def fullyRandomConstruction():
        random.shuffle(remaining_pieces)
        cornerIndex = 0
        edgeIndex = 0
        internalIndex = 0
        board = [None] * eternity_puzzle.n_piece
        board[0] = corner_rotation(fixedCorner, 0)
        for piece in remaining_pieces:
            if is_corner(piece):
                board[corner_positions[cornerIndex]] = corner_rotation(piece, corner_positions[cornerIndex])
                cornerIndex += 1
            elif is_side(piece):
                board[edge_positions[edgeIndex]] = edge_rotation(piece, edge_positions[edgeIndex])
                edgeIndex += 1
            else:
                board[internal_positions[internalIndex]] = piece
                internalIndex += 1
        return board

    def scanRowConstruction(maxTime: float):
        random.shuffle(remaining_pieces)
        beginTime = time.time()
        board = [None] * eternity_puzzle.n_piece
        board[0] = corner_rotation(fixedCorner, 0)

        def recursiveBuild(board: List[Tuple[int, int, int, int]], position: int, pieces: List[Tuple[int, int, int, int]]):
            if position == eternity_puzzle.n_piece:
                return board
            
            if position in corner_positions:
                for i in range(len(pieces)):
                    piece = pieces[i]
                    if is_corner(piece):
                        piece = corner_rotation(piece, position)
                        if (time.time() - beginTime > maxTime) or (position == eternity_puzzle.board_size-1 and piece[2] == board[position-1][3]) or (position == eternity_puzzle.n_piece - eternity_puzzle.board_size and piece[1] == board[position-eternity_puzzle.board_size][0]) or (position == eternity_puzzle.n_piece - 1 and piece[2] == board[position-1][3] and piece[1] == board[position-eternity_puzzle.board_size][0]):
                            board[position] = piece
                            result = recursiveBuild(board, position+1, pieces[:i] + pieces[i+1:])
                            if result:
                                return result
                        
            elif is_top(position) or is_bottom(position) or is_left(position) or is_right(position):
                for i in range(len(pieces)):
                    piece = pieces[i]
                    if is_side(piece):
                        piece = edge_rotation(piece, position)
                        if (time.time() - beginTime > maxTime) or (is_bottom(position) and piece[2] == board[position-1][3]) or (is_top(position) and piece[2] == board[position-1][3] and piece[1] == board[position-eternity_puzzle.board_size][0]) or (is_left(position) and piece[1] == board[position-eternity_puzzle.board_size][0]) or (is_right(position)  and piece[2] == board[position-1][3] and piece[1] == board[position-eternity_puzzle.board_size][0]):
                            board[position] =  piece
                            result = recursiveBuild(board, position+1, pieces[:i] + pieces[i+1:])
                            if result:
                                return result
            else:
                for i in range(len(pieces)):
                    piece = pieces[i]
                    if not is_corner(piece) and not is_side(piece):
                        for rotation in eternity_puzzle.generate_rotation(piece):
                            if (time.time() - beginTime > maxTime) or rotation[2] == board[position-1][3] and rotation[1] == board[position-eternity_puzzle.board_size][0]:
                                board[position] = rotation
                                result = recursiveBuild(board, position+1, pieces[:i] + pieces[i+1:])
                                if result:
                                    return result  

            return None
        
        return recursiveBuild(board, 1, remaining_pieces)

    def scanRowConstructionReverse(maxTime: float):
        pieces = copy.deepcopy(remaining_pieces)
        beginTime = time.time()
        board = [None] * eternity_puzzle.n_piece
        # Find another corner
        for piece in pieces:
            if is_corner(piece):
                board[0] = corner_rotation(piece, 0)
                pieces.remove(piece)
                break

        pieces.append(fixedCorner)
        random.shuffle(pieces)

        def recursiveBuild(board: List[Tuple[int, int, int, int]], position: int, pieces: List[Tuple[int, int, int, int]]):
            if position == eternity_puzzle.n_piece:
                return board
            
            if position in corner_positions:
                for i in range(len(pieces)):
                    piece = pieces[i]
                    if is_corner(piece):
                        piece = corner_rotation(piece, position)
                        if (time.time() - beginTime > maxTime) or (position == eternity_puzzle.board_size-1 and piece[2] == board[position-1][3]) or (position == eternity_puzzle.n_piece - eternity_puzzle.board_size and piece[1] == board[position-eternity_puzzle.board_size][0]) or (position == eternity_puzzle.n_piece - 1 and piece[2] == board[position-1][3] and piece[1] == board[position-eternity_puzzle.board_size][0]):
                            board[position] = piece
                            result = recursiveBuild(board, position+1, pieces[:i] + pieces[i+1:])
                            if result:
                                return result
                        
            elif is_top(position) or is_bottom(position) or is_left(position) or is_right(position):
                for i in range(len(pieces)):
                    piece = pieces[i]
                    if is_side(piece):
                        piece = edge_rotation(piece, position)
                        if (time.time() - beginTime > maxTime) or (is_bottom(position) and piece[2] == board[position-1][3]) or (is_top(position) and piece[2] == board[position-1][3] and piece[1] == board[position-eternity_puzzle.board_size][0]) or (is_left(position) and piece[1] == board[position-eternity_puzzle.board_size][0]) or (is_right(position)  and piece[2] == board[position-1][3] and piece[1] == board[position-eternity_puzzle.board_size][0]):
                            board[position] =  piece
                            result = recursiveBuild(board, position+1, pieces[:i] + pieces[i+1:])
                            if result:
                                return result
            else:
                for i in range(len(pieces)):
                    piece = pieces[i]
                    if not is_corner(piece) and not is_side(piece):
                        for rotation in eternity_puzzle.generate_rotation(piece):
                            if (time.time() - beginTime > maxTime) or rotation[2] == board[position-1][3] and rotation[1] == board[position-eternity_puzzle.board_size][0]:
                                board[position] = rotation
                                result = recursiveBuild(board, position+1, pieces[:i] + pieces[i+1:])
                                if result:
                                    return result  

            return None
        
        board = recursiveBuild(board, 1, pieces)[::-1]
        for position in range(eternity_puzzle.n_piece):
            if is_corner(board[position]):
                if eternity_puzzle.hash_piece(board[position]) == eternity_puzzle.hash_piece(fixedCorner):
                    board[position] = corner_rotation(board[0], position)
                board[position] = corner_rotation(board[position], position)
            elif is_side(board[position]):
                board[position] = edge_rotation(board[position], position)
            else:
                n,s,w,e = board[position]
                board[position] = (s,n,e,w)

        board[0] = corner_rotation(fixedCorner, 0)

        return board
    

    def getPieceConflicts(board: List[Tuple[int, int, int, int]], piece: Tuple[int, int, int, int], position: int, otherPositions: List[int] = []):

        nb_conflicts = 0
        if piece[2] and board[position-1][3] != piece[2]:
            nb_conflicts += 1 if position-1 not in otherPositions else 0.5

        if piece[3] and board[position+1][2] != piece[3]:
            nb_conflicts += 1 if position+1 not in otherPositions else 0.5

        if piece[1] and board[position-eternity_puzzle.board_size][0] != piece[1]:
            nb_conflicts += 1 if position-eternity_puzzle.board_size not in otherPositions else 0.5

        if piece[0] and board[position+eternity_puzzle.board_size][1] != piece[0]:
            nb_conflicts += 1 if position+eternity_puzzle.board_size not in otherPositions else 0.5

        return nb_conflicts

    def getInnerPieceConflicts(board: List[Tuple[int, int, int, int]], piece: Tuple[int, int, int, int], position: int, otherPositions: List[int] = []):
        """
        :param board: the board
        :param piece: the piece
        :param position: the position of the piece
        :return: the number of conflicts for the piece at position
        """
        nb_conflicts = 0
        if board[position-1][3] != piece[2]:
            nb_conflicts += 1 if position-1 not in otherPositions else 0.5

        if board[position+1][2] != piece[3]:
            nb_conflicts += 1 if position+1 not in otherPositions else 0.5

        if board[position-eternity_puzzle.board_size][0] != piece[1]:
            nb_conflicts += 1 if position-eternity_puzzle.board_size not in otherPositions else 0.5

        if board[position+eternity_puzzle.board_size][1] != piece[0]:
            nb_conflicts += 1 if position+eternity_puzzle.board_size not in otherPositions else 0.5

        return nb_conflicts
    
    def getCornerPieceConflicts(board: List[Tuple[int, int, int, int]], piece: Tuple[int, int, int, int], position: int):
        """
        :param board: the board
        :param piece: the piece
        :param position: the position of the piece
        :return: the number of conflicts for the piece at position
        """
        nb_conflicts = 0
        if position == 0:
            nb_conflicts += board[position+eternity_puzzle.board_size][1] != piece[0]
            nb_conflicts += board[position+1][2] != piece[3]
        elif position == eternity_puzzle.board_size - 1:
            nb_conflicts += board[position+eternity_puzzle.board_size][1] != piece[0]
            nb_conflicts += board[position-1][3] != piece[2]
        elif position == eternity_puzzle.board_size*(eternity_puzzle.board_size - 1):
            nb_conflicts += board[position-eternity_puzzle.board_size][0] != piece[1]
            nb_conflicts += board[position+1][2] != piece[3]
        else:
            nb_conflicts += board[position-eternity_puzzle.board_size][0] != piece[1]
            nb_conflicts += board[position-1][3] != piece[2]

        # print("piece", piece, "position", position, "nb_conflicts", nb_conflicts)   
        return nb_conflicts


    """
        Low-level heuristics
    """


    def swapAndRotateTwoCorners(board: List[Tuple[int, int, int, int]]):
        """
            Select two random corners
            Returns a list of moved pieces as tuples (rotated piece, position)
        """

        position1, position2 = random.sample(corner_positions, 2)
        move = [(corner_rotation(board[position2], position1), position1), (corner_rotation(board[position1], position2), position2)]

        save = []
        delta = 0
        for _, position in move:
            save.append(board[position])
            delta -= getCornerPieceConflicts(board, board[position], position)
    

        # c1 = eternity_puzzle.get_total_n_conflict(board)
        for piece, position in move:
            board[position] = piece
        # c2 = eternity_puzzle.get_total_n_conflict(board)
        
        for _, position in move:
            delta += getCornerPieceConflicts(board, board[position], position)

        for i, (_, position) in enumerate(move):
            board[position] = save[i] 

        # if c2-c1 != delta:
        #     raise Exception ("c2-c1 != delta in swapAndRotateTwoCorners")
        return move, delta
    
    def swapAndRotateTwoEdges(board: List[Tuple[int, int, int, int]]):
        """
            Select two random edges
            Returns a list of moved pieces as tuples (rotated piece, position)
        """
        position1, position2 = random.sample(edge_positions, 2)
        move = [(edge_rotation(board[position2], position1), position1), (edge_rotation(board[position1], position2), position2)]

        save = []
        delta = 0
        for _, position in move:
            save.append(board[position])
            delta -= getPieceConflicts(board, board[position], position, otherPositions=[position1, position2])
        
        # c1 = eternity_puzzle.get_total_n_conflict(board)
        for piece, position in move:
            board[position] = piece
        # c2 = eternity_puzzle.get_total_n_conflict(board)

        for _, position in move:
            delta += getPieceConflicts(board, board[position], position, otherPositions=[position1, position2])

        for i, (_, position) in enumerate(move):
            board[position] = save[i] 

        # if c2-c1 != delta:
        #     raise Exception ("c2-c1 != delta in swapAndRotateTwoEdges")
        return move, delta
    
    def swapAndRotateTwoInnerPieces(board: List[Tuple[int, int, int, int]]):
        """
            Select two random inner pieces
            Returns a list of moved pieces as tuples (rotated piece, position)
        """
        position1, position2 = random.sample(internal_positions, 2)
        move = [(random.choice(eternity_puzzle.generate_rotation(board[position2])), position1), (random.choice(eternity_puzzle.generate_rotation(board[position1])), position2)]

        save = []
        delta = 0
        for _, position in move:
            save.append(board[position])
            delta -= getPieceConflicts(board, board[position], position, otherPositions=[position1, position2])

        # c1 = eternity_puzzle.get_total_n_conflict(board)
        for piece, position in move:
            board[position] = piece
        # c2 = eternity_puzzle.get_total_n_conflict(board)

        for _, position in move:
            delta += getPieceConflicts(board, board[position], position, otherPositions=[position1, position2])

        for i, (_, position) in enumerate(move):
            board[position] = save[i] 

        # if c2-c1 != delta:
        #     raise Exception ("c2-c1 != delta in swapAndRotateTwoInnerPieces")
        return move, delta

    def swapOptimallyNonAdjacentBorderPieces(board: List[Tuple[int, int, int, int]], k: int):
        """"
            Select k random non-adjacent border pieces, and swap them optimally.
            Returns a list of moved pieces as tuples (rotated piece, position)
        """
        positions = set(edge_positions + corner_positions)
        weights = {pos: getPieceConflicts(board, board[pos], pos) + 1 for pos in positions}

        # print("[3] 1 ", eternity_puzzle.verify_solution(board))
        
        S = []
        for _ in range(k):
            if not positions:
                break
            positionsList = list(positions)
            weightsList = [weights[pos] for pos in positionsList]

            pos = random.choices(positionsList, weights=weightsList)[0]
            positions.remove(pos)
            S.append(pos)

            # Update weights, carefully handle 0
            if is_left(pos):
                if pos+eternity_puzzle.board_size in positions:
                    positions.remove(pos+eternity_puzzle.board_size)
                if pos-eternity_puzzle.board_size in positions:
                    positions.remove(pos-eternity_puzzle.board_size)
            elif is_right(pos):
                if pos+eternity_puzzle.board_size in positions:
                    positions.remove(pos+eternity_puzzle.board_size)
                if pos-eternity_puzzle.board_size in positions:
                    positions.remove(pos-eternity_puzzle.board_size)

            if is_top(pos):
                if pos+1 in positions:
                    positions.remove(pos+1)
                if pos-1 in positions:
                    positions.remove(pos-1)
            
            elif is_bottom(pos):
                if pos+1 in positions:
                    positions.remove(pos+1)
                if pos-1 in positions:
                    positions.remove(pos-1)

        k = len(S)

        rotations = [[board[S[i]] for _ in range(k)] for i in range(k)] 
        costs = np.zeros((k,k))
        for i, origin in enumerate(S):
            for j, destination in enumerate(S):
                # First check if this is a valide move:
                # A corner can only be move to another corner position, and an edge can only be moved to another edge position
                if (origin in corner_positions and destination not in corner_positions) or (origin not in corner_positions and destination in corner_positions):
                    continue

                rotation = border_rotation(board[origin], destination)
                cost = 4-getPieceConflicts(board, rotation, destination)
                if cost > costs[i][j]:
                    costs[i,j] = cost
                    rotations[i][j] = rotation
            
        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(costs, maximize=True)

        delta = 0
        for i in range(k):
            delta += getPieceConflicts(board, rotations[row_ind[i]][col_ind[i]], S[col_ind[i]]) - getPieceConflicts(board, board[S[col_ind[i]]], S[col_ind[i]])
        
        return [(rotations[row_ind[i]][col_ind[i]], S[col_ind[i]]) for i in range(k)], delta

    def swapOptimallyNonAdjacentInnerPieces(board: List[Tuple[int, int, int, int]], k: int):
        """"
            Select k random non-adjacent inner pieces, and swap them optimally.
            Returns a list of moved pieces as tuples (rotated piece, position)
        """
        positions = set(internal_positions)        
        weights = {pos: getPieceConflicts(board, board[pos], pos) + 1 for pos in positions}

        S = []
        for _ in range(k):
            if not positions:
                break
            positionsList = list(positions)
            weightsList = [weights[pos] for pos in positionsList]
            pos = random.choices(positionsList, weights=weightsList)[0]
            positions.remove(pos)
            S.append(pos)
            if pos-1 in positions:
                positions.remove(pos-1)
            if pos+1 in positions:
                positions.remove(pos+1)
            if pos-eternity_puzzle.board_size in positions:
                positions.remove(pos-eternity_puzzle.board_size)
            if pos+eternity_puzzle.board_size in positions:
                positions.remove(pos+eternity_puzzle.board_size)
        
        k = len(S)

        rotations = [[board[S[i]] for _ in range(k)] for i in range(k)] 
        costs = np.zeros((k,k))
        for i, origin in enumerate(S):
            for j, destination in enumerate(S):
                for rotation in eternity_puzzle.generate_rotation(board[origin]):
                    cost = 4-getInnerPieceConflicts(board, rotation, destination)
                    if cost > costs[i][j]:
                        costs[i,j] = cost
                        rotations[i][j] = rotation
            
        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(costs, maximize=True)

        delta = 0
        for i in range(k):
            delta += getPieceConflicts(board, rotations[row_ind[i]][col_ind[i]], S[col_ind[i]]) - getPieceConflicts(board, board[S[col_ind[i]]], S[col_ind[i]])

        return [(rotations[row_ind[i]][col_ind[i]], S[col_ind[i]]) for i in range(k)], delta

    def isCompleteSquare(board: List[Tuple[int, int, int, int]], position, squareSide):
        """
            :param board: the board
            :param position: the position of the bottom left corner of the square
        """
        # Bottom side
        for i in range(squareSide-1):
            if board[position+i][3] != board[position+i+1][2]:
                return False
            
        # Right side
        for i in range(squareSide-1):
            if board[position+i*eternity_puzzle.board_size][0] != board[position+(i+1)*eternity_puzzle.board_size][1]:
                return False
        
        # Top side
        for i in range(squareSide-1):
            if board[position+(squareSide-1)*eternity_puzzle.board_size+i][3] != board[position+(squareSide-1)*eternity_puzzle.board_size+i+1][2]:
                return False
        
        # Left side
        for i in range(squareSide-1):
            if board[position+i*eternity_puzzle.board_size][0] != board[position+(i+1)*eternity_puzzle.board_size][1]:
                return False
        
        # Inside
        for i in range(1, squareSide-1):
            for j in range(1, squareSide-1):
                if getInnerPieceConflicts(board, board[position+i*eternity_puzzle.board_size+j], position+i*eternity_puzzle.board_size+j) != 0:
                    return False
        
        return True

    def evaluateAndAccept_CompleteSquares(board: List[Tuple[int, int, int, int]], move: List[Tuple[Tuple[int, int, int, int], int]], squareSide: int):
        """
            Evaluates the delta of the number of complete squares after a move
        """

        # print("1", eternity_puzzle.verify_solution(board))

        delta = 0
        save = []
        squarePositions = set()
        for _, pos in move:
            for i in range(squareSide):
                for j in range(squareSide):
                    squarePosition = pos - i - j*eternity_puzzle.board_size
                    if squarePosition > 0 and squarePosition + (squareSide - 1) + (squareSide - 1)*eternity_puzzle.board_size < eternity_puzzle.board_size**2:
                        squarePositions.add(squarePosition)
            save.append(board[pos])

        # print("2", eternity_puzzle.verify_solution(board))

        for squarePosition in squarePositions:
            if isCompleteSquare(board, squarePosition, squareSide):
                delta -= 1
        # print("3", eternity_puzzle.verify_solution(board))
        
        for piece, pos in move:
            board[pos] = piece
            # print(piece, pos, "now", board[pos])

        # print("4", eternity_puzzle.verify_solution(board))

        for squarePosition in squarePositions:
            if isCompleteSquare(board, squarePosition, squareSide):
                delta += 1

        if delta >= 0:
            return board, delta, True
    
        for i, change in enumerate(move):
            board[change[1]] = save[i]

        # print("5", eternity_puzzle.verify_solution(board))
        return board, 0, False

    def pathRelinking(board1, board2):
        """
            Mixed path relinking algorithm
        """
        pieces1 = {}
        pieces2 = {}
        deprecated1 = set()
        deprecated2 = set()

        cost1 = eternity_puzzle.get_total_n_conflict(board1)
        cost2 = eternity_puzzle.get_total_n_conflict(board2)
        bestBoard = None
        bestCost = min(cost1, cost2)
        if cost1 < cost2:
            bestBoard = copy.deepcopy(board1)
        else:
            bestBoard = copy.deepcopy(board2)
            
        random.shuffle(all_positions)
        for n in all_positions:
            if board1[n] != board2[n]:
                pieces1[hash(eternity_puzzle.hash_piece(board1[n]))] = n
                pieces2[hash(eternity_puzzle.hash_piece(board2[n]))] = n

        parity = True
        steps = 0
        distance = len(pieces1)
        while steps + len(deprecated1) < distance and steps + len(deprecated2) < distance:
            # print("BEFORE : ", eternity_puzzle.get_total_n_conflict(board1))
            bestDelta = 9
            bestMove = None
            if parity: # board1 -> board2
                for h in pieces1:
                    if h in deprecated2:
                        continue
                    # Find a best move
                    position1 = pieces1[h]
                    position2 = pieces2[h]
                    
                    if board2[position2].count(0):
                        if board2[position2].count(0) == 2:
                            delta = -getCornerPieceConflicts(board1, board1[position1], position1)
                            delta -= getCornerPieceConflicts(board1, board1[position2], position2)
                            rotation1 = corner_rotation(board1[position2], position1)
                            delta += getCornerPieceConflicts(board1, rotation1, position1) 
                            delta += getCornerPieceConflicts(board1, board2[position2], position2)
                            if delta < bestDelta:
                                bestDelta = delta
                                bestMove = ((position1, rotation1), (position2, board2[position2]))
                                # print("Corner move for a delta of ", bestDelta, "swapping" , position1, "with", position2)
                        else:
                            delta = -getPieceConflicts(board1, board1[position1], position1, otherPositions=[position2])
                            delta -= getPieceConflicts(board1, board1[position2], position2, otherPositions=[position1])
                            rotation1 = edge_rotation(board1[position2], position1)
                            save1, save2 = board1[position1], board1[position2] #Here piece1 and piece2 save the state of pieces in board1
                            board1[position1], board1[position2] = rotation1, board2[position2]
                            delta += getPieceConflicts(board1, rotation1, position1, otherPositions=[position2]) 
                            delta += getPieceConflicts(board1, board2[position2], position2, otherPositions=[position1]) 
                            if delta < bestDelta:
                                bestDelta = delta
                                bestMove = ((position1, rotation1), (position2, board2[position2]))
                                # print("Edge move for a delta of ", bestDelta, "swapping" , position1, "with", position2)
                            board1[position1], board1[position2] = save1, save2

                    else:
                        delta = -getInnerPieceConflicts(board1, board1[position2], position2, otherPositions=[position1])
                        if position1 != position2:
                            delta -= getInnerPieceConflicts(board1, board1[position1], position1, otherPositions=[position2])

                        save1, save2 = board1[position1], board1[position2]
                        board1[position2] = board2[position2]
                        
                        if position1 != position2:
                            for rotation1 in eternity_puzzle.generate_rotation(save2):
                                board1[position1] = rotation1
                                conflicts1 = getInnerPieceConflicts(board1, board1[position1], position1, otherPositions=[position2])
                                conflicts2 = getInnerPieceConflicts(board1, board1[position2], position2, otherPositions=[position1])
                                if delta + conflicts1 + conflicts2 < bestDelta:
                                    bestDelta = delta + conflicts1 + conflicts2
                                    bestMove = ((position1, rotation1), (position2, board1[position2]))
                                    # print("Internal move for a delta of ", bestDelta, "swapping" , position1, "with", position2)

                        else:
                            conflicts2 = getInnerPieceConflicts(board1, board1[position2], position2, otherPositions=[position1])
                            if delta + conflicts2 < bestDelta:
                                bestDelta = delta + conflicts2
                                bestMove = ((position1, None), (position2, board1[position2]))
                                # print("Internal move for a delta of ", bestDelta, "swapping" , position1, "with", position2)

                        board1[position1], board1[position2] = save1, save2
            
                # Make the best move in board1
                if bestMove:
                    (position1, piece1), (position2, piece2) = bestMove
                    if position1 != position2:
                        board1[position1], board1[position2] = piece1, piece2
                        deprecated1.add(hash(eternity_puzzle.hash_piece(piece1)))
                        pieces1[hash(eternity_puzzle.hash_piece(piece1))] = position1
                        cost1 += bestDelta
                        # print("AFTER : ", eternity_puzzle.get_total_n_conflict(board1))
                        del pieces1[hash(eternity_puzzle.hash_piece(piece2))]
                        del pieces2[hash(eternity_puzzle.hash_piece(piece2))]
                    else:
                        board1[position2] = piece2
                        cost1 += bestDelta
                        # print("AFTER : ", eternity_puzzle.get_total_n_conflict(board1))
                        # print(bestDelta)
                        del pieces1[hash(eternity_puzzle.hash_piece(piece2))]
                        del pieces2[hash(eternity_puzzle.hash_piece(piece2))]

                    if cost1 < bestCost:
                        bestCost = cost1
                        bestBoard = copy.deepcopy(board1)
                        print(f"[pathRelinking] New best cost: {bestCost} / {eternity_puzzle.get_total_n_conflict(bestBoard)}")

                if not eternity_puzzle.verify_solution(board1) or not eternity_puzzle.verify_solution(board2):
                    print("ERROR2", steps)
                    print(board1)
                    print(board2)
            
            # Swap board1 and board2
            board1, board2 = board2, board1
            pieces1, pieces2 = pieces2, pieces1
            deprecated1, deprecated2 = deprecated2, deprecated1
            cost1, cost2 = cost2, cost1

            steps += 1

        return bestBoard

    
    def PR2(board1, board2):
        """
            Mixed path relinking algorithm
        """

        positions1 = {}
        positions2 = {}
        hashes = set()

        cost1 = eternity_puzzle.get_total_n_conflict(board1)
        cost2 = eternity_puzzle.get_total_n_conflict(board2)
        bestCost = min(cost1, cost2)
        bestBoard = copy.deepcopy(board1) if cost1 < cost2 else copy.deepcopy(board2)

        for n in all_positions:
            if board1[n] != board2[n]:
                hash1 = eternity_puzzle.hash_piece(board1[n])
                hash2 = eternity_puzzle.hash_piece(board2[n])
                positions1[hash1] = n
                positions2[hash2] = n

                hashes.add(hash1)

        while hashes:
            bestNextCost = float('inf')
            bestNextBoard = None
            bestCycle = None
            tempBoard = copy.deepcopy(board2)
            for pieceHash in hashes:
                cycle = [pieceHash]
                while True:
                    positionIn1 = positions1[pieceHash]
                    pieceHash = eternity_puzzle.hash_piece(board2[positionIn1]) # Store the piece popped in board2
                    board2[positionIn1] = board1[positionIn1] # Swap the piece in board2
                    if pieceHash == cycle[0]: # We completed a cycle
                        break
                    cycle.append(pieceHash) # Add the piece to the cycle

                cost = eternity_puzzle.get_total_n_conflict(board2)
                if cost < bestNextCost:
                    bestNextCost = cost
                    bestCycle = copy.deepcopy(cycle)
                    bestNextBoard = copy.deepcopy(board2)

                board2 = copy.deepcopy(tempBoard)

            for pieceHash in bestCycle:
                hashes.remove(pieceHash)
            board2 = bestNextBoard

            # print("[ PR2 ] Cycle complete", eternity_puzzle.verify_solution(board2), bestNextCost)
            if bestNextCost < bestCost:
                bestCost = bestNextCost
                bestBoard = copy.deepcopy(board2)
                print(f"[ PR2 ] New best cost: {bestCost}")

            board1, board2 = board2, board1
            positions1, positions2 = positions2, positions1

        return bestBoard

    def hyperSearch(board: List[int], max_iterations: int, plot: bool= False) -> List[int]:
        totalConflicts = eternity_puzzle.get_total_n_conflict(board)
        bestBoard = copy.deepcopy(board)
        bestConflicts = totalConflicts
        
        squareSide = 4
        totalPerfectSquares = 0

        for i in range(eternity_puzzle.board_size - squareSide + 1):
            for j in range(eternity_puzzle.board_size - squareSide + 1):
                position = i + j*eternity_puzzle.board_size
                if isCompleteSquare(board, position, squareSide):
                    totalPerfectSquares += 1
        if plot:
            perfectSquaresHistory = [totalPerfectSquares]
            conflictsHistory = [480-totalConflicts]

        for _ in tqdm(range(max_iterations)):
            
            heuristicSelection = random.randint(0, 4)
            if heuristicSelection == 0:
                move, conflictsDelta = swapAndRotateTwoCorners(board)
            elif heuristicSelection == 1:
                move, conflictsDelta = swapAndRotateTwoEdges(board)
            elif heuristicSelection == 2:
                move, conflictsDelta = swapAndRotateTwoInnerPieces(board)
            elif heuristicSelection == 3:
                move, conflictsDelta = swapOptimallyNonAdjacentBorderPieces(board, eternity_puzzle.board_size)
            elif heuristicSelection == 4:
                move, conflictsDelta = swapOptimallyNonAdjacentInnerPieces(board, int(eternity_puzzle.board_size*3/2))
                
            board, perfectSquaresDelta, accepted = evaluateAndAccept_CompleteSquares(board, move, squareSide)
            if not eternity_puzzle.verify_solution(board):
                print("Invalid board after ", heuristicSelection, "and evaluate")
                break

            totalPerfectSquares += perfectSquaresDelta
            if accepted:
                totalConflicts += conflictsDelta
                if totalConflicts < bestConflicts:
                    bestConflicts = totalConflicts
                    bestBoard = copy.deepcopy(board)
                    print("New best board with", bestConflicts, "conflicts and", totalPerfectSquares, "perfect squares")
                    print()

            if plot:
                perfectSquaresHistory.append(totalPerfectSquares)
                conflictsHistory.append(480-totalConflicts)

        if plot:
            plt.plot(perfectSquaresHistory)
            plt.plot(conflictsHistory)
            plt.legend(["Perfect squares", "Conflicts"])
            plt.show()
    
        return bestBoard

    print("##################")
    print(seed)
    print("##################")
    print()
    
    # pool = ElitePool(10)
    # bestBoard = fullyRandomConstruction()
    # bestConflicts = eternity_puzzle.get_total_n_conflict(bestBoard) 

    # for _ in tqdm(range(1)):
    #     board = scanRowConstructionReverse(10)
    #     conflicts = eternity_puzzle.get_total_n_conflict(board)
    #     if conflicts < bestConflicts:
    #         bestConflicts = conflicts
    #         bestBoard = copy.deepcopy(board)
    #         print(f"[ SCAN ] New best cost: {bestConflicts}")

    # bestBoard = None
    # bestConflicts = float('inf')
    # for _ in tqdm(range(1)):
    #     b1 = scanRowConstruction(10)
    #     print("b1", eternity_puzzle.get_total_n_conflict(b1))
    #     b1 = hyperSearch(b1, 10000)

    #     b2 = scanRowConstructionReverse(10)
    #     print("b2", eternity_puzzle.get_total_n_conflict(b2))
    #     b2 = hyperSearch(b2, 10000)

    #     for _ in range(1000):
    #         board = pathRelinking(b1, b2)
    #         conflicts = eternity_puzzle.get_total_n_conflict(board)
    #         if conflicts < bestConflicts:
    #             bestConflicts = conflicts
    #             bestBoard = copy.deepcopy(board)
    #             print(f"[ PR2 ] New best cost: {bestConflicts}")


    bestBoard = scanRowConstruction(10)
    bestBoard = hyperSearch(bestBoard, 10000)
    bestConflicts = eternity_puzzle.get_total_n_conflict(bestBoard)
    return bestBoard, bestConflicts

    # redo = True
    # while redo:
    #     redo = False
    #     for i in range(len(pool.pool)):
    #         for j in range(i+1, len(pool.pool)):
    #             board = PR1(pool.pool[i], pool.pool[j])
    #             conflicts = eternity_puzzle.get_total_n_conflict(board)
    #             if conflicts < bestConflicts:
    #                 pool.add(copy.deepcopy(board), conflicts)
    #                 redo = True
    #                 bestConflicts = conflicts
    #                 bestBoard = copy.deepcopy(board)
    #                 print(f"[ PR1 ] New best cost: {bestConflicts}")
    #                 break
    #         if redo:
    #             break

    return bestBoard, eternity_puzzle.get_total_n_conflict(bestBoard)
                