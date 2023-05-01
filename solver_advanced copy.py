from solver_heuristic import *
from typing import List, Tuple
import random
from tqdm import tqdm
from itertools import combinations, permutations
import math
import copy
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt

seed = random.randint(0, 999999999999999999)
# seed = 974870528835079714
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
        print("Adding element with score {}".format(score))
        if len(self.pool) < self.size:
            self.pool.append(copy.deepcopy(element))
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
                self.pool[scoreIndex] = copy.deepcopy(element)
                self.scores[scoreIndex] = score
            elif differences and maxDistance > 0:
                # It replaces the element with the maximum distance
                self.pool[maxIndex] = copy.deepcopy(element)
                self.scores[maxIndex] = score


class Tabu:
    def __init__(self, size: int):
        self.size = size
        self.tabuList = []

    def add(self, element):
        if len(self.tabuList) < self.size:
            self.tabuList.append(copy.deepcopy(element))
        else:
            self.tabuList.pop(0)
            self.tabuList.append(copy.deepcopy(element))

    def __contains__(self, element):
        return element in self.tabuList

def is_corner(piece: Tuple[int, int, int, int]):
    return piece.count(0) == 2

def is_side(piece: Tuple[int, int, int, int]):
    return piece.count(0) == 1

def solve_advanced(eternity_puzzle):
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

    def recursiveBuild(maxTime, positions):
        """
            General recursive function to build the board
            :param maxTime: maximum time to run the algorithm
            :param positions: list of positions to fill
            :return: a board
        """

        beginTime = time.time()
        random.shuffle(remaining_pieces)
        beginTime = time.time()
        board = [None] * eternity_puzzle.n_piece
        board[0] = corner_rotation(fixedCorner, 0)

        def recursion(board: List[Tuple[int, int, int, int]], posIndex: int, pieces: List[Tuple[int, int, int, int]], usedPositions: set[int]):
            if posIndex == len(positions):
                return board
            
            if positions[posIndex] in corner_positions:
                for i in range(len(pieces)):
                    piece = pieces[i]
                    if is_corner(piece):
                        piece = corner_rotation(piece, positions[posIndex])
                        if (time.time() - beginTime > maxTime):
                            board[positions[posIndex]] = piece
                            result = recursion(board, posIndex+1, pieces[:i] + pieces[i+1:], usedPositions | {positions[posIndex]})
                            if result:
                                return result
                        
                        elif positions[posIndex]-1 in usedPositions and piece[2] != board[positions[posIndex]-1][3]:
                            continue
                        elif positions[posIndex]+1 in usedPositions and piece[3] != board[positions[posIndex]+1][2]:
                            continue
                        elif positions[posIndex]-eternity_puzzle.board_size in usedPositions and piece[1] != board[positions[posIndex]-eternity_puzzle.board_size][0]:
                            continue
                        elif positions[posIndex]+eternity_puzzle.board_size in usedPositions and piece[0] != board[positions[posIndex]+eternity_puzzle.board_size][1]:
                            continue
                        
                        board[positions[posIndex]] = piece
                        result = recursion(board, posIndex+1, pieces[:i] + pieces[i+1:], usedPositions | {positions[posIndex]})
                        if result:
                            return result
                        
            elif is_top(positions[posIndex]) or is_bottom(positions[posIndex]) or is_left(positions[posIndex]) or is_right(positions[posIndex]):
                for i in range(len(pieces)):
                    piece = pieces[i]
                    if is_side(piece):
                        piece = edge_rotation(piece, positions[posIndex])
                        if (time.time() - beginTime > maxTime):
                            board[positions[posIndex]] = piece
                            result = recursion(board, posIndex+1, pieces[:i] + pieces[i+1:], usedPositions | {positions[posIndex]})
                            if result:
                                return result
                        
                        elif positions[posIndex]-1 in usedPositions and piece[2] != board[positions[posIndex]-1][3]:
                            continue
                        elif positions[posIndex]+1 in usedPositions and piece[3] != board[positions[posIndex]+1][2]:
                            continue
                        elif positions[posIndex]-eternity_puzzle.board_size in usedPositions and piece[1] != board[positions[posIndex]-eternity_puzzle.board_size][0]:
                            continue
                        elif positions[posIndex]+eternity_puzzle.board_size in usedPositions and piece[0] != board[positions[posIndex]+eternity_puzzle.board_size][1]:
                            continue
                        
                        board[positions[posIndex]] = piece
                        result = recursion(board, posIndex+1, pieces[:i] + pieces[i+1:], usedPositions | {positions[posIndex]})
                        if result:
                            return result
            else:
                for i in range(len(pieces)):
                    piece = pieces[i]
                    if not is_corner(piece) and not is_side(piece):
                        for rotation in eternity_puzzle.generate_rotation(piece):
                            if (time.time() - beginTime > maxTime):
                                board[positions[posIndex]] = piece
                                result = recursion(board, posIndex+1, pieces[:i] + pieces[i+1:], usedPositions | {positions[posIndex]})
                                if result:
                                    return result
                                
                            elif positions[posIndex]-1 in usedPositions and rotation[2] != board[positions[posIndex]-1][3]:
                                continue
                            elif positions[posIndex]+1 in usedPositions and rotation[3] != board[positions[posIndex]+1][2]:
                                continue
                            elif positions[posIndex]-eternity_puzzle.board_size in usedPositions and rotation[1] != board[positions[posIndex]-eternity_puzzle.board_size][0]:
                                continue
                            elif positions[posIndex]+eternity_puzzle.board_size in usedPositions and rotation[0] != board[positions[posIndex]+eternity_puzzle.board_size][1]:
                                continue
                            
                            board[positions[posIndex]] = rotation
                            result = recursion(board, posIndex+1, pieces[:i] + pieces[i+1:], usedPositions | {positions[posIndex]})
                            if result:
                                return result

            return None
        
        return recursion(board, 0, remaining_pieces, {0})
    

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

    def swapOptimallyNonAdjacentBorderPieces(board: List[Tuple[int, int, int, int]], k: int, diversify: bool = False):
        """"
            Select k random non-adjacent border pieces, and swap them optimally.
            Returns a list of moved pieces as tuples (rotated piece, position)
        """
        positions = set(edge_positions + corner_positions)
        if diversify: weights = {pos: getPieceConflicts(board, board[pos], pos) + 1 for pos in positions}
        else: weights = {pos: (10*getPieceConflicts(board, board[pos], pos)) + 1 for pos in positions}

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

    def swapOptimallyNonAdjacentInnerPieces(board: List[Tuple[int, int, int, int]], k: int, diversify: bool = False):
        """"
            Select k random non-adjacent inner pieces, and swap them optimally.
            Returns a list of moved pieces as tuples (rotated piece, position)
        """
        positions = set(internal_positions)
        if diversify: weights = {pos: getPieceConflicts(board, board[pos], pos) + 1 for pos in positions}
        else: weights = {pos: (10*getPieceConflicts(board, board[pos], pos)) + 1 for pos in positions}

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

        # print("2", eternity_puzzle.verify_ solution(board))

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
        return board, delta, False

    def pathRelinking(board1, board2, plot=False):
        """
            Mixed path relinking algorithm
        """
        improved = False

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

        conflicts1History = []
        conflicts2History = []
        if plot:
            conflicts1History.append(cost1)
            conflicts2History.append(cost2)


        print("[pathRelinking] Initial bestCost", bestCost)
            
        random.shuffle(all_positions)
        for n in all_positions:
            if board1[n] != board2[n]:
                pieces1[eternity_puzzle.hash_piece(board1[n])] = n
                pieces2[eternity_puzzle.hash_piece(board2[n])] = n

        parity = True
        steps = 0
        distance = len(pieces1)

        # print("> ", set(pieces1[h] for h in pieces1))
    
        while steps + len(deprecated1) < distance and steps + len(deprecated2) < distance:
            # print("BEFORE : ", eternity_puzzle.get_total_n_conflict(board1))
            bestDelta = 9
            bestMove = None
            if parity:
                startBoard = board2
                endBoard = board1
                deprecated = deprecated1
                startPieces = pieces2
                endPieces = pieces1
            else:
                startBoard = board1
                endBoard = board2
                deprecated = deprecated2
                startPieces = pieces1
                endPieces = pieces2

            for h in endPieces:
                if h in deprecated:
                    continue
                # Find a best move
                endPosition = endPieces[h]
                startPosition = startPieces[h]

                if startBoard[endPosition].count(0):
                    if startBoard[endPosition].count(0) == 2:
                        delta = -getCornerPieceConflicts(startBoard, startBoard[endPosition], endPosition)
                        delta -= getCornerPieceConflicts(startBoard, startBoard[startPosition], startPosition)
                        startRotation = corner_rotation(startBoard[endPosition], startPosition)
                        delta += getCornerPieceConflicts(startBoard, endBoard[endPosition], endPosition)
                        delta += getCornerPieceConflicts(startBoard, startRotation, startPosition)
                        if delta < bestDelta:
                            bestDelta = delta
                            bestMove = ((endBoard[endPosition], endPosition), (startRotation, startPosition))
                            # print("AAA", eternity_puzzle.hash_piece(endBoard[endPosition]) in endPieces)
                            # print("Corner move for a delta of ", bestDelta, "swapping" , startPosition, "with", endPosition)
                    else:
                        delta = -getPieceConflicts(startBoard, startBoard[endPosition], endPosition, otherPositions=[startPosition])
                        delta -= getPieceConflicts(startBoard, startBoard[startPosition], startPosition, otherPositions=[endPosition])
                        startRotation = edge_rotation(startBoard[endPosition], startPosition)
                        startSave, endSave = startBoard[startPosition], startBoard[endPosition]
                        startBoard[startPosition], startBoard[endPosition] = startRotation, endBoard[endPosition]
                        delta += getPieceConflicts(startBoard, startBoard[endPosition], endPosition, otherPositions=[startPosition])
                        delta += getPieceConflicts(startBoard, startBoard[startPosition], startPosition, otherPositions=[endPosition])
                        if delta < bestDelta:
                            bestDelta = delta
                            bestMove = ((startBoard[endPosition], endPosition), (startBoard[startPosition], startPosition))
                            # print("AAA", eternity_puzzle.hash_piece(endBoard[endPosition]) in endPieces)
                            # print("Edge move for a delta of ", bestDelta, "swapping" , startPosition, "with", endPosition)
                        startBoard[startPosition], startBoard[endPosition] = startSave, endSave

                else:
                    
                    negativeDelta = -getPieceConflicts(startBoard, startBoard[endPosition], endPosition, otherPositions=[startPosition])
                    if startPosition != endPosition:
                        negativeDelta -= getPieceConflicts(startBoard, startBoard[startPosition], startPosition, otherPositions=[endPosition])

                    startSave, endSave = startBoard[startPosition], startBoard[endPosition]

                    for startRotation in eternity_puzzle.generate_rotation(startBoard[endPosition]):

                        startBoard[startPosition], startBoard[endPosition] = startRotation, endBoard[endPosition]
                        positiveDelta = getPieceConflicts(startBoard, endBoard[endPosition], endPosition, otherPositions=[startPosition])
                        if startPosition != endPosition:
                            positiveDelta += getPieceConflicts(startBoard, startBoard, startPosition, otherPositions=[endPosition])

                        if negativeDelta + positiveDelta < bestDelta:
                            bestDelta = negativeDelta + positiveDelta
                            bestMove = ((startBoard[endPosition], endPosition), (startBoard[startPosition], startPosition))
                            # print("AAAA", eternity_puzzle.hash_piece(endBoard[endPosition]) in endPieces)
                            # print("Inner move for a delta of ", bestDelta, "swapping" , startPosition, "with", endPosition)
                    startBoard[startPosition], startBoard[endPosition] = startSave, endSave
        
            # Make the best move in board1
            if bestMove:
                (forcedPiece, forcedPosition), (movedPiece, movedPosition) = bestMove
                movedHash = eternity_puzzle.hash_piece(movedPiece)
                forcedHash = eternity_puzzle.hash_piece(forcedPiece)
                if forcedPosition != movedPosition:
                    if parity:
                        board2[forcedPosition], board2[movedPosition] = forcedPiece, movedPiece
                        deprecated2.add(movedHash)
                        if movedHash in pieces2:
                            pieces2[movedHash] = movedPosition

                    else:
                        board1[forcedPosition], board1[movedPosition] = forcedPiece, movedPiece
                        deprecated1.add(movedHash)
                        if movedHash in pieces1:
                            pieces1[movedHash] = movedPosition
                else:
                    if parity:
                        board2[forcedPosition] = forcedPiece
                    else:
                        board1[forcedPosition] = forcedPiece

                # print(forcedHash in pieces1, forcedHash in pieces2)
                del pieces1[forcedHash]
                del pieces2[forcedHash]

                if parity:
                    cost2 += bestDelta
                    if cost2 < bestCost:
                        bestCost = cost2
                        bestBoard = copy.deepcopy(board2)
                        print(f"[pathRelinking] New best cost: {bestCost} / {eternity_puzzle.get_total_n_conflict(bestBoard)}")
                        improved = True
                else:
                    cost1 += bestDelta
                    if cost1 < bestCost:
                        bestCost = cost1
                        bestBoard = copy.deepcopy(board1)
                        print(f"[pathRelinking] New best cost: {bestCost} / {eternity_puzzle.get_total_n_conflict(bestBoard)}")
                        improved = True

            steps += 1
            parity = not parity
            
            if plot:
                conflicts1History.append(cost1)
                conflicts2History.append(cost2)
                

        if plot:
            plt.plot(conflicts1History)
            plt.plot(conflicts2History)
            plt.show()

        return bestBoard, improved

    def PR2(board1, board2, plot=False):
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
        
        conflictsHistory = []
        if plot:
            conflictsHistory.append(cost1)

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

            if plot:
                conflictsHistory.append(cost)

            board1, board2 = copy.deepcopy(board2), copy.deepcopy(board1)
            positions1, positions2 = copy.deepcopy(positions2), copy.deepcopy(positions1)
            cost1, cost2 = cost2, cost1


        if plot:
            plt.plot(conflictsHistory)
            plt.show()

        return bestBoard

    def hyperSearch(board: List[int], max_iterations: int, plot: bool= False, getHistory: bool = False) -> List[int]:
        totalConflicts = eternity_puzzle.get_total_n_conflict(board)
        bestBoard = copy.deepcopy(board)
        bestConflicts = totalConflicts

        if getHistory:
            conflictsHistory = [(0, totalConflicts)]
        
        squareSide = 3
        totalPerfectSquares = 0

        for i in range(eternity_puzzle.board_size - squareSide + 1):
            for j in range(eternity_puzzle.board_size - squareSide + 1):
                position = i + j*eternity_puzzle.board_size
                if isCompleteSquare(board, position, squareSide):
                    totalPerfectSquares += 1
        if plot:
            perfectSquaresHistory = [totalPerfectSquares]
            conflictsHistory = [480-totalConflicts]

        for step in tqdm(range(max_iterations)):
            move = None
            heuristicSelection = 0
            while not move:
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

            if accepted:
                totalPerfectSquares += perfectSquaresDelta
                totalConflicts += conflictsDelta
                if totalConflicts < bestConflicts:
                    bestConflicts = totalConflicts
                    bestBoard = copy.deepcopy(board)
                    print("New best board with", bestConflicts, "conflicts and", totalPerfectSquares, "perfect squares")
                    print()
                    if getHistory:
                        conflictsHistory.append((step+1, bestConflicts))

            if plot:
                perfectSquaresHistory.append(totalPerfectSquares)
                conflictsHistory.append(480-totalConflicts)

        if plot:
            plt.plot(perfectSquaresHistory)
            plt.plot(conflictsHistory)
            plt.legend(["Perfect squares", "Conflicts"])
            plt.show()

        totalConflicts = bestConflicts
        board = copy.deepcopy(bestBoard)
        for step in tqdm(range(max_iterations)):
            move = None
            heuristicSelection = 0
            while not move:
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

            
            if conflictsDelta <= 0:
                totalConflicts += conflictsDelta
                for piece, position in move:
                    board[position] = piece
                if totalConflicts < bestConflicts:
                    bestConflicts = totalConflicts
                    bestBoard = copy.deepcopy(board)
                    print("New best board with", bestConflicts, "conflicts")
                    print()
                    if getHistory:
                        conflictsHistory.append((step+1+max_iterations, totalConflicts))
    
        if getHistory:
            return bestBoard, conflictsHistory

        return bestBoard

    print("##################")
    print(seed)
    print("##################")
    print()

    bottomToTopScanRowPositions = [i for i in range(1, eternity_puzzle.n_piece)] #49

    topToBottomScanRowPositions = [i for i in range(eternity_puzzle.n_piece-1, 0, -1)] #52

    doubleScanRowPositions = [topToBottomScanRowPositions[i//2 + eternity_puzzle.n_piece//2 if i%2 else i//2] for i in range(len(topToBottomScanRowPositions))] #

    spiralPositions = []
    for k in range(eternity_puzzle.board_size//2):
        # Top row
        spiralPositions += [i + k*eternity_puzzle.board_size for i in range(k, eternity_puzzle.board_size-k-1)]
        # Right column
        spiralPositions += [eternity_puzzle.board_size-k-1 + i*eternity_puzzle.board_size for i in range(k, eternity_puzzle.board_size-k-1)]
        # Bottom row
        spiralPositions += [i + (eternity_puzzle.board_size-k-1)*eternity_puzzle.board_size for i in range(eternity_puzzle.board_size-k-1, k, -1)]
        # Left column
        spiralPositions += [k + i*eternity_puzzle.board_size for i in range(eternity_puzzle.board_size-k-1, k, -1)]
    if eternity_puzzle.board_size % 2 == 1:
        # Add center position for odd-sized grid
        spiralPositions.append(eternity_puzzle.board_size//2 + (eternity_puzzle.board_size//2)*eternity_puzzle.board_size)
    spiralPositions = spiralPositions[1:]
    
    reverseSpiralPositions = spiralPositions[::-1]

    doubleSpiralPositions = []
    for i in range(len(spiralPositions)//2):
        doubleSpiralPositions.append(spiralPositions[i])
        doubleSpiralPositions.append(reverseSpiralPositions[i])
    if len(spiralPositions) % 2:
        doubleSpiralPositions.append(spiralPositions[len(spiralPositions)//2])

    
    bestBoard = None
    bestConflicts = 1000000
    
    for i, positions in enumerate([bottomToTopScanRowPositions, topToBottomScanRowPositions, bottomToTopScanRowPositions, topToBottomScanRowPositions, bottomToTopScanRowPositions, topToBottomScanRowPositions]):
        board = recursiveBuild(10, positions)
        # board = fullyRandomConstruction()

        board_no_tabu, conflictsHistory = hyperSearch(copy.deepcopy(board), 20000, plot = False, getHistory = True)
        with open(f"conflictsHistory_no_ILTA_{seed}_{i}.txt", "a+") as f:
            f.write(",".join(str(step) for step, _ in conflictsHistory))
            f.write("\n")
            f.write(",".join(str(conflict) for _, conflict in conflictsHistory))
            f.write("\n")

        board_with_tabu, conflictsHistory = hyperSearch(copy.deepcopy(board), 20000, plot = False, getHistory = True)
        with open(f"conflictsHistory_with_ILTA_{seed}_{i}.txt", "a+") as f:
            f.write(",".join(str(step) for step, _ in conflictsHistory))
            f.write("\n")
            f.write(",".join(str(conflict) for _, conflict in conflictsHistory))
            f.write("\n")

        conflicts = eternity_puzzle.get_total_n_conflict(board_no_tabu)
        if conflicts < bestConflicts:
            bestBoard = copy.deepcopy(board_no_tabu)
            bestConflicts = conflicts

        conflicts = eternity_puzzle.get_total_n_conflict(board_with_tabu)
        if conflicts < bestConflicts:
            bestBoard = copy.deepcopy(board_with_tabu)
            bestConflicts = conflicts

    # COMPARE WITH PATH RELINKING

    # pool = ElitePool(5)
    # for positions in [bottomToTopScanRowPositions, topToBottomScanRowPositions, bottomToTopScanRowPositions, topToBottomScanRowPositions, bottomToTopScanRowPositions, topToBottomScanRowPositions, bottomToTopScanRowPositions, topToBottomScanRowPositions, bottomToTopScanRowPositions, topToBottomScanRowPositions]:
    #     board = recursiveBuild(10, positions)
    #     conflicts = eternity_puzzle.get_total_n_conflict(board)
    #     pool.add(copy.deepcopy(board), conflicts)
    #     if conflicts < bestConflicts:
    #         bestBoard = copy.deepcopy(board)
    #         bestConflicts = conflicts

    # for i in range(len(pool.pool)):
    #     board, conflictsHistory = hyperSearch(copy.deepcopy(pool.pool[i]), 100000, getHistory=True)
    #     conflicts = eternity_puzzle.get_total_n_conflict(board)
    #     if conflicts < bestConflicts:
    #         bestBoard = copy.deepcopy(board)
    #         bestConflicts = conflicts

    #     with open(f"conflictsHistory_beforePathRelinking_{seed}.txt", "a+") as f:
    #         f.write(",".join(str(step) for step, _ in conflictsHistory))
    #         f.write("\n")
    #         f.write(",".join(str(conflict) for _, conflict in conflictsHistory))
    #         f.write("\n")

    # with open(f"eliteScores_{seed}.txt", "a+") as f: # For plotting
    #     redo = True
    #     while redo:
            
    #         f.write(",".join(str(score) for score in sorted(pool.scores)))
    #         f.write("\n")
            
    #         for i in range(len(pool.pool)):
    #             redo = False
    #             for j in range(i+1, len(pool.pool)):
    #                 print("Trying boards", i, "and", j)
    #                 for _ in range(10):
    #                     board, improved = pathRelinking(copy.deepcopy(pool.pool[i]), copy.deepcopy(pool.pool[j]))
    #                     conflicts = eternity_puzzle.get_total_n_conflict(board)
    #                     if improved:
    #                         pool.add(board, conflicts)
    #                         redo = True
    #                         break
    #             if redo:
    #                 break


    # for i in range(len(pool.pool)):
    #     board, conflictsHistory = hyperSearch(copy.deepcopy(pool.pool[i]), 100000, getHistory=True)
    #     conflicts = eternity_puzzle.get_total_n_conflict(board)
    #     if conflicts < bestConflicts:
    #         bestBoard = copy.deepcopy(board)
    #         bestConflicts = conflicts

    #     with open(f"conflictsHistory_afterPathRelinking_{seed}.txt", "a+") as f:
    #         f.write(",".join(str(step) for step, _ in conflictsHistory))
    #         f.write("\n")
    #         f.write(",".join(str(conflict) for _, conflict in conflictsHistory))
    #         f.write("\n")

    return bestBoard, bestConflicts