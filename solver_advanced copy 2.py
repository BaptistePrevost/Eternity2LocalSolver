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
        try:
            nb_conflicts = 0
            if piece[2] and board[position-1][3] != piece[2]:
                nb_conflicts += 1 if position-1 not in otherPositions else 0.5

            if piece[3] and board[position+1][2] != piece[3]:
                nb_conflicts += 1 if position+1 not in otherPositions else 0.5

            if piece[1] and board[position-eternity_puzzle.board_size][0] != piece[1]:
                nb_conflicts += 1 if position-eternity_puzzle.board_size not in otherPositions else 0.5

            if piece[0] and board[position+eternity_puzzle.board_size][1] != piece[0]:
                nb_conflicts += 1 if position+eternity_puzzle.board_size not in otherPositions else 0.5
        except:
            print(piece)
            print(position)
            print(otherPositions)
            raise

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
        Destroy operators

        Each of them returns positions of pieces to move
    """
    def fullRandomTwoCorners(board: List[Tuple[int, int, int, int]]):
        return random.sample(corner_positions, 2)

    def fullRandomTwoEdges(board: List[Tuple[int, int, int, int]]):
        return random.sample(edge_positions, 2)

    def fullRandomTwoInnerPieces(board: List[Tuple[int, int, int, int]]):
        return random.sample(internal_positions, 2)
    
    def rouletteRandomTwoCorners(board: List[Tuple[int, int, int, int]]):
        weights = [10*getCornerPieceConflicts(board, board[position], position) + 1 for position in corner_positions]
        return random.choices(corner_positions, k=2, weights=weights)
    
    def rouletteRandomTwoEdges(board: List[Tuple[int, int, int, int]]):
        weights = [10*getPieceConflicts(board, board[position], position) + 1 for position in edge_positions]
        return random.choices(edge_positions, k=2, weights=weights)
    
    def rouletteRandomTwoInnerPieces(board: List[Tuple[int, int, int, int]]):
        weights = [10*getInnerPieceConflicts(board, board[position], position) + 1 for position in internal_positions]
        return random.choices(internal_positions, k=2, weights=weights)
    
    def fullRandomNonAdjacentBorderPieces(board: List[Tuple[int, int, int, int]]):
        """
        :return: 16 non adjacent border positions selected uniformly at random
        """
        toSelect = 16

        positions = set(edge_positions + corner_positions)
        
        S = []
        for _ in range(toSelect):
            if not positions:
                break
            positionsList = list(positions)
            pos = random.sample(positionsList, 1)[0]
            positions.remove(pos)
            S.append(pos)

            # Update positions, carefully handle 0
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
        
        return S
    
    def rouletteRandomNonAdjacentBorderPieces(board: List[Tuple[int, int, int, int]]):
        """
        :return: 16 non adjacent border positions selected uniformly at random
        """
        toSelect = 16

        positions = set(edge_positions + corner_positions)
        weights = {pos: (10*getPieceConflicts(board, board[pos], pos)) + 1 for pos in positions}
        
        S = []
        for _ in range(toSelect):
            if not positions:
                break
            positionsList = list(positions)
            weightsList = [weights[pos] for pos in positionsList]

            pos = random.choices(positionsList, weights=weightsList)[0]
            positions.remove(pos)
            S.append(pos)

            # Update positions, carefully handle 0
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
        
        return S
    
    def fullRandomNonAdjacentInnerPieces(board: List[Tuple[int, int, int, int]]):
        """"
            Select 24 random non-adjacent inner pieces, and swap them optimally.
        """
        toSelect = 24

        positions = set(internal_positions)

        S = []
        for _ in range(toSelect):
            if not positions:
                break
            positionsList = list(positions)
            pos = random.sample(positionsList, 1)[0]
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
    
        return S

    def rouletteRandomNonAdjacentInnerPieces(board: List[Tuple[int, int, int, int]]):
        """"
            Select 24 random non-adjacent inner pieces, and swap them optimally.
        """
        toSelect = 24

        positions = set(internal_positions)
        weights = {pos: (10*getPieceConflicts(board, board[pos], pos)) + 1 for pos in positions}

        S = []
        for _ in range(toSelect):
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
    
        return S
    

    """
        Repair methods
    """

    def randomRepair(board, positions):
        """
            Randomly repair the given positions
        """

        if len(positions) > 2:
            return optimalRepair(board, positions)

        pieces = [board[pos] for pos in positions]
        saves = copy.deepcopy(pieces)
        delta = 0

        corners = []
        cornersPos = []
        edges = []
        edgesPos = []
        internals = []
        internalsPos = []
        for i, pos in enumerate(positions):
            delta -= getPieceConflicts(board, board[pos], pos, otherPositions=positions)
            if pieces[i].count(0) == 2:
                corners.append(pieces[i])
                cornersPos.append(pos)
            elif pieces[i].count(0) == 1:
                edges.append(pieces[i])
                edgesPos.append(pos)
            else:
                internals.append(pieces[i])
                internalsPos.append(pos)

        random.shuffle(corners)
        random.shuffle(edges)
        random.shuffle(internals)

        for i, pos in enumerate(cornersPos):
            board[pos] = corner_rotation(corners[i], pos)
            pieces[i] = board[pos]
        for i, pos in enumerate(edgesPos):
            board[pos] = edge_rotation(edges[i], pos)
            pieces[i] = board[pos]
        for i, pos in enumerate(internalsPos):
            board[pos] = random.choice(eternity_puzzle.generate_rotation(internals[i]))
            pieces[i] = board[pos]
        
        for i, pos in enumerate(positions):
            delta += getPieceConflicts(board, board[pos], pos, otherPositions=positions)

        for i, piece in enumerate(saves):
            board[positions[i]] = piece

        return [(pieces[i], positions[i]) for i in range(len(pieces))], delta
    
    def optimalRepair(board, positions):
        """
            Optimal repair of the given positions
        """

        if len(positions) == 2:
            saves = copy.deepcopy([board[pos] for pos in positions])
            bestDelta = 9
            negativeDelta = 0
            for pos in positions:
                negativeDelta -= getPieceConflicts(board, board[pos], pos, otherPositions=positions)
            
            if board[positions[0]].count(0) == 2:
                rotation1 = corner_rotation(board[positions[0]], positions[0])
                rotation2 = corner_rotation(board[positions[1]], positions[1])
                board[positions[0]] = rotation1
                board[positions[1]] = rotation2
                positiveDelta = getCornerPieceConflicts(board, board[positions[0]], positions[0]) + getCornerPieceConflicts(board, board[positions[1]], positions[1])
                if negativeDelta + positiveDelta < bestDelta:
                    res = [(rotation1, positions[0]), (rotation2, positions[1])]
                    bestDelta = negativeDelta + positiveDelta

                rotation1 = corner_rotation(board[positions[0]], positions[1])
                rotation2 = corner_rotation(board[positions[1]], positions[0])
                board[positions[0]] = rotation2
                board[positions[1]] = rotation1
                positiveDelta = getCornerPieceConflicts(board, board[positions[0]], positions[0]) + getCornerPieceConflicts(board, board[positions[1]], positions[1])
                if negativeDelta + positiveDelta < bestDelta:
                    res = [(rotation2, positions[0]), (rotation1, positions[1])]
                    bestDelta = negativeDelta + positiveDelta
            elif board[positions[0]].count(0) == 1:
                rotation1 = edge_rotation(board[positions[0]], positions[0])
                rotation2 = edge_rotation(board[positions[1]], positions[1])
                board[positions[0]] = rotation1
                board[positions[1]] = rotation2
                positiveDelta = getPieceConflicts(board, board[positions[0]], positions[0], otherPositions=positions) + getPieceConflicts(board, board[positions[1]], positions[1], otherPositions=positions)
                if negativeDelta + positiveDelta < bestDelta:
                    res = [(rotation1, positions[0]), (rotation2, positions[1])]
                    bestDelta = negativeDelta + positiveDelta

                rotation1 = edge_rotation(board[positions[0]], positions[1])
                rotation2 = edge_rotation(board[positions[1]], positions[0])
                board[positions[0]] = rotation2
                board[positions[1]] = rotation1
                positiveDelta = getPieceConflicts(board, board[positions[0]], positions[0], otherPositions=positions) + getPieceConflicts(board, board[positions[1]], positions[1], otherPositions=positions)
                if negativeDelta + positiveDelta < bestDelta:
                    res = [(rotation2, positions[0]), (rotation1, positions[1])]
                    bestDelta = negativeDelta + positiveDelta

            else:
                for rotation1 in eternity_puzzle.generate_rotation(board[positions[0]]):
                    for rotation2 in eternity_puzzle.generate_rotation(board[positions[1]]):
                        board[positions[0]] = rotation1
                        board[positions[1]] = rotation2
                        positiveDelta = getInnerPieceConflicts(board, board[positions[0]], positions[0], otherPositions=positions) + getInnerPieceConflicts(board, board[positions[1]], positions[1], otherPositions=positions)
                        if negativeDelta + positiveDelta < bestDelta:
                            res = [(rotation1, positions[0]), (rotation2, positions[1])]
                            bestDelta = negativeDelta + positiveDelta

                        board[positions[0]] = rotation2
                        board[positions[1]] = rotation1
                        positiveDelta = getInnerPieceConflicts(board, board[positions[0]], positions[0], otherPositions=positions) + getInnerPieceConflicts(board, board[positions[1]], positions[1], otherPositions=positions)
                        if negativeDelta + positiveDelta < bestDelta:
                            res = [(rotation2, positions[0]), (rotation1, positions[1])]
                            bestDelta = negativeDelta + positiveDelta
            
            for i, piece in enumerate(saves):
                board[positions[i]] = piece

            return res, bestDelta

        elif board[positions[0]].count(0):
            k = len(positions)

            rotations = [[board[positions[i]] for _ in range(k)] for i in range(k)] 
            costs = np.zeros((k,k))
            for i, origin in enumerate(positions):
                for j, destination in enumerate(positions):
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
                delta += getPieceConflicts(board, rotations[row_ind[i]][col_ind[i]], positions[col_ind[i]]) - getPieceConflicts(board, board[positions[col_ind[i]]], positions[col_ind[i]])
            
            return [(rotations[row_ind[i]][col_ind[i]], positions[col_ind[i]]) for i in range(k)], delta
    
        else:
            k = len(positions)

            rotations = [[board[positions[i]] for _ in range(k)] for i in range(k)] 
            costs = np.zeros((k,k))
            for i, origin in enumerate(positions):
                for j, destination in enumerate(positions):
                    for rotation in eternity_puzzle.generate_rotation(board[origin]):
                        cost = 4-getInnerPieceConflicts(board, rotation, destination)
                        if cost > costs[i][j]:
                            costs[i,j] = cost
                            rotations[i][j] = rotation
                
            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(costs, maximize=True)

            delta = 0
            for i in range(k):
                delta += getPieceConflicts(board, rotations[row_ind[i]][col_ind[i]], positions[col_ind[i]]) - getPieceConflicts(board, board[positions[col_ind[i]]], positions[col_ind[i]])

            return [(rotations[row_ind[i]][col_ind[i]], positions[col_ind[i]]) for i in range(k)], delta

    """
        Acceptance criteria
    """

    # def 


    """
        Guiding heuristics
    """

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

    def hyperSearch(board: List[int], max_iterations: int, plot: bool= False, getHistory: bool = False, ALNS: bool = False) -> List[int]:
        totalConflicts = eternity_puzzle.get_total_n_conflict(board)
        bestBoard = copy.deepcopy(board)
        bestConflicts = totalConflicts

        if getHistory:
            conflictsHistory = [(0, totalConflicts)]
        
        squareSide = 3
        totalPerfectSquares = 0

        destroyFunctions = [
                # fullRandomTwoCorners,
                # fullRandomTwoEdges,
                # fullRandomTwoInnerPieces,
                # rouletteRandomTwoCorners,
                # rouletteRandomTwoEdges,
                # rouletteRandomTwoInnerPieces,
                # fullRandomNonAdjacentBorderPieces,
                # fullRandomNonAdjacentInnerPieces,
                rouletteRandomNonAdjacentBorderPieces,
                rouletteRandomNonAdjacentInnerPieces
            ]
        repairFunctions = [
                # randomRepair,
                optimalRepair
            ]
        destroyWeights = [1 for i in range(len(destroyFunctions))]
        repairWeights = [1 for i in range(len(repairFunctions))]
        scores = [
            5, # new best
            2, # better
            1, # accepted
            0.5 # rejected
        ]
        decay = 0.8

        for i in range(eternity_puzzle.board_size - squareSide + 1):
            for j in range(eternity_puzzle.board_size - squareSide + 1):
                position = i + j*eternity_puzzle.board_size
                if isCompleteSquare(board, position, squareSide):
                    totalPerfectSquares += 1
        if plot:
            perfectSquaresHistory = [totalPerfectSquares]
            conflictsHistory = [480-totalConflicts]

        for step in tqdm(range(max_iterations)):

            # Destroy
            destroyIndex = random.choices([i for i in range(len(destroyFunctions))], weights=destroyWeights)[0]
            repairIndex = random.choices([i for i in range(len(repairFunctions))], weights=repairWeights)[0]

            # print("Destroy:", destroyFunctions[destroyIndex].__name__, "Repair:", repairFunctions[repairIndex].__name__)

            positions = destroyFunctions[destroyIndex](board)
            move, conflictsDelta = repairFunctions[repairIndex](board, positions)

            # Evaluate
            board, perfectSquaresDelta, accepted = evaluateAndAccept_CompleteSquares(board, move, squareSide)

            if ALNS:
                if totalConflicts + conflictsDelta < bestConflicts:
                    destroyWeights[destroyIndex] = decay*destroyWeights[destroyIndex] + (1-decay)*scores[0]
                    repairWeights[repairIndex] = decay*repairWeights[repairIndex] + (1-decay)*scores[0]
                elif conflictsDelta < 0:
                    destroyWeights[destroyIndex] = decay*destroyWeights[destroyIndex] + (1-decay)*scores[1]
                    repairWeights[repairIndex] = decay*repairWeights[repairIndex] + (1-decay)*scores[1]
                elif conflictsDelta == 0:
                    destroyWeights[destroyIndex] = decay*destroyWeights[destroyIndex] + (1-decay)*scores[2]
                    repairWeights[repairIndex] = decay*repairWeights[repairIndex] + (1-decay)*scores[2]
                else:
                    destroyWeights[destroyIndex] = decay*destroyWeights[destroyIndex] + (1-decay)*scores[3]
                    repairWeights[repairIndex] = decay*repairWeights[repairIndex] + (1-decay)*scores[3]

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
        
        destroyWeights = [1 for i in range(10)]
        repairWeights = [1 for i in range(2)]

        for step in tqdm(range(max_iterations)):
            
            destroyIndex = random.choices([i for i in range(len(destroyFunctions))], weights=destroyWeights)[0]
            repairIndex = random.choices([i for i in range(len(repairFunctions))], weights=repairWeights)[0]

            positions = destroyFunctions[destroyIndex](board)
            move, conflictsDelta = repairFunctions[repairIndex](board, positions)

            if ALNS:
                if totalConflicts + conflictsDelta < bestConflicts:
                    destroyWeights[destroyIndex] = decay*destroyWeights[destroyIndex] + (1-decay)*scores[0]
                    repairWeights[repairIndex] = decay*repairWeights[repairIndex] + (1-decay)*scores[0]
                elif conflictsDelta < 0:
                    destroyWeights[destroyIndex] = decay*destroyWeights[destroyIndex] + (1-decay)*scores[1]
                    repairWeights[repairIndex] = decay*repairWeights[repairIndex] + (1-decay)*scores[1]
                elif conflictsDelta == 0:
                    destroyWeights[destroyIndex] = decay*destroyWeights[destroyIndex] + (1-decay)*scores[2]
                    repairWeights[repairIndex] = decay*repairWeights[repairIndex] + (1-decay)*scores[2]
                else:
                    destroyWeights[destroyIndex] = decay*destroyWeights[destroyIndex] + (1-decay)*scores[3]
                    repairWeights[repairIndex] = decay*repairWeights[repairIndex] + (1-decay)*scores[3]

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
        with open(f"conflictsHistory_no_ALNS_{seed}_{i}.txt", "a+") as f:
            f.write(",".join(str(step) for step, _ in conflictsHistory))
            f.write("\n")
            f.write(",".join(str(conflict) for _, conflict in conflictsHistory))
            f.write("\n")

        board_with_tabu, conflictsHistory = hyperSearch(copy.deepcopy(board), 20000, plot = False, getHistory = True, ALNS = True)
        with open(f"conflictsHistory_with_ALNS_{seed}_{i}.txt", "a+") as f:
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

    return bestBoard, bestConflicts