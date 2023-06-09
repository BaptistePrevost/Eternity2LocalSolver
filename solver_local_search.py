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
# seed = 771096499388344588
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

    def swapOptimallyNonAdjacentBorderPieces(board: List[Tuple[int, int, int, int]], k: int):
        """"
            Select k random non-adjacent border pieces, and swap them optimally.
            Returns a list of moved pieces as tuples (rotated piece, position)
        """
        positions = set(edge_positions + corner_positions)
        weights = {pos: 10*getPieceConflicts(board, board[pos], pos) + 1 for pos in positions}

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
        weights = {pos: 10*getPieceConflicts(board, board[pos], pos) + 1 for pos in positions}

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

    def localSearch(board: List[int], max_iterations: int, plot: bool= False) -> List[int]:
        totalConflicts = eternity_puzzle.get_total_n_conflict(board)
        bestBoard = copy.deepcopy(board)
        bestConflicts = totalConflicts
        
        if plot:
            conflictsHistory = [480-totalConflicts]

        maxBeforeRestart = 5000
        for restart in range(5):
            lastImprovementStep = 0
            lastImprovementPeriod = 0
            for step in tqdm(range(max_iterations)):
                
                if random.randint(0, 1):
                    move, conflictsDelta = swapOptimallyNonAdjacentBorderPieces(board, eternity_puzzle.board_size)
                else:
                    move, conflictsDelta = swapOptimallyNonAdjacentInnerPieces(board, int(eternity_puzzle.board_size*3/2))

                if conflictsDelta <= 0:
                    for piece, pos in move:
                        board[pos] = piece
                    totalConflicts += conflictsDelta
                    if conflictsDelta < 0:
                        lastImprovementPeriod = step - lastImprovementPeriod
                        lastImprovementStep = step
                        if totalConflicts < bestConflicts:
                            bestBoard = copy.deepcopy(board)
                            bestConflicts = totalConflicts
                        print("New best board with", totalConflicts, "conflicts")
                        print()

                if step - lastImprovementStep > maxBeforeRestart and step - lastImprovementStep > 2*lastImprovementPeriod:
                    lastImprovementStep = 0
                    lastImprovementPeriod = 0
                    board = recursiveBuild(10, topToBottomScanRowPositions)
                    totalConflicts = eternity_puzzle.get_total_n_conflict(board)
                    break
            
                if plot:
                    conflictsHistory.append(480-bestConflicts)

        if plot:
            plt.plot(conflictsHistory)
            plt.legend(["Conflicts"])
            plt.show()
    
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

    bestBoard = recursiveBuild(10, topToBottomScanRowPositions)
    bestBoard = localSearch(bestBoard, 20000)
    bestConflicts = eternity_puzzle.get_total_n_conflict(bestBoard)
    return bestBoard, bestConflicts