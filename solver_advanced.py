from solver_heuristic import *
from typing import List, Tuple
import random
from tqdm import tqdm
import copy
from scipy.optimize import linear_sum_assignment
import numpy as np


seed = random.randint(0, 999999999999999999)
# seed = 629410397156499646
random.seed(seed)

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
    border_positions = edge_positions + corner_positions
    all_positions = internal_positions + edge_positions + corner_positions

    
    searchBeginTime = time.time()
    searchEndTime = searchBeginTime + 60*57 # 57 minutes, just to be sure

    corners = []
    other_pieces = []
    for piece in eternity_puzzle.piece_list:
        if piece.count(0) == 2:
            corners.append(piece)
        else:
            other_pieces.append(piece)
    
    """
        Construction functions
    """

    def fullyRandomConstruction():
        remaining_pieces = other_pieces + corners[1:]
        random.shuffle(remaining_pieces)
        cornerIndex = 0
        edgeIndex = 0
        internalIndex = 0
        board = [None] * eternity_puzzle.n_piece
        board[0] = corner_rotation(corners[0], 0)
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

    def recursiveBuild(maxTime, positions, fixedCornersPositions: List[int] = []):
        """
            General recursive function to build the board
            :param maxTime: maximum time to run the algorithm
            :param positions: list of positions to fill
            :return: a board
        """
        beginTime = time.time()
        board = [None] * eternity_puzzle.n_piece

        remaining_pieces = None
        if fixedCornersPositions:
            remaining_pieces = other_pieces
            for i in range(len(fixedCornersPositions)):
                board[fixedCornersPositions[i]] = corner_rotation(corners[i+1], fixedCornersPositions[i])
        else:
            remaining_pieces = other_pieces + corners[1:]

        board[0] = corner_rotation(corners[0], 0)

        random.shuffle(remaining_pieces)

        def recursion(board: List[Tuple[int, int, int, int]], posIndex: int, pieces: List[Tuple[int, int, int, int]], usedPositions: set[int]):
            if posIndex == len(positions):
                return board
            
            if positions[posIndex] in usedPositions:
                result = recursion(board, posIndex+1, pieces, usedPositions)
                if result:
                    return result

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

            return None #Proved to be non feasible
        
        return recursion(board, 0, remaining_pieces, {0} | set(fixedCornersPositions))
    

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
        Neighbor functions
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


    """
        Perturbation
    """
    def randomPerturbation(board: List[Tuple[int, int, int, int]], k: int):
        """
            Select k random pieces, and rotate them randomly
        """
        edgesPos = []
        edges = []
        innerPos = []
        inner = []
        positions = random.sample(edge_positions + internal_positions, k)

        for position in positions:
                if is_side(board[position]):
                    edgesPos.append(position)
                    edges.append(board[position])
                else:
                    innerPos.append(position)
                    inner.append(board[position])

        random.shuffle(edgesPos)
        random.shuffle(innerPos)

        for i, position in enumerate(edgesPos):
            board[position] = edge_rotation(edges[i], position)
        
        for i, position in enumerate(innerPos):
            board[position] = random.choice(eternity_puzzle.generate_rotation(inner[i]))

        return board
    

    def squarePerturbation(board: List[Tuple[int, int, int, int]], k: int):
        """
            Select a random square of size kxk, and rotate it randomly
        """
        # bottom left corner of the square
        botLeft = random.randint(0, eternity_puzzle.board_size**2 - k - eternity_puzzle.board_size*(k-1))

        edgesPos = []
        edges = []
        innerPos = []
        inner = []
        for i in range(k):
            for j in range(k):
                position = botLeft+i*eternity_puzzle.board_size+j
                if is_side(board[position]):
                    edgesPos.append(position)
                    edges.append(board[position])
                elif not is_corner(board[position]):
                    innerPos.append(position)
                    inner.append(board[position])

        random.shuffle(edgesPos)
        random.shuffle(innerPos)

        for i, position in enumerate(edgesPos):
            board[position] = edge_rotation(edges[i], position)
        
        for i, position in enumerate(innerPos):
            board[position] = random.choice(eternity_puzzle.generate_rotation(inner[i]))

        return board
    
    def randomSwapCorners(board: List[Tuple[int, int, int, int]]):
        """
            Select two random corners
            Returns a list of moved pieces as tuples (rotated piece, position)
        """

        positions = copy.deepcopy(corner_positions)
        random.shuffle(positions)

        for i in range(len(corner_positions)):
            board[positions[i]] = corner_rotation(board[corner_positions[i]], positions[i])

        return board

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

        for squarePosition in squarePositions:
            if isCompleteSquare(board, squarePosition, squareSide):
                delta -= 1
        
        for piece, pos in move:
            board[pos] = piece


        for squarePosition in squarePositions:
            if isCompleteSquare(board, squarePosition, squareSide):
                delta += 1

        if delta >= 0:
            return board, delta, True
    
        for i, change in enumerate(move):
            board[change[1]] = save[i]

        return board, delta, False

    def surrogateSearch(board: List[int], max_iterations: int, squareSide: int) -> List[int]:

        totalConflicts = eternity_puzzle.get_total_n_conflict(board)
        bestBoard = copy.deepcopy(board)
        bestConflicts = totalConflicts
        initialConflicts = totalConflicts

        print(f"[ surrogateSearch ({squareSide}x{squareSide}) ] Starting with {totalConflicts} conflicts.")

        totalPerfectSquares = 0
        for i in range(eternity_puzzle.board_size - squareSide + 1):
            for j in range(eternity_puzzle.board_size - squareSide + 1):
                position = i + j*eternity_puzzle.board_size
                if isCompleteSquare(board, position, squareSide):
                    totalPerfectSquares += 1

        for step in tqdm(range(max_iterations)):
            neighborhoodSelection = random.randint(0, 4)
            if neighborhoodSelection == 0:
                move, conflictsDelta = swapAndRotateTwoCorners(board)
            elif neighborhoodSelection == 1:
                move, conflictsDelta = swapAndRotateTwoEdges(board)
            elif neighborhoodSelection == 2:
                move, conflictsDelta = swapAndRotateTwoInnerPieces(board)
            elif neighborhoodSelection == 3:
                move, conflictsDelta = swapOptimallyNonAdjacentBorderPieces(board, eternity_puzzle.board_size)
            elif neighborhoodSelection == 4:
                move, conflictsDelta = swapOptimallyNonAdjacentInnerPieces(board, int(eternity_puzzle.board_size*3/2))

            somethingMoved = False
            if conflictsDelta == 0:
                for piece, pos in move:
                    somethingMoved |= board[pos] != piece
                if not somethingMoved:
                    continue

            # Evaluate
            board, perfectSquaresDelta, accepted = evaluateAndAccept_CompleteSquares(board, move, squareSide)

            if accepted:
                totalPerfectSquares += perfectSquaresDelta
                totalConflicts += conflictsDelta
                if totalConflicts < bestConflicts:
                    bestConflicts = totalConflicts
                    print(f"[ surrogateSearch ({squareSide}x{squareSide}) ]New best board with {bestConflicts} conflicts and {totalPerfectSquares} perfect  squares")
                    bestBoard = copy.deepcopy(board)
                    print()
                    
                    if bestConflicts == 0:
                        print("Found solution in", step+1, "iterations.")
                        return bestBoard, (bestConflicts < initialConflicts)

        return bestBoard, (bestConflicts < initialConflicts)

    def localSearch(board: List[Tuple[int, int, int, int]], max_iterations: int):
        
        totalConflicts = eternity_puzzle.get_total_n_conflict(board)
        bestConflicts = totalConflicts
        bestBoard = copy.deepcopy(board)

        for step in tqdm(range(max_iterations)):
            if time.time() > searchEndTime:
                return bestBoard
            neighborhoodSelection = random.randint(0, 4)
            if neighborhoodSelection == 0:
                move, conflictsDelta = swapAndRotateTwoCorners(board)
            elif neighborhoodSelection == 1:
                move, conflictsDelta = swapAndRotateTwoEdges(board)
            elif neighborhoodSelection == 2:
                move, conflictsDelta = swapAndRotateTwoInnerPieces(board)
            elif neighborhoodSelection == 3:
                move, conflictsDelta = swapOptimallyNonAdjacentBorderPieces(board, eternity_puzzle.board_size)
            elif neighborhoodSelection == 4:
                move, conflictsDelta = swapOptimallyNonAdjacentInnerPieces(board, int(eternity_puzzle.board_size*3/2))

            somethingMoved = False
            if conflictsDelta == 0:
                for piece, pos in move:
                    somethingMoved |= board[pos] != piece
                if not somethingMoved:
                    continue

            if conflictsDelta <= 0:
                totalConflicts += conflictsDelta
                for piece, pos in move:
                    board[pos] = piece
                if totalConflicts < bestConflicts:
                    bestConflicts = totalConflicts
                    bestBoard = copy.deepcopy(board)
                    print(f"[ localSearch ] New best board with {bestConflicts} conflicts")
                    print()
                    if bestConflicts == 0:
                        print("Found solution in", step+1, "iterations.")
                        return bestBoard
                totalConflicts = eternity_puzzle.get_total_n_conflict(board)

        return bestBoard

    def largeSearch(board: List[Tuple[int, int, int, int]], max_iterations: int):
        squareSide = max(2,eternity_puzzle.board_size // 5)
        justGotBetter = True
        bestBoard = copy.deepcopy(board)

        for _ in range(10):
            if time.time() > searchEndTime:
                return bestBoard, bestConflicts
            print(f"[ completeSearch ] Trying {squareSide}x{squareSide} squares")
            bestConflictsBoard, improved = surrogateSearch(board, max_iterations, squareSide)

            if improved:
                bestBoard = copy.deepcopy(bestConflictsBoard)
                if squareSide < eternity_puzzle.board_size:
                    squareSide += 1
                    justGotBetter = True
                else:
                    break
            else:
                if justGotBetter and squareSide > 3:
                    squareSide -= 2
                elif squareSide > 2:
                    squareSide -= 1
                else:
                    squareSide = max(2,eternity_puzzle.board_size // 5)

                justGotBetter = False

            board = copy.deepcopy(bestConflictsBoard)

        return bestBoard

    print("##################")
    print(seed)
    print("##################")
    print()

    topToBottomScanRowPositions = [i for i in range(eternity_puzzle.n_piece-1, 0, -1)]
    
    bestBoard = None
    bestConflicts = 1000000

    import itertools
    corner_permutations = list(itertools.permutations(corner_positions))
    scores = [0 for _ in range(len(corner_permutations))]
    calls = [1 for _ in range(len(corner_permutations))]

    for i in range(len(corner_permutations)):
        board = recursiveBuild(20, topToBottomScanRowPositions, corner_permutations[i])
        if board:
            board = localSearch(board, 25000)
            conflicts = eternity_puzzle.get_total_n_conflict(board)
            scores[i] += 480-conflicts
            if conflicts < bestConflicts:
                bestConflicts = conflicts
                bestBoard = copy.deepcopy(board)
                print(f"[ main ] New best board with {bestConflicts} conflicts")
                print()
                if bestConflicts == 0:
                    print("Found solution in", i+1, "iterations.")
                    return bestBoard, bestConflicts

    for restartIndex in range(20):
        if time.time() > searchEndTime:
            return bestBoard, bestConflicts
        print(f"###############################\n###############################\nRestart {restartIndex}\n###############################\n###############################\n")
        if random.random() < 0.4:
            permutationIndex = max([i for i in range(len(corner_permutations))], key=lambda x: scores[x]/calls[x])
        else:
            permutationIndex = random.randint(0, len(corner_permutations)-1)

        calls[permutationIndex] += 1
        
        initialBoard = recursiveBuild(20, topToBottomScanRowPositions, corner_permutations[permutationIndex])
        if not initialBoard:
            continue

        board = largeSearch(initialBoard, 15000)
        board = localSearch(board, 250000)

        conflicts = eternity_puzzle.get_total_n_conflict(board)

        scores[permutationIndex] += 480-conflicts

        if conflicts < bestConflicts:
            bestConflicts = conflicts
            bestBoard = copy.deepcopy(board)
            print("[ main ] New best board with", bestConflicts, "conflicts")
            print()
            if bestConflicts == 0:
                return bestBoard, bestConflicts

    return bestBoard, bestConflicts