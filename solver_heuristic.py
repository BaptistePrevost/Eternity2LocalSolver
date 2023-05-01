'''
    Piece are square and have 4 sides, each side is a color. The puzzle is a square of n*n pieces. The pieces are represented as tuples, with the following format:
    (north_side, south_side, west_side, east_side)
    Solution format : list of pieces from bottom-left to top-right

'''
import random
from typing import Tuple, List

def is_corner(piece: Tuple[int, int, int, int]):
    return piece.count(0) == 2

def is_side(piece: Tuple[int, int, int, int]):
    return piece.count(0) == 1

def solve_heuristic(eternity_puzzle):
    """
    Heuristic solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    # bits = [0] + [2**eternity_puzzle - 1 for _ in range(eternity_puzzle.n_color) - 1] # All positions are set to be adjacent to all colors, except for the first one representing the sides
    # # Setup sides
    # for i in range(eternity_puzzle.board_size):
    #     bits[0] ^= 1 << i
    #     bits[0] ^= 1 << (i * eternity_puzzle.board_size)
    #     bits[0] ^= 1 << (i * eternity_puzzle.board_size + eternity_puzzle.board_size - 1)
    #     bits[0] ^= 1 << (i + eternity_puzzle.board_size*(eternity_puzzle.board_size - 1))
    
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

    remaining_pieces = []
    fixedCorner = None
    for piece in eternity_puzzle.piece_list:
        if not fixedCorner and piece.count(0) == 2:
            fixedCorner = piece
        else:
            remaining_pieces.append(piece)
    
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

    topToBottomScanRowPositions = [i for i in range(1, eternity_puzzle.n_piece)]

    bottomToTopScanRowPositions = [i for i in range(eternity_puzzle.n_piece-1, 0, -1)]

    doubleScanRowPositions = [topToBottomScanRowPositions[i//2 + eternity_puzzle.n_piece//2 if i%2 else i//2] for i in range(len(topToBottomScanRowPositions))]

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

    board = recursiveBuild(10, reverseSpiralPositions)

    return board, eternity_puzzle.get_total_n_conflict(board)