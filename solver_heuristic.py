'''
    Piece are square and have 4 sides, each side is a color. The puzzle is a square of n*n pieces. The pieces are represented as tuples, with the following format:
    (north_side, south_side, west_side, east_side)
    Solution format : list of pieces from bottom-left to top-right

'''

import copy
from typing import Tuple
import random


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


class Solution:
    def __init__(self, eternity_puzzle):
        self.board = [(-1, -1, -1, -1) for _ in range(eternity_puzzle.n_piece)]
        self.remaining_pieces = copy.deepcopy(eternity_puzzle.piece_list)

        self.internal_positions = [pos for pos in range(eternity_puzzle.board_size + 1, eternity_puzzle.n_piece - eternity_puzzle.board_size - 1) if pos % eternity_puzzle.board_size != 0 and pos % eternity_puzzle.board_size != eternity_puzzle.board_size - 1 and pos // eternity_puzzle.board_size != 0 and pos // eternity_puzzle.board_size != eternity_puzzle.board_size - 1]
        self.edge_positions = [pos for pos in range(1, eternity_puzzle.board_size-1)] + [pos*eternity_puzzle.board_size for pos in range(1, eternity_puzzle.board_size-1)] + [pos + eternity_puzzle.board_size*(eternity_puzzle.board_size-1) for pos in range(1, eternity_puzzle.board_size-1)] + [pos*eternity_puzzle.board_size + eternity_puzzle.board_size-1 for pos in range(1, eternity_puzzle.board_size-1)]
        self.corner_positions = [eternity_puzzle.board_size - 1, eternity_puzzle.n_piece - eternity_puzzle.board_size, eternity_puzzle.n_piece - 1]


def GreedyRandomizedCompletion(eternity_puzzle, solution):
    # Complete the solution with its remaining pieces

        random.shuffle(solution.remaining_pieces)

        def get_nb_conflicts(piece: Tuple[int, int, int, int], position: int, solution: list):
            nb_conflicts = 0
            if piece.count(0):
                if piece[0] == 0 and position // eternity_puzzle.board_size == 0:
                    nb_conflicts -= 4
                if piece[1] == 0 and position // eternity_puzzle.board_size == eternity_puzzle.board_size - 1:
                    nb_conflicts -= 4
                if piece[2] == 0 and position % eternity_puzzle.board_size == 0:
                    nb_conflicts -= 4
                if piece[3] == 0 and position % eternity_puzzle.board_size == eternity_puzzle.board_size - 1:
                    nb_conflicts -= 4

            if position % eternity_puzzle.board_size != 0:
                if solution[position - 1][3] != -1 and piece[2] != solution[position - 1][3]:
                    nb_conflicts += 1
            if position % eternity_puzzle.board_size != eternity_puzzle.board_size - 1:
                if solution[position + 1][2] != -1 and piece[3] != solution[position + 1][2]:
                    nb_conflicts += 1
            if position // eternity_puzzle.board_size != 0:
                if solution[position - eternity_puzzle.board_size][1] != -1 and piece[0] != solution[position - eternity_puzzle.board_size][1]:
                    nb_conflicts += 1
            if position // eternity_puzzle.board_size != eternity_puzzle.board_size - 1:
                if solution[position + eternity_puzzle.board_size][0] != -1 and piece[1] != solution[position + eternity_puzzle.board_size][0]:
                    nb_conflicts += 1
            return nb_conflicts
        
        def rotations_according_to_position(piece: Tuple[int, int, int, int], position: int):
            if not piece.count(0):
                return eternity_puzzle.generate_rotation(piece)
            elif is_side(piece):
                for rotation in eternity_puzzle.generate_rotation(piece):
                    if position % eternity_puzzle.board_size == 0:
                        if rotation[2] == 0:
                            return [rotation]
                    elif position % eternity_puzzle.board_size == eternity_puzzle.board_size - 1:
                        if rotation[3] == 0:
                            return [rotation]
                    elif position // eternity_puzzle.board_size == 0:
                        if rotation[1] == 0:
                            return [rotation]
                    elif position // eternity_puzzle.board_size == eternity_puzzle.board_size - 1:
                        if rotation[0] == 0:
                            return [rotation]
            elif is_corner(piece):
                for rotation in eternity_puzzle.generate_rotation(piece):
                    if position == 0:
                        if rotation[1] == 0 and rotation[2] == 0:
                            return [rotation]
                    elif position == eternity_puzzle.board_size - 1:
                        if rotation[1] == 0 and rotation[3] == 0:
                            return [rotation]
                    elif position == eternity_puzzle.board_size*(eternity_puzzle.board_size - 1):
                        if rotation[0] == 0 and rotation[2] == 0:
                            return [rotation]
                    elif position == eternity_puzzle.board_size*eternity_puzzle.board_size - 1:
                        if rotation[0] == 0 and rotation[3] == 0:
                            return [rotation]
                        


        for i in range(len(solution.remaining_pieces)):
            best_conflicts = 5
            best_position = -1
            best_rotation = None

            if not solution.remaining_pieces[i].count(0):
                positions = solution.internal_positions
            elif solution.remaining_pieces[i].count(0) == 1:
                positions = solution.edge_positions
            else:
                positions = solution.corner_positions

            random.shuffle(positions)
            
            for j in range(len(positions)):
                if solution.board[positions[j]] != (-1, -1, -1, -1):
                    continue
                for rotation in rotations_according_to_position(solution.remaining_pieces[i], positions[j]):
                    c = get_nb_conflicts(rotation, positions[j], solution.board)
                    # print(f"\tPiece {remaining_pieces[i]} at position {j} with conflicts {c}")
                
                    if c < best_conflicts:
                        best_conflicts = c
                        best_position = j
                        best_rotation = rotation
                        if c == 0:
                            break
                
                if best_conflicts == 0:
                    break

            solution.board[positions[best_position]] = best_rotation
            positions.pop(best_position)

def GreedyRandomizedConstruction(eternity_puzzle):
    solution = Solution(eternity_puzzle)

    '''
        First, fix the first piece in the bottom left corner
    '''
    
    corner = None
    for i in range(len(solution.remaining_pieces)):
        if is_corner(solution.remaining_pieces[i]):
            corner = solution.remaining_pieces.pop(i)
            break
    
    for rotation in eternity_puzzle.generate_rotation(corner):
        if rotation[1] == 0 and rotation[2] == 0:
            corner = rotation
            break
    
    solution.board[0] = corner

    GreedyRandomizedCompletion(eternity_puzzle, solution)
    return solution

@timer
def solve_heuristic(eternity_puzzle):
    """
    Heuristic solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    # TODO implement here your solution

    

    # bits = [0] + [2**eternity_puzzle - 1 for _ in range(eternity_puzzle.n_color) - 1] # All positions are set to be adjacent to all colors, except for the first one representing the sides
    # # Setup sides
    # for i in range(eternity_puzzle.board_size):
    #     bits[0] ^= 1 << i
    #     bits[0] ^= 1 << (i * eternity_puzzle.board_size)
    #     bits[0] ^= 1 << (i * eternity_puzzle.board_size + eternity_puzzle.board_size - 1)
    #     bits[0] ^= 1 << (i + eternity_puzzle.board_size*(eternity_puzzle.board_size - 1))

    solution = GreedyRandomizedConstruction(eternity_puzzle)

    return solution.board, eternity_puzzle.get_total_n_conflict(solution.board)
