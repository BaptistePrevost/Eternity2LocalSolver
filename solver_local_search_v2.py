from solver_heuristic import *
from typing import List, Tuple
import random
from tqdm import tqdm
from itertools import combinations, permutations
import math
import copy

from scipy.optimize import linear_sum_assignment
import numpy as np

seed = random.randint(0, 999999999999999999)
# seed = 368236563069293615
random.seed(seed)

class Tabu:
    def __init__(self, size: int):
        self.size = size
        self.tabu_list = []
        self.tabu_set = set()

    def add(self, element):
        if len(self.tabu_list) >= self.size:
            self.tabu_set.remove(self.tabu_list.pop(0))

        self.tabu_list.append(element)
        self.tabu_set.add(element)

    def __contains__(self, element):
        return element in self.tabu_set
    

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

    def getInternalPieceConflicts(board: List[Tuple[int, int, int, int]], piece: Tuple[int, int, int, int], position: int, ignorePositions: List[int] = []):
        """
        :param board: the board
        :param piece: the piece
        :param position: the position of the piece
        :return: the number of conflicts for the piece at position
        """
        nb_conflicts = 0
        if board[position-1][3] != piece[2]:
            nb_conflicts += 1 if position-1 not in ignorePositions else 0.5

        if board[position+1][2] != piece[3]:
            nb_conflicts += 1 if position+1 not in ignorePositions else 0.5

        if board[position-eternity_puzzle.board_size][0] != piece[1]:
            nb_conflicts += 1 if position-eternity_puzzle.board_size not in ignorePositions else 0.5

        if board[position+eternity_puzzle.board_size][1] != piece[0]:
            nb_conflicts += 1 if position+eternity_puzzle.board_size not in ignorePositions else 0.5

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
    
    def getEdgePieceConflictsWithoutInterior(board: List[Tuple[int, int, int, int]], piece: Tuple[int, int, int, int], position: int, ignorePositions: set[int]=set()):
        """
        :param board: the board
        :param piece: the piece
        :param position: the position of the piece
        :return: the number of conflicts for the piece at position
        """
        nb_conflicts = 0
        if position < eternity_puzzle.board_size: # bottom
            if board[position-1][3] != piece[2]:
                if position-1 not in ignorePositions:
                    nb_conflicts += 1
                else:
                    nb_conflicts += 0.5
            if  board[position+1][2] != piece[3]:
                if position+1 not in ignorePositions:
                    nb_conflicts += 1
                else:
                    nb_conflicts += 0.5

        elif position % eternity_puzzle.board_size == 0: # left
            if board[position-eternity_puzzle.board_size][0] != piece[1]:
                if position-eternity_puzzle.board_size not in ignorePositions:
                    nb_conflicts += 1
                else:
                    nb_conflicts += 0.5
            if board[position+eternity_puzzle.board_size][1] != piece[0]:
                if position+eternity_puzzle.board_size not in ignorePositions:
                    nb_conflicts += 1
                else:
                    nb_conflicts += 0.5
        
        elif position % eternity_puzzle.board_size == eternity_puzzle.board_size-1: # right
            if board[position-eternity_puzzle.board_size][0] != piece[1]:
                if position-eternity_puzzle.board_size not in ignorePositions:
                    nb_conflicts += 1
                else:
                    nb_conflicts += 0.5
            if board[position+eternity_puzzle.board_size][1] != piece[0]:
                if position+eternity_puzzle.board_size not in ignorePositions:
                    nb_conflicts += 1
                else:
                    nb_conflicts += 0.5

        else: # top
            if board[position-1][3] != piece[2]:
                if position-1 not in ignorePositions:
                    nb_conflicts += 1
                else:
                    nb_conflicts += 0.5
            if board[position+1][2] != piece[3]:
                if position+1 not in ignorePositions:
                    nb_conflicts += 1
                else:
                    nb_conflicts += 0.5
        # print("ignorePositions", ignorePositions)
        # print("piece", piece, "position", position, "nb_conflicts", nb_conflicts)
        return nb_conflicts    

    def getAllBorderConflicts(board: List[Tuple[int, int, int, int]]):
        """
            Count all conflicts for the border pieces, using getBorderPieceConflicts for the corners, and getEdgePieceConflictsWithoutInterior for the edges
        """
        nb_conflicts = 0
        for cornerPosition in [0] + corner_positions:
            nb_conflicts += getCornerPieceConflicts(board, board[cornerPosition], cornerPosition)
        for edgePosition in edge_positions:
            nb_conflicts += getEdgePieceConflictsWithoutInterior(board, board[edgePosition], edgePosition)
        
        return nb_conflicts/2



    def nSwapBorderNeighborhood(localBoard: List[Tuple[int, int, int, int]], swapSize: int):
        """
        :param localBoard: the actual board
        :param swapSize: the size of the swap
        :yield: the neighborhood of the solution by swapping swapSize edges, with its associated delta
        """
        N = []
        for cornerIndex1 in range(len(corner_positions)):
            cornerPosition1 = corner_positions[cornerIndex1]
            corner1NegativeDelta = -getCornerPieceConflicts(localBoard, localBoard[cornerPosition1], cornerPosition1)
            for cornerIndex2 in range(cornerIndex1+1, len(corner_positions)):
                cornerPosition2 = corner_positions[cornerIndex2]
                negativeDelta = corner1NegativeDelta - getCornerPieceConflicts(localBoard, localBoard[cornerPosition2], cornerPosition2)
                rotation_1 = corner_rotation(localBoard[cornerPosition1], cornerPosition2)
                positiveDelta = getCornerPieceConflicts(localBoard, rotation_1, cornerPosition2)
                rotation_2 = corner_rotation(localBoard[cornerPosition2], cornerPosition1)
                positiveDelta += getCornerPieceConflicts(localBoard, rotation_2, cornerPosition1)
                delta = positiveDelta + negativeDelta
                # print("positiveDelta: ", positiveDelta, "negativeDelta: ", negativeDelta, "delta: ", delta, "swapSize: ", swapSize, (rotation_1, cornerPosition2), (rotation_2, cornerPosition1))
                N.append([[(rotation_2, cornerPosition1), (rotation_1, cornerPosition2)], delta])

        random.shuffle(edge_positions)
        for initial_positions in combinations(edge_positions, swapSize):
            ignorePositions = set(initial_positions)
            pieces = [localBoard[position] for position in initial_positions]
            negativeDelta = 0
            for i in range(swapSize):
                negativeDelta -= getEdgePieceConflictsWithoutInterior(localBoard, pieces[i], initial_positions[i], ignorePositions=ignorePositions)

            for final_positions in permutations(initial_positions):
                positiveDelta = 0
                for i in range(swapSize):
                    localBoard[final_positions[i]] = edge_rotation(pieces[i], final_positions[i])
                for i in range(swapSize):
                    positiveDelta += getEdgePieceConflictsWithoutInterior(localBoard, localBoard[final_positions[i]], final_positions[i], ignorePositions=ignorePositions)

                
                delta = positiveDelta + negativeDelta
                # print(getAllBorderConflicts(localBoard))
                # print("positiveDelta: ", positiveDelta, "negativeDelta: ", negativeDelta, "delta: ", delta, "swapSize: ", swapSize, "initial_positions: ", initial_positions, "final_positions: ", final_positions)
                N.append([[(localBoard[final_positions[i]], final_positions[i]) for i in range(swapSize)], delta])

            for i in range(swapSize):
                localBoard[initial_positions[i]] = pieces[i]
        
        return N
    
    def nSwapInteriorNeighborhood(localBoard: List[Tuple[int, int, int, int]], swapSize: int, neighborhoodSize: int):
        """
        :param localBoard: the actual board
        :param swapSize: the size of the swap
        :yield: the neighborhood of the solution by swapping swapSize edges, with its associated delta
        """
        N = []
        random.shuffle(internal_positions)
        for initial_positions in combinations(internal_positions[:neighborhoodSize], swapSize):
            ignorePositions = set(initial_positions)
            pieces = [localBoard[position] for position in initial_positions]
            negativeDelta = 0
            for i in range(swapSize):
                negativeDelta -= getInternalPieceConflicts(localBoard, pieces[i], initial_positions[i], ignorePositions=ignorePositions)

            for final_positions in permutations(initial_positions):

                def generate_combinations(lst, variation_func, curr_combination=[], index=0, all_combinations=[]):
                    if index == len(lst):
                        all_combinations.append(curr_combination.copy())
                        # print("adding", curr_combination.copy())
                        return

                    for variation in variation_func(lst[index]):
                        curr_combination.append(variation)
                        generate_combinations(lst, variation_func, curr_combination, index+1, all_combinations)
                        curr_combination.pop()
                    return all_combinations
                
                for rotations in generate_combinations(pieces, eternity_puzzle.generate_rotation):
                    positiveDelta = 0
                    for i in range(swapSize):
                        localBoard[final_positions[i]] = rotations[i]
                    for i in range(swapSize):
                        positiveDelta += getInternalPieceConflicts(localBoard, localBoard[final_positions[i]], final_positions[i], ignorePositions=ignorePositions)

                    N.append([[(localBoard[final_positions[i]], final_positions[i]) for i in range(swapSize)], positiveDelta + negativeDelta])

            for i in range(swapSize):
                localBoard[initial_positions[i]] = pieces[i]

        return N
        
    def fastBorderNeighborhoodSearch(localBoard: List[Tuple[int, int, int, int]], swapSize: int):
        """
        :param solution: a solution
        :return: the 2-opt neighborhood of the solution by swapping edges
        """
        for move, delta in nSwapBorderNeighborhood(localBoard, swapSize):
            if delta < 0:
                return move
        return None
    
    def borderNeighborhoodSearch(localBoard: List[Tuple[int, int, int, int]], swapSize: int):
        """
        :param solution: a solution
        :return: the 2-opt neighborhood of the solution by swapping edges
        """
        bestDelta = 9
        bestMove = None
        for move, delta in nSwapBorderNeighborhood(localBoard, swapSize):
            if delta < bestDelta:
                bestDelta = delta
                bestMove = move
        return bestMove

    def fastInteriorNeighborhoodSearch(localBoard: List[Tuple[int, int, int, int]], swapSize: int):
        """
        :param solution: a solution
        :return: the 2-opt neighborhood of the solution by swapping edges
        """
        for move, delta in nSwapInteriorNeighborhood(localBoard, swapSize, neighborhoodSize=20):
            if delta < 0:
                return move
        return None
    
    def interiorNeighborhoodSearch(localBoard: List[Tuple[int, int, int, int]], swapSize: int):
        """
        :param solution: a solution
        :return: the 2-opt neighborhood of the solution by swapping edges
        """
        bestDelta = 9
        bestMove = None
        for move, delta in nSwapInteriorNeighborhood(localBoard, swapSize, neighborhoodSize=20):
            if delta < bestDelta:
                bestDelta = delta
                bestMove = move
        return bestMove
    
    
    def localSearch(board: List[Tuple[int, int, int, int]], max_iterations: int, neighborhood_selection_function):
        """
            Fast local search algorithm for the border of the board
            First improving neigbor is selected
            During the search, only the conflicts of the border are computed
        """
        localBoard = board
        for _ in tqdm(range(max_iterations)):
            move = neighborhood_selection_function(localBoard, 2)
            if not move:
                return localBoard
        
            for piece, position in move:
                localBoard[position] = piece

        return localBoard
    
    def borderTabuSearch(board: List[Tuple[int, int, int, int]], max_iterations: int, tabu_size: int):
        """
            Fast local search algorithm for the border of the board
            First improving neigbor is selected
            During the search, only the conflicts of the border are computed
        """
        localBoard = board
        borderConflicts = getAllBorderConflicts(localBoard)
        bestBoard = copy.deepcopy(localBoard)
        bestBorderConflicts = borderConflicts
        tabu = {}
        tabuCancellations = 0

        for i in tqdm(range(max_iterations)):
            candidates = nSwapBorderNeighborhood(copy.deepcopy(localBoard), 2)
            # return localBoard
            candidates.sort(key=lambda x: x[1])
            for move, delta in candidates:
                key = "-".join(str(pos) for piece, pos in move)
                if delta < 0 or key not in tabu or tabu[key] < i:
                    for piece, position in move:
                        localBoard[position] = piece
                    tabu[key] = i + tabu_size
                    borderConflicts += delta
                    if borderConflicts < bestBorderConflicts:
                        bestBorderConflicts = borderConflicts
                        bestBoard = copy.deepcopy(localBoard)
                    break
                tabuCancellations += 1
            
            if bestBorderConflicts == 0:
                print(f"[ borderTabuSearch] Border complete in : {i}")
                break

            
        print(f"[ borderTabuSearch] Tabu cancellations: {tabuCancellations}")
        print(f"[ borderTabuSearch] Border conflicts: {borderConflicts}")

        return bestBoard
    
    def interiorTabuSearch(board: List[Tuple[int, int, int, int]], max_iterations: int, tabu_size: int, neighborhood_size: int):
        """
            Fast local search algorithm for the border of the board
            First improving neigbor is selected
            During the search, only the conflicts of the border are computed
        """
        localBoard = board
        tabu = {}
        totalConflicts = eternity_puzzle.get_total_n_conflict(localBoard)
        bestBoard = copy.deepcopy(localBoard)
        bestTotalConflicts = totalConflicts
        tabuCancellations = 0

        for i in tqdm(range(max_iterations)):
            candidates = nSwapInteriorNeighborhood(copy.deepcopy(localBoard), 2, neighborhood_size)
            # return localBoard
            candidates.sort(key=lambda x: x[1])
            for move, delta in candidates:
                key = "-".join(str(pos) for piece, pos in move)
                if delta < 0 or key not in tabu or tabu[key] < i:
                    for piece, position in move:
                        localBoard[position] = piece
                    tabu[key] = i + tabu_size
                    totalConflicts += delta
                    if totalConflicts < bestTotalConflicts:
                        bestTotalConflicts = totalConflicts
                        bestBoard = copy.deepcopy(localBoard)
                    break
                tabuCancellations += 1
            if bestTotalConflicts == 0:
                print(f"[ interiorTabuSearch ] Interior complete in : {i}")
                break

            
        print(f"[ interiorTabuSearch ] Tabu cancellations: {tabuCancellations}")
        print(f"[ interiorTabuSearch ] Total conflicts: {bestTotalConflicts}")

        return bestBoard


    def simulatedAnneallingStep(localBoard: List[Tuple[int, int, int, int]], initialTemperature, neighborhood_size: int):
        """
            Simulated annealling algorithm
        """
        for move, delta in nSwapInteriorNeighborhood(copy.deepcopy(localBoard), 2, neighborhood_size):
            if delta < 0 or random.random() < math.exp(-delta / initialTemperature):
                for piece, position in move:
                    localBoard[position] = piece
                return localBoard
    
        return localBoard
    
    def perturbationStep(localBoard: List[Tuple[int, int, int, int]], gamma: float):
        """
            Selects eternity_puzzle.board_size * eternity_puzzle.board_size * gamma pieces and swaps them randomly
            Half of them from the border, half of them from the interior
        """
        nPieces = int(eternity_puzzle.board_size * eternity_puzzle.board_size * gamma)       
        for n in range(nPieces // 2):
            borderPieces = random.sample(edge_positions, 2)
            localBoard[borderPieces[0]], localBoard[borderPieces[1]] = edge_rotation(localBoard[borderPieces[1]], borderPieces[1]), edge_rotation(localBoard[borderPieces[0]], borderPieces[0])
        for n in range(nPieces // 2):
            interiorPieces = random.sample(internal_positions, 2)
            localBoard[interiorPieces[0]], localBoard[interiorPieces[1]] = random.sample(eternity_puzzle.generate_rotation(localBoard[interiorPieces[1]]), 1)[0], random.sample(eternity_puzzle.generate_rotation(localBoard[interiorPieces[0]]), 1)[0]
        return localBoard
    
    def interiorPerturbation(board: List[Tuple[int, int, int, int]], size: int):
        """
            Selects size pieces and swaps them randomly
        """
        positions = random.sample(internal_positions, size)
        pieces = [random.choice(eternity_puzzle.generate_rotation(board[pos])) for pos in positions]
        random.shuffle(pieces)
        for i in range(size):
            board[positions[i]] = pieces[i]
        return board


    def PR1(board1, board2):
        """
            Mixed path relinking algorithm
        """
        # First, we search among all positions what pieces are different
        # For each position, we store in a dictionary the (key, value) = (hash(piece), [position]), eventually adding the position to the list
        # We also store in a list the positions that are different

        positions1 = {}
        positions2 = {}
        hashes = []

        cost1 = eternity_puzzle.get_total_n_conflict(board1)
        cost2 = eternity_puzzle.get_total_n_conflict(board2)
        bestCost = min(cost1, cost2)
        bestBoard = copy.deepcopy(board1) if cost1 < cost2 else copy.deepcopy(board2)

        for n in all_positions:
            if board1[n] != board2[n]:
                hash1 = eternity_puzzle.hash_piece(board1[n])
                hash2 = eternity_puzzle.hash_piece(board2[n])
                if hash1 not in positions1:
                    positions1[hash1] = [n]
                else:
                    positions1[hash1].append(n)
                if hash2 not in positions2:
                    positions2[hash2] = [n]
                else:
                    positions2[hash2].append(n)

                hashes.append(hash1)

        while hashes:
            # Select a piece from hashes randomly
            pieceHash = hashes.pop(random.randint(0, len(hashes)-1))

            firstHash = pieceHash
            while True:
                positionIn1 = positions1[pieceHash].pop()
                positions2[pieceHash].pop() # useless ?
                pieceHash = eternity_puzzle.hash_piece(board2[positionIn1]) # Store the piece popped in board2
                board2[positionIn1] = board1[positionIn1] # Swap the piece in board2

                if pieceHash == firstHash: # We completed a cycle
                    break
                hashes.remove(pieceHash) # Remove the piece from the list of pieces to be swapped

            print("[ PR1 ] Cycle complete", eternity_puzzle.verify_solution(board2), eternity_puzzle.get_total_n_conflict(board2))
            cost = eternity_puzzle.get_total_n_conflict(board2)
            if cost < bestCost:
                bestCost = cost
                bestBoard = copy.deepcopy(board2)
                print(f"[ PR1 ] New best cost: {bestCost}")

            board1, board2 = board2, board1
            positions1, positions2 = positions2, positions1
        
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

            print("[ PR2 ] Cycle complete", eternity_puzzle.verify_solution(board2), bestNextCost)
            if bestNextCost < bestCost:
                bestCost = bestNextCost
                bestBoard = copy.deepcopy(board2)
                print(f"[ PR2 ] New best cost: {bestCost}")

            board1, board2 = board2, board1
            positions1, positions2 = positions2, positions1

        return bestBoard

    @timer
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
                            delta = -getEdgePieceConflictsWithoutInterior(board1, board1[position1], position1, ignorePositions=[position2])
                            delta -= getEdgePieceConflictsWithoutInterior(board1, board1[position2], position2, ignorePositions=[position1])
                            rotation1 = edge_rotation(board1[position2], position1)
                            save1, save2 = board1[position1], board1[position2] #Here piece1 and piece2 save the state of pieces in board1
                            board1[position1], board1[position2] = rotation1, board2[position2]
                            delta += getEdgePieceConflictsWithoutInterior(board1, rotation1, position1, ignorePositions=[position2]) 
                            delta += getEdgePieceConflictsWithoutInterior(board1, board2[position2], position2, ignorePositions=[position1]) 
                            if delta < bestDelta:
                                bestDelta = delta
                                bestMove = ((position1, rotation1), (position2, board2[position2]))
                                # print("Edge move for a delta of ", bestDelta, "swapping" , position1, "with", position2)
                            board1[position1], board1[position2] = save1, save2

                    else:
                        delta = -getInternalPieceConflicts(board1, board1[position2], position2, ignorePositions=[position1])
                        if position1 != position2:
                            delta -= getInternalPieceConflicts(board1, board1[position1], position1, ignorePositions=[position2])

                        save1, save2 = board1[position1], board1[position2]
                        board1[position2] = board2[position2]
                        
                        if position1 != position2:
                            for rotation1 in eternity_puzzle.generate_rotation(save2):
                                board1[position1] = rotation1
                                conflicts1 = getInternalPieceConflicts(board1, board1[position1], position1, ignorePositions=[position2])
                                conflicts2 = getInternalPieceConflicts(board1, board1[position2], position2, ignorePositions=[position1])
                                if delta + conflicts1 + conflicts2 < bestDelta:
                                    bestDelta = delta + conflicts1 + conflicts2
                                    bestMove = ((position1, rotation1), (position2, board1[position2]))
                                    # print("Internal move for a delta of ", bestDelta, "swapping" , position1, "with", position2)

                        else:
                            conflicts2 = getInternalPieceConflicts(board1, board1[position2], position2, ignorePositions=[position1])
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

            else: # board2 -> board1
                for h in pieces2:
                    if h in deprecated1:
                        continue
                    # Find a best move
                    position1 = pieces1[h]
                    position2 = pieces2[h]
                    
                    if board1[position1].count(0):
                        if board1[position1].count(0) == 2:
                            delta = -getEdgePieceConflictsWithoutInterior(board2, board2[position1], position1, ignorePositions=[position2])
                            delta -= getEdgePieceConflictsWithoutInterior(board2, board2[position2], position2, ignorePositions=[position1])
                            rotation2 = corner_rotation(board2[position1], position2)
                            delta += getEdgePieceConflictsWithoutInterior(board2, board1[position1], position1, ignorePositions=[position2]) 
                            delta += getEdgePieceConflictsWithoutInterior(board2, rotation2, position2, ignorePositions=[position1])
                            if delta < bestDelta:
                                bestDelta = delta
                                bestMove = ((position1, board1[position1]), (position2, rotation2))
                                # print("Corner move for a delta of ", bestDelta, "swapping" , position1, "with", position2)
                        else:
                            delta = -getEdgePieceConflictsWithoutInterior(board2, board2[position1], position1, ignorePositions=[position2])
                            delta -= getEdgePieceConflictsWithoutInterior(board2, board2[position2], position2, ignorePositions=[position1])
                            rotation2 = edge_rotation(board2[position1], position2)
                            save1, save2 = board2[position1], board2[position2] #Here piece1 and piece2 save the state of pieces in board1
                            board2[position1], board2[position2] = board1[position1], rotation2
                            delta += getEdgePieceConflictsWithoutInterior(board2, board2[position1], position1, ignorePositions=[position2]) 
                            delta += getEdgePieceConflictsWithoutInterior(board2, board2[position2], position2, ignorePositions=[position1]) 
                            if delta < bestDelta:
                                bestDelta = delta
                                bestMove = ((position1, board2[position1]), (position2, board2[position2]))
                                # print("Edge move for a delta of ", bestDelta, "swapping" , position1, "with", position2)
                            board2[position1], board2[position2] = save1, save2

                    else:
                        delta = -getInternalPieceConflicts(board2, board2[position1], position2, ignorePositions=[position1])
                        if position1 != position2:
                            delta -= getInternalPieceConflicts(board2, board2[position2], position2, ignorePositions=[position1])

                        save1, save2 = board2[position1], board2[position2]
                        board2[position1] = board1[position1]
                        
                        if position1 != position2:
                            for rotation2 in eternity_puzzle.generate_rotation(save1):
                                board2[position2] = rotation2
                                conflicts1 = getInternalPieceConflicts(board2, board2[position1], position1, ignorePositions=[position2])
                                conflicts2 = getInternalPieceConflicts(board2, board2[position2], position2, ignorePositions=[position1])
                                if delta + conflicts1 + conflicts2 < bestDelta:
                                    bestDelta = delta + conflicts1 + conflicts2
                                    bestMove = ((position1, board2[position1]), (position2, board2[position2]))
                                    # print("Internal move for a delta of ", bestDelta, "swapping" , position1, "with", position2)

                        else:
                            conflicts2 = getInternalPieceConflicts(board2, board2[position1], position1, ignorePositions=[position2])
                            if delta + conflicts2 < bestDelta:
                                bestDelta = delta + conflicts2
                                bestMove = ((position1, board1[position1]), (position2, None))
                                # print("Internal move for a delta of ", bestDelta, "swapping" , position1, "with", position2)

                        board2[position1], board2[position2] = save1, save2
            
                # Make the best move in board1
                if bestMove:
                    # print("BEST MOVE : ", bestMove, "with a delta of ", bestDelta)
                    (position1, piece1), (position2, piece2) = bestMove
                    if position1 != position2:
                        board2[position1], board2[position2] = piece1, piece2
                        deprecated2.add(hash(eternity_puzzle.hash_piece(piece2)))
                        pieces2[hash(eternity_puzzle.hash_piece(piece2))] = position2
                        cost1 += bestDelta
                        # print("AFTER : ", eternity_puzzle.get_total_n_conflict(board1))
                        del pieces1[hash(eternity_puzzle.hash_piece(piece1))]
                        del pieces2[hash(eternity_puzzle.hash_piece(piece1))]
                    else:
                        board2[position1] = piece1
                        cost2 += bestDelta
                        # print("AFTER : ", eternity_puzzle.get_total_n_conflict(board1))
                        # print(bestDelta)
                        del pieces1[hash(eternity_puzzle.hash_piece(piece1))]
                        del pieces2[hash(eternity_puzzle.hash_piece(piece1))]

                    if cost2 < bestCost:
                        bestCost = cost2
                        bestBoard = copy.deepcopy(board2)

            parity = not parity
            steps += 1

        return bestBoard


    def VLNTabuSearch(board: List[Tuple[int, int, int, int]], setSize, tabu1, tabu1Size, tabu2, tabu2Size, stepsForDiversification: int, stepsForIntensification: int):
        """
            Restricted to interior yet
        """
        bestBoard = copy.deepcopy(board)
        bestCost = eternity_puzzle.get_total_n_conflict(board)
        diversificationCountdown = 0
        intensificationCountdown = 0

        pool = ElitePool(10)


        for iter in tqdm(range(400)):

            # Select a subset S of non adjacent of setSize positions not in tabu1
            """
                - Here, most violtaed positions should be preferably chosen
                - After a given number of non improving iterations, diversification should occur : a random swap and rotations of a given number of pieces, and clear tabu1 and tabu2
                - After a greater threshold of non improving iterations, intensification occurs : back to the best solution
            """
            N = []
            for _ in range(1000):
                """
                    Select a non adjacent set of positions
                """
                S = []
                adj = set()
                weights = [getInternalPieceConflicts(board, board[pos], pos) for pos in internal_positions]
                while len(S) < setSize:
                    position = random.choices(internal_positions, weights=weights, k=1)[0]
                    if position not in tabu1 or tabu1[position] < iter:
                        if position not in adj:
                            S.append(position)
                            adj.update({position, position-1, position+1, position-eternity_puzzle.board_size, position+eternity_puzzle.board_size})

                # Strategy 2 : Faire un damier
            
            # checkers = [i for i in range(1, eternity_puzzle.board_size**2-4*eternity_puzzle.board_size-4)]
            # S = None
            # for i in checkers:
            #     S = []
            #     correct = True
            #     for x in range(2):
            #         for y in range(4):
            #             position = i + (2*x + y%2) + y*eternity_puzzle.board_size
            #             if position not in tabu1 or tabu1[position] < iter:
            #                 S.append(position)
            #             else:
            #                 correct = False
                
            #     if not correct:
            #         continue

                # print("S", S)
                # input()

                # Also penilize costs with edges in tabu2
                rotations = [[board[S[i]] for _ in range(setSize)] for i in range(setSize)] 
                costs = np.zeros((setSize,setSize))
                for i, origin in enumerate(S):
                    for j, destination in enumerate(S):
                        if (i,j) in tabu2 and tabu2[(i,j)] > iter:
                            costs[i,j] = -1000
                        else:
                            for rotation in eternity_puzzle.generate_rotation(board[origin]):
                                cost = 4-getInternalPieceConflicts(board, rotation, destination)
                                if cost > costs[i][j]:
                                    costs[i,j] = cost
                                    rotations[i][j] = rotation
                    
                # Solve the assignment problem
                row_ind, col_ind = linear_sum_assignment(costs, maximize=True)

                delta = 0
                for i in range(setSize):
                    delta += getInternalPieceConflicts(board, rotations[row_ind[i]][col_ind[i]], S[col_ind[i]]) - getInternalPieceConflicts(board, board[S[col_ind[i]]], S[col_ind[i]])
                
                N.append((delta, [S[col_ind[i]] for i in range(setSize)], [rotations[row_ind[i]][col_ind[i]] for i in range(setSize)]))

            # Sort N by delta
            N.sort(key=lambda x: x[0])

            # Select the best assignment
            _, positions, pieces = N[0]

            for i in range(setSize):
                board[positions[i]] = pieces[i]
                # print("Setting " + str(S[col_ind[i]]) + " to " + str(rotations[row_ind[i]][col_ind[i]]))

            # Add elemenets of S in tabu1 for tabu1Size iterations
            for position in positions:
                tabu1[position] = iter + tabu1Size
            # Add the reverse arcs of moved pieces to tabu2
            # for i in range(setSize):
            #     tabu2[(col_ind[i], row_ind[i])] = iter + tabu2Size

            cost = eternity_puzzle.get_total_n_conflict(board)
            if cost < bestCost:
                bestCost = cost
                bestBoard = copy.deepcopy(board)
                # pool.add(bestBoard, bestCost)
                # for eliteBoard in pool.pool:
                #     otherBoard = pathRelinking(bestBoard, eliteBoard)
                #     if eternity_puzzle.get_total_n_conflict(otherBoard) < bestCost:
                #         bestCost = eternity_puzzle.get_total_n_conflict(otherBoard)
                #         bestBoard = copy.deepcopy(otherBoard)
                #         print(f"[ VLNTabuSearch ] Path relinking found a better board")

                diversificationCountdown = iter
                intensificationCountdown = iter
                print(f"[ VLNTabuSearch ] Best cost: {bestCost}")

            if iter - diversificationCountdown > stepsForDiversification:
                board = interiorPerturbation(board, 10)
                # tabu1 = {}
                # tabu2 = {}
                diversificationCountdown = iter
                print(f"[ VLNTabuSearch ] Diversification after reaching cost : ", cost)
            elif iter - intensificationCountdown > stepsForIntensification:
                board = copy.deepcopy(bestBoard)
                tabu1 = {}
                tabu2 = {}
                diversificationCountdown = iter
                intensificationCountdown = iter
                print(f"[ VLNTabuSearch ] Intensification")

        return bestBoard

    
    print("##################")
    print(seed)
    print("##################")
    print()
    
    # pool = ElitePool(10)
    # bestBoard = fullyRandomConstruction()
    # bestCost = eternity_puzzle.get_total_n_conflict(bestBoard)
    # pool.add(bestBoard, bestCost)

    # anomaly = 0
    # for i in range(eternity_puzzle.n_piece):
    #     for j in range(i+1, eternity_puzzle.n_piece):
    #         anomaly += hash(eternity_puzzle.hash_piece(bestBoard[i])) == hash(eternity_puzzle.hash_piece(bestBoard[j]))
    # print(f"Anomaly: {anomaly} / {eternity_puzzle.n_piece} ({anomaly/eternity_puzzle.n_piece*100}%)")
    
    # for _ in range(10):
    #     b1 = fullyRandomConstruction()
    #     b2 = fullyRandomConstruction()
    #     pathRelinking(b1, b2)
        # print(f"Path relinking cost: {eternity_puzzle.get_total_n_conflict(pathRelinking(b1, b2))}")


    board = fullyRandomConstruction()
    bestBoard = copy.deepcopy(board)
    bestCost = eternity_puzzle.get_total_n_conflict(board)

    """
        Parameters
    """
    TABU_SIZE = 360
    MAX_BORDER_TABU_ITERATIONS = 400
    NON_IMPROVING_THRESHOLD_FOR_SA = 400
    I3 = 3500
    MAX_INTERIOR_NEIGHBORS = 20
    ALPHA = 0.99
    T0 = 100
    TE = 0
    GAMMA = 0.5

    """
        Search
    """

    pool = ElitePool(10)

    for _ in range(5):
        board = fullyRandomConstruction()
        board = borderTabuSearch(board, MAX_BORDER_TABU_ITERATIONS, TABU_SIZE)
        board = interiorTabuSearch(board, MAX_BORDER_TABU_ITERATIONS, TABU_SIZE, MAX_INTERIOR_NEIGHBORS)
        board = VLNTabuSearch(board, 16, {}, 5, {}, 10, 100, 200)
        cost = eternity_puzzle.get_total_n_conflict(board)

        pool.add(copy.deepcopy(board), cost)

        if cost < bestCost:
            bestCost = cost
            bestBoard = copy.deepcopy(board)
            print(f"[ SA ] New best cost: {bestCost}")

        


    """
        Post-optimization
    """


    # print("Post-optimization")
    # redo = True
    # while redo:
    #     redo = False
    #     for elite1 in tqdm(pool.pool):
    #         for elite2 in pool.pool:
    #             board = PR2(copy.deepcopy(elite1), copy.deepcopy(elite2))
    #             cost = eternity_puzzle.get_total_n_conflict(board)
    #             if cost < bestCost:
    #                 bestCost = cost
    #                 bestBoard = copy.deepcopy(board)
    #                 print(f"(2) New best cost: {bestCost}")
    #                 pool.add(bestBoard, bestCost)
    #                 redo = True

    return bestBoard, eternity_puzzle.get_total_n_conflict(bestBoard)
                