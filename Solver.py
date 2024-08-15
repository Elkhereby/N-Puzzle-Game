import numpy as np
import heapq
from collections import deque
import time
class Node:# Node representation as an object
    def __init__(self, board, parent=None, move=None, g=0, h=0,depth=0):
        self.board = board# current state
        self.parent = parent
        self.move = move
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost to goal
        self.f = g + h  # Total cost
        self.depth = depth#depth limit only used in DLS

    def __eq__(self, other):#operator overloading in equality and comparison and indexing
        return np.array_equal(self.board, other.board)

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.board.tostring())



class Puzzle:
    def __init__(self, initial_state, goal_state):# initilize puzzle class
        self.initial_state = np.array(initial_state)
        self.goal_state = np.array(goal_state)
        self.size = self.initial_state.shape[0]
        self.goal_positions = self.calculate_goal_positions()

    def calculate_goal_positions(self):
        positions = {}
        for r in range(self.goal_state.shape[0]):
            for c in range(self.goal_state.shape[1]):
                positions[self.goal_state[r, c]] = (r, c)
        #print(f"psoitions =  {positions}")
        return positions

    def get_neighbors(self, node):#possible moves
        neighbors = []
        zero_pos = np.argwhere(node.board == 0)[0]#position of zero tile
        possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

        for move in possible_moves:
            new_pos = zero_pos + move
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:#in board borders
                new_board = node.board.copy()
                new_board[zero_pos[0], zero_pos[1]], new_board[new_pos[0], new_pos[1]] = new_board[new_pos[0], new_pos[1]], new_board[zero_pos[0], zero_pos[1]]#replace
                neighbors.append(Node(new_board, node, move))

        return neighbors

    def heuristic(self, board):#manhatain heurastic calculate total distance of displaced tiles
        total_distance = 0
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] != 0:  # Assuming 0 is the blank tile
                    goal_r, goal_c = self.goal_positions[board[r, c]]
                    total_distance += abs(r - goal_r) + abs(c - goal_c)#calculate absoulute diffrence between goal and current row,col
        return total_distance

    def heuristic_2(self, board):
        total_distance = 0
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] != 0:  # Assuming 0 is the blank tile
                    goal_r, goal_c = self.goal_positions[board[r, c]]
                    total_distance += np.sqrt((r - goal_r) ** 2 + (c - goal_c) ** 2)
        return total_distance

    def is_solved(self, board):
        return np.array_equal(board, self.goal_state)

    def getInvCount(self, flat_board):
        inv_count = 0
        empty_value = 0  # Assuming 0 is used for the empty tile
        for i in range(len(flat_board)):
            for j in range(i + 1, len(flat_board)):
                if flat_board[j] != empty_value and flat_board[i] != empty_value and flat_board[i] > flat_board[j]:# if board[i] >board[j]
                    inv_count += 1
        return inv_count

    def is_solvable(self):
        flat_board = self.initial_state.flatten()
        inv_count = self.getInvCount(flat_board)

        return inv_count % 2 == 0# if inv_number is even then true

class Solver:
    def __init__(self, puzzle):
        self.puzzle = puzzle#

    def solve_bfs(self):
        initial_node = Node(self.puzzle.initial_state)
        if self.puzzle.is_solved(initial_node.board):
            return self.reconstruct_path(initial_node)

        frontier = deque([initial_node])# queue holding values
        explored = set()# visted states unique so no repeated states
        explored.add(initial_node)

        while frontier:# while quue still not empty
            current_node = frontier.popleft()
            if self.puzzle.is_solved(current_node.board):
                return self.reconstruct_path(current_node)

            for neighbor in self.puzzle.get_neighbors(current_node):# explre neighbors
                if neighbor not in explored:
                    frontier.append(neighbor)# add to frontier list
                    explored.add(neighbor)# add to explored

        return None  # No solution found

    def solve_dfs(self):
        initial_node = Node(self.puzzle.initial_state)
        if self.puzzle.is_solved(initial_node.board):
            return self.reconstruct_path(initial_node)

        frontier = [initial_node]# stack
        explored = set()
        explored.add(initial_node)

        while frontier:
            current_node = frontier.pop()# pop top element
            if self.puzzle.is_solved(current_node.board):#   if solved
                return self.reconstruct_path(current_node)# return path

            for neighbor in self.puzzle.get_neighbors(current_node):# check neighbors
                if neighbor not in explored:
                    frontier.append(neighbor)
                    explored.add(neighbor)

        return None  # No solution found

    def solve_dfs_limited(self,limit=32):
        c = 0
        initial_node = Node(self.puzzle.initial_state)
        if self.puzzle.is_solved(initial_node.board):
            return self.reconstruct_path(initial_node)

        frontier = [initial_node]
        explored = set()
        explored.add(initial_node)

        while frontier:
            c+=1
            current_node = frontier.pop()
            if self.puzzle.is_solved(current_node.board):
                print("c = ",c)
                return self.reconstruct_path(current_node)
            if current_node.depth>=limit:
                continue


            for neighbor in self.puzzle.get_neighbors(current_node):
                if neighbor not in explored:
                    neighbor.depth=current_node.depth+1
                    frontier.append(neighbor)
                    explored.add(neighbor)
        print("c = ", c)
        return None  # No solution found

    def solve_astar(self):
        initial_node = Node(self.puzzle.initial_state, h=self.puzzle.heuristic(self.puzzle.initial_state))
        open_set = []
        heapq.heappush(open_set, initial_node)# initilize openset as a heap queue(priority queue)
        closed_set = set()#e=visited

        while open_set:
            current_node = heapq.heappop(open_set)

            if self.puzzle.is_solved(current_node.board):
                return self.reconstruct_path(current_node)

            closed_set.add(current_node)

            for neighbor in self.puzzle.get_neighbors(current_node):
                if neighbor in closed_set:
                    continue

                neighbor.g = current_node.g + 1
                neighbor.h = self.puzzle.heuristic(neighbor.board)# calculate heurastic
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)
                else:# update weights (not needeed)
                    for open_node in open_set:#update weight if cost is less
                        if neighbor == open_node and neighbor.g < open_node.g:
                            open_node.g = neighbor.g
                            open_node.f = neighbor.f
                            open_node.parent = current_node
                            break

        return None  # No solution found

    def solve_astar_eucleadian(self):
        initial_node = Node(self.puzzle.initial_state, h=self.puzzle.heuristic_2(self.puzzle.initial_state))
        open_set = []
        heapq.heappush(open_set, initial_node)
        closed_set = set()

        while open_set:
            current_node = heapq.heappop(open_set)

            if self.puzzle.is_solved(current_node.board):
                return self.reconstruct_path(current_node)

            closed_set.add(current_node)

            for neighbor in self.puzzle.get_neighbors(current_node):
                if neighbor in closed_set:
                    continue

                neighbor.g = current_node.g + 1
                neighbor.h = self.puzzle.heuristic_2(neighbor.board)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)
                else:
                    for open_node in open_set:
                        if neighbor == open_node and neighbor.g < open_node.g:
                            open_node.g = neighbor.g
                            open_node.f = neighbor.f
                            open_node.parent = current_node
                            break

        return None  # No solution found

    def reconstruct_path(self, node):# reconstruct path
        path = []
        while node:# keep track of each node parent

            path.append(node.board)
            node = node.parent

        return path[::-1]

if __name__ == "__main__":
    initial_state = [
        [0, 3, 8],
        [5, 6, 7],
        [1, 4, 2]
    ]

    goal_state = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]

    puzzle = Puzzle(initial_state, goal_state)
    solver = Solver(puzzle)



    print("\nA* Solution:")
    st = time.time()
    solution = solver.solve_astar()
    end = time.time()
    print(end-st)
    if solution:
        print(len(solution))
    else:
        print("No solution found.")
