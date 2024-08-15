import numpy as np
import matplotlib.pyplot as plt
from Solver import *
import time
from tqdm import tqdm

if __name__=="__main__":
    initial_states = [[
        [6, 4, 2],
        [1, 3, 7],
        [0, 5, 8]
    ],
    [
        [8, 1, 3],
        [5, 7, 4],
        [2, 6, 0]
    ],
    [
        [0, 3, 8],
        [5, 6, 7],
        [1, 4, 2]
    ],




    ]

    goal_state = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])

bfs = []
dfs = []
Astar = []
Astar_2 = []
for state in tqdm(initial_states):
    puzzle = Puzzle(state, goal_state)
    solver = Solver(puzzle)

    start_time = time.time()
    solution = solver.solve_bfs()
    end_time = time.time()
    bfs.append((len(solution),abs(end_time-start_time)))



    puzzle = Puzzle(state, goal_state)
    solver = Solver(puzzle)

    start_time = time.time()
    solution = solver.solve_dfs()
    end_time = time.time()
    dfs.append((len(solution),abs(end_time-start_time)))



    puzzle = Puzzle(state, goal_state)
    solver = Solver(puzzle)

    start_time = time.time()
    solution = solver.solve_astar()
    end_time = time.time()
    Astar.append((len(solution),abs(end_time-start_time)))

    puzzle = Puzzle(state, goal_state)
    solver = Solver(puzzle)

    start_time = time.time()
    solution = solver.solve_astar_eucleadian()
    end_time = time.time()
    Astar_2.append((len(solution), abs(end_time - start_time)))


print(bfs)
print(dfs)
print(Astar)
print(Astar_2)



# Extracting the data for plotting
bfs_lengths, bfs_times = zip(*bfs)
dfs_lengths, dfs_times = zip(*dfs)
astar_lengths, astar_times = zip(*Astar)
astar_lengths_2, astar_times_2 = zip(*Astar_2)
# Number of initial states
N = len(initial_states)

# Setting the positions and width for the bars
ind = np.arange(N)
width = 0.20

# Plotting the data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Top subplot: Solution Lengths with logarithmic scale
ax1.bar(ind - width, bfs_lengths, width, label='BFS')
ax1.bar(ind, dfs_lengths, width, label='DFS')

ax1.bar(ind + width, astar_lengths, width, label='A* Manhaten')
ax1.bar(ind + width*2, astar_lengths_2, width, label='A* Eucleadean')

ax1.set_ylabel('Number of Movements (log scale)')
ax1.set_yscale('log')
ax1.set_title('Number of Movements and Time Taken by BFS, DFS, and A* Algorithms')
ax1.set_xticks(ind)
ax1.set_xticklabels([f'State {i+1}' for i in range(N)])
ax1.legend()

# Bottom subplot: Solution Times
ax2.bar(ind - width, bfs_times, width, label='BFS')
ax2.bar(ind, dfs_times, width, label='DFS')
ax2.bar(ind + width, astar_times, width, label='A* Manhaten')
ax2.bar(ind + width*2, astar_times_2, width, label='A* Eucleadean')

ax2.set_ylabel('Time Taken (s)')
ax2.set_xticks(ind)
ax2.set_xticklabels([f'State {i+1}' for i in range(N)])
ax2.legend()

plt.tight_layout()

# Save the figure
plt.savefig('algorithm_comparison_path.png')

plt.show()