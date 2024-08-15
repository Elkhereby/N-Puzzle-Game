import pygame
import numpy as np
import random
import time
from collections import deque
from Solver import Node, Puzzle, Solver

class NPuzzleGame:
    def __init__(self, initial_state, goal_state, size, tile_size, margin):
        self.initial_state = np.array(initial_state)
        self.goal_state = np.array(goal_state)
        self.current_state = self.initial_state.copy()#current value of board
        self.size = size# number of tiles in each row/col
        self.tile_size = tile_size
        self.margin = margin
        self.screen_size = size * tile_size + (size + 1) * margin
        self.puzzle = Puzzle(self.current_state.copy(), self.goal_state)
        self.solver = Solver(self.puzzle)# initilize solver
        self.moves = 0# done by user
        self.solve_moves = 0# done by algorithm
        self.solved = False
        self.solution_path = deque()# dequeu of solution returned
        self.solution_index = 0
        self.solver_type = "BFS"  # Default solver type
        self.time_taken = 0

    def shuffle(self):
        self.current_state = self.initial_state.copy()
        for _ in range(1000):
            neighbors = self.puzzle.get_neighbors(Node(self.current_state))
            self.current_state = random.choice(neighbors).board

        self.puzzle.initial_state = self.current_state.copy()
        self.reset_values()
        print("Shuffled to:\n", self.current_state)
    def reset_values(self):
        self.moves = 0
        self.solve_moves = 0
        self.solved = False
        self.solution_path.clear()
        self.solution_index = 0
    def reset(self):
        self.current_state = self.initial_state.copy()
        self.reset_values()
        print("Reset to:\n", self.current_state)

    def solve(self):
        if not self.puzzle.is_solvable():
            print("This puzzle is not solvable.")
            return

        self.puzzle.initial_state = self.current_state.copy()
        #print(self.puzzle.initial_state)
        self.solver = Solver(self.puzzle)
        try:
            start_time = time.time()
            if self.solver_type == "BFS":
                self.solution_path = deque(self.solver.solve_bfs())
            elif self.solver_type == "DFS":

                self.solution_path = self.solver.solve_dfs_limited()
                if not self.solution_path:
                    print("hereeeee")
                    self.solution_path = self.solver.solve_dfs_limited(limit=50)
                if self.solution_path:
                    self.solution_path = deque(self.solution_path)
                else:
                    self.solution_path = None
            elif self.solver_type == "A*":
                self.solution_path = deque(self.solver.solve_astar())
            end_time = time.time()
        except Exception as e:
            print(e)
            self.solution_path.clear()
            self.solve_moves = 0
            self.moves = 0
            print("No Solution Found")

        self.solution_index = 0
        if self.solution_path:
            print(f"Solution found using {self.solver_type}!")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            self.time_taken = float(end_time-start_time)
        else:
            print("No solution found!")

    def move(self, pos):
        zero_pos = np.argwhere(self.current_state == 0)[0]
        if pos[0] == zero_pos[0] and abs(pos[1] - zero_pos[1]) == 1:
            self._swap(zero_pos, pos)
        elif pos[1] == zero_pos[1] and abs(pos[0] - zero_pos[0]) == 1:
            self._swap(zero_pos, pos)
        self.moves += 1
        if np.array_equal(self.current_state, self.goal_state):
            self.solved = True
        print(f"Moved: {self.moves}, Current state:\n{self.current_state}")

    def _swap(self, pos1, pos2):
        self.current_state[pos1[0], pos1[1]], self.current_state[pos2[0], pos2[1]] = self.current_state[
            pos2[0], pos2[1]], self.current_state[pos1[0], pos1[1]]

    def draw_board(self, screen, font):
        start_x = (screen.get_width() - self.screen_size) // 2
        start_y = (screen.get_height() - self.screen_size) // 2 - 50
        for i in range(self.size):
            for j in range(self.size):
                if self.current_state[i, j] != 0:
                    pygame.draw.rect(screen, (0, 128, 255), (start_x + j * self.tile_size + (j + 1) * self.margin,
                                                             start_y + i * self.tile_size + (i + 1) * self.margin,
                                                             self.tile_size, self.tile_size))
                    text = font.render(str(self.current_state[i, j]), True, (255, 255, 255))
                    screen.blit(text, (start_x + j * self.tile_size + (j + 1) * self.margin + 30,
                                       start_y + i * self.tile_size + (i + 1) * self.margin + 30))

    def draw_moves(self, screen, font, screen_width):
        moves_text = font.render(f"Moves: {self.moves}", True, (0, 0, 0))
        solve_moves_text = font.render(f"Solve Moves: {self.solve_moves}", True, (0, 0, 0))
        screen.blit(moves_text, (screen_width // 2 - 330, screen.get_height() - 50))
        screen.blit(solve_moves_text, (screen_width // 2 - 100, screen.get_height() - 50))

    def get_tile_position(self, mouse_pos, screen):
        x, y = mouse_pos
        start_x = (screen.get_width() - self.screen_size) // 2
        start_y = (screen.get_height() - self.screen_size) // 2 - 50
        col = (x - start_x) // (self.tile_size + self.margin)
        row = (y - start_y) // (self.tile_size + self.margin)
        if 0 <= col < self.size and 0 <= row < self.size:
            tile_rect = pygame.Rect(
                start_x + col * self.tile_size + (col + 1) * self.margin,
                start_y + row * self.tile_size + (row + 1) * self.margin,
                self.tile_size, self.tile_size
            )
            return row, col, tile_rect
        return None

    def animate_shuffle(self, screen, font):
        for _ in range(100):
            neighbors = self.puzzle.get_neighbors(Node(self.current_state))
            self.current_state = random.choice(neighbors).board
            screen.fill((255, 255, 255))  # Clear the screen before drawing
            self.draw_board(screen, font)
            pygame.display.flip()
            pygame.time.delay(20)
        self.moves = 0
        self.solve_moves = 0
        print("Shuffle animation completed, Current state:\n", self.current_state)

    def animate_solve(self, screen, font):
        print("Animating solution...")
        if self.solution_path:
            while self.solution_index < len(self.solution_path):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

                self.current_state = self.solution_path[self.solution_index]

                self.solution_index += 1
                self.solve_moves +=1

                screen.fill((255, 255, 255))  # Clear the screen before drawing
                self.draw_board(screen, font)
                pygame.display.flip()
                pygame.time.delay(200)

        print("Solution animation completed, Current state:\n", self.current_state)

    def step_forward(self, screen, font):
        if self.solution_index < len(self.solution_path):
            self.current_state = self.solution_path[self.solution_index]
            self.solution_index += 1
            self.solve_moves += 1
            screen.fill((255, 255, 255))  # Clear the screen before drawing
            self.draw_board(screen, font)
            pygame.display.flip()

    def step_backward(self, screen, font):
        if self.solution_index > 0:
            self.solution_index -= 1
            self.current_state = self.solution_path[self.solution_index]
            self.solve_moves -= 1
            screen.fill((255, 255, 255))  # Clear the screen before drawing
            self.draw_board(screen, font)
            pygame.display.flip()


class NpuzzleUI:
    def __init__(self):
        pygame.init()
        self.size = 3  # Default puzzle size (3x3)
        self.tile_size = 100
        self.margin = 5
        self.screen_width = 1000
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("N-Puzzle Solver")
        self.font = pygame.font.Font(None, 40)

        # Load icons
        self.back_icon = pygame.image.load('left.png')
        self.forward_icon = pygame.image.load('right.png')
        icon_size = (50, 50)
        self.back_icon = pygame.transform.scale(self.back_icon, icon_size)
        self.forward_icon = pygame.transform.scale(self.forward_icon, icon_size)

        initial_state = [
            [1, 2, 3],
            [4, 0, 5],
            [7, 8, 6]
        ]

        goal_state = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ]

        self.game = NPuzzleGame(initial_state, goal_state, self.size, self.tile_size, self.margin)
        self.game.shuffle()

        self.solution_queue = None
        self.solving = False

    def open_initial_state_input(self):
        editing = True
        input_box = pygame.Rect(0, 0, 50, 50)
        color_inactive = pygame.Color('lightskyblue3')
        color_active = pygame.Color('dodgerblue2')
        color = color_inactive
        active = False
        text = ''
        last_click_time = 0
        double_click_delay = 500  # 500 ms for double-click detection
        row, col = None, None

        while editing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    current_time = pygame.time.get_ticks()
                    pos = self.game.get_tile_position(event.pos, self.screen)
                    if pos:
                        row, col, tile_rect = pos
                        if current_time - last_click_time < double_click_delay:
                            input_box.topleft = tile_rect.topleft
                            active = True
                            color = color_active
                        else:
                            active = False
                            color = color_inactive
                        last_click_time = current_time
                if event.type == pygame.KEYDOWN:
                    if active:
                        if event.key == pygame.K_RETURN:
                            try:
                                value = int(text)
                                if 0 <= value < self.game.size * self.game.size:
                                    self.game.current_state[row, col] = value
                                    self.solving = False
                            except ValueError:
                                pass
                            text = ''
                            active = False
                            color = color_inactive
                        elif event.key == pygame.K_BACKSPACE:
                            text = text[:-1]
                        else:
                            text += event.unicode
                    elif event.key == pygame.K_ESCAPE:
                        self.game.moves = 0
                        self.game.solve_moves = 0
                        editing = False

            self.screen.fill((255, 255, 255))
            self.game.draw_board(self.screen, self.font)

            if active:
                txt_surface = self.font.render(text, True, color)
                width = max(50, txt_surface.get_width() + 10)
                input_box.w = width
                self.screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
                pygame.draw.rect(self.screen, color, input_box, 2)

            pygame.display.flip()
            pygame.time.Clock().tick(30)

    def draw_selector(self, selected_solver):
        bfs_text = self.font.render("BFS", True, (255, 255, 255))
        dfs_text = self.font.render("DFS", True, (255, 255, 255))
        astar_text = self.font.render("A*", True, (255, 255, 255))
        time_text = self.font.render(f"Time: {self.game.time_taken:.2f} seconds", True,
                                (0, 0, 0))  # Render the time taken as text

        pygame.draw.rect(self.screen, (0, 128, 0) if selected_solver == "BFS" else (128, 128, 128),
                         (self.screen_width // 2 - 480, 5, 100, 40))
        pygame.draw.rect(self.screen, (0, 128, 0) if selected_solver == "DFS" else (128, 128, 128),
                         (self.screen_width // 2 - 350, 5, 100, 40))
        pygame.draw.rect(self.screen, (0, 128, 0) if selected_solver == "A*" else (128, 128, 128),
                         (self.screen_width // 2 - 220, 5, 100, 40))

        self.screen.blit(bfs_text, (self.screen_width // 2 - 455, 5 + 5))
        self.screen.blit(dfs_text, (self.screen_width // 2 - 325, 5 + 5))
        self.screen.blit(astar_text, (self.screen_width // 2 - 195, 5 + 5))
        self.screen.blit(time_text,(self.screen_width - 350, 5 + 5))

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = self.game.get_tile_position(event.pos, self.screen)
                    if pos:
                        self.game.move(pos[:2])
                    else:
                        x, y = event.pos
                        if self.screen_height - 100 < y < self.screen_height:
                            if self.screen_width // 2 - 430 < x < self.screen_width // 2 - 230:
                                self.game.animate_shuffle(self.screen, self.font)
                                self.solving = False
                            elif self.screen_width // 2 - 200 < x < self.screen_width // 2:
                                if not self.solving:
                                    self.game.solve()
                                    self.game.animate_solve(self.screen,self.font)
                                    self.solving = True
                            elif self.screen_width // 2 + 30 < x < self.screen_width // 2 + 230:
                                self.game.reset()
                                self.solving = False
                            elif self.screen_width // 2 + 250 < x < self.screen_width // 2 + 450:
                                self.open_initial_state_input()
                        elif self.screen_width - 60 < x < self.screen_width and self.screen_height // 2 - 25 < y < self.screen_height // 2 + 25:
                            self.game.step_forward(self.screen, self.font)
                        elif 0 < x < 50 and self.screen_height // 2 - 25 < y < self.screen_height // 2 + 25:
                            self.game.step_backward(self.screen, self.font)
                        elif self.screen_width // 2 - 480 < x < self.screen_width // 2 - 380 and 5 < y < 45:
                            self.game.solver_type = "BFS"
                            self.solving = False
                            self.game.solve_moves = 0
                        elif self.screen_width // 2 - 350 < x < self.screen_width // 2 - 250 and 5 < y < 45:
                            self.game.solver_type = "DFS"
                            self.solving = False
                            self.game.solve_moves = 0
                        elif self.screen_width // 2 - 220 < x < self.screen_width // 2 - 120 and 5 < y < 45:
                            self.game.solver_type = "A*"
                            self.solving = False
                            self.game.solve_moves = 0

            self.screen.fill((255, 255, 255))  # Clear the screen before drawing
            self.game.draw_board(self.screen, self.font)

            # Draw the icons
            self.screen.blit(self.back_icon, (10, self.screen_height // 2 - 25))
            self.screen.blit(self.forward_icon, (self.screen_width - 60, self.screen_height // 2 - 25))

            self.game.draw_moves(self.screen, self.font, self.screen_width)

            # Draw the buttons
            button_y = self.screen_height - 90
            pygame.draw.rect(self.screen, (0, 128, 0), (self.screen_width // 2 - 430, button_y, 200, 40))
            pygame.draw.rect(self.screen, (128, 0, 0), (self.screen_width // 2 - 200, button_y, 200, 40))
            pygame.draw.rect(self.screen, (0, 0, 128), (self.screen_width // 2 + 30, button_y, 200, 40))
            pygame.draw.rect(self.screen, (128, 128, 0), (self.screen_width // 2 + 250, button_y, 200, 40))

            shuffle_text = self.font.render("Shuffle", True, (255, 255, 255))
            solve_text = self.font.render("Solve", True, (255, 255, 255))
            reset_text = self.font.render("Reset", True, (255, 255, 255))
            custom_text = self.font.render("Custom Init", True, (255, 255, 255))

            self.screen.blit(shuffle_text, (self.screen_width // 2 - 400, button_y + 5))
            self.screen.blit(solve_text, (self.screen_width // 2 - 170, button_y + 5))
            self.screen.blit(reset_text, (self.screen_width // 2 + 60, button_y + 5))
            self.screen.blit(custom_text, (self.screen_width // 2 + 280, button_y + 5))

            self.draw_selector(self.game.solver_type)

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    npuzzle_ui = NpuzzleUI()
    npuzzle_ui.run()
