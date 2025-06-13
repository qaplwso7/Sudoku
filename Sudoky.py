import numpy as np
import pygame
import random
from typing import List, Tuple, Optional
import time
import threading

# Настройки PyGame
pygame.init()
WIDTH, HEIGHT = 550, 800
CELL_SIZE = WIDTH // 9
DARK_BG = (30, 30, 40)          # Темный фон (основной фон интерфейса)
WHITE = (255, 255, 255)         # Белый
BLACK = (0, 0, 0)               # Черный
TURQUOISE = (64, 224, 208)      # Бирюзовый
GRAY = (100, 100, 110)          # Серый
BLUE = (0, 120, 215)            # Синий
RED = (255, 80, 80)             # Красный
GREEN = (80, 200, 80)           # Зеленый
YELLOW = (255, 215, 0)          # Желтый
PURPLE = (180, 0, 180)          # Фиолетовый
ORANGE = (255, 140, 0)          # Оранжевый
LIGHT_GREEN = (144, 238, 144)   # Светло-зеленый
BRIGHT_PINK = (255, 0, 128)     # Ярко-розовый
SUDOKU_BG = (40, 40, 50)        # Фон игрового поля судоку (немного светлее основного фона)
GRID_LINE_COLOR = (80, 80, 90)  # Цвет линий сетки судоку

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sudoku")
font = pygame.font.SysFont("Arial", 40)
small_font = pygame.font.SysFont("Arial", 24)
timer_font = pygame.font.SysFont("Arial", 20)


class SudokuGame:

    def __init__(self):
        self.grid: np.ndarray = np.zeros((9, 9), dtype=int)  # Текущее состояние игрового поля
        self.initial_grid: np.ndarray = np.zeros((9, 9), dtype=int)  # Начальное состояние
        self.selected: Optional[Tuple[int, int]] = None  # Координаты выбранной клетки (строка, столбец) или None
        self.solution_count: int = 0  # Количество найденных решений для текущего судоку
        self.error_cells: List[Tuple[int, int]] = []  # Список клеток с ошибками
        self.max_solutions: int = 2000  # Лимит решений для предотвращения зависания
        self.solution_limit_reached: bool = False  # Флаг: достигнут ли лимит решений (max_solutions)
        self.message: str = ""  # Текст сообщения для игрока
        self.message_timer: float = 0  # Таймер отображения сообщения
        self.active_button: Optional[str] = None  # Текущая активная кнопка ("solve", "new" и т.д.)
        self.searching_solutions: bool = False  # Флаг: выполняется ли поиск решений (для анимации)
        self.animation_phase: int = 0  # Текущая фаза анимации (например, вращение индикатора)
        self.last_animation_time: float = 0  # Время последнего обновления анимации
        self.edit_mode: bool = False  # Режим редактирования (True = можно менять любые клетки)
        self.editable_color = YELLOW  # Цвет цифр в редактируемых клетках

    def generate_sudoku(self) -> None:
        """Генерация случайного судоку"""
        self.grid = np.zeros((9, 9), dtype=int)
        self.solve_sudoku()
        self.initial_grid = self.grid.copy()

        cells = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(cells)
        to_remove = random.randint(41, 55)

        for row, col in cells[:to_remove]:
            self.grid[row, col] = 0

        self.initial_grid = self.grid.copy()
        self.reset_solution_info()
        self.show_message("Новая игра создана!")
        self.edit_mode = False

    def reset_solution_info(self):
        """Сброс информации о решениях"""
        self.solution_count = 0
        self.error_cells = []
        self.solution_limit_reached = False
        self.searching_solutions = False

    def solve_sudoku(self) -> bool:
        """Находит случайное решение судоку"""
        self.reset_solution_info()
        temp_grid = self.initial_grid.copy()

        # Запускаем поиск случайного решения в отдельном потоке
        self.searching_solutions = True
        threading.Thread(target=self._find_random_solution_thread, args=(temp_grid,), daemon=True).start()
        return True

    def _find_random_solution_thread(self, grid: np.ndarray):
        """Находит случайное решение в отдельном потоке."""
        if self._solve_random(grid):
            self.grid = grid.copy()
            self.show_message("Случайное решение найдено!")
        else:
            self.show_message("Решение не найдено!")
        self.searching_solutions = False

    def _solve_random(self, grid: np.ndarray) -> bool:
        """Рекурсивный бэктрекинг со случайным порядком чисел"""

        empty = self._find_empty(grid)  # Находим первую пустую клетку
        if not empty:
            return True

        row, col = empty

        numbers = list(range(1, 10))    # Перемешиваем числа от 1 до 9 для случайного порядка
        random.shuffle(numbers)

        for num in numbers:
            if self._is_valid(grid, row, col, num):
                grid[row, col] = num

                if self._solve_random(grid):
                    return True

                grid[row, col] = 0

        return False

    def _find_empty(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Находит первую пустую клетку в сетке."""
        for row in range(9):
            for col in range(9):
                if grid[row, col] == 0:
                    return (row, col)
        return None

    def _solve(self, grid: np.ndarray, row: int, col: int) -> bool:
        """Рекурсивный бэктрекинг"""
        if row == 9:
            return True
        if col == 9:
            return self._solve(grid, row + 1, 0)
        if grid[row, col] != 0:
            return self._solve(grid, row, col + 1)

        for num in range(1, 10):
            if self._is_valid(grid, row, col, num):
                grid[row, col] = num
                if self._solve(grid, row, col + 1):
                    return True
                grid[row, col] = 0
        return False

    def _is_valid(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Проверка вставки числа"""
        if num in grid[row] or num in grid[:, col]:
            return False

        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_row:box_row + 3, box_col:box_col + 3]:
            return False

        return True

    def count_solutions(self) -> None:
        """Запускает подсчет решений в отдельном потоке"""
        self.reset_solution_info()
        self.searching_solutions = True
        self.last_animation_time = time.time()

        # Запускаем подсчет в отдельном потоке
        threading.Thread(target=self._count_solutions_thread, daemon=True).start()

    def _count_solutions_thread(self):
        """Метод для запуска в отдельном потоке"""
        self._count_solutions(self.initial_grid.copy())
        self.searching_solutions = False
        if self.solution_limit_reached:
            self.show_message(f"Достигнут лимит ({self.max_solutions} решений)")
        else:
            self.show_message(f"Найдено решений: {self.solution_count}")

    def _count_solutions(self, grid: np.ndarray) -> None:
        """Метод для подсчёта решений"""
        def backtrack(temp_grid: np.ndarray, row: int, col: int):
            if self.solution_count >= self.max_solutions:
                self.solution_limit_reached = True
                return

            if row == 9:
                self.solution_count += 1
                return

            if col == 9:
                backtrack(temp_grid, row + 1, 0)
                return

            if temp_grid[row, col] != 0:
                backtrack(temp_grid, row, col + 1)
                return

            for num in range(1, 10):
                if self._is_valid(temp_grid, row, col, num):
                    temp_grid[row, col] = num
                    backtrack(temp_grid, row, col + 1)
                    temp_grid[row, col] = 0

        self.solution_count = 0
        self.solution_limit_reached = False
        backtrack(grid.copy(), 0, 0)

    def check_errors(self) -> None:
        """Проверяет конфликты чисел"""
        self.error_cells = []
        for row in range(9):
            for col in range(9):
                num = self.grid[row, col]
                if num == 0:
                    continue

                if (np.count_nonzero(self.grid[row] == num) > 1 or
                        np.count_nonzero(self.grid[:, col] == num) > 1):
                    self.error_cells.append((row, col))

                box_row, box_col = 3 * (row // 3), 3 * (col // 3)
                box = self.grid[box_row:box_row + 3, box_col:box_col + 3]
                if np.count_nonzero(box == num) > 1:
                    self.error_cells.append((row, col))

    def show_message(self, text: str, duration: float = 6.0) -> None:
        """Показывает сообщение на экране, пропадающие через время"""
        self.message = text
        self.message_timer = duration

    def update_animation(self):
        """Обновляет анимацию индикатора поиска"""
        current_time = time.time()
        if current_time - self.last_animation_time > 0.1:
            self.animation_phase = (self.animation_phase + 1) % 8
            self.last_animation_time = current_time

    def draw_loading_indicator(self, x: int, y: int):
        """Рисует вращающийся индикатор"""
        if not self.searching_solutions:
            return

        self.update_animation()
        radius = 15
        start_angle = self.animation_phase * 45  # Начальный угол
        arc_length = 180 + 45 * np.sin(self.animation_phase * 0.5)  # Длина дуги
        color_value = int(100 + 155 * abs(np.sin(self.animation_phase * 0.3)))
        color = (color_value, 200, 255 - color_value // 2)
        pygame.draw.arc(screen, color,(x - radius, y - radius, radius * 2, radius * 2), np.radians(start_angle), np.radians(start_angle + arc_length),4)

    def draw_interface(self) -> None:
        """Интерфейс"""
        pygame.draw.rect(screen, DARK_BG, (0, WIDTH, WIDTH, HEIGHT - WIDTH))
        button_width = 100
        button_height = 50
        button_margin = 10
        start_x = (WIDTH - (5 * button_width + 4 * button_margin)) // 2

        buttons = [
            {"rect": pygame.Rect(start_x, WIDTH + 20, button_width, button_height),
             "text": "Решить", "key": "solve", "color": GREEN, "key_shortcut": "R"},
            {"rect": pygame.Rect(start_x + button_width + button_margin, WIDTH + 20, button_width, button_height),
             "text": "Решения", "key": "count", "color": BLUE, "key_shortcut": "C"},
            {"rect": pygame.Rect(start_x + 2 * (button_width + button_margin), WIDTH + 20, button_width, button_height),
             "text": "Новая", "key": "new", "color": ORANGE, "key_shortcut": "N"},
            {"rect": pygame.Rect(start_x + 3 * (button_width + button_margin), WIDTH + 20, button_width, button_height),
             "text": "Сброс", "key": "reset", "color": RED, "key_shortcut": "D"},
            {"rect": pygame.Rect(start_x + 4 * (button_width + button_margin), WIDTH + 20, button_width, button_height),
             "text": "Редакт.", "key": "edit", "color": PURPLE, "key_shortcut": "E"},
        ]

        for button in buttons:
            color = button["color"]
            if self.active_button == button["key"]:
                color = (min(color[0] + 40, 255), min(color[1] + 40, 255), min(color[2] + 40, 255))
            elif button["key"] == "edit" and self.edit_mode:
                color = YELLOW

            pygame.draw.rect(screen, color, button["rect"], border_radius=5)
            pygame.draw.rect(screen, WHITE, button["rect"], 2, border_radius=5)

            # Текст кнопки
            text = small_font.render(button["text"], True, WHITE)
            text_rect = text.get_rect(center=button["rect"].center)
            screen.blit(text, text_rect)

            # Горячая клавиша
            shortcut_text = small_font.render(f"({button['key_shortcut']})", True, WHITE)
            shortcut_rect = shortcut_text.get_rect(midtop=(button["rect"].centerx, button["rect"].bottom + 5))
            screen.blit(shortcut_text, shortcut_rect)

        status_y = WIDTH + 100

        # Индикатор и счетчик решений
        if self.searching_solutions:
            self.draw_loading_indicator(30, status_y + 15)
            progress_text = f"Поиск решений: {self.solution_count}"
            text = timer_font.render(progress_text, True, TURQUOISE)
            screen.blit(text, (60, status_y + 10))
        elif self.solution_count > 0:
            result_text = f"Найдено решений: {self.solution_count}"
            if self.solution_limit_reached:
                result_text += " (лимит)"
            text = timer_font.render(result_text, True, TURQUOISE)
            screen.blit(text, (20, status_y + 10))

        if self.message_timer > 0:
            self.message_timer -= 0.02
            msg_text = small_font.render(self.message, True, YELLOW)
            msg_rect = msg_text.get_rect(midtop=(WIDTH // 2, status_y + 40))
            screen.blit(msg_text, msg_rect)

        controls_y = HEIGHT - 10
        controls_text = timer_font.render("Управление: Цифры 1-9 - ввод, Del - удалить, Стрелки - перемещение",True, WHITE)
        controls_rect = controls_text.get_rect(midbottom=(WIDTH // 2, controls_y))
        screen.blit(controls_text, controls_rect)

    def draw(self) -> None:
        """Отрисовка всей игры"""
        screen.fill(DARK_BG)
        # Отрисовка доски
        board_bg = pygame.Rect(0, 0, WIDTH, WIDTH)
        pygame.draw.rect(screen, SUDOKU_BG, board_bg)
        # Сетка
        for i in range(10):
            thickness = 3 if i % 3 == 0 else 1
            pygame.draw.line(screen, BLUE, (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE), thickness)
            pygame.draw.line(screen, BLUE, (i * CELL_SIZE, 0), (i * CELL_SIZE, WIDTH), thickness)

        # Цифры
        for row in range(9):
            for col in range(9):
                num = self.grid[row, col]
                if num != 0:
                    # Определяем цвет цифры
                    if (row, col) in self.error_cells:
                        color = RED
                    elif self.edit_mode:
                        color = self.editable_color  # Специальный цвет в режиме редактирования
                    elif self.initial_grid[row, col] != 0:
                        color = LIGHT_GREEN  # Исходные цифры
                    else:
                        color = TURQUOISE  # Введенные пользователем цифры

                    text = font.render(str(num), True, color)
                    screen.blit(text, (col * CELL_SIZE + 18, row * CELL_SIZE + 10))

        # Выделение клетки
        if self.selected:
            row, col = self.selected
            pygame.draw.rect(screen, GREEN, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 3)

        self.draw_interface()

def main():
    game = SudokuGame()
    game.generate_sudoku()
    running = True

    while running:
        game.active_button = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if y < WIDTH:  # Клик по игровому полю
                    col = x // CELL_SIZE
                    row = y // CELL_SIZE
                    game.selected = (row, col)
                else:  # Клик по кнопкам
                    for button in [
                        ("solve", game.solve_sudoku),
                        ("count", game.count_solutions),
                        ("new", game.generate_sudoku),
                        ("reset", lambda: [
                            setattr(game, 'grid', np.zeros((9, 9), dtype=int)),
                            setattr(game, 'initial_grid', np.zeros((9, 9), dtype=int)),
                            game.reset_solution_info(),
                            game.show_message("Поле очищено!")
                        ]),
                        ("edit", lambda: [
                            setattr(game, 'edit_mode', not game.edit_mode),
                            game.show_message("Режим редактирования!" if game.edit_mode else "Режим решения!"),
                            setattr(game, 'initial_grid', game.grid.copy()) if not game.edit_mode else None
                        ])
                    ]:
                        btn_key, action = button
                        btn_rect = pygame.Rect(
                            (WIDTH - (5 * 100 + 4 * 10)) // 2 + list(range(5)).index(
                                btn_key == "edit" and 4 or ["solve", "count", "new", "reset"].index(btn_key)) * (100 + 10), WIDTH + 20, 100, 50)

                        if btn_rect.collidepoint(x, y):
                            game.active_button = btn_key
                            action()
                            break

            elif event.type == pygame.KEYDOWN:
                if game.selected:
                    row, col = game.selected
                    if event.key == pygame.K_UP and row > 0:
                        game.selected = (row - 1, col)
                    elif event.key == pygame.K_DOWN and row < 8:
                        game.selected = (row + 1, col)
                    elif event.key == pygame.K_LEFT and col > 0:
                        game.selected = (row, col - 1)
                    elif event.key == pygame.K_RIGHT and col < 8:
                        game.selected = (row, col + 1)

                if event.unicode.isdigit() and 1 <= int(event.unicode) <= 9:
                    if game.selected:
                        row, col = game.selected
                        if game.edit_mode or game.initial_grid[row, col] == 0:
                            game.grid[row, col] = int(event.unicode)
                            game.check_errors()

                elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                    if game.selected:
                        row, col = game.selected
                        if game.edit_mode or game.initial_grid[row, col] == 0:
                            game.grid[row, col] = 0
                            game.check_errors()

                elif not game.selected and (event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT)):
                    game.selected = (0, 0)

                elif event.key == pygame.K_r:
                    game.active_button = "solve"
                    game.solve_sudoku()
                elif event.key == pygame.K_c:
                    game.active_button = "count"
                    game.count_solutions()
                elif event.key == pygame.K_n:
                    game.active_button = "new"
                    game.generate_sudoku()
                elif event.key == pygame.K_d:
                    game.active_button = "reset"
                    game.grid = np.zeros((9, 9), dtype=int)
                    game.initial_grid = np.zeros((9, 9), dtype=int)
                    game.reset_solution_info()
                    game.show_message("Поле очищено!")
                elif event.key == pygame.K_e:
                    game.active_button = "edit"
                    game.edit_mode = not game.edit_mode
                    if not game.edit_mode:
                        game.initial_grid = game.grid.copy()  # Фиксируем изменения
                        game.show_message("Режим решения! Начальное поле сохранено")
                    else:
                        game.show_message("Режим редактирования!")

        game.draw()
        pygame.display.flip()
        pygame.time.delay(20)

    pygame.quit()

if __name__ == "__main__":
    main()