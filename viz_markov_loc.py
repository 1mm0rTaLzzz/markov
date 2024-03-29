import pygame
import numpy as np
import pygame_widgets
from pygame_widgets.textbox import TextBox
import random
import time
from collections import deque
from tkinter import *
from tkinter import messagebox

pHit = 0.6
pMiss = 0.2
pExact = 0.8
pOvershoot = 0.1
pUndershoot = 0.1
DISPLAY_PROBABILITY = True

# --------------------------------------------
pygame.init()
# --------------------------------------------
BLACK = (0, 0, 0)
DARK_RED = (153, 0, 51)
DARK_GREEN = (51, 102, 51)
WHITE = (255, 255, 255)
RED = (255, 0, 51)
GREEN = (51, 204, 51)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (204, 204, 204)
DARK_GREY = (100, 100, 100)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)
CYAN = (0, 255, 255)
BROWN = (165, 42, 42)

win_size = (800, 600)
screen = pygame.display.set_mode(win_size)
pygame.display.set_caption("Робот 2-D")

screen.fill(GRAY)

# --------------------------------------------
font = pygame.font.SysFont(None, 20)
font2 = pygame.font.SysFont(None, 25)
font3 = pygame.font.SysFont(None, 30)


def output():
    rows = textbox.getText()
    cols = textbox1.getText()
    if rows.isdigit() and cols.isdigit():
        grid(int(rows), int(cols), p=np.zeros((int(rows), int(cols))), world=np.array([]))


def grid(rows, cols, p, world=np.array([]), real=(0, 0), prediction=(0, 0), start_point=(0, 0), end_point=(0, 0)):
    square_size_x = int(600 / rows)
    square_size_y = int(600 / cols)

    square_size = min(square_size_x, square_size_y)

    square = pygame.Rect((200, 0, 600, 600))
    pygame.draw.rect(screen, GRAY, square)

    x = 201
    y = 1
    row, col = 0, 0
    flag = False

    if world.size == 0:
        world = np.zeros((rows, cols), 'str')
        flag = True
    while x < 800 and col != cols:
        while y < 600 and row != rows:
            square = pygame.Rect((x, y, square_size - 2, square_size - 2))
            if DISPLAY_PROBABILITY:
                if flag:
                    rand_color = random.randint(0, 100)
                    if rand_color < 10 and real[0] != row and real[1] != col:
                        pygame.draw.rect(screen, DARK_GREY, square)
                        world[row, col] = 'w'
                    elif rand_color < 65:
                        pygame.draw.rect(screen, GREEN, square)
                        world[row, col] = 'g'
                    else:
                        pygame.draw.rect(screen, RED, square)
                        world[row, col] = 'r'
                else:
                    if world[row, col] == 'w':
                        pygame.draw.rect(screen, DARK_GREY, square)
                    elif world[row, col] == 'g':
                        pygame.draw.rect(screen, GREEN, square)
                    else:
                        pygame.draw.rect(screen, RED, square)
                    if real[0] == row and real[1] == col: pygame.draw.rect(screen, BLUE, square, 9)
                    if prediction[0] == row and prediction[1] == col: pygame.draw.rect(screen, YELLOW, square, 5)

                y += square_size
                row += 1
            else:
                color_intensity = int(255 - (p[row][col] * 25) * 10)
                color = (color_intensity, color_intensity, color_intensity)
                clr = (255 - color_intensity, 255 - color_intensity, 255 - color_intensity)
                pygame.draw.rect(screen, color, square)
                font = pygame.font.SysFont(None, 20)
                img = font.render(str("{:.3f}".format(p[row, col])), True, clr)
                screen.blit(img, (x + square_size / 3, y + square_size / 2))
                if real[0] == row and real[1] == col: pygame.draw.rect(screen, BLUE, square, 9)
                if prediction[0] == row and prediction[1] == col: pygame.draw.rect(screen, YELLOW, square, 5)
                y += square_size
                row += 1

        x += square_size
        y = 1
        row = 0
        col += 1

    pygame.display.flip()
    return world


def find_path(start, end, world):
    rows, cols = len(world), len(world[0])
    visited = set()
    queue = deque([(start, [])])
    while queue:
        current, path = queue.popleft()
        if np.all(current == end):
            return path + [current]

        if current in visited:
            continue
        visited.add(current)

        row, col = current
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1),
                      (-1, 1)]  # Вправо, влево, вниз, вверх

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols and world[new_row][new_col] != 'w':
                new_point = (new_row, new_col)
                queue.append((new_point, path + [current]))

    return None


def pprint(v):
    for row in v:
        print(", ".join(['{:.3f}'.format(x) for x in row]))


def sense_color(p, Z, rows, cols, world):
    p_new = np.copy(p)

    for i in range(rows):
        for j in range(cols):
            hit = (Z == world[i, j])
            p_new[i, j] = p[i, j] * (hit * pHit + (1 - hit) * pMiss)

    s = np.sum(p_new)

    if s != 0:
        p_new /= s

    return p_new


def sense_lidar(p, u, max_index, rows, cols, world):  # пока считаем, что лидар идеален
    p_new = np.copy(p)
    flag = True
    wall_detect = (world[(u[0] + max_index[0]) % rows, (u[1] + max_index[1]) % cols] == 'w')
    if wall_detect:
        p_new[(u[0] + max_index[0]) % rows, (u[1] + max_index[1]) % cols] = 0
    else:
        flag = False

    s = np.sum(p_new)

    if s != 0:
        p_new /= s

    return p_new, flag


def move(p, U, rows, cols):
    p_new = np.copy(p)

    for i in range(rows):
        for j in range(cols):
            s = pExact * p[(i - U[0]) % rows, (j - U[1]) % cols]
            s += pOvershoot * p[(i - U[0] - 1 * (U[0] != 0)) % rows, (j - U[1] - 1 * (U[1] != 0)) % cols]
            s += pUndershoot * p[(i - U[0] + 1 * (U[0] != 0)) % rows, (j - U[1] + 1 * (U[1] != 0)) % cols]
            p_new[i, j] = s

    return p_new


def opposite(place):
    if place == 'g':
        return 'r'
    else:
        return 'g'


def parse_string_to_arr(input_string):
    arr = []
    pairs = input_string.strip('()').split(',')
    pairs = [tuple(map(int, pair.strip('()').split(','))) for pair in pairs]
    array = np.array(pairs).reshape(-1, 2)
    for i in array:
        arr.append(i[::-1])
    arr = np.array(arr)
    return arr


def probability(world):
    font = pygame.font.SysFont(None, 20)
    rows = textbox1.getText()
    cols = textbox.getText()
    start_x = textbox2.getText()
    start_y = textbox3.getText()
    repeat = textbox4.getText()
    str_move = textbox5.getText()
    realSensorError = textbox6.getText()

    if rows.isdigit() and cols.isdigit():
        rows, cols = int(rows), int(cols)
        p = np.zeros((rows, cols), dtype='float')
    else:
        rows, cols = 5, 5
        p = np.zeros((5, 5), dtype='float')

    start_point = (0, 0)
    end_point = (cols - 1, rows - 1)

    if start_x.isdigit() and start_y.isdigit():
        if int(start_x) < rows and int(start_y) < cols:
            start_x, start_y = int(start_x), int(start_y)

        else:
            Tk().wm_withdraw()
            messagebox.showinfo('Результат ввода', 'Стартовые координаты находятся вне сетки или введены с ошибкой')

            return
    else:
        Tk().wm_withdraw()
        messagebox.showinfo('Результат ввода',
                            'Стартовые координаты находятся вне сетки или введены с ошибкой. Стартовые координаты заменены на (0,0)')

        start_x, start_y = 0, 0

    repeat = int(repeat) if (repeat.isdigit() and int(repeat) != 0) else 1

    real = (start_y, start_x)

    p[start_y, start_x] = 1

    try:
        U = parse_string_to_arr(str_move)
    except:
        U = np.array([[0, 0]])
        Tk().wm_withdraw()
        messagebox.showerror('Ошибка',
                             'Движения введены с ошибкой. Весь список был заменен на (0,0) ')

    realSensorError = int(realSensorError) if realSensorError.isdigit() else 0.5

    false_count = 0
    for k in range(repeat):
        print("Проход № ", k + 1)
        print(world)
        if k == 0:
            grid(rows, cols, p, world=world, real=real)

        for u in U:
            max_index = np.unravel_index(np.argmax(p), p.shape)
            p, flag_wall = sense_lidar(p, u, max_index, rows, cols, world)

            flag = random.random() < realSensorError
            if flag:
                p = sense_color(p, world[real[0]][real[1]], rows, cols, world)
            else:
                p = sense_color(p, opposite(world[real[0]][real[1]]), rows, cols, world)

            prediction = (max_index[0], max_index[1])
            print('sense_color: \n', end='')
            pprint(p)
            print(f' {real=}, {prediction=}')

            if flag_wall == False:
                p = move(p, u, rows, cols)
                real = ((real[0] + u[0]) % rows, (real[1] + u[1]) % cols)
                max_index = np.unravel_index(np.argmax(p), p.shape)
                prediction = (max_index[0], max_index[1])
                grid(rows, cols, p, world=world, real=real, prediction=prediction)
                print('move: \n', end='')
                pprint(p)
                print(f' {real=}, {prediction=}\n')
            else:
                p[(max_index[0] + u[0] * 2) % rows, (max_index[1] + u[1] * 2) % cols] = 0

                s = np.sum(p)
                if s != 0:
                    p /= s

                print('Встретил стенку')
                prediction = (max_index[0], max_index[1])

                print('wall: \n', end='')
                pprint(p)
                print(f' {real=}, {prediction=}\n')

            if real != prediction:
                false_count += 1

            time.sleep(2.5)

    robot_position = start_point
    end = end_point
    print('false_count =', false_count)
    print('=================================', '\n')
    mess = pygame.Rect((5, 800, 190, 110))
    pygame.draw.rect(screen, WHITE, mess)

    real_p = real[::-1]
    prediction_p = real[::-1]
    Tk().wm_withdraw()
    messagebox.showinfo('Результат ввода',
                        f'{real_p = } - синяя рамка\n' f'{prediction_p = } - жёлтая\n' f'{false_count = }')

    return p, rows, cols, world, robot_position, end, prediction, U


def print_path(world, robot_position, U):
    repeat = textbox4.getText()
    repeat = int(repeat) if (repeat.isdigit() and int(repeat) != 0) else 1

    rows, cols = len(world), len(world[0])
    square_size_x = int(600 / rows)
    square_size_y = int(600 / cols)
    s = sum(U) * repeat
    if s[0] > rows:
        s[0] = s[0] % rows
    if s[1] > cols:
        s[1] = s[1] % cols
    print(type(s))
    if world[robot_position[0]][robot_position[1]] != 'w':
        path = find_path(robot_position, s, world)
    else:
        return
    square_size = min(square_size_x, square_size_y)
    # Если путь найден, выведем его и обновим визуализацию
    if path:
        print("Найденный путь:", path)
        for point in path:
            print(point)
            if world[point[0]][point[1]] == 'r':
                pygame.draw.rect(screen, DARK_RED,
                                 pygame.Rect(201 + point[1] * square_size, 1 + point[0] * square_size,
                                             square_size - 2,
                                             square_size - 2))
            elif world[point[0]][point[1]] == 'g':
                pygame.draw.rect(screen, DARK_GREEN,
                                 pygame.Rect(201 + point[1] * square_size, 1 + point[0] * square_size,
                                             square_size - 2,
                                             square_size - 2))
        pygame.draw.rect(screen, CYAN,
                         pygame.Rect(201 + path[0][0] * square_size, 1 + path[0][1] * square_size,
                                     square_size - 2,
                                     square_size - 2), width=3)

    else:
        print("Путь не найден.")
        Tk().wm_withdraw()  # прячем окно ткинтера
        messagebox.showwarning('Внимание', 'Путь не найден')


# _______________________-

img = font.render('Размер поля:        x ', True, BLACK)
screen.blit(img, (10, 20))

textbox = TextBox(screen, 105, 10, 20, 30, fontSize=20,
                  borderColour=(0, 0, 0), textColour=(0, 0, 0),
                  onSubmit=output, radius=1, borderThickness=1)

textbox1 = TextBox(screen, 145, 10, 20, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)

img = font.render('Координаты начала: ', True, BLACK)
screen.blit(img, (10, 60))

img = font2.render('(x0,y0) = (      ;      )', True, BLACK)
screen.blit(img, (10, 80))
textbox2 = TextBox(screen, 95, 80, 20, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)

textbox3 = TextBox(screen, 130, 80, 20, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)

img = font.render('Количество итераций: ', True, BLACK)
screen.blit(img, (10, 140))
textbox4 = TextBox(screen, 160, 140, 20, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)

img = font.render('Перечень движений: ', True, BLACK)
screen.blit(img, (10, 190))
textbox5 = TextBox(screen, 1, 210, 199, 30, fontSize=18,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)
img = font.render('формат: (1,1),(-1,1),(0,1)', True, BLACK)
screen.blit(img, (30, 245))

img = font.render('Коэф. точности: ', True, BLACK)
screen.blit(img, (10, 300))
textbox6 = TextBox(screen, 130, 290, 35, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)

# --------------------------------------------
world = grid(rows=5, cols=5, p=np.zeros((5, 5)), world=np.array([]))
pygame.display.flip()

fl = 0
x_pos, y_pos = 0, 0
running = True

while running:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False

        pos = pygame.mouse.get_pos()
        # кнопка запуска
        if 50 < pos[0] < 160 and 560 < pos[1] < 590:
            button = pygame.Rect((50, 560, 110, 30))
            pygame.draw.rect(screen, DARK_GREY, button)
        else:
            button = pygame.Rect((50, 560, 110, 30))
            pygame.draw.rect(screen, WHITE, button)

        img = font2.render('Запуск', True, BLACK)
        screen.blit(img, (65, 565))

        # кнопка пути
        if 50 < pos[0] < 160 and 520 < pos[1] < 550:
            button = pygame.Rect((50, 520, 110, 30))
            pygame.draw.rect(screen, DARK_GREY, button)
        else:
            button = pygame.Rect((50, 520, 110, 30))
            pygame.draw.rect(screen, WHITE, button)

        img = font.render('Отрисовать путь', True, BLACK)
        screen.blit(img, (53, 525))

        # кнопка обновления
        if 50 < pos[0] < 160 and 480 < pos[1] < 510:
            button = pygame.Rect((50, 480, 110, 30))
            pygame.draw.rect(screen, DARK_GREY, button)
        else:
            button = pygame.Rect((50, 480, 110, 30))
            pygame.draw.rect(screen, WHITE, button)

        img = font2.render('Обновить', True, BLACK)
        screen.blit(img, (65, 485))

        # кнопка переключения экранов
        if 50 < pos[0] < 160 and 440 < pos[1] < 470:
            button = pygame.Rect((50, 440, 110, 30))
            pygame.draw.rect(screen, DARK_GREY, button)
        else:
            button = pygame.Rect((50, 440, 110, 30))
            pygame.draw.rect(screen, WHITE, button)

        img = font.render('Переключить', True, BLACK)
        screen.blit(img, (65, 445))
        img = font.render('экран', True, BLACK)
        screen.blit(img, (65, 455))

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if event.button == 1:
                # запуск
                if 50 < pos[0] < 160 and 560 < pos[1] < 590:
                    fl = 0
                    rows = textbox1.getText()
                    cols = textbox.getText()
                    if rows.isdigit() and cols.isdigit():
                        rows, cols = int(rows), int(cols)
                        world = grid(rows=rows, cols=cols, p=np.zeros((rows, cols)), world=np.array([]))
                    elif rows == '' and cols == '':
                        world = grid(rows=5, cols=5, p=np.zeros((5, 5)), world=np.array([]))
                    p, rows, cols, world, robot_position, end, prediction, U = probability(world)
                    fl = 1
                # путь
                if 50 < pos[0] < 160 and 520 < pos[1] < 550:
                    start_point_str = (textbox2.getText(), textbox3.getText())
                    print(start_point_str)
                    if start_point_str[0].isdigit() and start_point_str[1].isdigit():
                        start_point = tuple(map(int, start_point_str))

                        fl += 1
                    else:
                        Tk().wm_withdraw()  # прячем окно ткинтера
                        messagebox.showerror('Ошибка',
                                             'Координаты начала и конца должны быть целыми числами')
                    if fl == 2:
                        print_path(world, start_point, U)
                    else:
                        Tk().wm_withdraw()  # прячем окно ткинтера
                        messagebox.showerror('Ошибка',
                                             'Пожалуйста, сначала запустите основную программу навигации')

            # обновление
            if 50 < pos[0] < 160 and 480 < pos[1] < 510:
                rows = textbox1.getText()
                cols = textbox.getText()
                if rows.isdigit() and cols.isdigit():
                    rows, cols = int(rows), int(cols)
                    world = grid(rows=rows, cols=cols, p=np.zeros((rows, cols)), world=np.array([]))
                elif rows == '' and cols == '':
                    world = grid(rows=5, cols=5, p=np.zeros((5, 5)), world=np.array([]))
                else:
                    Tk().wm_withdraw()
                    messagebox.showerror('Ошибка', 'Количество строк и столбцов матрицы должны быть целыми числами')

            # переключение
            if 50 < pos[0] < 160 and 440 < pos[1] < 470:
                DISPLAY_PROBABILITY = not DISPLAY_PROBABILITY
                p, rows, cols, world, robot_position, end, prediction, U = probability(world)

    pygame_widgets.update(events)
    pygame.display.update()

pygame.quit()
