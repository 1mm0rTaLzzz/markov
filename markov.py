import pygame
import numpy as np
import pygame_widgets
from pygame_widgets.textbox import TextBox
import random
import time
from collections import deque

pHit = 0.6
pMiss = 0.2
pExact = 0.8
pOvershoot = 0.1
pUndershoot = 0.1
DISPLAY_PROBABILITY = True

def output():
    rows = textbox.getText()
    cols = textbox1.getText()
    if rows.isdigit() and cols.isdigit():
        grid(int(rows), int(cols), p=np.zeros((int(rows), int(cols))), pos=(0, 0), button=0, world=np.array([]))


def grid(rows, cols, p, pos, button=0, world=np.array([]), real=(0, 0), prediction=(0, 0), start_point=(0, 0),
         end_point=(0, 0)):
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
            if DISPLAY_PROBABILITY:
                font = pygame.font.SysFont(None, 20)

                img = font.render(str("{:.3f}".format(p[row, col])), True, BLACK)
                screen.blit(img, (x + square_size / 3, y + square_size / 2))
                y += square_size
                row += 1
        x += square_size
        y = 1
        row = 0
        col += 1
    pygame.draw.rect(screen, CYAN,
                     pygame.Rect(201 + start_point[1] * square_size, 1 + start_point[0] * square_size, square_size - 2,
                                 square_size - 2), width=3)
    pygame.draw.rect(screen, BROWN,
                     pygame.Rect(201 + end_point[1] * square_size, 1 + end_point[0] * square_size, square_size - 2,
                                 square_size - 2), width=3)

    pygame.display.flip()
    return world


def find_path(start, end, world):
    rows, cols = len(world), len(world[0])
    visited = set()
    queue = deque([(start, [])])

    while queue:
        current, path = queue.popleft()
        if current == end:
            return path + [current]

        if current in visited:
            continue
        visited.add(current)

        row, col = current
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Вправо, влево, вниз, вверх

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


def parse_string_to_array(input_string):
    pairs = input_string.strip('()').split(',')
    pairs = [tuple(map(int, pair.strip('()').split(','))) for pair in pairs]
    array = np.array(pairs).reshape(-1, 2)

    return array


def probability():
    rows = textbox.getText()
    cols = textbox1.getText()
    start_x = textbox2.getText()
    start_y = textbox3.getText()
    repeat = textbox4.getText()
    str_move = textbox5.getText()
    realSensorError = textbox6.getText()
    start_point_str = textbox7.getText()
    end_point_str = textbox8.getText()

    if ',' in start_point_str and ',' in end_point_str:
        start_point = tuple(map(int, start_point_str.split(',')))
        end_point = tuple(map(int, end_point_str.split(',')))


    if rows.isdigit() and cols.isdigit():
        rows, cols = int(rows), int(cols)
        p = np.zeros((rows, cols), dtype='float')
    else:
        rows, cols = 5, 5
        p = np.zeros((5, 5), dtype='float')

    if start_x.isdigit() and start_y.isdigit():
        if int(start_x) < rows and int(start_y) < cols:
            start_x, start_y = int(start_x), int(start_y)
            mess = pygame.Rect((5, 400, 190, 100))
            pygame.draw.rect(screen, GRAY, mess)
        else:
            mess = pygame.Rect((5, 400, 190, 100))
            pygame.draw.rect(screen, WHITE, mess)
            font = pygame.font.SysFont(None, 20)
            img = font.render('Стартовые координаты', True, BLACK)
            screen.blit(img, (10, 410))
            img = font.render('находятся вне сетки или', True, BLACK)
            screen.blit(img, (10, 430))
            return
    else:
        start_x, start_y = 0, 0

    repeat = int(repeat) if repeat.isdigit() else 1

    real = (start_y, start_x)

    p[start_y, start_x] = 1
    world = grid(rows, cols, p, pos=(0, 0), button=0, world=np.array([]), real=real)

    try:
        print(str_move)
        U = parse_string_to_array(str_move)
        print(U)
    except:
        U = np.array([[1, 0]])

    realSensorError = int(realSensorError) if realSensorError.isdigit() else 0.5

    false_count = 0
    for k in range(repeat):
        print("Проход № ", k + 1)
        print(world)
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
                grid(rows, cols, p, pos=(0, 0), button=0, world=world, real=real, prediction=prediction)
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
    robot_position = (start_point[1], start_point[0])
    end = (end_point[1], end_point[0])

    # Найдем путь от начальной до конечной точки
    path = find_path(robot_position, end, world)

    # Если путь найден, выведем его и обновим визуализацию
    if path:
        print("Найденный путь:", path)
        for point in path:
            print(point)
        grid(rows, cols, p, pos=(0, 0), button=0, world=world, real=robot_position, prediction=prediction,
             start_point=start_point, end_point=end_point)
    else:
        print("Путь не найден.")

    print('false_count =', false_count)
    print('=================================', '\n')
    mess = pygame.Rect((5, 400, 190, 110))
    pygame.draw.rect(screen, WHITE, mess)
    font = pygame.font.SysFont(None, 20)
    img = font.render('Результат (формат (y, x)): ', True, BLACK)
    screen.blit(img, (10, 410))
    img = font.render(f'{real = } - синяя рамка', True, BLACK)
    screen.blit(img, (20, 430))
    img = font.render(f'{prediction = } - жёлтая', True, BLACK)
    screen.blit(img, (20, 450))
    img = font.render('рамка', True, BLACK)
    screen.blit(img, (10, 470))
    img = font.render(f'{false_count = }', True, BLACK)
    screen.blit(img, (20, 490))
    #grid(rows, cols, p, pos=(0, 0), button=0, world=world, real=real, prediction=prediction)


# --------------------------------------------
pygame.init()
# --------------------------------------------
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (130, 130, 130)
DARK_GREY = (100, 100, 100)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)
CYAN = (0, 255, 255)
BROWN = (165, 42, 42)

win_size = (1000, 600)
screen = pygame.display.set_mode(win_size)
pygame.display.set_caption("Робот 2-D")

screen.fill(GRAY)

button = pygame.Rect((50, 540, 100, 40))
pygame.draw.rect(screen, WHITE, button)

# --------------------------------------------
font = pygame.font.SysFont(None, 20)

img = font.render('Количество: ', True, BLACK)
screen.blit(img, (10, 10))

img = font.render('- cтрок = ', True, BLACK)
screen.blit(img, (30, 40))
textbox = TextBox(screen, 90, 30, 20, 30, fontSize=20,
                  borderColour=(0, 0, 0), textColour=(0, 0, 0),
                  onSubmit=output, radius=1, borderThickness=1)

img = font.render('- столбцов = ', True, BLACK)
screen.blit(img, (30, 80))
textbox1 = TextBox(screen, 120, 70, 20, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)

img = font.render('Координаты начала: ', True, BLACK)
screen.blit(img, (10, 120))

img = font.render('X = ', True, BLACK)
screen.blit(img, (30, 150))
textbox2 = TextBox(screen, 60, 140, 20, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)

img = font.render('Y = ', True, BLACK)
screen.blit(img, (30, 190))
textbox3 = TextBox(screen, 60, 180, 20, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)

img = font.render('Количество итераций: ', True, BLACK)
screen.blit(img, (10, 230))
textbox4 = TextBox(screen, 160, 220, 20, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)

img = font.render('Перечень движений: ', True, BLACK)
screen.blit(img, (10, 270))
textbox5 = TextBox(screen, 1, 290, 199, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)
img = font.render('формат: (1,1),(-1,1),(0,1)', True, BLACK)
screen.blit(img, (30, 320))

img = font.render('Коэф. точности датчкиов: ', True, BLACK)
screen.blit(img, (10, 350))
textbox6 = TextBox(screen, 120, 370, 35, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)

font = pygame.font.SysFont(None, 30)
img = font.render('Запуск', True, BLACK)
screen.blit(img, (65, 550))
textbox7 = TextBox(screen, 800, 500, 35, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)
img = font.render('Start Point (x,y): ', True, BLACK)
# screen.blit(img, (1000, 300))

textbox8 = TextBox(screen, 800, 400, 35, 30, fontSize=20,
                   borderColour=(0, 0, 0), textColour=(0, 0, 0),
                   onSubmit=output, radius=1, borderThickness=1)
img = font.render('End Point (x,y): ', True, BLACK)
# screen.blit(img, (800, 600))
# --------------------------------------------
grid(rows=5, cols=5, p=np.zeros((5, 5)), pos=(0, 0), button=0, world=np.array([]))
pygame.display.flip()

x_pos, y_pos = 0, 0
running = True

while running:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False

        pos = pygame.mouse.get_pos()
        if 50 < pos[0] < 150 and 540 < pos[1] < 580:
            button = pygame.Rect((50, 540, 100, 40))
            pygame.draw.rect(screen, DARK_GREY, button)
        else:
            button = pygame.Rect((50, 540, 100, 40))
            pygame.draw.rect(screen, WHITE, button)
        font = pygame.font.SysFont(None, 30)
        img = font.render('Запуск', True, BLACK)
        screen.blit(img, (65, 550))

        # for event in events:
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_SPACE:
        #             DISPLAY_PROBABILITY = not DISPLAY_PROBABILITY
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if event.button == 1:
                if 50 < pos[0] < 150 and 540 < pos[1] < 580:
                    probability()
            #     elif 200 < pos[0] < 800 and 0 < pos[1] < 600:
            #        world = grid(int(rows), int(cols), p=np.zeros((int(rows), int(cols))), pos=(0, 0), button=0, world=world)
            # if event.button == 2 and 200 < pos[0] < 800 and 0 < pos[1] < 600:
            #      world = grid(int(rows), int(cols), p=np.zeros((int(rows), int(cols))), pos=(0, 0), button=0, world=world)
            # if event.button == 3 and 200 < pos[0] < 800 and 0 < pos[1] < 600:
            #      world = grid(int(rows), int(cols), p=np.zeros((int(rows), int(cols))), pos=(0, 0), button=0, world=world)

    pygame_widgets.update(events)
    pygame.display.update()

pygame.quit()
