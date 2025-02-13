import numpy as np
from PIL import Image, ImageOps
from math import sin, cos, pi, sqrt

def dotted_line(image, x0, y0, x1, y1, count, color):
    step = 1.0/count
    for t in np.arange (0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color

def dotted_line_v2(image, x0, y0, x1, y1, color):
    count = sqrt((x0 - x1)**2 + (y0 - y1)**2)
    step = 1.0/count
    for t in np.arange (0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color

def x_loop_line(image, x0, y0, x1, y1, color):
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color

def x_loop_line_fix1(image, x0, y0, x1, y1, color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color
    
def x_loop_line_fix2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color


def x_loop_line_v2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

def x_loop_line_no_y_calc(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0
    y_update = 1 if y1 > y0 else -1
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if (derror > 0.5):
            derror -= 1
            y += y_update

def bresenham_line(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0
    y_update = 1 if y1 > y0 else -1
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if (derror > (x1- x0)):
            derror -= 2 * (x1 - x0)
            y += y_update


def draw_star(filename, method, count=0):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:200, :200, 2] = 255
    for i in range(13):
        x0 = 100
        y0 = 100
        x1 = int(100 + 95 * cos(i * 2 * pi/13))
        y1 = int(100 + 95 * sin(i * 2 * pi / 13))
        if (count > 0):
            method(img, x0, y0, x1, y1, count, (255, 255, 255))
        else:
            method(img, x0, y0, x1, y1, (255, 255, 255))
    image = Image.fromarray(img, mode="RGB")
    image.save(filename)


# TASK 3
def vertex_parser(filename):
    f = open(filename)

    arr = []
    for s in f:
        splitted = s.split(" ")
        if (splitted[0] == 'v'):
            arr.append([float(el) for el in splitted[1:]])
    f.close()
    return arr

# TASK 4
def draw_vertex(vertex):
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for coord in vertex:
        x = int(5000 * coord[0] + 500)
        y = int(5000 * coord[1] + 500)
        img[y][x] = [255, 255, 255]
    image = Image.fromarray(img)
    image = ImageOps.flip(image)
    image.save("dots.png")


def polygons_parser(filename):
    f = open(filename)

    res = []
    for s in f:
        splitted = s.split(" ")
        if (splitted[0] == 'f'):
            res.append([int(x.split("/")[0]) - 1 for x in splitted[1:]])
    return res   

def draw_polygons(vertex, polygons):
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for vert in polygons:
        x0, y0 = int(5000 * vertex[vert[0]][0] + 500), int(5000 * vertex[vert[0]][1] + 500)
        x1, y1 = int(5000 * vertex[vert[1]][0] + 500), int(5000 * vertex[vert[1]][1] + 500)
        x2, y2 = int(5000 * vertex[vert[2]][0] + 500), int(5000 * vertex[vert[2]][1] + 500)
        bresenham_line(img, x0, y0, x1, y1, (255, 255, 255))
        bresenham_line(img, x0, y0, x2, y2, (255, 255, 255))
        bresenham_line(img, x1, y1, x2, y2, (255, 255, 255))
    image = Image.fromarray(img)
    image = ImageOps.flip(image)
    image.save("polygons.png")



img = np.zeros((600, 800, 3), dtype=np.uint8)
img[0:600, 0:800, 2] = 255
for i in range(600):
    for j in range(800):
        img[i, j] = [255 * (i) / 600, 255 * int((i**2 + j ** 2) ** 0.5) / 1400, 255 * (j) / 800,]
image = Image.fromarray(img, mode="RGB")
image.save("1_3.png")

draw_star(filename="1method.png", method=dotted_line, count=100)
draw_star(filename="2method.png", method=dotted_line_v2)
draw_star(filename="3method.png", method=x_loop_line)
draw_star(filename="4method.png", method=x_loop_line_fix1)
draw_star(filename="5method.png", method=x_loop_line_fix2)
draw_star(filename="6method.png", method=x_loop_line_v2)
draw_star(filename="7method.png", method=x_loop_line_no_y_calc)
draw_star(filename="8method.png", method=bresenham_line)

vertex, polygons = vertex_parser("model_1.obj"), polygons_parser("model_1.obj")
draw_vertex(vertex)
draw_polygons(vertex, polygons)
