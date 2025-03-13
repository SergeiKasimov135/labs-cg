import numpy as np
from PIL import Image, ImageOps
from math import sin, cos, pi, sqrt
import math
import random
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
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for vert in polygons:
        x0, y0 = (5000 * vertex[vert[0]][0] + 5000), (5000 * vertex[vert[0]][1] + 5000)
        x1, y1 = (5000 * vertex[vert[1]][0] + 5000), (5000 * vertex[vert[1]][1] + 5000)
        x2, y2 = (5000 * vertex[vert[2]][0] + 5000), (5000 * vertex[vert[2]][1] + 5000)
        print(x0, y0,  x1, y1, x2, y2)
        draw_triangle(img, x0, y0, x1, y1, x2, y2, [random.randrange(30, 256) for i in range(3)])
    image = Image.fromarray(img)
    image = ImageOps.flip(image)
    image.save("polygons.png")


def draw_polygons_n(vertex, polygons, n = 0):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    l = np.array([0, 0, 1])
    for vert in polygons:
        x0, y0, z0 = vertex[vert[0]][0], vertex[vert[0]][1], vertex[vert[0]][2]
        x1, y1, z1 = vertex[vert[1]][0], vertex[vert[1]][1], vertex[vert[1]][2]
        x2, y2, z2 = vertex[vert[2]][0], vertex[vert[2]][1], vertex[vert[2]][2]
        norm = norm_cal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        cosN = np.dot(norm, l) / (np.linalg.norm(norm) * np.linalg.norm(l))
        if cosN < 0:
            draw_triangle_n(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, [cosN * -255, cosN * -255, cosN * -255])
    image = Image.fromarray(img)
    image = ImageOps.flip(image)
    image.save(f"polygons_{n}.png")


def bary(x0, y0, x1, y1, x2, y2, x, y):
    l0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l2 = 1.0 - l0 - l1
    return l0, l1, l2

#))))
def draw_triangle(img, x0, y0, x1, y1, x2, y2, light):
    x0_proj = 5000 * x0 + 2000
    y0_proj = 5000 * y0 + 2000
    x1_proj = 5000 * x1 + 2000
    y1_proj = 5000 * y1 + 2000
    x2_proj = 5000 * x2 + 2000
    y2_proj = 5000 * y2 + 2000

    xmin =  math.floor(min(x0_proj, x1_proj, x2_proj))
    xmax =  math.ceil(max(x0_proj, x1_proj, x2_proj))
    ymin =  math.floor(min(y0_proj, y1_proj, y2_proj))
    ymax =  math.ceil(max(y0_proj, y1_proj, y2_proj))
    if xmin < 0: xmin = 0
    if xmax >= H: xmax = H
    if ymin < 0: ymin = 0
    if ymax >= W: ymax = W
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l0, l1, l2 = bary(x0, y0, x1, y1, x2, y2, x, y)
            if l0 >= 0 and l1 >= 0 and l2 >= 0:
                img[y, x] = light


def draw_triangle_n(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, light):
    x0_proj = 4000 * x0 / z0 + W / 2
    y0_proj = 4000 * y0 / z0 + W / 2
    x1_proj = 4000 * x1 /z1 + W / 2
    y1_proj = 4000 * y1 / z1+ W / 2
    x2_proj = 4000 * x2 / z2+ W / 2
    y2_proj = 4000 * y2 /z2 + W / 2
    xmin =  math.floor(min(x0_proj, x1_proj, x2_proj))
    xmax =  math.ceil(max(x0_proj, x1_proj, x2_proj))
    ymin =  math.floor(min(y0_proj, y1_proj, y2_proj))
    ymax =  math.ceil(max(y0_proj, y1_proj, y2_proj))
    if xmin < 0: xmin = 0
    if xmax >= H: xmax = H
    if ymin < 0: ymin = 0
    if ymax >= W: ymax = W
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l0, l1, l2 = bary(x0_proj, y0_proj, x1_proj, y1_proj, x2_proj, y2_proj, x, y)
            z_x = l0 * z0 + l1 * z1 + l2 * z2
            if z_x < z_buffer[y, x] and l0 >= 0 and l1 >= 0 and l2 >= 0:
                img[y, x] = light
                z_buffer[y, x] = z_x

def norm_cal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n1 = np.array([x1 - x2, y1 - y2, z1 - z2])
    n2 = np.array([x1 - x0, y1 - y0, z1 - z0])
    n = np.cross(n1, n2)
    return n

def rotate(x, y, z, alpha, beta, gamma, tx, ty, tz):
    R = np.array([[1, 0, 0], [0, cos(alpha), sin(alpha)], [0, -sin(alpha), cos(alpha)]]) @ \
        np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]]) @ \
        np.array([[cos(gamma), sin(gamma), 0], [-sin(gamma), cos(gamma), 0], [0, 0, 1]])
    return np.dot(R, np.array([x, y, z])) + np.array([tx, ty, tz])

def rotate_all(vertex, alpha, beta, gamma, tx, ty, tz):
    for vert in range(len(vertex)):
        print(vertex[vert], end=' ')
        vertex[vert] = rotate(*vertex[vert], alpha, beta, gamma, tx, ty, tz)
        print(vertex[vert])
    return vertex


W, H = 1000, 1000

img = np.zeros((H, W, 3), dtype=np.uint8)
z_buffer = np.full((H, W), np.inf)
img[0:H, 0:W] = 255
vertex, polygons = vertex_parser("model_1.obj"), polygons_parser("model_1.obj")
vertex = rotate_all(vertex, 0, 1.5, 0, 0, -0.03, 0.05)
draw_polygons_n(vertex, polygons, 1)
print(rotate(1, 2, 3, math.pi, 0, 0, 0, 0, 0))


image = Image.fromarray(img, mode="RGB")
image.save("triangle.png")
