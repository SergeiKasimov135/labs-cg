import numpy as np
from PIL import Image, ImageOps
from math import sin, cos, pi, sqrt
import math
import random
from tqdm import tqdm

class RenderParams:
    def __init__(self, width=4000, height=4000, proj_coef=10000, proj_shift=2):
        self.width = width
        self.height = height
        self.proj_coef = proj_coef
        self.proj_shift = proj_shift
        self.z_buffer = np.full((height, width), np.inf)
        self.img = np.zeros((height, width, 3), dtype=np.uint8)

def bresenham_line(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
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
    for x in range(x0, x1):
        t = (x-x0)/(x1 - x0) if (x1 - x0) != 0 else 0
        y = round((1.0 - t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if (derror > (x1- x0)):
            derror -= 2 * (x1 - x0)
            y += y_update

def vertex_parser(filename):
    f = open(filename)

    arr = []
    for s in f:
        splitted = s.split(" ")
        if (splitted[0] == 'v'):
            arr.append([float(el) for el in splitted[1:]])
    f.close()
    return arr

def draw_vertex(vertex, params):
    img = np.zeros((params.height, params.width, 3), dtype=np.uint8)
    for coord in vertex:
        x = int(params.proj_coef * coord[0] + params.width/2)
        y = int(params.proj_coef * coord[1] + params.height/2)
        if 0 <= x < params.width and 0 <= y < params.height:
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
            vertices = []
            for elem in splitted[1:]:
                if elem:
                    vertices.append(int(elem.split("/")[0]) - 1)

            if len(vertices) > 3:
                for i in range(1, len(vertices) - 1):
                    res.append([vertices[0], vertices[i], vertices[i+1]])
            else:
                res.append(vertices)
    f.close()
    return res

def textures_parser(filename):
    f = open(filename)
    arr = []
    for s in f:
        splitted = s.split(" ")
        if (splitted[0] == 'vt'):
            coords = [float(el) for el in splitted[1:] if el]
            arr.append(coords)
    f.close()
    return arr

def textures_c_parser(filename):
    f = open(filename)
    res = []
    for s in f:
        splitted = s.split(" ")
        if (splitted[0] == 'f'):
            vertices = []
            for elem in splitted[1:]:
                if elem:
                    parts = elem.split("/")
                    if len(parts) > 1 and parts[1]:
                        vertices.append(int(parts[1]) - 1)
                    else:
                        vertices.append(0)
            if len(vertices) > 3:
                for i in range(1, len(vertices) - 1):
                    res.append([vertices[0], vertices[i], vertices[i+1]])
            else:
                res.append(vertices)
    f.close()
    return res


def draw_polygons(vertex, polygons, params):
    img = params.img
    for vert in polygons:
        x0, y0 = (1000 * vertex[vert[0]][0] + 500), (1000 * vertex[vert[0]][1] + 500)
        x1, y1 = (1000 * vertex[vert[1]][0] + 500), (1000 * vertex[vert[1]][1] + 500)
        x2, y2 = (1000 * vertex[vert[2]][0] + 500), (1000 * vertex[vert[2]][1] + 500)
        draw_triangle(img, x0, y0, x1, y1, x2, y2, [random.randrange(30, 256) for i in range(3)], params)
    image = Image.fromarray(img)
    image = ImageOps.flip(image)
    image.save("polygons.png")

def draw_polygons_n(vertex, polygons, params, n=0):
    img = params.img
    l = np.array([0, 0, 1])

    for vert in polygons:
        x0, y0, z0 = vertex[vert[0]][0], vertex[vert[0]][1], vertex[vert[0]][2]
        x1, y1, z1 = vertex[vert[1]][0], vertex[vert[1]][1], vertex[vert[1]][2]
        x2, y2, z2 = vertex[vert[2]][0], vertex[vert[2]][1], vertex[vert[2]][2]
        norm = norm_cal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        cosN = np.dot(norm, l) / (np.linalg.norm(norm) * np.linalg.norm(l))
        if cosN < 0:
            draw_triangle_n(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, [cosN * -255, cosN * -255, cosN * -255], params)
    image = Image.fromarray(img)
    image = ImageOps.flip(image)
    image.save(f"polygons_{n}.png")

def draw_polygons_guru(vertex, polygons, vn_calc, params, n=1):
    img = params.img
    l = np.array([0, 0, 1])

    for index, vert in enumerate(tqdm(polygons, desc="Рендерим Методом Гуру:")):
        x0, y0, z0 = vertex[vert[0]][0], vertex[vert[0]][1], vertex[vert[0]][2]
        x1, y1, z1 = vertex[vert[1]][0], vertex[vert[1]][1], vertex[vert[1]][2]
        x2, y2, z2 = vertex[vert[2]][0], vertex[vert[2]][1], vertex[vert[2]][2]
        I0 = np.dot(vn_calc[vert[0]], l) /(np.linalg.norm(vn_calc[vert[0]]) * np.linalg.norm(l))
        I1 = np.dot(vn_calc[vert[1]], l) /(np.linalg.norm(vn_calc[vert[1]]) * np.linalg.norm(l))
        I2 = np.dot(vn_calc[vert[2]], l) /(np.linalg.norm(vn_calc[vert[2]]) * np.linalg.norm(l))
        norm = norm_cal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        cosN = np.dot(norm, l) / (np.linalg.norm(norm) * np.linalg.norm(l))
        if cosN < 0:
            draw_triangle_guru(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, I0, I1, I2, params)
    image = Image.fromarray(img)
    image = ImageOps.flip(image)
    image.save(f"polygons_{n}.png")

def draw_polygons_t(vertex, polygons, vn_calc, textures, textures_c, params, texture_img, n=2):
    img = params.img
    l = np.array([0, 0, 1])
    wt, ht = texture_img.size

    for index, vert in enumerate(tqdm(polygons, desc="Рендерим с текстурами")):
        x0, y0, z0 = vertex[vert[0]][0], vertex[vert[0]][1], vertex[vert[0]][2]
        x1, y1, z1 = vertex[vert[1]][0], vertex[vert[1]][1], vertex[vert[1]][2]
        x2, y2, z2 = vertex[vert[2]][0], vertex[vert[2]][1], vertex[vert[2]][2]

        tex_coords = textures_c[index]

        u0, v0 = textures[tex_coords[0]][0], textures[tex_coords[0]][1]
        u1, v1 = textures[tex_coords[1]][0], textures[tex_coords[1]][1]
        u2, v2 = textures[tex_coords[2]][0], textures[tex_coords[2]][1]
        I0 = np.dot(vn_calc[vert[0]], l) / (np.linalg.norm(vn_calc[vert[0]]) * np.linalg.norm(l))
        I1 = np.dot(vn_calc[vert[1]], l) / (np.linalg.norm(vn_calc[vert[1]]) * np.linalg.norm(l))
        I2 = np.dot(vn_calc[vert[2]], l) / (np.linalg.norm(vn_calc[vert[2]]) * np.linalg.norm(l))
        norm = norm_cal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        cosN = np.dot(norm, l) / (np.linalg.norm(norm) * np.linalg.norm(l))
        if cosN < 0:
            draw_triangle_t(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, I0, I1, I2, u0, v0, u1, v1, u2, v2, params, texture_img, wt, ht)
    image = Image.fromarray(img)
    image = ImageOps.flip(image)
    image.save(f"polygons_{n}.png")

def bary(x0, y0, x1, y1, x2, y2, x, y):
    l0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l2 = 1.0 - l0 - l1
    return l0, l1, l2

def draw_triangle(img, x0, y0, x1, y1, x2, y2, light, params):
    x0_proj = 1000 * x0 + 500
    y0_proj = 1000 * y0 + 500
    x1_proj = 1000 * x1 + 500
    y1_proj = 1000 * y1 + 500
    x2_proj = 1000 * x2 + 500
    y2_proj = 1000 * y2 + 500

    xmin = math.floor(min(x0_proj, x1_proj, x2_proj))
    xmax = math.ceil(max(x0_proj, x1_proj, x2_proj))
    ymin = math.floor(min(y0_proj, y1_proj, y2_proj))
    ymax = math.ceil(max(y0_proj, y1_proj, y2_proj))
    if xmin < 0: xmin = 0
    if xmax >= params.height: xmax = params.height
    if ymin < 0: ymin = 0
    if ymax >= params.width: ymax = params.width

    for x in range(xmin, xmax+1):
        for y in range(ymin, ymax+1):
            l0, l1, l2 = bary(x0_proj, y0_proj, x1_proj, y1_proj, x2_proj, y2_proj, x, y)
            if l0 >= 0 and l1 >= 0 and l2 >= 0:
                img[y, x] = light


def draw_triangle_guru(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, I0, I1, I2, params):
    if z0 <= 0 or z1 <= 0 or z2 <= 0:
        return

    x0_proj = params.proj_coef * x0 / z0 + params.width / params.proj_shift
    y0_proj = params.proj_coef * y0 / z0 + params.height / params.proj_shift
    x1_proj = params.proj_coef * x1 / z1 + params.width / params.proj_shift
    y1_proj = params.proj_coef * y1 / z1 + params.height / params.proj_shift
    x2_proj = params.proj_coef * x2 / z2 + params.width / params.proj_shift
    y2_proj = params.proj_coef * y2 / z2 + params.height / params.proj_shift

    xmin = math.floor(min(x0_proj, x1_proj, x2_proj))
    xmax = math.ceil(max(x0_proj, x1_proj, x2_proj))
    ymin = math.floor(min(y0_proj, y1_proj, y2_proj))
    ymax = math.ceil(max(y0_proj, y1_proj, y2_proj))
    if xmin < 0: xmin = 0
    if xmax >= params.height: xmax = params.height
    if ymin < 0: ymin = 0
    if ymax >= params.width: ymax = params.width

    for x in range(xmin, xmax+1):
        for y in range(ymin, ymax+1):
            l0, l1, l2 = bary(x0_proj, y0_proj, x1_proj, y1_proj, x2_proj, y2_proj, x, y)
            if l0 < 0 or l1 < 0 or l2 < 0:
                continue

            z_x = l0 * z0 + l1 * z1 + l2 * z2

            if z_x < params.z_buffer[y, x] and l0 >= 0 and l1 >= 0 and l2 >= 0:
                intensity = -225 * (I0 * l0 + I1 * l1 + I2 * l2)

                intensity = max(0, min(255, intensity))
                img[y, x] = [intensity, intensity, intensity]
                params.z_buffer[y, x] = z_x

def draw_triangle_n(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, light, params):
    if z0 <= 0 or z1 <= 0 or z2 <= 0:
        return

    x0_proj = params.proj_coef * x0 / z0 + params.width / params.proj_shift
    y0_proj = params.proj_coef * y0 / z0 + params.height / params.proj_shift
    x1_proj = params.proj_coef * x1 / z1 + params.width / params.proj_shift
    y1_proj = params.proj_coef * y1 / z1 + params.height / params.proj_shift
    x2_proj = params.proj_coef * x2 / z2 + params.width / params.proj_shift
    y2_proj = params.proj_coef * y2 / z2 + params.height / params.proj_shift

    xmin = math.floor(min(x0_proj, x1_proj, x2_proj))
    xmax = math.ceil(max(x0_proj, x1_proj, x2_proj))
    ymin = math.floor(min(y0_proj, y1_proj, y2_proj))
    ymax = math.ceil(max(y0_proj, y1_proj, y2_proj))
    if xmin < 0: xmin = 0
    if xmax >= params.height: xmax = params.height
    if ymin < 0: ymin = 0
    if ymax >= params.width: ymax = params.width

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l0, l1, l2 = bary(x0_proj, y0_proj, x1_proj, y1_proj, x2_proj, y2_proj, x, y)
            if l0 < 0 or l1 < 0 or l2 < 0:
                continue

            z_x = l0 * z0 + l1 * z1 + l2 * z2

            if z_x < params.z_buffer[y, x] and l0 >= 0 and l1 >= 0 and l2 >= 0:
                img[y, x] = light
                params.z_buffer[y, x] = z_x


def draw_triangle_t(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, I0, I1, I2, u0, v0, u1, v1, u2, v2, params, texture_img, wt, ht):
    if z0 <= 0 or z1 <= 0 or z2 <= 0:
        return

    x0_proj = params.proj_coef * x0 / z0 + params.width / params.proj_shift
    y0_proj = params.proj_coef * y0 / z0 + params.height / params.proj_shift
    x1_proj = params.proj_coef * x1 / z1 + params.width / params.proj_shift
    y1_proj = params.proj_coef * y1 / z1 + params.height / params.proj_shift
    x2_proj = params.proj_coef * x2 / z2 + params.width / params.proj_shift
    y2_proj = params.proj_coef * y2 / z2 + params.height / params.proj_shift

    xmin = math.floor(min(x0_proj, x1_proj, x2_proj))
    xmax = math.ceil(max(x0_proj, x1_proj, x2_proj))
    ymin = math.floor(min(y0_proj, y1_proj, y2_proj))
    ymax = math.ceil(max(y0_proj, y1_proj, y2_proj))
    if xmin < 0: xmin = 0
    if xmax >= params.height: xmax = params.height
    if ymin < 0: ymin = 0
    if ymax >= params.width: ymax = params.width

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l0, l1, l2 = bary(x0_proj, y0_proj, x1_proj, y1_proj, x2_proj, y2_proj, x, y)
            if l0 < 0 or l1 < 0 or l2 < 0:
                continue

            z_x = l0 * z0 + l1 * z1 + l2 * z2

            if z_x < params.z_buffer[y, x]:
                u_r, v_r = int(wt * (l0 * u0 + l1 * u1 + l2 * u2)), int(ht * (l0 * v0 + l1 * v1 + l2 * v2))
                intensity = -(I0 * l0 + I1 * l1 + I2 * l2)
                intensity = max(0, min(255, intensity))

                img[y, x] = np.array(texture_img.getpixel((u_r, v_r))) * intensity
                params.z_buffer[y, x] = z_x

def norm_cal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n1 = np.array([x1 - x2, y1 - y2, z1 - z2])
    n2 = np.array([x1 - x0, y1 - y0, z1 - z0])
    n = np.cross(n1, n2)
    return n

def rotate_euler(x, y, z, alpha, beta, gamma, tx, ty, tz, scale=1.0):
    R_x = np.array([[1, 0, 0],
                     [0, cos(alpha), sin(alpha)],
                     [0, -sin(alpha), cos(alpha)]])

    R_y = np.array([[cos(beta), 0, sin(beta)],
                     [0, 1, 0],
                     [-sin(beta), 0, cos(beta)]])

    R_z = np.array([[cos(gamma), sin(gamma), 0],
                     [-sin(gamma), cos(gamma), 0],
                     [0, 0, 1]])

    R = R_x @ R_y @ R_z

    point = np.array([x, y, z]) * scale
    rotated = R @ point

    return rotated + np.array([tx, ty, tz])


def quaternion_to_rotation_matrix(q):
    q = q / np.linalg.norm(q)
    a, b, c, d = q

    rotation_matrix = np.array([
        [1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
        [2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b],
        [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, 1 - 2 * b * b - 2 * c * c]
    ])

    return rotation_matrix


def rotate_quaternion(x, y, z, q, tx, ty, tz, scale=1.0):
    rotation_matrix = quaternion_to_rotation_matrix(q)
    point = np.array([x, y, z]) * scale
    rotated_point = np.dot(rotation_matrix, point)
    return rotated_point + np.array([tx, ty, tz])


def transform_vertex(vertex, alpha=0, beta=0, gamma=0, tx=0, ty=0, tz=0, scale=1.0, use_quaternion=False, quaternion=None):
    transformed = []

    if use_quaternion:
        if quaternion is None:
            quaternion =  np.array([1, 0, 0, 0])

        for v in vertex:
            if len(v) == 2:
                v = [v[0], v[1], 0]
            while len(v) < 3:
                v.append(0)

            transformed_v = rotate_quaternion(v[0], v[1], v[2], quaternion, tx, ty, tz, scale)
            transformed.append(transformed_v)
    else:
        for v in vertex:
            if len(v) == 2:
                v = [v[0], v[1], 0]
            while len(v) < 3:
                v.append(0)

            transformed_v = rotate_euler(v[0], v[1], v[2], alpha, beta, gamma, tx, ty, tz, scale)
            transformed.append(transformed_v)

    return transformed


def vn_calc(vertex, polygons):
    vn_calc = np.zeros((len(vertex), 3))
    for vert in polygons:
        x0, y0, z0 = vertex[vert[0]][0], vertex[vert[0]][1], vertex[vert[0]][2]
        x1, y1, z1 = vertex[vert[1]][0], vertex[vert[1]][1], vertex[vert[1]][2]
        x2, y2, z2 = vertex[vert[2]][0], vertex[vert[2]][1], vertex[vert[2]][2]
        norm = norm_cal(x0, y0, z0, x1, y1, z1, x2, y2, z2)

        vn_calc[vert[0]] += norm
        vn_calc[vert[1]] += norm
        vn_calc[vert[2]] += norm

    for i in range(len(vn_calc)):
        if np.linalg.norm(vn_calc[i]) > 0:
            vn_calc[i] = vn_calc[i] / np.linalg.norm(vn_calc[i])

    return vn_calc

def load_model(filename, params, alpha=0, beta=0, gamma=0, tx=0, ty=0, tz=0, scale=1.0, use_quaternion=False, quaternion=None, texture_file=None):
    vertex = vertex_parser(filename)
    polygons = polygons_parser(filename)

    vertex = transform_vertex(vertex, alpha, beta, gamma, tx, ty, tz, scale, use_quaternion, quaternion)

    vert_normals = vn_calc(vertex, polygons)

    texture_img = None
    textures = []
    textures_c = []

    if texture_file:
        texture_img = Image.open(texture_file)
        texture_img = ImageOps.flip(texture_img)
        texture_img = texture_img.convert('RGB')

        textures = textures_parser(filename)
        textures_c = textures_c_parser(filename)

    return {
        'vertex': vertex,
        'polygons': polygons,
        'normals': vert_normals,
        'textures': textures,
        'texture_coords': textures_c,
        'texture_img': texture_img
    }


def render_model(model, params, render_mode='guru', output_file=None):
    if render_mode == 'wireframe':
        for vert in model['polygons']:
            if len(vert) < 3:
                continue
            for i in range(len(vert)):
                j = (i + 1) % len(vert)
                v0 = model['vertex'][vert[i]]
                v1 = model['vertex'][vert[j]]
                x0, y0 = v0[0], v0[1]
                x1, y1 = v1[0], v1[1]
                bresenham_line(params.img, int(x0), int(y0), int(x1), int(y1), [255, 255, 255])

    elif render_mode == 'flat':
        draw_polygons_n(model['vertex'], model['polygons'], params)

    elif render_mode == 'guru':
        draw_polygons_guru(model['vertex'], model['polygons'], model['normals'], params)

    elif render_mode == 'textured':
        if model['texture_img'] is not None:
            draw_polygons_t(model['vertex'], model['polygons'], model['normals'], model['textures'],
                            model['texture_coords'], params, model['texture_img'])
        else:
            draw_polygons_guru(model['vertex'], model['polygons'], model['normals'], params)

    if output_file:
        image = Image.fromarray(params.img)
        image = ImageOps.flip(image)
        image.save(output_file)

    return params.img

def main():
    params = RenderParams(width=2000, height=2000, proj_coef=10000, proj_shift=2)

    model1 = load_model(
        "model.obj",
        params,
        alpha=0,
        beta=math.pi,
        gamma=0,
        tx=0,
        ty=0,
        tz=50.8,
        scale=1.0,
        texture_file='model.bmp'
    )


    render_model(model1, params, render_mode='textured', output_file="model1.png")

    model2 = load_model(
        "cat.obj",
        params,
        alpha=0,
        beta=1.5 * math.pi,
        gamma=0,
        tx=0.0,
        ty=0,
        tz=2000,
        scale=2,
        texture_file='cat-atlas.jpg'
    )

    model3 = load_model(
        "model_1.obj",
        params,
        alpha=0,
        beta=math.pi,
        gamma=0,
        tx=0.0,
        ty=-1.5,
        tz=400,
        scale=0.7,
        texture_file="bunny-atlas.jpg"
    )


    model4 = load_model(
        "model.obj",
        params,
        alpha=0,
        beta=math.pi,
        gamma=0,
        tx=3,
        ty=0,
        tz=50.8,
        scale=1.0,
        texture_file='model.bmp'
    )


    render_model(model2, params, render_mode='textured', output_file="combined_models.png")

    render_model(model4, params, render_mode='guru', output_file="combined_models1.png")
    render_model(model3, params, render_mode='guru',  output_file="combined_models2.png")


if __name__ == "__main__":
    main()