# Nome: Vítor Beneti Martins
# N°USP: 11877635

# A ideia da cena é representar a logo de 5 marcas de carros: Citroen, Mitsubishi, Chevrolet, Audi e Renault
# As logos foram feitas a partir de poligonos
# Citroen: Elipse (ao redor) e 2 'Chevrons' 
# Mitsubishi: 3 losangos (diamonds), sendo o segundo uma rotação de 120° do primeiro, e o terceiro uma rotação de 240° do primeiro
# Chevrolet: 2 paralelepipedos formando uma cruz e 2 outros paralelepipedos na mesma forma, mas em escala maior, a fim de dar o efeito de contorno
# Audi: 4 rosquinhas (circulos) entrelaçadas
# Renault: 1 losango e um hexagono externo ao losango

# Controles: 
# W, A, S, D: translação
# Setas: rotação em X e Y
# P: Mostrar/esconder malha poligonal
# Z: aumentar escala
# X: reduzir escala

# Todos os objetos se mexem da mesma forma, pois a ideia é explorar a forma de cada logo e enxerga-las juntamente

# Os comentarios no codigo são em inglês, pois é um costume pessoal e gosto de postar meus trabalhos no github 

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math

def init_glfw():
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(1000, 1000, "Car Logos in 3D", None, None)
    glfw.make_context_current(window)
    return window

def mult_matrix(a,b):
    m_a = a.reshape(4,4)
    m_b = b.reshape(4,4)
    m_c = np.dot(m_a,m_b)
    c = m_c.reshape(1,16)
    return c

# Creates a rotation matrix based on the angle and axis of rotation
# Each axis has a different rotation matrix 
def create_rotation_matrix(angle, axis):
    rad = math.radians(angle)
    cos_theta = math.cos(rad)
    sin_theta = math.sin(rad)
    if axis == 'x':
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta, 0.0],
            [0.0, sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    elif axis == 'y':
        return np.array([
            [cos_theta, 0.0, sin_theta, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_theta, 0.0, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    elif axis == 'z':
        return np.array([
            [cos_theta, -sin_theta, 0.0, 0.0],
            [sin_theta, cos_theta, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

# Creates a translation matrix based on the translation values for each axis
def create_translation_matrix(tx, ty, tz):
    return np.array([
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

# Creates a scaling matrix based on the scale values for each axis
# Actually I only used regular scaling 
def create_scale_matrix(sx, sy, sz):
    return np.array([
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)    

def create_shaders():
    vertex_code = """
        attribute vec3 position;
        uniform mat4 mat_transformation;
        void main(){
            gl_Position = mat_transformation * vec4(position, 1.0);
        }
        """

    fragment_code = """
        uniform vec4 color;
        void main(){
            gl_FragColor = color;
        }
        """

    program = glCreateProgram()
    vertex = glCreateShader(GL_VERTEX_SHADER)
    fragment = glCreateShader(GL_FRAGMENT_SHADER)

    glShaderSource(vertex, vertex_code)
    glShaderSource(fragment, fragment_code)

    glCompileShader(vertex)
    if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(vertex).decode()
        print(error)
        raise RuntimeError("Erro de compilacao do Vertex Shader")

    glCompileShader(fragment)
    if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(fragment).decode()
        print(error)
        raise RuntimeError("Erro de compilacao do Fragment Shader")

    glAttachShader(program, vertex)
    glAttachShader(program, fragment)

    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        print(glGetProgramInfoLog(program))
        raise RuntimeError('Linking error')
    
    glUseProgram(program)
    return program

# Used for many logos both mitsubishi and citroen logos
def generate_diamond_vertices(center_x, center_y, width, height, depth):
    half_width = width / 2
    half_height = height / 2
    half_depth = depth / 2
    vertices = np.array([
        # Front face
        [center_x, center_y + half_height, half_depth],
        [center_x + half_width, center_y, half_depth],
        [center_x, center_y - half_height, half_depth],
        [center_x - half_width, center_y, half_depth],
        # Back face
        [center_x, center_y + half_height, -half_depth],
        [center_x + half_width, center_y, -half_depth],
        [center_x, center_y - half_height, -half_depth],
        [center_x - half_width, center_y, -half_depth],
    ], dtype=np.float32)
    return vertices

def generate_diamond_indices():
    return np.array([
        # Front face
        0, 1, 2,
        0, 2, 3,
        # Back face
        4, 6, 5,
        4, 7, 6,
        # Top face
        0, 4, 5,
        0, 5, 1,
        # Bottom face
        2, 6, 7,
        2, 7, 3,
        # Right face
        1, 5, 6,
        1, 6, 2,
        # Left face
        0, 3, 7,
        0, 7, 4
    ], dtype=np.uint32)

# Used for Chevrolet logo
def generate_cross_vertices(center_x, center_y, width, height, depth):
    half_width = width/2
    half_height = height/2
    half_depth = depth/2
    vertices = np.array([
        # Vertical bar
        [center_x - 0.1, center_y + half_height, half_depth],
        [center_x + 0.1, center_y + half_height, half_depth],
        [center_x + 0.1, center_y - half_height, half_depth],
        [center_x - 0.1, center_y - half_height, half_depth],
        [center_x - 0.1, center_y + half_height, -half_depth],
        [center_x + 0.1, center_y + half_height, -half_depth],
        [center_x + 0.1, center_y - half_height, -half_depth],
        [center_x - 0.1, center_y - half_height, -half_depth],
        # Horizontal bar
        [center_x - half_width, center_y + 0.1, half_depth],
        [center_x + half_width, center_y + 0.1, half_depth],
        [center_x + half_width, center_y - 0.1, half_depth],
        [center_x - half_width, center_y - 0.1, half_depth],
        [center_x - half_width, center_y + 0.1, -half_depth],
        [center_x + half_width, center_y + 0.1, -half_depth],
        [center_x + half_width, center_y - 0.1, -half_depth],
        [center_x - half_width, center_y - 0.1, -half_depth],
    ], dtype=np.float32)
    return vertices

def generate_cross_indices():
    return np.array([
        # Vertical box
        0, 1, 2, 0, 2, 3,  # Front
        4, 6, 5, 4, 7, 6,  # Back
        0, 4, 5, 0, 5, 1,  # Top
        2, 6, 7, 2, 7, 3,  # Bottom
        1, 5, 6, 1, 6, 2,  # Right
        0, 3, 7, 0, 7, 4,  # Left
        # Horizontal box
        8, 9, 10, 8, 10, 11,  # Front
        12, 14, 13, 12, 15, 14,  # Back
        8, 12, 13, 8, 13, 9,  # Top
        10, 14, 15, 10, 15, 11,  # Bottom
        9, 13, 14, 9, 14, 10,  # Right
        8, 11, 15, 8, 15, 12,  # Left
    ], dtype=np.uint32)

# Used for Audi logo
def generate_circle_vertices(center_x, center_y, outer_radius, inner_radius, depth, num_segments):
    vertices = []
    for i in range(num_segments):
        theta = 2.0 * math.pi * i / num_segments
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Outer vertices (front and back)
        vertices.append([center_x + outer_radius * cos_theta, center_y + outer_radius * sin_theta, depth/2])
        vertices.append([center_x + outer_radius * cos_theta, center_y + outer_radius * sin_theta, -depth/2])

        # Inner vertices (front and back)
        vertices.append([center_x + inner_radius * cos_theta, center_y + inner_radius * sin_theta, depth/2])
        vertices.append([center_x + inner_radius * cos_theta, center_y + inner_radius * sin_theta, -depth/2])

    return np.array(vertices, dtype=np.float32)

def generate_circle_indices(num_segments):
    indices = []
    for i in range(num_segments):
        base = i * 4
        next_base = ((i + 1) % num_segments) * 4

        # Outer side
        indices.extend([base, next_base, base + 1])
        indices.extend([next_base, next_base + 1, base + 1])

        # Inner side
        indices.extend([base + 2, base + 3, next_base + 2])
        indices.extend([next_base + 2, base + 3, next_base + 3])

        # Front face
        indices.extend([base, base + 2, next_base])
        indices.extend([next_base, base + 2, next_base + 2])

        # Back face
        indices.extend([base + 1, next_base + 1, base + 3])
        indices.extend([next_base + 1, next_base + 3, base + 3])

    return np.array(indices, dtype=np.uint32)

# Used for Renault logo (outer)
def generate_hexagon_vertices(center_x, center_y, width, height, depth):
    half_width = width / 2
    half_height = height / 2
    half_depth = depth / 2
    vertices = np.array([
        # Front face
        [center_x - half_width, center_y, half_depth],
        [center_x - half_width/2, center_y + half_height, half_depth],
        [center_x + half_width/2, center_y + half_height, half_depth],
        [center_x + half_width, center_y, half_depth],
        [center_x + half_width/2, center_y - half_height, half_depth],
        [center_x - half_width/2, center_y - half_height, half_depth],
        # Back face
        [center_x - half_width, center_y, -half_depth],
        [center_x - half_width/2, center_y + half_height, -half_depth],
        [center_x + half_width/2, center_y + half_height, -half_depth],
        [center_x + half_width, center_y, -half_depth],
        [center_x + half_width/2, center_y - half_height, -half_depth],
        [center_x - half_width/2, center_y - half_height, -half_depth],
    ], dtype=np.float32)
    return vertices

def generate_hexagon_indices():
    return np.array([
        # Front face
        0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5,
        # Back face
        6, 8, 7, 6, 9, 8, 6, 10, 9, 6, 11, 10,
        # Side faces
        0, 6, 7, 0, 7, 1,
        1, 7, 8, 1, 8, 2,
        2, 8, 9, 2, 9, 3,
        3, 9, 10, 3, 10, 4,
        4, 10, 11, 4, 11, 5,
        5, 11, 6, 5, 6, 0
    ], dtype=np.uint32)

# Used for Citroen logo (outer) 
# Actually I could have used it for audi logo as well 
def generate_ellipse_vertices(x, y, width, height, thickness, num_segments):
    vertices = []
    for i in range(num_segments + 1):
        theta = 2.0 * np.pi * i / num_segments
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Outer edge of ellipse
        x_outer = x + width * 0.47 * cos_theta
        y_outer = y + height * 0.47 * sin_theta
        vertices.append([x_outer, y_outer, thickness * 0.47])
        vertices.append([x_outer, y_outer, -thickness * 0.47])
        
        # Inner edge of ellipse
        x_inner = x + (width - thickness) * 0.5 * cos_theta
        y_inner = y + (height - thickness) * 0.5 * sin_theta
        vertices.append([x_inner, y_inner, thickness * 0.5])
        vertices.append([x_inner, y_inner, -thickness * 0.5])
    
    return np.array(vertices, dtype=np.float32)

def generate_ellipse_indices(num_segments):
    indices = []
    for i in range(num_segments):
        base = i * 4
        indices.extend([
            base, base+1, base+4,
            base+1, base+5, base+4,
            base+2, base+3, base+6,
            base+3, base+7, base+6,
            base, base+2, base+4,
            base+2, base+6, base+4,
            base+1, base+3, base+5,
            base+3, base+7, base+5
        ])
    return np.array(indices, dtype=np.uint32)

# Used for Citroen logo (inner)
def generate_chevron_vertices(x, y, width, height, thickness, num_segments):
    vertices = []
    for i in range(num_segments + 1):
        t = i / num_segments
        # Outer edge of chevron
        x_outer = x + width * t
        y_outer = y + height * (0.5 - abs(t - 0.5))
        vertices.append([x_outer, y_outer, 0.0])
        # Inner edge of chevron
        x_inner = x + width * t
        y_inner = y + (height - thickness) * (0.5 - abs(t - 0.5))
        vertices.append([x_inner, y_inner, 0.0])
    return np.array(vertices, dtype=np.float32)

def generate_chevron_indices(num_segments):
    indices = []
    for i in range(num_segments):
        indices.extend([
            2*i, 2*i+1, 2*i+2,
            2*i+1, 2*i+3, 2*i+2
        ])
    return np.array(indices, dtype=np.uint32)

def setup_buffer(vertices, indices, program):
    # Vertex buffer
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
    loc = glGetAttribLocation(program, "position")
    glEnableVertexAttribArray(loc)
    glVertexAttribPointer(loc, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

    # Index buffer
    ibo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

# Mitsubishi logo is composed of 3 diamonds
def setup_mitsubishi_logo():
    diamond1 = generate_diamond_vertices(0.0, 0.2, 0.2, 0.4, 0.1)
    diamond2 = generate_diamond_vertices(0.0, 0.2, 0.2, 0.4, 0.1)
    diamond3 = generate_diamond_vertices(0.0, 0.2, 0.2, 0.4, 0.1)
    mitsubishi_logo = np.vstack((diamond1, diamond2, diamond3))
    mitsubishi_indices = np.concatenate([generate_diamond_indices(), 
                                         generate_diamond_indices() + 8, 
                                         generate_diamond_indices() + 16])
    return mitsubishi_logo, mitsubishi_indices


# Chevrolet logo is composed of a cross 
def setup_chevrolet_logo():
    chevrolet_logo = generate_cross_vertices(0.0, 0.0, 0.6, 0.35, 0.2)
    chevrolet_indices = generate_cross_indices()
    return chevrolet_logo, chevrolet_indices

# Audi logo is composed of 4 circles
def setup_audi_logo():
    num_segments = 64
    circle_vertices = []
    circle_indices = []
    offset = 0
    for i in range(4):
        x = 0.20 * i - 0.225  # Horizontally aligned
        y = 0.0
        vertices = generate_circle_vertices(x, y, 0.15, 0.12, 0.1, num_segments)
        indices = generate_circle_indices(num_segments) + offset
        circle_vertices.append(vertices)
        circle_indices.append(indices)
        offset += len(vertices)
    return np.vstack(circle_vertices), np.concatenate(circle_indices)

# Renault logo is composed of a hexagon and a diamond
def setup_renault_logo():
    renault_logo = generate_hexagon_vertices(0.0, 0.0, 0.35, 0.6, 0.1)
    renault_indices = generate_hexagon_indices()
    diamond = generate_diamond_vertices(0.0, 0.0, 0.15, 0.4, 0.15)
    renault_logo = np.vstack((renault_logo, diamond))
    diamond_indices = generate_diamond_indices() + len(renault_logo) - len(diamond)
    renault_indices = np.concatenate((renault_indices, diamond_indices))
    return renault_logo, renault_indices

# Citroen logo is composed of 2 chevrons and an ellipse
def setup_citroen_logo():
    num_segments = 64 # good number to make it look round
    left_chevron = generate_chevron_vertices(-0.2, 0.0, 0.4, 0.2, 0.1, num_segments)
    right_chevron = generate_chevron_vertices(-0.2, -0.1, 0.4, 0.2, 0.1, num_segments)
    ellipse = generate_ellipse_vertices(0.0, 0.0, 0.5, 0.6, 0.1, num_segments)
    citroen_logo = np.vstack((left_chevron, right_chevron, ellipse))
    left_indices = generate_chevron_indices(num_segments)
    right_indices = generate_chevron_indices(num_segments) + len(left_chevron)
    ellipse_indices = generate_ellipse_indices(num_segments) + len(left_chevron) + len(right_chevron)
    citroen_indices = np.concatenate((left_indices, right_indices, ellipse_indices))
    return citroen_logo, citroen_indices

# Draws the Mitsubishi logo
# Picks second diamond, rotates it 120 degrees and draws it
# Picks third diamond, rotates it 240 degrees and draws it
def draw_mitsubishi_logo(loc_mat_transformation, loc_color, mat_transformation_base):
    for i in range(3):
        mat_rotation_diamond = create_rotation_matrix(120 * i, 'z')
        mat_transformation_diamond = mult_matrix(mat_transformation_base, mat_rotation_diamond)
        glUniformMatrix4fv(loc_mat_transformation, 1, GL_TRUE, mat_transformation_diamond)
        glUniform4f(loc_color, 1.0, 0.0, 0.0, 1.0)  # Red color for Mitsubishi
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, ctypes.c_void_p(i * 36 * 4))

# Draws the Chevrolet logo
def draw_chevrolet_logo(loc_mat_transformation, loc_color, mat_transformation_base):
    
    # Draws the contour (slightly larger) than the inner part
    contour_scale = create_scale_matrix(1.1, 1.4, 0.99)
    mat_transformation_contour = mult_matrix(mat_transformation_base, contour_scale)
    mat_transformation_contour = mult_matrix(mat_transformation_contour, create_rotation_matrix(180, 'z'))
    glUniformMatrix4fv(loc_mat_transformation, 1, GL_TRUE, mat_transformation_contour)
    glUniform4f(loc_color, 0.75, 0.75, 0.75, 1.0)  # Silver color 
    glDrawElements(GL_TRIANGLES, 72, GL_UNSIGNED_INT, ctypes.c_void_p(108 * 4))

    # Draw the main logo
    mat_transformation_chevrolet = mult_matrix(mat_transformation_base, create_rotation_matrix(180, 'z'))
    glUniformMatrix4fv(loc_mat_transformation, 1, GL_TRUE, mat_transformation_chevrolet)
    glUniform4f(loc_color, 1.0, 0.843, 0.0, 1.0)  # Golden color 
    glDrawElements(GL_TRIANGLES, 72, GL_UNSIGNED_INT, ctypes.c_void_p(108 * 4))  # offset after Mitsubishi

def draw_audi_logo(loc_mat_transformation, loc_color, mat_transformation_base, num_audi_indices):
    glUniformMatrix4fv(loc_mat_transformation, 1, GL_TRUE, mat_transformation_base)
    glUniform4f(loc_color, 0.75, 0.75, 0.75, 1.0)  # grey (almost silver) 
    glDrawElements(GL_TRIANGLES, num_audi_indices, GL_UNSIGNED_INT, ctypes.c_void_p((108 + 72) * 4))  # Offset after Mitsubishi and Chevrolet

def draw_renault_logo(loc_mat_transformation, loc_color, mat_transformation_base, num_renault_indices, offset):
    # Hexagon Part
    glUniformMatrix4fv(loc_mat_transformation, 1, GL_TRUE, mat_transformation_base)
    glUniform4f(loc_color, 1.0, 0.84, 0.0, 1.0)  # Yellow for the hexagon
    glDrawElements(GL_TRIANGLES, num_renault_indices - 36, GL_UNSIGNED_INT, ctypes.c_void_p(offset * 4))
    
    # Diamond Part
    glUniform4f(loc_color, 1.0, 1.0, 1.0, 1.0)  # White color for the inner diamond
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, ctypes.c_void_p((offset + num_renault_indices - 36) * 4))

def draw_citroen_logo(loc_mat_transformation, loc_color, mat_transformation_base, num_citroen_indices, offset):
    glUniformMatrix4fv(loc_mat_transformation, 1, GL_TRUE, mat_transformation_base)
    glUniform4f(loc_color, 0.529, 0.808, 0.922, 1.0)  # Light blue 
    glDrawElements(GL_TRIANGLES, num_citroen_indices, GL_UNSIGNED_INT, ctypes.c_void_p(offset * 4))

def main():
    window = init_glfw()
    program = create_shaders()
    
    mitsubishi_logo, mitsubishi_indices = setup_mitsubishi_logo()
    chevrolet_logo, chevrolet_indices = setup_chevrolet_logo()
    audi_logo, audi_indices = setup_audi_logo()
    renault_logo, renault_indices = setup_renault_logo()
    citroen_logo, citroen_indices = setup_citroen_logo()
    
    all_vertices = np.vstack((mitsubishi_logo, chevrolet_logo, audi_logo, renault_logo, citroen_logo))
    all_indices = np.concatenate([mitsubishi_indices, 
                                  chevrolet_indices + len(mitsubishi_logo), 
                                  audi_indices + len(mitsubishi_logo) + len(chevrolet_logo),
                                  renault_indices + len(mitsubishi_logo) + len(chevrolet_logo) + len(audi_logo),
                                  citroen_indices + len(mitsubishi_logo) + len(chevrolet_logo) + len(audi_logo) + len(renault_logo)])
    
    setup_buffer(all_vertices, all_indices, program)
    
    loc_color = glGetUniformLocation(program, "color")
    loc_mat_transformation = glGetUniformLocation(program, "mat_transformation")
    
    glfw.show_window(window)
    
    rotation_angle_x = 0.0
    rotation_angle_y = 0.0
    rotation_speed = 2.0
    show_mesh = False
    scale_factor = 1.0
    scale_speed = 0.1
    translation_x = 0.0
    translation_y = 0.0
    translation_z = 0.0
    translation_speed = 0.05
    
    def key_callback(window, key, scancode, action, mods):
        nonlocal rotation_angle_x, rotation_angle_y, show_mesh, scale_factor, translation_x, translation_y, translation_z
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_LEFT:
                rotation_angle_y += rotation_speed
            elif key == glfw.KEY_RIGHT:
                rotation_angle_y -= rotation_speed
            elif key == glfw.KEY_UP:
                rotation_angle_x += rotation_speed
            elif key == glfw.KEY_DOWN:
                rotation_angle_x -= rotation_speed
            elif key == glfw.KEY_P:
                show_mesh = not show_mesh
            elif key == glfw.KEY_Z:
                scale_factor += scale_speed
            elif key == glfw.KEY_X:
                scale_factor = max(0.1, scale_factor - scale_speed)
            elif key == glfw.KEY_W:
                translation_y += translation_speed
            elif key == glfw.KEY_S:
                translation_y -= translation_speed
            elif key == glfw.KEY_A:
                translation_x -= translation_speed
            elif key == glfw.KEY_D:
                translation_x += translation_speed
    
    glfw.set_key_callback(window, key_callback)
    
    glEnable(GL_DEPTH_TEST)
    
    while not glfw.window_should_close(window):
        glfw.poll_events()
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
        # Matrices for Mitsubishi Logo --------------------------------
        # Translation
        M_tx, M_ty, M_tz = -0.5 + translation_x, 0.5 + translation_y, 0.0 + translation_z
        M_mat_translation = create_translation_matrix(M_tx, M_ty, M_tz)
        # Rotation
        M_mat_rotation_x = create_rotation_matrix(rotation_angle_x, 'x')
        M_mat_rotation_y = create_rotation_matrix(rotation_angle_y, 'y')
        M_mat_rotation = mult_matrix(M_mat_rotation_x, M_mat_rotation_y)
        # Scale
        M_s_x, M_s_y, M_s_z = 0.25 * scale_factor, 0.25 * scale_factor, 0.25 * scale_factor   
        M_mat_scale = create_scale_matrix(M_s_x, M_s_y, M_s_z)
        # Transformation
        M_mat_transformation = mult_matrix(M_mat_translation, M_mat_rotation)
        M_mat_transformation = mult_matrix(M_mat_scale, M_mat_transformation)
        # ------------------------------------------------------------

        # Matrices for Chevrolet Logo -------------------------------
        # Translation   
        C_tx, C_ty, C_tz = 0.5 + translation_x, 0.5 + translation_y, 0.0 + translation_z
        C_mat_translation = create_translation_matrix(C_tx, C_ty, C_tz)
        # Rotation
        C_mat_rotation_x = create_rotation_matrix(rotation_angle_x, 'x')
        C_mat_rotation_y = create_rotation_matrix(rotation_angle_y, 'y')
        C_mat_rotation = mult_matrix(C_mat_rotation_x, C_mat_rotation_y)
        # Scale
        C_s_x, C_s_y, C_s_z = 0.25 * scale_factor, 0.25 * scale_factor, 0.25 * scale_factor   
        C_mat_scale = create_scale_matrix(C_s_x, C_s_y, C_s_z)
        # Transformation
        C_mat_transformation = mult_matrix(C_mat_translation, C_mat_rotation)
        C_mat_transformation = mult_matrix(C_mat_scale, C_mat_transformation)
        # ------------------------------------------------------------

        # Matrices for Audi Logo ----------------------------------------
        # Translation
        A_tx, A_ty, A_tz = -0.5 + translation_x, -0.5 + translation_y, 0.0 + translation_z
        A_mat_translation = create_translation_matrix(A_tx, A_ty, A_tz)
        # Rotation
        A_mat_rotation_x = create_rotation_matrix(rotation_angle_x, 'x')
        A_mat_rotation_y = create_rotation_matrix(rotation_angle_y, 'y')
        A_mat_rotation = mult_matrix(A_mat_rotation_x, A_mat_rotation_y)
        # Scale
        A_s_x, A_s_y, A_s_z = 0.25 * scale_factor, 0.25 * scale_factor, 0.25 * scale_factor
        A_mat_scale = create_scale_matrix(A_s_x, A_s_y, A_s_z)
        # Transformation
        A_mat_transformation = mult_matrix(A_mat_translation, A_mat_rotation)
        A_mat_transformation = mult_matrix(A_mat_scale, A_mat_transformation)
        # ------------------------------------------------------------

        # Matrices for Renault Logo ----------------------------------------
        # Translation
        R_tx, R_ty, R_tz = 0.5 + translation_x, -0.5 + translation_y, 0.0 + translation_z
        R_mat_translation = create_translation_matrix(R_tx, R_ty, R_tz)
        # Rotation
        R_mat_rotation_x = create_rotation_matrix(rotation_angle_x, 'x')
        R_mat_rotation_y = create_rotation_matrix(rotation_angle_y, 'y')
        R_mat_rotation = mult_matrix(R_mat_rotation_x, R_mat_rotation_y)
        # Scale
        R_s_x, R_s_y, R_s_z = 0.25 * scale_factor, 0.25 * scale_factor, 0.25 * scale_factor
        R_mat_scale = create_scale_matrix(R_s_x, R_s_y, R_s_z)
        # Transformation
        R_mat_transformation = mult_matrix(R_mat_translation, R_mat_rotation)
        R_mat_transformation = mult_matrix(R_mat_scale, R_mat_transformation)
        # ------------------------------------------------------------

        # Matrices for Citroen Logo ----------------------------------------
        # Translation
        Ci_tx, Ci_ty, Ci_tz = -1.5 + translation_x, 0.5 + translation_y, 0.0 + translation_z
        Ci_mat_translation = create_translation_matrix(Ci_tx, Ci_ty, Ci_tz)
        # Rotation
        Ci_mat_rotation_x = create_rotation_matrix(rotation_angle_x, 'x')
        Ci_mat_rotation_y = create_rotation_matrix(rotation_angle_y, 'y')
        Ci_mat_rotation = mult_matrix(Ci_mat_rotation_x, Ci_mat_rotation_y)
        # Scale
        Ci_s_x, Ci_s_y, Ci_s_z = 0.25 * scale_factor, 0.25 * scale_factor, 0.25 * scale_factor
        Ci_mat_scale = create_scale_matrix(Ci_s_x, Ci_s_y, Ci_s_z)
        # Transformation
        Ci_mat_transformation = mult_matrix(Ci_mat_translation, Ci_mat_rotation)
        Ci_mat_transformation = mult_matrix(Ci_mat_scale, Ci_mat_transformation)
        # ------------------------------------------------------------
        # Choose between wireframe and solid mesh
        if show_mesh:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # Draw all logos (the order of drawing is not the order that they appear in the scene) 
        # Because I started drawing then in this order
        draw_mitsubishi_logo(loc_mat_transformation, loc_color, M_mat_transformation)
        draw_chevrolet_logo(loc_mat_transformation, loc_color, C_mat_transformation)
        draw_audi_logo(loc_mat_transformation, loc_color, A_mat_transformation, len(audi_indices))
        draw_renault_logo(loc_mat_transformation, loc_color, R_mat_transformation, len(renault_indices), 108 + 72 + len(audi_indices))
        draw_citroen_logo(loc_mat_transformation, loc_color, Ci_mat_transformation, len(citroen_indices), 108 + 72 + len(audi_indices) + len(renault_indices))
        
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
