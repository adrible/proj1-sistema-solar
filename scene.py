import math
import numpy as np


# ============================================================
# Matrizes 2D homogêneas (4x4)
# ============================================================

def identity():
    return np.identity(4, dtype=np.float32)


def translation(x, y, z=0.0):
    m = identity()
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(sx, sy=None, sz=1.0):
    if sy is None:
        sy = sx

    m = identity()
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotation_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)

    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s,  c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def ortho(left, right, bottom, top, near=-1.0, far=1.0):
    m = identity()

    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)

    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)

    return m


def gpu_matrix_bytes(matrix):
    # WGSL usa armazenamento por colunas.
    return np.asarray(matrix, dtype=np.float32).T.tobytes()


# ============================================================
# Transform
# ============================================================

class Transform:
    def __init__(self):
        self.mat = identity()

    def reset(self):
        self.mat = identity()
        return self

    def translate(self, x, y, z=0.0):
        self.mat = self.mat @ translation(x, y, z)
        return self

    def scale(self, sx, sy=None, sz=1.0):
        self.mat = self.mat @ scale(sx, sy, sz)
        return self

    def rotate(self, angle):
        self.mat = self.mat @ rotation_z(angle)
        return self

    def get_matrix(self):
        return self.mat

    def load(self, state):
        state.push_matrix(self.mat)

    def unload(self, state):
        state.pop_matrix()


# ============================================================
# Grafo de cena
# ============================================================

class Node:
    """
    Estrutura usada no material da disciplina:

    parent
    pipeline
    trf
    apps[]
    shps[]
    nodes[]
    """

    def __init__(
        self,
        name="node",
        pipeline=None,
        trf=None,
        apps=None,
        shps=None,
        nodes=None,
    ):
        self.name = name
        self.parent = None
        self.pipeline = pipeline
        self.trf = trf
        self.apps = [] if apps is None else list(apps)
        self.shps = [] if shps is None else list(shps)
        self.nodes = []

        if nodes:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node):
        node.parent = self
        self.nodes.append(node)
        return node

    def get_model_matrix(self):
        chain = []
        node = self

        while node is not None:
            if node.trf is not None:
                chain.append(node.trf.get_matrix())
            node = node.parent

        model = identity()

        for matrix in reversed(chain):
            model = model @ matrix

        return model

    def render(self, state):
        # 1. carregar
        if self.pipeline is not None:
            self.pipeline.load(state)

        if self.trf is not None:
            self.trf.load(state)

        for app in self.apps:
            app.load(state)

        state.load_matrices()

        # 2. desenhar
        for shape in self.shps:
            shape.draw(state)

        # 3. percorrer filhos
        for node in self.nodes:
            node.render(state)

        # 4. descarregar em ordem reversa
        for app in reversed(self.apps):
            app.unload(state)

        if self.trf is not None:
            self.trf.unload(state)

        if self.pipeline is not None:
            self.pipeline.unload(state)


class Engine:
    def update(self, dt):
        raise NotImplementedError


class Scene:
    def __init__(self, root):
        self.root = root
        self.engines = []

    def add_engine(self, engine):
        self.engines.append(engine)

    def update(self, dt):
        for engine in self.engines:
            engine.update(dt)

    def render(self, state):
        self.root.render(state)


class Camera2D:
    def __init__(self, left, right, bottom, top):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

    def get_projection_matrix(self):
        return ortho(
            self.left,
            self.right,
            self.bottom,
            self.top,
        )
