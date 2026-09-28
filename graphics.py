from pathlib import Path

import numpy as np
from PIL import Image
import wgpu

from scene import identity, gpu_matrix_bytes


MAX_MATRICES = 128


# ============================================================
# Aparências
# ============================================================

class ColorMaterial:
    GROUP = 1

    def __init__(self, r=1.0, g=1.0, b=1.0, a=1.0):
        self.color = np.array([r, g, b, a], dtype=np.float32)
        self.bind_groups = {}

    def register(self, shader):
        shader.add_material(self)

    def load(self, state):
        shader = state.get_shader()
        state.push_bind_group(
            self.GROUP,
            self.bind_groups[id(shader)],
        )

    def unload(self, state):
        state.pop_bind_group(self.GROUP)


class Texture:
    def __init__(self, device, filename):
        image = Image.open(filename).convert("RGBA")
        pixels = np.asarray(image, dtype=np.uint8)

        width, height = image.size

        self.texture = device.create_texture(
            size=(width, height, 1),
            dimension="2d",
            format="rgba8unorm-srgb",
            usage=(
                wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_DST
            ),
        )

        device.queue.write_texture(
            {"texture": self.texture},
            pixels.tobytes(),
            {
                "bytes_per_row": width * 4,
                "rows_per_image": height,
            },
            (width, height, 1),
        )

        self.view = self.texture.create_view()


class Sampler:
    def __init__(self, device):
        self.sampler = device.create_sampler(
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            mag_filter="linear",
            min_filter="linear",
        )


class TextureSet:
    GROUP = 2

    def __init__(self, texture, sampler):
        self.texture = texture
        self.sampler = sampler
        self.bind_groups = {}

    def register(self, shader):
        shader.add_texture_set(self)

    def load(self, state):
        shader = state.get_shader()
        state.push_bind_group(
            self.GROUP,
            self.bind_groups[id(shader)],
        )

    def unload(self, state):
        state.pop_bind_group(self.GROUP)


# ============================================================
# Formas
# ============================================================

class Shape:
    def draw(self, state):
        raise NotImplementedError


class IndexedShape(Shape):
    def __init__(self, device, vertices, indices):
        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.uint32)

        self.vertex_buffer = device.create_buffer_with_data(
            data=vertices.tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )

        self.index_buffer = device.create_buffer_with_data(
            data=indices.tobytes(),
            usage=wgpu.BufferUsage.INDEX,
        )

        self.index_count = len(indices)

    def draw(self, state):
        shader = state.get_shader()

        instance = shader.commit_matrix(state)

        state.render_pass.set_vertex_buffer(
            0,
            self.vertex_buffer,
        )

        state.render_pass.set_index_buffer(
            self.index_buffer,
            wgpu.IndexFormat.uint32,
        )

        state.render_pass.draw_indexed(
            self.index_count,
            1,
            0,
            0,
            instance,
        )


class Disk(IndexedShape):
    def __init__(self, device, segments=128):
        vertices = [[0.0, 0.0, 0.5, 0.5]]

        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments

            x = np.cos(angle)
            y = np.sin(angle)

            s = 0.5 * (x + 1.0)
            t = 0.5 * (1.0 - y)

            vertices.append([x, y, s, t])

        indices = []

        for i in range(1, segments + 1):
            indices.extend([0, i, i + 1])

        super().__init__(device, vertices, indices)


class Quad(IndexedShape):
    def __init__(self, device):
        vertices = [
            [-1.0, -1.0, 0.0, 1.0],
            [ 1.0, -1.0, 1.0, 1.0],
            [ 1.0,  1.0, 1.0, 0.0],
            [-1.0,  1.0, 0.0, 0.0],
        ]

        indices = [
            0, 1, 2,
            0, 2, 3,
        ]

        super().__init__(device, vertices, indices)


# ============================================================
# Estado da travessia
# ============================================================

class State:
    def __init__(self, camera, render_pass):
        self.camera = camera
        self.render_pass = render_pass

        self.matrix_stack = [identity()]
        self.pipeline_stack = []
        self.bind_group_stacks = {}

        self.vertex = identity()
        self.used_shaders = []

    def push_matrix(self, matrix):
        self.matrix_stack.append(
            self.matrix_stack[-1] @ matrix
        )

    def pop_matrix(self):
        self.matrix_stack.pop()

    def load_matrices(self):
        self.vertex = self.matrix_stack[-1]

    def push_pipeline(self, pipeline):
        self.pipeline_stack.append(pipeline)
        self.render_pass.set_pipeline(
            pipeline.gpu_pipeline
        )

    def pop_pipeline(self):
        self.pipeline_stack.pop()

        if self.pipeline_stack:
            self.render_pass.set_pipeline(
                self.pipeline_stack[-1].gpu_pipeline
            )

    def get_shader(self):
        return self.pipeline_stack[-1].shader

    def push_bind_group(self, group, bind_group):
        stack = self.bind_group_stacks.setdefault(
            group,
            [],
        )

        stack.append(bind_group)

        self.render_pass.set_bind_group(
            group,
            bind_group,
        )

    def pop_bind_group(self, group):
        stack = self.bind_group_stacks[group]
        stack.pop()

        if stack:
            self.render_pass.set_bind_group(
                group,
                stack[-1],
            )

    def mark_shader(self, shader):
        if shader not in self.used_shaders:
            self.used_shaders.append(shader)


# ============================================================
# Shader / Pipeline
# ============================================================

class Shader:
    MATRIX_GROUP = 0
    MATERIAL_GROUP = 1
    TEXTURE_GROUP = 2
    GLOBAL_GROUP = 3

    def __init__(self, device, filename):
        self.device = device

        code = Path(filename).read_text(
            encoding="utf-8"
        )

        self.module = device.create_shader_module(
            code=code
        )

        self.pipeline = None
        self.frame_matrices = []

    def attach_pipeline(self, pipeline):
        self.pipeline = pipeline

        self.matrix_buffer = self.device.create_buffer(
            size=MAX_MATRICES * 64,
            usage=(
                wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_DST
            ),
        )

        self.matrix_bind_group = (
            self.device.create_bind_group(
                layout=(
                    pipeline.gpu_pipeline
                    .get_bind_group_layout(
                        self.MATRIX_GROUP
                    )
                ),
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self.matrix_buffer,
                            "offset": 0,
                            "size": MAX_MATRICES * 64,
                        },
                    }
                ],
            )
        )

        self.global_buffer = self.device.create_buffer(
            size=64,
            usage=(
                wgpu.BufferUsage.UNIFORM
                | wgpu.BufferUsage.COPY_DST
            ),
        )

        self.global_bind_group = (
            self.device.create_bind_group(
                layout=(
                    pipeline.gpu_pipeline
                    .get_bind_group_layout(
                        self.GLOBAL_GROUP
                    )
                ),
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self.global_buffer,
                            "offset": 0,
                            "size": 64,
                        },
                    }
                ],
            )
        )

    def begin_frame(self, state):
        self.frame_matrices = []

        projection = (
            state.camera.get_projection_matrix()
        )

        self.device.queue.write_buffer(
            self.global_buffer,
            0,
            gpu_matrix_bytes(projection),
        )

        state.render_pass.set_bind_group(
            self.MATRIX_GROUP,
            self.matrix_bind_group,
        )

        state.render_pass.set_bind_group(
            self.GLOBAL_GROUP,
            self.global_bind_group,
        )

        state.mark_shader(self)

    def commit_matrix(self, state):
        index = len(self.frame_matrices)

        if index >= MAX_MATRICES:
            raise RuntimeError(
                "Número máximo de matrizes excedido."
            )

        self.frame_matrices.append(
            state.vertex.copy()
        )

        return index

    def flush_matrices(self):
        if not self.frame_matrices:
            return

        data = b"".join(
            gpu_matrix_bytes(matrix)
            for matrix in self.frame_matrices
        )

        self.device.queue.write_buffer(
            self.matrix_buffer,
            0,
            data,
        )

    def add_material(self, material):
        key = id(self)

        if key in material.bind_groups:
            return

        buffer = self.device.create_buffer(
            size=16,
            usage=(
                wgpu.BufferUsage.UNIFORM
                | wgpu.BufferUsage.COPY_DST
            ),
        )

        self.device.queue.write_buffer(
            buffer,
            0,
            material.color.tobytes(),
        )

        bind_group = self.device.create_bind_group(
            layout=(
                self.pipeline.gpu_pipeline
                .get_bind_group_layout(
                    self.MATERIAL_GROUP
                )
            ),
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": buffer,
                        "offset": 0,
                        "size": 16,
                    },
                }
            ],
        )

        material.bind_groups[key] = bind_group

    def add_texture_set(self, texture_set):
        key = id(self)

        if key in texture_set.bind_groups:
            return

        bind_group = self.device.create_bind_group(
            layout=(
                self.pipeline.gpu_pipeline
                .get_bind_group_layout(
                    self.TEXTURE_GROUP
                )
            ),
            entries=[
                {
                    "binding": 0,
                    "resource": texture_set.texture.view,
                },
                {
                    "binding": 1,
                    "resource": texture_set.sampler.sampler,
                },
            ],
        )

        texture_set.bind_groups[key] = bind_group


class Pipeline:
    def __init__(
        self,
        device,
        shader,
        target_format,
    ):
        self.shader = shader

        self.gpu_pipeline = (
            device.create_render_pipeline(
                layout="auto",
                vertex={
                    "module": shader.module,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": 16,
                            "step_mode": "vertex",
                            "attributes": [
                                {
                                    "shader_location": 0,
                                    "offset": 0,
                                    "format": "float32x2",
                                },
                                {
                                    "shader_location": 1,
                                    "offset": 8,
                                    "format": "float32x2",
                                },
                            ],
                        }
                    ],
                },
                primitive={
                    "topology": "triangle-list",
                    "front_face": "ccw",
                    "cull_mode": "none",
                },
                fragment={
                    "module": shader.module,
                    "entry_point": "fs_main",
                    "targets": [
                        {"format": target_format}
                    ],
                },
            )
        )

        shader.attach_pipeline(self)

    def load(self, state):
        state.push_pipeline(self)
        self.shader.begin_frame(state)

    def unload(self, state):
        state.pop_pipeline()


# ============================================================
# Renderer
# ============================================================

class Renderer:
    def __init__(self, device, context):
        self.device = device
        self.context = context

    def render(self, scene, camera):
        texture = (
            self.context.get_current_texture()
        )

        view = texture.create_view()

        encoder = (
            self.device.create_command_encoder()
        )

        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": (
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                }
            ]
        )

        state = State(
            camera,
            render_pass,
        )

        scene.render(state)

        render_pass.end()

        for shader in state.used_shaders:
            shader.flush_matrices()

        self.device.queue.submit(
            [encoder.finish()]
        )
