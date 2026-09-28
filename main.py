from pathlib import Path
import math
import time

import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from scene import (
    Camera2D,
    Engine,
    Node,
    Scene,
    Transform,
)

from graphics import (
    ColorMaterial,
    Disk,
    Pipeline,
    Quad,
    Renderer,
    Sampler,
    Shader,
    Texture,
    TextureSet,
)


BASE_DIR = Path(__file__).resolve().parent
ASSETS = BASE_DIR / "assets"
SHADER_FILE = BASE_DIR / "shaders" / "solar.wgsl"


# ============================================================
# Parâmetros da cena
# ============================================================

WORLD_W = 12.0
WORLD_H = 8.0

EARTH_ORBIT_RADIUS = 7.0
MERCURY_ORBIT_RADIUS = 3.4
MOON_ORBIT_RADIUS = 1.65

SUN_RADIUS = 1.55
EARTH_RADIUS = 0.82
MOON_RADIUS = 0.30
MERCURY_RADIUS = 0.48

EARTH_ORBIT_SPEED = math.radians(18.0)
EARTH_SPIN_SPEED = math.radians(105.0)
MOON_ORBIT_SPEED = math.radians(85.0)
MERCURY_ORBIT_SPEED = math.radians(52.0)
MERCURY_SPIN_SPEED = math.radians(35.0)
SUN_SPIN_SPEED = math.radians(8.0)


# ============================================================
# Engine de animação
# ============================================================

class SolarSystemEngine(Engine):
    def __init__(
        self,
        earth_orbit,
        earth_spin,
        moon_orbit,
        mercury_orbit,
        mercury_spin,
        sun_spin,
    ):
        self.earth_orbit = earth_orbit
        self.earth_spin = earth_spin
        self.moon_orbit = moon_orbit
        self.mercury_orbit = mercury_orbit
        self.mercury_spin = mercury_spin
        self.sun_spin = sun_spin

        self.earth_orbit_angle = 0.0
        self.earth_spin_angle = 0.0
        self.moon_orbit_angle = math.radians(35.0)
        self.mercury_orbit_angle = math.radians(145.0)
        self.mercury_spin_angle = 0.0
        self.sun_spin_angle = 0.0

    def update(self, dt):
        dt = min(dt, 0.1)

        self.earth_orbit_angle += (
            EARTH_ORBIT_SPEED * dt
        )

        self.earth_spin_angle += (
            EARTH_SPIN_SPEED * dt
        )

        self.moon_orbit_angle += (
            MOON_ORBIT_SPEED * dt
        )

        self.mercury_orbit_angle += (
            MERCURY_ORBIT_SPEED * dt
        )

        self.mercury_spin_angle += (
            MERCURY_SPIN_SPEED * dt
        )

        self.sun_spin_angle += (
            SUN_SPIN_SPEED * dt
        )

        self.earth_orbit.reset().rotate(
            self.earth_orbit_angle
        )

        self.earth_spin.reset().rotate(
            self.earth_spin_angle
        )

        self.moon_orbit.reset().rotate(
            self.moon_orbit_angle
        )

        self.mercury_orbit.reset().rotate(
            self.mercury_orbit_angle
        )

        self.mercury_spin.reset().rotate(
            self.mercury_spin_angle
        )

        self.sun_spin.reset().rotate(
            self.sun_spin_angle
        )


# ============================================================
# Janela + WebGPU
# ============================================================

canvas = RenderCanvas(
    size=(1200, 800),
    title="INF1761 - Mini Sistema Solar 2D",
    update_mode="continuous",
    max_fps=60,
)

context = canvas.get_context("wgpu")

adapter = wgpu.gpu.request_adapter_sync(
    power_preference="high-performance",
    canvas=context,
)

device = adapter.request_device_sync()

target_format = context.get_preferred_format(
    adapter
)

context.configure(
    device=device,
    format=target_format,
    alpha_mode="opaque",
)


# ============================================================
# Pipeline / shader
# ============================================================

shader = Shader(
    device,
    SHADER_FILE,
)

pipeline = Pipeline(
    device,
    shader,
    target_format,
)


# ============================================================
# Geometrias
# ============================================================

disk = Disk(
    device,
    segments=128,
)

quad = Quad(device)


# ============================================================
# Texturas PNG
# ============================================================

sampler = Sampler(device)

texture_sets = {}

for name in [
    "space",
    "sun",
    "earth",
    "moon",
    "mercury",
]:
    texture = Texture(
        device,
        ASSETS / f"{name}.png",
    )

    texture_sets[name] = TextureSet(
        texture,
        sampler,
    )


# ============================================================
# Material
# ============================================================

white = ColorMaterial(
    1.0,
    1.0,
    1.0,
    1.0,
)

white.register(shader)

for texture_set in texture_sets.values():
    texture_set.register(shader)


# ============================================================
# Grafo de cena
#
# root
# ├── background
# ├── sun_spin -> sun
# ├── mercury_orbit -> mercury_offset
# │   └── mercury_spin -> mercury
# └── earth_orbit -> earth_offset
#     ├── earth_spin -> earth
#     └── moon_orbit -> moon_offset -> moon
#
# A Lua NÃO é filha de earth_spin.
# ============================================================

root = Node(
    name="root",
    pipeline=pipeline,
    apps=[white],
)


# Fundo
root.add_node(
    Node(
        name="background",
        trf=(
            Transform()
            .scale(WORLD_W, WORLD_H)
        ),
        apps=[texture_sets["space"]],
        shps=[quad],
    )
)


# Sol
sun_spin = Transform()

sun_spin_node = root.add_node(
    Node(
        name="sun_spin",
        trf=sun_spin,
    )
)

sun_spin_node.add_node(
    Node(
        name="sun",
        trf=(
            Transform()
            .scale(SUN_RADIUS)
        ),
        apps=[texture_sets["sun"]],
        shps=[disk],
    )
)


# Mercúrio
mercury_orbit = Transform()

mercury_orbit_node = root.add_node(
    Node(
        name="mercury_orbit",
        trf=mercury_orbit,
    )
)

mercury_offset_node = (
    mercury_orbit_node.add_node(
        Node(
            name="mercury_offset",
            trf=(
                Transform()
                .translate(
                    MERCURY_ORBIT_RADIUS,
                    0.0,
                )
            ),
        )
    )
)

mercury_spin = Transform()

mercury_spin_node = (
    mercury_offset_node.add_node(
        Node(
            name="mercury_spin",
            trf=mercury_spin,
        )
    )
)

mercury_spin_node.add_node(
    Node(
        name="mercury",
        trf=(
            Transform()
            .scale(MERCURY_RADIUS)
        ),
        apps=[texture_sets["mercury"]],
        shps=[disk],
    )
)


# Terra
earth_orbit = Transform()

earth_orbit_node = root.add_node(
    Node(
        name="earth_orbit",
        trf=earth_orbit,
    )
)

earth_offset_node = (
    earth_orbit_node.add_node(
        Node(
            name="earth_offset",
            trf=(
                Transform()
                .translate(
                    EARTH_ORBIT_RADIUS,
                    0.0,
                )
            ),
        )
    )
)


# Rotação própria da Terra
earth_spin = Transform()

earth_spin_node = (
    earth_offset_node.add_node(
        Node(
            name="earth_spin",
            trf=earth_spin,
        )
    )
)

earth_spin_node.add_node(
    Node(
        name="earth",
        trf=(
            Transform()
            .scale(EARTH_RADIUS)
        ),
        apps=[texture_sets["earth"]],
        shps=[disk],
    )
)


# Lua
# Ela é filha de earth_offset, e não de earth_spin.
moon_orbit = Transform()

moon_orbit_node = (
    earth_offset_node.add_node(
        Node(
            name="moon_orbit",
            trf=moon_orbit,
        )
    )
)

moon_offset_node = (
    moon_orbit_node.add_node(
        Node(
            name="moon_offset",
            trf=(
                Transform()
                .translate(
                    MOON_ORBIT_RADIUS,
                    0.0,
                )
            ),
        )
    )
)

moon_offset_node.add_node(
    Node(
        name="moon",
        trf=(
            Transform()
            .scale(MOON_RADIUS)
        ),
        apps=[texture_sets["moon"]],
        shps=[disk],
    )
)


# ============================================================
# Scene / Engine / Camera / Renderer
# ============================================================

scene = Scene(root)

scene.add_engine(
    SolarSystemEngine(
        earth_orbit,
        earth_spin,
        moon_orbit,
        mercury_orbit,
        mercury_spin,
        sun_spin,
    )
)

camera = Camera2D(
    -WORLD_W,
    WORLD_W,
    -WORLD_H,
    WORLD_H,
)

renderer = Renderer(
    device,
    context,
)


# ============================================================
# Loop
# ============================================================

last_time = time.perf_counter()


def draw_frame():
    global last_time

    now = time.perf_counter()
    dt = now - last_time
    last_time = now

    scene.update(dt)
    renderer.render(
        scene,
        camera,
    )


canvas.request_draw(draw_frame)

loop.run()
