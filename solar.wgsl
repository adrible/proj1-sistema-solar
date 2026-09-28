struct Matrix {
    vertex: mat4x4<f32>,
};

@group(0) @binding(0)
var<storage, read> matrices: array<Matrix>;


struct Material {
    color: vec4<f32>,
};

@group(1) @binding(0)
var<uniform> material: Material;


@group(2) @binding(0)
var image_texture: texture_2d<f32>;

@group(2) @binding(1)
var image_sampler: sampler;


struct Global {
    projection: mat4x4<f32>,
};

@group(3) @binding(0)
var<uniform> global: Global;


struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
};


@vertex
fn vs_main(
    @builtin(instance_index) i: u32,
    @location(0) position: vec2<f32>,
    @location(1) texcoord: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    let p =
        matrices[i].vertex *
        vec4<f32>(
            position,
            0.0,
            1.0
        );

    out.position =
        global.projection * p;

    out.texcoord = texcoord;

    return out;
}


@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    let texel = textureSample(
        image_texture,
        image_sampler,
        in.texcoord
    );

    if (texel.a < 0.05) {
        discard;
    }

    return texel * material.color;
}
