//! GLSL shader sources and compilation helpers.

use glow::HasContext;

/// Vertex shader for filled/stroked paths.
///
/// Transforms vertices by the entity's scale+translate transform, and passes
/// the *local-space* position to the fragment shader for gradient
/// interpolation.
pub const PATH_VERTEX_SRC: &str = r#"#version 140

in vec2 a_position;

// Entity transform: output = offset + scale * input
uniform vec2 u_scale;
uniform vec2 u_offset;

// Viewport resolution for NDC conversion
uniform vec2 u_resolution;

// Local-space position for gradient interpolation
out vec2 v_local;

void main() {
    v_local = a_position;

    vec2 world = u_offset + u_scale * a_position;

    // Convert from [0, resolution] to [-1, 1] (flip Y for GL)
    vec2 ndc = (world / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
"#;

/// Fragment shader for filled paths.
///
/// Supports three shader modes via `u_shader_type`:
///   0 = solid color
///   1 = vertical gradient (top→bottom in local space)
///   2 = horizontal gradient (left→right in local space)
pub const PATH_FRAGMENT_SRC: &str = r#"#version 140

in vec2 v_local;

uniform int u_shader_type;
uniform vec4 u_color_a;   // solid color, or gradient start
uniform vec4 u_color_b;   // gradient end (unused for solid)
uniform vec2 u_bounds;    // [min, max] for gradient axis

out vec4 frag_color;

void main() {
    if (u_shader_type == 0) {
        // Solid color
        frag_color = u_color_a;
    } else {
        // Gradient
        float coord;
        if (u_shader_type == 1) {
            coord = v_local.y;  // vertical
        } else {
            coord = v_local.x;  // horizontal
        }
        float range = u_bounds.y - u_bounds.x;
        float t = (range > 0.0) ? clamp((coord - u_bounds.x) / range, 0.0, 1.0) : 0.0;
        frag_color = mix(u_color_a, u_color_b, t);
    }

    // Premultiply alpha for correct blending
    frag_color.rgb *= frag_color.a;
}
"#;

/// Vertex shader for textured quads (images).
///
/// The image entity uses the scene's unit rectangle [0,1]x[0,1] transformed
/// by the entity transform. We generate UV coordinates from the local position.
pub const IMAGE_VERTEX_SRC: &str = r#"#version 140

in vec2 a_position;

uniform vec2 u_scale;
uniform vec2 u_offset;
uniform vec2 u_resolution;
uniform bool u_flip_uv_y;

out vec2 v_uv;

void main() {
    v_uv = a_position;
    if (u_flip_uv_y) {
        v_uv.y = 1.0 - v_uv.y;
    }

    vec2 world = u_offset + u_scale * a_position;
    vec2 ndc = (world / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
"#;

/// Fragment shader for textured quads.
pub const IMAGE_FRAGMENT_SRC: &str = r#"#version 140

in vec2 v_uv;

uniform sampler2D u_texture;

out vec4 frag_color;

void main() {
    frag_color = texture(u_texture, v_uv);
    // Premultiply alpha
    frag_color.rgb *= frag_color.a;
}
"#;

/// Compile a shader program from vertex + fragment source.
///
/// # Safety
/// Requires a valid GL context to be current.
pub unsafe fn compile_program(
    gl: &glow::Context,
    vertex_src: &str,
    fragment_src: &str,
) -> Result<glow::Program, String> {
    let program = gl.create_program()?;

    let vs = compile_shader(gl, glow::VERTEX_SHADER, vertex_src)?;
    let fs = compile_shader(gl, glow::FRAGMENT_SHADER, fragment_src)?;

    gl.attach_shader(program, vs);
    gl.attach_shader(program, fs);
    gl.link_program(program);

    if !gl.get_program_link_status(program) {
        let log = gl.get_program_info_log(program);
        gl.delete_program(program);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        return Err(format!("Program link error: {log}"));
    }

    // Shaders can be detached after linking
    gl.detach_shader(program, vs);
    gl.detach_shader(program, fs);
    gl.delete_shader(vs);
    gl.delete_shader(fs);

    Ok(program)
}

unsafe fn compile_shader(
    gl: &glow::Context,
    shader_type: u32,
    source: &str,
) -> Result<glow::Shader, String> {
    let shader = gl.create_shader(shader_type)?;
    gl.shader_source(shader, source);
    gl.compile_shader(shader);

    if !gl.get_shader_compile_status(shader) {
        let log = gl.get_shader_info_log(shader);
        gl.delete_shader(shader);
        return Err(format!("Shader compile error: {log}"));
    }

    Ok(shader)
}
