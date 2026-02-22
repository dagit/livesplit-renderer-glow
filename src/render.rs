//! The main renderer: owns GL state, consumes livesplit-core [`Scene`] data,
//! and issues draw calls.

use glow::{HasContext, PixelUnpackData};
use livesplit_core::layout::LayoutState;
use livesplit_core::rendering::{
    Background, Entity, FillShader, Handle, LabelHandle, SceneManager, Transform,
};
use livesplit_core::settings::ImageCache;
use std::sync::Arc;

use crate::allocator::GlAllocator;
use crate::shaders;
use crate::types::*;

/// Shadow offset in component coordinate space, matching livesplit-core's
/// internal constant (which is not publicly exported).
const SHADOW_OFFSET: f32 = 0.05;

/// Cached uniform locations for the path shader program.
struct PathUniforms {
    scale: glow::UniformLocation,
    offset: glow::UniformLocation,
    resolution: glow::UniformLocation,
    shader_type: glow::UniformLocation,
    color_a: glow::UniformLocation,
    color_b: glow::UniformLocation,
    bounds: glow::UniformLocation,
}

/// Cached uniform locations for the image shader program.
struct ImageUniforms {
    scale: glow::UniformLocation,
    offset: glow::UniformLocation,
    resolution: glow::UniformLocation,
    texture: glow::UniformLocation,
    flip_uv_y: glow::UniformLocation,
}

/// A GPU-accelerated renderer for livesplit-core layouts.
///
/// # Usage
///
/// ```no_run
/// # use livesplit_renderer_glow::GlowRenderer;
/// // During setup (with a current GL context):
/// let renderer = unsafe { GlowRenderer::new(gl) };
///
/// // Each frame:
/// unsafe {
///     renderer.render(&layout_state, &image_cache, [width, height]);
/// }
/// ```
/// Number of MSAA samples for antialiasing.
const MSAA_SAMPLES: i32 = 4;

pub struct GlowRenderer {
    gl: Arc<glow::Context>,
    allocator: GlAllocator,
    scene_manager: SceneManager<Option<GlPath>, GlImage, GlFont, GlLabel>,

    // GL resources
    path_program: glow::Program,
    path_uniforms: PathUniforms,
    image_program: glow::Program,
    image_uniforms: ImageUniforms,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    ebo: glow::Buffer,

    // Bottom-layer caching (non-MSAA, used as texture source for blitting)
    fbo: glow::Framebuffer,
    fbo_texture: glow::Texture,

    // MSAA rendering target
    msaa_fbo: glow::Framebuffer,
    msaa_rbo: glow::Renderbuffer,

    fbo_size: [u32; 2],
    bottom_layer_dirty: bool,
}

impl GlowRenderer {
    /// Create a new renderer. Must be called with a current GL context.
    ///
    /// # Safety
    /// The `gl` context must be current and valid.
    pub unsafe fn new(gl: Arc<glow::Context>) -> Result<Self, String> {
        let path_program =
            shaders::compile_program(&gl, shaders::PATH_VERTEX_SRC, shaders::PATH_FRAGMENT_SRC)?;
        let image_program = shaders::compile_program(
            &gl,
            shaders::IMAGE_VERTEX_SRC,
            shaders::IMAGE_FRAGMENT_SRC,
        )?;

        let path_uniforms = PathUniforms {
            scale: gl.get_uniform_location(path_program, "u_scale").unwrap(),
            offset: gl.get_uniform_location(path_program, "u_offset").unwrap(),
            resolution: gl
                .get_uniform_location(path_program, "u_resolution")
                .unwrap(),
            shader_type: gl
                .get_uniform_location(path_program, "u_shader_type")
                .unwrap(),
            color_a: gl
                .get_uniform_location(path_program, "u_color_a")
                .unwrap(),
            color_b: gl
                .get_uniform_location(path_program, "u_color_b")
                .unwrap(),
            bounds: gl
                .get_uniform_location(path_program, "u_bounds")
                .unwrap(),
        };

        let image_uniforms = ImageUniforms {
            scale: gl.get_uniform_location(image_program, "u_scale").unwrap(),
            offset: gl
                .get_uniform_location(image_program, "u_offset")
                .unwrap(),
            resolution: gl
                .get_uniform_location(image_program, "u_resolution")
                .unwrap(),
            texture: gl
                .get_uniform_location(image_program, "u_texture")
                .unwrap(),
            flip_uv_y: gl
                .get_uniform_location(image_program, "u_flip_uv_y")
                .unwrap(),
        };

        let vao = gl.create_vertex_array()?;
        let vbo = gl.create_buffer()?;
        let ebo = gl.create_buffer()?;

        // Set up VAO with position attribute
        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(
            0,               // attribute index
            2,               // components (vec2)
            glow::FLOAT,     // type
            false,           // normalized
            8,               // stride (2 * f32)
            0,               // offset
        );
        gl.bind_vertex_array(None);

        // Create FBO for bottom-layer caching (resolve target)
        let fbo = gl.create_framebuffer()?;
        let fbo_texture = gl.create_texture()?;

        // Create MSAA FBO for antialiased rendering
        let msaa_fbo = gl.create_framebuffer()?;
        let msaa_rbo = gl.create_renderbuffer()?;

        let mut allocator = GlAllocator::new();
        let scene_manager = SceneManager::new(&mut allocator);

        Ok(Self {
            gl,
            allocator,
            scene_manager,
            path_program,
            path_uniforms,
            image_program,
            image_uniforms,
            vao,
            vbo,
            ebo,
            fbo,
            fbo_texture,
            msaa_fbo,
            msaa_rbo,
            fbo_size: [0, 0],
            bottom_layer_dirty: true,
        })
    }

    /// Render the layout into the currently-bound framebuffer (typically the
    /// default framebuffer / screen).
    ///
    /// Returns an optional new resolution hint from livesplit-core's layout
    /// engine, indicating the layout's preferred size changed.
    ///
    /// # Safety
    /// Requires a current GL context matching the one passed to [`new`].
    pub unsafe fn render(
        &mut self,
        state: &LayoutState,
        image_cache: &ImageCache,
        [width, height]: [u32; 2],
    ) -> Option<[f32; 2]> {
        let resolution = [width as f32, height as f32];

        // Ensure FBO is the right size before borrowing the scene
        if self.fbo_size != [width, height] {
            self.resize_fbo(width, height);
            self.bottom_layer_dirty = true;
        }

        let new_resolution = self.scene_manager.update_scene(
            &mut self.allocator,
            resolution,
            state,
            image_cache,
        );

        let scene = self.scene_manager.scene();
        let bottom_layer_changed = scene.bottom_layer_changed();

        let gl = &self.gl;

        // Set up blending for premultiplied alpha
        gl.enable(glow::BLEND);
        gl.blend_func(glow::ONE, glow::ONE_MINUS_SRC_ALPHA);

        let w = width as i32;
        let h = height as i32;

        if bottom_layer_changed || self.bottom_layer_dirty {
            // Render bottom layer into MSAA FBO
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.msaa_fbo));
            gl.viewport(0, 0, w, h);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            // Render background
            if let Some(bg) = scene.background() {
                self.render_background(bg, resolution);
            }

            // Render bottom layer entities
            for entity in scene.bottom_layer() {
                self.render_entity(entity, resolution);
            }

            // Resolve MSAA → cached texture
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.msaa_fbo));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.fbo));
            gl.blit_framebuffer(0, 0, w, h, 0, 0, w, h, glow::COLOR_BUFFER_BIT, glow::NEAREST);

            self.bottom_layer_dirty = false;
        }

        // Composite: blit cached bottom layer + render top layer, all into MSAA FBO
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.msaa_fbo));
        gl.viewport(0, 0, w, h);
        gl.clear_color(0.0, 0.0, 0.0, 0.0);
        gl.clear(glow::COLOR_BUFFER_BIT);

        // Draw cached bottom layer texture into MSAA FBO
        self.blit_fbo(resolution);

        // Render top layer into MSAA FBO
        for entity in scene.top_layer() {
            self.render_entity(entity, resolution);
        }

        // Resolve MSAA → default framebuffer (screen)
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.msaa_fbo));
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
        gl.blit_framebuffer(0, 0, w, h, 0, 0, w, h, glow::COLOR_BUFFER_BIT, glow::NEAREST);

        gl.disable(glow::BLEND);

        new_resolution
    }

    /// Render a single entity.
    unsafe fn render_entity(
        &self,
        entity: &Entity<Option<GlPath>, GlImage, GlLabel>,
        resolution: [f32; 2],
    ) {
        match entity {
            Entity::FillPath(path, shader, transform) => {
                if let Some(path) = path.as_ref() {
                    self.draw_path(path, shader, transform, resolution);
                }
            }
            Entity::StrokePath(path, _stroke_width, color, transform) => {
                // TODO: proper stroke tessellation. For now, render as a thin
                // filled path which works for livesplit's common case
                // (separators are filled rectangles).
                if let Some(path) = path.as_ref() {
                    let shader = FillShader::SolidColor(*color);
                    self.draw_path(path, &shader, transform, resolution);
                }
            }
            Entity::Image(image, transform) => {
                self.draw_image(image, transform, resolution);
            }
            Entity::Label(label, shader, text_shadow, transform) => {
                self.draw_label(label, shader, text_shadow.as_ref(), transform, resolution);
            }
        }
    }

    /// Draw a filled path with the given shader and transform.
    unsafe fn draw_path(
        &self,
        path: &GlPath,
        shader: &FillShader,
        transform: &Transform,
        resolution: [f32; 2],
    ) {
        let gl = &self.gl;

        gl.use_program(Some(self.path_program));
        gl.uniform_2_f32(Some(&self.path_uniforms.resolution), resolution[0], resolution[1]);
        gl.uniform_2_f32(
            Some(&self.path_uniforms.scale),
            transform.scale_x,
            transform.scale_y,
        );
        gl.uniform_2_f32(Some(&self.path_uniforms.offset), transform.x, transform.y);

        self.set_shader_uniforms(shader, path);

        self.upload_and_draw(path);
    }

    /// Configure the fragment shader uniforms for a fill shader.
    ///
    /// For gradient shaders, we compute the bounding box of the path vertices
    /// in local space to determine the interpolation range.
    unsafe fn set_shader_uniforms(&self, shader: &FillShader, path: &GlPath) {
        let gl = &self.gl;
        let u = &self.path_uniforms;

        match shader {
            FillShader::SolidColor(color) => {
                gl.uniform_1_i32(Some(&u.shader_type), 0);
                gl.uniform_4_f32(Some(&u.color_a), color[0], color[1], color[2], color[3]);
            }
            FillShader::VerticalGradient(top, bottom) => {
                let [min, max] = vertex_bounds_y(&path.vertices);
                gl.uniform_1_i32(Some(&u.shader_type), 1);
                gl.uniform_4_f32(Some(&u.color_a), top[0], top[1], top[2], top[3]);
                gl.uniform_4_f32(Some(&u.color_b), bottom[0], bottom[1], bottom[2], bottom[3]);
                gl.uniform_2_f32(Some(&u.bounds), min, max);
            }
            FillShader::HorizontalGradient(left, right) => {
                let [min, max] = vertex_bounds_x(&path.vertices);
                gl.uniform_1_i32(Some(&u.shader_type), 2);
                gl.uniform_4_f32(Some(&u.color_a), left[0], left[1], left[2], left[3]);
                gl.uniform_4_f32(Some(&u.color_b), right[0], right[1], right[2], right[3]);
                gl.uniform_2_f32(Some(&u.bounds), min, max);
            }
        }
    }

    /// Upload vertex/index data and issue the draw call.
    unsafe fn upload_and_draw(&self, path: &GlPath) {
        let gl = &self.gl;

        gl.bind_vertex_array(Some(self.vao));

        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytemuck::cast_slice(&path.vertices),
            glow::STREAM_DRAW,
        );

        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo));
        gl.buffer_data_u8_slice(
            glow::ELEMENT_ARRAY_BUFFER,
            bytemuck::cast_slice(&path.indices),
            glow::STREAM_DRAW,
        );

        gl.draw_elements(
            glow::TRIANGLES,
            path.indices.len() as i32,
            glow::UNSIGNED_INT,
            0,
        );

        gl.bind_vertex_array(None);
    }

    /// Draw a text label (each glyph is a filled path).
    unsafe fn draw_label(
        &self,
        label: &LabelHandle<GlLabel>,
        shader: &FillShader,
        text_shadow: Option<&[f32; 4]>,
        transform: &Transform,
        resolution: [f32; 2],
    ) {
        let label = label.read().unwrap();

        // Render shadow pass first
        if let Some(shadow_color) = text_shadow {
            let alpha = match shader {
                FillShader::SolidColor([.., a]) => *a,
                FillShader::VerticalGradient([.., a1], [.., a2])
                | FillShader::HorizontalGradient([.., a1], [.., a2]) => 0.5 * (a1 + a2),
            };
            let shadow_rgba = [
                shadow_color[0],
                shadow_color[1],
                shadow_color[2],
                shadow_color[3] * alpha,
            ];
            let shadow_shader = FillShader::SolidColor(shadow_rgba);
            let shadow_transform = transform.pre_translate(SHADOW_OFFSET, SHADOW_OFFSET);

            for glyph in label.glyphs() {
                if let Some(path) = &glyph.path {
                    let t = shadow_transform
                        .pre_translate(glyph.x, glyph.y)
                        .pre_scale(glyph.scale, glyph.scale);
                    self.draw_path(path, &shadow_shader, &t, resolution);
                }
            }
        }

        // Render glyphs
        for glyph in label.glyphs() {
            if let Some(path) = &glyph.path {
                let t = transform
                    .pre_translate(glyph.x, glyph.y)
                    .pre_scale(glyph.scale, glyph.scale);
                let glyph_shader = if let Some(color) = &glyph.color {
                    FillShader::SolidColor(*color)
                } else {
                    *shader
                };
                self.draw_path(path, &glyph_shader, &t, resolution);
            }
        }
    }

    /// Draw an image entity.
    unsafe fn draw_image(
        &self,
        image: &Handle<GlImage>,
        transform: &Transform,
        resolution: [f32; 2],
    ) {
        let gl = &self.gl;
        let texture = self.ensure_texture(image);

        gl.use_program(Some(self.image_program));
        gl.uniform_2_f32(
            Some(&self.image_uniforms.resolution),
            resolution[0],
            resolution[1],
        );
        gl.uniform_2_f32(
            Some(&self.image_uniforms.scale),
            transform.scale_x,
            transform.scale_y,
        );
        gl.uniform_2_f32(Some(&self.image_uniforms.offset), transform.x, transform.y);
        gl.uniform_1_i32(Some(&self.image_uniforms.flip_uv_y), 0);

        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.uniform_1_i32(Some(&self.image_uniforms.texture), 0);

        // Draw the scene's unit rectangle with this texture
        let scene = self.scene_manager.scene();
        let rect = scene.rectangle();
        if let Some(path) = rect.as_ref() {
            self.upload_and_draw(path);
        }

        gl.bind_texture(glow::TEXTURE_2D, None);
    }

    /// Ensure an image's pixel data is uploaded as a GL texture.
    unsafe fn ensure_texture(&self, image: &Handle<GlImage>) -> glow::Texture {
        let data = &image.data;
        let mut tex_lock = data.texture.write().unwrap();

        if let Some(tex) = *tex_lock {
            return tex;
        }

        let gl = &self.gl;
        let texture = gl.create_texture().unwrap();
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            data.width as i32,
            data.height as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            PixelUnpackData::Slice(Some(&data.pixels)),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.bind_texture(glow::TEXTURE_2D, None);

        *tex_lock = Some(texture);
        texture
    }

    /// Render the scene background (solid/gradient fill).
    unsafe fn render_background(
        &self,
        background: &Background<GlImage>,
        resolution: [f32; 2],
    ) {
        match background {
            Background::Shader(shader) => {
                // Full-screen quad using the scene rectangle
                let transform = Transform {
                    scale_x: resolution[0],
                    scale_y: resolution[1],
                    x: 0.0,
                    y: 0.0,
                };
                let scene = self.scene_manager.scene();
                let rect = scene.rectangle();
                if let Some(path) = rect.as_ref() {
                    self.draw_path(path, shader, &transform, resolution);
                }
            }
            Background::Image(_, _) => {
                // TODO: background image support (blur, brightness, opacity)
                // For now, skip background images.
            }
        }
    }

    /// Blit the cached FBO texture to the current framebuffer as a fullscreen
    /// textured quad.
    unsafe fn blit_fbo(&self, resolution: [f32; 2]) {
        let gl = &self.gl;

        gl.use_program(Some(self.image_program));
        gl.uniform_2_f32(
            Some(&self.image_uniforms.resolution),
            resolution[0],
            resolution[1],
        );
        gl.uniform_2_f32(Some(&self.image_uniforms.scale), resolution[0], resolution[1]);
        gl.uniform_2_f32(Some(&self.image_uniforms.offset), 0.0, 0.0);
        gl.uniform_1_i32(Some(&self.image_uniforms.flip_uv_y), 1);

        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(self.fbo_texture));
        gl.uniform_1_i32(Some(&self.image_uniforms.texture), 0);

        let scene = self.scene_manager.scene();
        let rect = scene.rectangle();
        if let Some(path) = rect.as_ref() {
            self.upload_and_draw(path);
        }

        gl.bind_texture(glow::TEXTURE_2D, None);
    }

    /// Resize (or initially create) both the resolve FBO and MSAA FBO.
    unsafe fn resize_fbo(&mut self, width: u32, height: u32) {
        let gl = &self.gl;

        // Set up the resolve target (non-MSAA texture)
        gl.bind_texture(glow::TEXTURE_2D, Some(self.fbo_texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            width as i32,
            height as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            PixelUnpackData::Slice(None),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );

        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(self.fbo_texture),
            0,
        );

        // Set up the MSAA renderbuffer
        gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.msaa_rbo));
        gl.renderbuffer_storage_multisample(
            glow::RENDERBUFFER,
            MSAA_SAMPLES,
            glow::RGBA8,
            width as i32,
            height as i32,
        );

        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.msaa_fbo));
        gl.framebuffer_renderbuffer(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::RENDERBUFFER,
            Some(self.msaa_rbo),
        );

        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.bind_renderbuffer(glow::RENDERBUFFER, None);
        gl.bind_texture(glow::TEXTURE_2D, None);

        self.fbo_size = [width, height];
    }

    /// Clean up GL resources.
    ///
    /// # Safety
    /// Must be called with the same GL context that was used to create the
    /// renderer, and only once.
    pub unsafe fn destroy(&self) {
        let gl = &self.gl;
        gl.delete_program(self.path_program);
        gl.delete_program(self.image_program);
        gl.delete_vertex_array(self.vao);
        gl.delete_buffer(self.vbo);
        gl.delete_buffer(self.ebo);
        gl.delete_framebuffer(self.fbo);
        gl.delete_texture(self.fbo_texture);
        gl.delete_framebuffer(self.msaa_fbo);
        gl.delete_renderbuffer(self.msaa_rbo);
    }
}

/// Compute min/max Y of path vertices (for vertical gradient bounds).
fn vertex_bounds_y(vertices: &[Vertex]) -> [f32; 2] {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for v in vertices {
        min = min.min(v.position[1]);
        max = max.max(v.position[1]);
    }
    if max < min {
        [0.0, 0.0]
    } else {
        [min, max]
    }
}

/// Compute min/max X of path vertices (for horizontal gradient bounds).
fn vertex_bounds_x(vertices: &[Vertex]) -> [f32; 2] {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for v in vertices {
        min = min.min(v.position[0]);
        max = max.max(v.position[0]);
    }
    if max < min {
        [0.0, 0.0]
    } else {
        [min, max]
    }
}
