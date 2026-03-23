//! GPU-accelerated renderers for [livesplit-core] layout scenes.
//!
//! This crate provides two renderers, each behind a cargo feature flag
//! (both enabled by default):
//!
//! - [`GlowRenderer`] (`glow` feature) — renders to an OpenGL framebuffer
//!   via [glow].
//! - [`WgpuRenderer`] (`wgpu` feature) — renders via [wgpu], supporting
//!   Vulkan, Metal, DX12, and OpenGL backends.
//!
//! Both renderers share the same architecture: paths are tessellated via
//! [lyon] at creation time and drawn as indexed triangle meshes. The
//! livesplit-core scene's two-layer design is honored — the bottom layer is
//! cached in an off-screen texture and only re-rendered when it changes.
//!
//! # Features
//!
//! - **4× MSAA** antialiasing on all rendered content.
//! - **Two-layer caching**: the bottom layer is rendered to an off-screen
//!   texture and reused across frames when unchanged.
//! - **Gradient fills**: solid, vertical, and horizontal gradients are
//!   supported natively in the fragment shader.
//! - **Text rendering** via livesplit-core's built-in text engine, with
//!   optional text shadows.
//! - **Lazy texture upload**: images are decoded on the CPU and uploaded to
//!   the GPU only when first drawn.
//! - **Background blur**: optional gaussian blur on background images,
//!   computed on the CPU and cached.
//!
//! # Choosing a renderer
//!
//! | | [`GlowRenderer`] | [`WgpuRenderer`] |
//! |---|---|---|
//! | API | Raw OpenGL via glow | wgpu (Vulkan/Metal/DX12/GL) |
//! | Safety | `unsafe` — caller must manage GL context | Safe public API |
//! | Cleanup | Manual ([`destroy`](GlowRenderer::destroy)) | Automatic (Drop) |
//! | Requirements | OpenGL 3.1+, `Arc<glow::Context>` | `wgpu::Device` + `wgpu::Queue` |
//!
//! # Cargo features
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `glow`  | yes     | Enables [`GlowRenderer`] (OpenGL backend) |
//! | `wgpu`  | yes     | Enables [`WgpuRenderer`] (wgpu backend) |
//!
//! # Safety
//!
//! [`GlowRenderer`] requires a valid, current OpenGL context. Its
//! construction and rendering methods are `unsafe` because they issue raw GL
//! calls. [`WgpuRenderer`] has a fully safe public API.
//!
//! [livesplit-core]: https://github.com/LiveSplit/livesplit-core
//! [glow]: https://docs.rs/glow
//! [wgpu]: https://docs.rs/wgpu
//! [lyon]: https://docs.rs/lyon

mod common;

#[cfg(feature = "glow")]
mod allocator;
#[cfg(feature = "glow")]
mod render;
#[cfg(feature = "glow")]
mod shaders;
#[cfg(feature = "glow")]
mod types;

#[cfg(feature = "wgpu")]
mod wgpu_allocator;
#[cfg(feature = "wgpu")]
mod wgpu_buffer_pool;
#[cfg(feature = "wgpu")]
mod wgpu_render;
#[cfg(feature = "wgpu")]
mod wgpu_shaders;
#[cfg(feature = "wgpu")]
mod wgpu_types;

#[cfg(feature = "glow")]
pub use render::GlowRenderer;
#[cfg(feature = "wgpu")]
pub use wgpu_render::WgpuRenderer;
