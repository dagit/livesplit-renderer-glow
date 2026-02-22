//! A GPU-accelerated renderer for livesplit-core using OpenGL via glow.
//!
//! This crate provides [`GlowRenderer`], which implements the same rendering
//! output as livesplit-core's software renderer but uses OpenGL for
//! rasterization. Paths are tessellated via lyon at creation time and rendered
//! as vertex buffers. The scene's two-layer design is honored: the bottom layer
//! is cached in a framebuffer object and only re-rendered when it changes.

mod allocator;
mod render;
mod shaders;
mod types;

pub use render::GlowRenderer;
