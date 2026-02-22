//! Resource types for the GL renderer.
//!
//! Each type corresponds to an associated type on [`ResourceAllocator`].

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use livesplit_core::rendering::{self, SharedOwnership};

/// A vertex in a tessellated path, ready for the GPU.
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex {
    pub position: [f32; 2],
}

/// A tessellated path stored as indexed triangle data.
///
/// Created at `PathBuilder::finish()` time via lyon tessellation.
/// Drawn with `glDrawElements(GL_TRIANGLES, ...)`.
#[derive(Clone)]
pub struct GlPath {
    pub vertices: Arc<Vec<Vertex>>,
    pub indices: Arc<Vec<u32>>,
}

impl SharedOwnership for GlPath {
    fn share(&self) -> Self {
        self.clone()
    }
}

/// A decoded image ready for GL texture upload.
#[derive(Clone)]
pub struct GlImage {
    pub data: Arc<GlImageData>,
}

pub struct GlImageData {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub aspect_ratio: f32,
    /// GL texture name, lazily uploaded. `None` until first draw.
    pub texture: std::sync::RwLock<Option<glow::Texture>>,
}

impl rendering::Image for GlImage {
    fn aspect_ratio(&self) -> f32 {
        self.data.aspect_ratio
    }
}

impl SharedOwnership for GlImage {
    fn share(&self) -> Self {
        self.clone()
    }
}

/// Font handle. We delegate all font/shaping work to livesplit-core's default
/// text engine, so this is just a re-export of its font type.
pub type GlFont = livesplit_core::rendering::default_text_engine::Font;

/// Label handle. Same delegation — the default text engine produces labels
/// containing glyph paths (which are our `GlPath` type). The `Label` and
/// `SharedOwnership` traits are already implemented for this type inside
/// livesplit-core, so we don't need (and can't, due to orphan rules) implement
/// them here.
pub type GlLabel = livesplit_core::rendering::default_text_engine::Label<Option<GlPath>>;

/// The locked label type, for reading glyph data. Useful for consumers
/// that need to inspect label contents directly.
#[allow(dead_code)]
pub type GlLockedLabel =
    livesplit_core::rendering::default_text_engine::LockedLabel<Option<GlPath>>;
