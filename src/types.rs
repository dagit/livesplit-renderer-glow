//! Resource types for the GL renderer.
//!
//! Each type corresponds to an associated type on
//! [`ResourceAllocator`](livesplit_core::rendering::ResourceAllocator).

use std::sync::{Arc, RwLock};

use bytemuck::{Pod, Zeroable};
use livesplit_core::rendering::{self, SharedOwnership};
use lyon::path::Path as LyonPath;

/// A vertex in a tessellated path, ready for the GPU.
///
/// Uses `#[repr(C)]` and derives [`Pod`] so the vertex array can be
/// reinterpreted as a byte slice for upload via `glBufferData`.
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex {
    /// X and Y position in local (pre-transform) coordinate space.
    pub position: [f32; 2],
}

/// A tessellated path stored as indexed triangle data.
///
/// Created at `PathBuilder::finish()` time via lyon tessellation.
/// Drawn with `glDrawElements(GL_TRIANGLES, ...)`.
///
/// Vertices and indices are wrapped in [`Arc`] so that
/// [`SharedOwnership::share`] is a cheap reference-count bump.
/// The original lyon path is retained for stroke tessellation.
pub struct GlPath {
    /// Triangle vertices in local coordinate space.
    pub vertices: Arc<Vec<Vertex>>,
    /// Triangle indices into [`vertices`](Self::vertices).
    pub indices: Arc<Vec<u32>>,
    /// The original lyon path, retained for stroke tessellation.
    pub lyon_path: Arc<LyonPath>,
    /// Cached stroke tessellation, keyed by stroke width.
    stroke_cache: RwLock<Option<StrokeCache>>,
}

/// Shared vertex and index buffers for a tessellated path.
type StrokeGeometry = (Arc<Vec<Vertex>>, Arc<Vec<u32>>);

/// Cached stroke tessellation data for a specific line width.
struct StrokeCache {
    /// The stroke width this cache was tessellated for.
    width: f32,
    /// Stroke triangle vertices.
    vertices: Arc<Vec<Vertex>>,
    /// Stroke triangle indices.
    indices: Arc<Vec<u32>>,
}

impl GlPath {
    /// Create a new `GlPath` from tessellated geometry and the original path.
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>, lyon_path: Arc<LyonPath>) -> Self {
        Self {
            vertices: Arc::new(vertices),
            indices: Arc::new(indices),
            lyon_path,
            stroke_cache: RwLock::new(None),
        }
    }

    /// Create a `GlPath` from pre-shared vertex and index buffers.
    pub fn from_arcs(
        vertices: Arc<Vec<Vertex>>,
        indices: Arc<Vec<u32>>,
        lyon_path: Arc<LyonPath>,
    ) -> Self {
        Self {
            vertices,
            indices,
            lyon_path,
            stroke_cache: RwLock::new(None),
        }
    }

    /// Get the cached stroke tessellation for a given width, or `None` if
    /// the cache is empty or was tessellated for a different width.
    pub fn cached_stroke(&self, width: f32) -> Option<StrokeGeometry> {
        let cache = self
            .stroke_cache
            .read()
            .expect("stroke cache RwLock poisoned");
        cache.as_ref().and_then(|c| {
            if (c.width - width).abs() < f32::EPSILON {
                Some((Arc::clone(&c.vertices), Arc::clone(&c.indices)))
            } else {
                None
            }
        })
    }

    /// Store a stroke tessellation in the cache for a given width.
    pub fn set_stroke_cache(&self, width: f32, vertices: Arc<Vec<Vertex>>, indices: Arc<Vec<u32>>) {
        let mut cache = self
            .stroke_cache
            .write()
            .expect("stroke cache RwLock poisoned");
        *cache = Some(StrokeCache {
            width,
            vertices,
            indices,
        });
    }
}

impl Clone for GlPath {
    fn clone(&self) -> Self {
        Self {
            vertices: Arc::clone(&self.vertices),
            indices: Arc::clone(&self.indices),
            lyon_path: Arc::clone(&self.lyon_path),
            // Start with an empty cache — it will be populated on first stroke draw.
            stroke_cache: RwLock::new(None),
        }
    }
}

impl std::fmt::Debug for GlPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlPath")
            .field("vertices", &self.vertices.len())
            .field("indices", &self.indices.len())
            .finish_non_exhaustive()
    }
}

impl SharedOwnership for GlPath {
    fn share(&self) -> Self {
        self.clone()
    }
}

/// A decoded image ready for GL texture upload.
///
/// The raw pixel data is shared via [`Arc`] so that cloning an image is
/// cheap. The GL texture handle is lazily created on first draw.
#[derive(Clone)]
pub struct GlImage {
    /// Shared image data (pixels, dimensions, and cached texture handle).
    pub data: Arc<GlImageData>,
}

/// Backing store for a [`GlImage`].
///
/// Contains the decoded RGBA pixel data and an optional GL texture handle
/// that is populated on first use.
pub struct GlImageData {
    /// Raw pixel data in RGBA8 format, row-major, top-to-bottom.
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Precomputed width / height, used by the scene layout engine.
    pub aspect_ratio: f32,
    /// GL texture name, lazily uploaded on first draw. `None` until then.
    pub texture: RwLock<Option<glow::Texture>>,
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

/// Font handle.
///
/// All font loading and shaping is delegated to livesplit-core's default
/// text engine, so this is a re-export of its font type.
pub type GlFont = livesplit_core::rendering::default_text_engine::Font;

/// Label handle.
///
/// The default text engine produces labels containing glyph outlines
/// tessellated as [`GlPath`] values. The [`Label`](rendering::Label) and
/// [`SharedOwnership`] traits are already implemented for this type inside
/// livesplit-core, so we cannot (and need not) implement them here.
pub type GlLabel = livesplit_core::rendering::default_text_engine::Label<Option<GlPath>>;

/// The locked (read-guard) form of a [`GlLabel`].
///
/// Useful for consumers that need to inspect label glyph data directly.
/// Obtained by calling `label.read()` on a label handle.
#[allow(dead_code)]
pub type GlLockedLabel =
    livesplit_core::rendering::default_text_engine::LockedLabel<Option<GlPath>>;
