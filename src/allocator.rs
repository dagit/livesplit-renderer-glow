//! [`ResourceAllocator`] implementation that tessellates paths via lyon and
//! delegates text shaping to livesplit-core's default text engine.

use livesplit_core::{
    rendering::{self, default_text_engine::TextEngine, FontKind, ResourceAllocator},
    settings,
};
use lyon::{
    math::point,
    path::Path as LyonPath,
    tessellation::{
        BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, StrokeOptions,
        StrokeTessellator, StrokeVertex, VertexBuffers,
    },
};
use std::sync::Arc;

use crate::types::{GlFont, GlImage, GlImageData, GlLabel, GlPath, Vertex};

/// Lyon-backed path builder that produces a [`GlPath`] on `finish()`.
///
/// Implements livesplit-core's [`PathBuilder`](rendering::PathBuilder) trait,
/// converting path commands (move/line/quad/curve/close) into a lyon path and
/// then tessellating it into an indexed triangle mesh.
pub struct GlPathBuilder {
    /// The underlying lyon path builder.
    builder: lyon::path::path::Builder,
}

impl rendering::PathBuilder for GlPathBuilder {
    type Path = Option<GlPath>;

    fn move_to(&mut self, x: f32, y: f32) {
        self.builder.begin(point(x, y));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(point(x, y));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder.quadratic_bezier_to(point(x1, y1), point(x, y));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder
            .cubic_bezier_to(point(x1, y1), point(x2, y2), point(x, y));
    }

    fn close(&mut self) {
        self.builder.close();
    }

    fn finish(self) -> Self::Path {
        let path = self.builder.build();
        tessellate_path(&path)
    }
}

/// Tessellate a lyon path into an indexed triangle mesh (fill).
///
/// Uses a fill tessellator with the non-zero fill rule and a tolerance of
/// 0.01 (suitable for the small coordinate spaces livesplit-core uses).
///
/// Returns `None` if tessellation fails or produces no vertices.
fn tessellate_path(path: &LyonPath) -> Option<GlPath> {
    let mut geometry: VertexBuffers<Vertex, u32> = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();

    let result = tessellator.tessellate_path(
        path,
        &FillOptions::tolerance(0.01).with_fill_rule(FillRule::NonZero),
        &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex| Vertex {
            position: vertex.position().to_array(),
        }),
    );

    match result {
        Ok(()) if !geometry.vertices.is_empty() => Some(GlPath::new(
            geometry.vertices,
            geometry.indices,
            Arc::new(path.clone()),
        )),
        _ => None,
    }
}

/// Tessellate a path outline (stroke) into an indexed triangle mesh.
///
/// Uses lyon's [`StrokeTessellator`] with the given `stroke_width` and a
/// tolerance of 0.01. Results are cached inside the [`GlPath`]'s stroke
/// cache so that repeated draws at the same width do not re-tessellate.
///
/// Returns `None` if tessellation fails or produces no geometry.
pub(crate) fn tessellate_stroke(path: &GlPath, stroke_width: f32) -> Option<GlPath> {
    // Check the cache first.
    if let Some((verts, idxs)) = path.cached_stroke(stroke_width) {
        return Some(GlPath::from_arcs(verts, idxs, Arc::clone(&path.lyon_path)));
    }

    // Cache miss — tessellate the stroke.
    let mut geometry: VertexBuffers<Vertex, u32> = VertexBuffers::new();
    let mut tessellator = StrokeTessellator::new();

    let result = tessellator.tessellate_path(
        &*path.lyon_path,
        &StrokeOptions::tolerance(0.01).with_line_width(stroke_width),
        &mut BuffersBuilder::new(&mut geometry, |vertex: StrokeVertex| Vertex {
            position: vertex.position().to_array(),
        }),
    );

    match result {
        Ok(()) if !geometry.vertices.is_empty() => {
            let verts = Arc::new(geometry.vertices);
            let idxs = Arc::new(geometry.indices);

            // Populate the cache for next time.
            path.set_stroke_cache(stroke_width, Arc::clone(&verts), Arc::clone(&idxs));

            Some(GlPath::from_arcs(verts, idxs, Arc::clone(&path.lyon_path)))
        }
        _ => None,
    }
}

/// The resource allocator that wires together path tessellation (via lyon)
/// and text shaping (via livesplit-core's default text engine).
pub struct GlAllocator {
    /// Text engine instance used for font loading, glyph shaping, and label
    /// management.
    pub text_engine: TextEngine<Option<GlPath>>,
}

impl GlAllocator {
    /// Create a new allocator with a fresh text engine.
    pub fn new() -> Self {
        Self {
            text_engine: TextEngine::new(),
        }
    }
}

impl Default for GlAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceAllocator for GlAllocator {
    type PathBuilder = GlPathBuilder;
    type Path = Option<GlPath>;
    type Image = GlImage;
    type Font = GlFont;
    type Label = GlLabel;

    fn path_builder(&mut self) -> Self::PathBuilder {
        GlPathBuilder {
            builder: LyonPath::builder(),
        }
    }

    fn create_image(&mut self, data: &[u8]) -> Option<Self::Image> {
        let img = image::load_from_memory(data).ok()?.to_rgba8();
        let (width, height) = img.dimensions();
        Some(GlImage {
            data: Arc::new(GlImageData {
                pixels: img.into_raw(),
                width,
                height,
                // Precision loss is acceptable: viewport dimensions are small
                // relative to f32 mantissa range.
                #[expect(clippy::cast_precision_loss)]
                aspect_ratio: width as f32 / height as f32,
                texture: std::sync::RwLock::new(None),
            }),
        })
    }

    fn create_font(&mut self, font: Option<&settings::Font>, kind: FontKind) -> Self::Font {
        self.text_engine.create_font(font, kind)
    }

    fn create_label(
        &mut self,
        text: &str,
        font: &mut Self::Font,
        max_width: Option<f32>,
    ) -> Self::Label {
        self.text_engine
            .create_label(gl_path_builder, text, font, max_width)
    }

    fn update_label(
        &mut self,
        label: &mut Self::Label,
        text: &str,
        font: &mut Self::Font,
        max_width: Option<f32>,
    ) {
        self.text_engine
            .update_label(gl_path_builder, label, text, font, max_width);
    }
}

/// Factory function matching the signature [`TextEngine`] expects for creating
/// path builders on demand during glyph outline extraction.
fn gl_path_builder() -> GlPathBuilder {
    GlPathBuilder {
        builder: LyonPath::builder(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use image::ImageEncoder;

    use super::*;
    use livesplit_core::rendering::{PathBuilder, SharedOwnership};

    #[test]
    fn tessellate_unit_rectangle() {
        let mut builder = LyonPath::builder();
        builder.begin(point(0.0, 0.0));
        builder.line_to(point(1.0, 0.0));
        builder.line_to(point(1.0, 1.0));
        builder.line_to(point(0.0, 1.0));
        builder.close();
        let path = builder.build();

        let result = tessellate_path(&path);
        assert!(result.is_some(), "rectangle should tessellate successfully");

        let gl_path = result.unwrap();
        assert!(
            !gl_path.vertices.is_empty(),
            "rectangle should have vertices"
        );
        assert!(!gl_path.indices.is_empty(), "rectangle should have indices");
        assert_eq!(
            gl_path.indices.len() % 3,
            0,
            "index count must be a multiple of 3 (triangles)"
        );
    }

    #[test]
    fn tessellate_empty_path_returns_none() {
        let builder = LyonPath::builder();
        let path = builder.build();

        assert!(
            tessellate_path(&path).is_none(),
            "empty path should return None"
        );
    }

    #[test]
    fn tessellate_degenerate_single_point_returns_none() {
        let mut builder = LyonPath::builder();
        builder.begin(point(1.0, 1.0));
        builder.close();
        let path = builder.build();

        assert!(
            tessellate_path(&path).is_none(),
            "single-point path should return None"
        );
    }

    #[test]
    fn path_builder_trait_builds_triangle() {
        let mut pb = gl_path_builder();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(0.5, 1.0);
        pb.close();
        let result = pb.finish();

        assert!(result.is_some(), "triangle should tessellate successfully");
        let path = result.unwrap();
        assert_eq!(path.indices.len(), 3, "triangle should have 3 indices");
    }

    #[test]
    fn path_builder_quadratic_curve() {
        let mut pb = gl_path_builder();
        pb.move_to(0.0, 0.0);
        pb.quad_to(0.5, 1.0, 1.0, 0.0);
        pb.close();
        let result = pb.finish();

        assert!(
            result.is_some(),
            "quadratic curve path should tessellate successfully"
        );
    }

    #[test]
    fn path_builder_cubic_curve() {
        let mut pb = gl_path_builder();
        pb.move_to(0.0, 0.0);
        pb.curve_to(0.25, 1.0, 0.75, 1.0, 1.0, 0.0);
        pb.close();
        let result = pb.finish();

        assert!(
            result.is_some(),
            "cubic curve path should tessellate successfully"
        );
    }

    #[test]
    fn gl_path_share_clones_arcs() {
        let mut pb = gl_path_builder();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(0.5, 1.0);
        pb.close();
        let path = pb.finish().unwrap();

        let shared = SharedOwnership::share(&path);
        assert!(Arc::ptr_eq(&path.vertices, &shared.vertices));
        assert!(Arc::ptr_eq(&path.indices, &shared.indices));
    }

    #[test]
    fn stroke_tessellate_rectangle() {
        let mut pb = gl_path_builder();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(1.0, 1.0);
        pb.line_to(0.0, 1.0);
        pb.close();
        let path = pb.finish().unwrap();

        let stroked = tessellate_stroke(&path, 0.1);
        assert!(
            stroked.is_some(),
            "rectangle stroke should produce geometry"
        );
        let stroked = stroked.unwrap();
        assert!(!stroked.vertices.is_empty());
        assert!(!stroked.indices.is_empty());
    }

    #[test]
    fn stroke_tessellate_open_line() {
        // An open path (line segment) should also stroke successfully.
        let mut builder = LyonPath::builder();
        builder.begin(point(0.0, 0.0));
        builder.line_to(point(1.0, 0.0));
        builder.end(false); // open path, not closed
        let lyon_path = builder.build();
        let fill = tessellate_path(&lyon_path);
        // Fill of an open line may be None, but we need the lyon_path for stroke.
        // Create a GlPath manually if fill is None.
        let path = fill.unwrap_or_else(|| GlPath::new(vec![], vec![], Arc::new(lyon_path)));

        let stroked = tessellate_stroke(&path, 0.1);
        assert!(
            stroked.is_some(),
            "open line stroke should produce geometry"
        );
    }

    #[test]
    fn stroke_cache_hit() {
        let mut pb = gl_path_builder();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(0.5, 1.0);
        pb.close();
        let path = pb.finish().unwrap();

        // First call populates the cache.
        let first = tessellate_stroke(&path, 0.1);
        assert!(first.is_some());

        // Second call with same width should hit the cache.
        let cached = path.cached_stroke(0.1);
        assert!(
            cached.is_some(),
            "cache should be populated after first stroke"
        );
    }

    #[test]
    fn stroke_cache_miss_on_different_width() {
        let mut pb = gl_path_builder();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(0.5, 1.0);
        pb.close();
        let path = pb.finish().unwrap();

        let _ = tessellate_stroke(&path, 0.1);

        // Different width should miss the cache.
        let cached = path.cached_stroke(0.2);
        assert!(cached.is_none(), "cache should miss for different width");
    }

    #[test]
    fn create_image_with_invalid_data_returns_none() {
        let mut alloc = GlAllocator::new();
        assert!(alloc.create_image(b"not an image").is_none());
    }

    #[test]
    fn create_image_with_valid_png() {
        // Encode a 2x1 PNG at runtime so the bytes are always correct.
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let encoder = image::codecs::png::PngEncoder::new(&mut buf);
            // 2x1 RGBA image: red pixel, blue pixel
            let pixels: &[u8] = &[255, 0, 0, 255, 0, 0, 255, 255];
            encoder
                .write_image(pixels, 2, 1, image::ExtendedColorType::Rgba8)
                .unwrap();
        }

        let mut alloc = GlAllocator::new();
        let image = alloc.create_image(buf.get_ref());
        assert!(image.is_some(), "valid PNG should produce an image");

        let img = image.unwrap();
        assert_eq!(img.data.width, 2);
        assert_eq!(img.data.height, 1);
        assert!((img.data.aspect_ratio - 2.0).abs() < f32::EPSILON);
        // RGBA: 4 bytes per pixel x 2 pixels
        assert_eq!(img.data.pixels.len(), 8);
    }

    #[test]
    fn allocator_default_matches_new() {
        let _alloc: GlAllocator = GlAllocator::default();
    }
}
