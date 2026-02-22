//! [`ResourceAllocator`] implementation that tessellates paths via lyon and
//! delegates text shaping to livesplit-core's default text engine.

use livesplit_core::rendering::default_text_engine::TextEngine;
use livesplit_core::rendering::{self, FontKind, ResourceAllocator};
use livesplit_core::settings;
use lyon::math::point;
use lyon::path::Path as LyonPath;
use lyon::tessellation::*;
use std::sync::Arc;

use crate::types::*;

/// Lyon-backed path builder that produces [`GlPath`] on `finish()`.
pub struct GlPathBuilder {
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
        self.builder
            .quadratic_bezier_to(point(x1, y1), point(x, y));
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

/// Tessellate a lyon path into an indexed triangle mesh.
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
        Ok(_) if !geometry.vertices.is_empty() => Some(GlPath {
            vertices: Arc::new(geometry.vertices),
            indices: Arc::new(geometry.indices),
        }),
        _ => None,
    }
}

/// Tessellate a path outline (stroke) into an indexed triangle mesh.
#[allow(dead_code)]
pub fn tessellate_stroke(path: &Option<GlPath>, stroke_width: f32) -> Option<GlPath> {
    // For strokes we can't re-tessellate from the original path data since we
    // only stored the fill tessellation. As a pragmatic first pass, we render
    // strokes by re-using the fill mesh. This is incorrect for thin strokes on
    // open paths but works for the common livesplit case (separators, graph
    // lines) which are typically filled rectangles anyway.
    //
    // TODO: Store the original lyon::path::Path alongside the tessellated mesh
    // so we can properly stroke-tessellate here.
    let _ = stroke_width;
    path.as_ref().map(|p| p.clone())
}

/// The resource allocator that wires everything together.
pub struct GlAllocator {
    pub text_engine: TextEngine<Option<GlPath>>,
}

impl GlAllocator {
    pub fn new() -> Self {
        Self {
            text_engine: TextEngine::new(),
        }
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
            .update_label(gl_path_builder, label, text, font, max_width)
    }
}

/// Factory function matching the signature `TextEngine` expects for creating
/// path builders on demand during glyph outline extraction.
fn gl_path_builder() -> GlPathBuilder {
    GlPathBuilder {
        builder: LyonPath::builder(),
    }
}
