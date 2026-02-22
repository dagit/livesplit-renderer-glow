# livesplit-renderer-glow

A GPU-accelerated renderer for [livesplit-core](https://github.com/LiveSplit/livesplit-core) using OpenGL via [glow](https://docs.rs/glow).

## Features

- **GPU-accelerated rendering** of livesplit-core layout scenes via OpenGL.
- **4x MSAA** antialiasing on all rendered content.
- **Two-layer caching**: the bottom layer (backgrounds, static elements) is rendered to an off-screen framebuffer and reused across frames when unchanged.
- **Gradient fills**: solid, vertical, and horizontal gradients are handled natively in the fragment shader.
- **Text rendering** via livesplit-core's built-in text engine, with optional text shadows.
- **Lazy texture upload**: images are decoded on the CPU and uploaded to the GPU only when first drawn.
- Path tessellation via [lyon](https://docs.rs/lyon) at creation time for efficient per-frame rendering.

## Usage

```rust
use livesplit_renderer_glow::GlowRenderer;
use std::sync::Arc;

// During setup (with a current GL context):
let mut renderer = unsafe { GlowRenderer::new(gl) }.unwrap();

// Each frame:
let new_size = unsafe { renderer.render(&layout_state, &image_cache, [width, height]) };

// On shutdown:
unsafe { renderer.destroy() };
```

## Requirements

- OpenGL 3.1+ (GLSL 1.40)
- A valid, current OpenGL context wrapped in `Arc<glow::Context>`

## Dependencies

| Crate | Purpose |
|-------|---------|
| [livesplit-core](https://github.com/LiveSplit/livesplit-core) | Layout state, scene management, text engine |
| [glow](https://docs.rs/glow) | OpenGL bindings |
| [lyon](https://docs.rs/lyon) | Path tessellation |
| [bytemuck](https://docs.rs/bytemuck) | Safe byte reinterpretation for GPU uploads |
| [image](https://docs.rs/image) | PNG/JPEG decoding |

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/dagit/livesplit-renderer-glow).
