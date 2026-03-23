# livesplit-renderer-glow

GPU-accelerated renderers for [livesplit-core](https://github.com/LiveSplit/livesplit-core) layout scenes, with two backend options:

- **`GlowRenderer`** — OpenGL via [glow](https://docs.rs/glow)
- **`WgpuRenderer`** — [wgpu](https://docs.rs/wgpu) (Vulkan, Metal, DX12, OpenGL)

## Features

- **4x MSAA** antialiasing on all rendered content.
- **Two-layer caching**: the bottom layer (backgrounds, static elements) is rendered to an off-screen texture and reused across frames when unchanged.
- **Gradient fills**: solid, vertical, and horizontal gradients are handled natively in the fragment shader.
- **Text rendering** via livesplit-core's built-in text engine, with optional text shadows.
- **Background blur**: optional gaussian blur on background images, computed on the CPU and cached.
- **Lazy texture upload**: images are decoded on the CPU and uploaded to the GPU only when first drawn.
- Path tessellation via [lyon](https://docs.rs/lyon) at creation time for efficient per-frame rendering.

## Usage

### OpenGL (GlowRenderer)

```rust
use livesplit_renderer_gpu::GlowRenderer;
use std::sync::Arc;

// During setup (with a current GL context):
let mut renderer = unsafe { GlowRenderer::new(gl) }.unwrap();

// Each frame:
let new_size = unsafe { renderer.render(&layout_state, &image_cache, [width, height], true) };

// On shutdown:
unsafe { renderer.destroy() };
```

Requires OpenGL 3.1+ (GLSL 1.40) and a valid `Arc<glow::Context>`.

### wgpu (WgpuRenderer)

```rust
use livesplit_renderer_gpu::WgpuRenderer;

// During setup:
let mut renderer = WgpuRenderer::new(&device, surface_format);

// Each frame:
let new_size = renderer.render(&device, &queue, &layout_state, &image_cache,
                               [width, height], &output_view, true);
```

Requires a `wgpu::Device` and `wgpu::Queue`. No unsafe code needed.

## Dependencies

| Crate | Purpose |
|-------|---------|
| [livesplit-core](https://github.com/LiveSplit/livesplit-core) | Layout state, scene management, text engine |
| [glow](https://docs.rs/glow) | OpenGL bindings |
| [wgpu](https://docs.rs/wgpu) | Cross-platform GPU API |
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
