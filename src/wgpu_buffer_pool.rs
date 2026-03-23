//! Per-frame GPU buffer pool for reducing wgpu allocations.
//!
//! Instead of calling `device.create_buffer_init()` for every draw call
//! (creating thousands of tiny buffers per second), this module provides
//! growable ring buffers that are reused across frames. Data is written
//! via `queue.write_buffer()` at monotonically increasing offsets within
//! each frame.

/// A pool of GPU buffers that are reused across frames.
///
/// Call [`begin_frame`](Self::begin_frame) at the start of each frame to
/// reset offsets and release retired buffers. Then use
/// [`alloc_uniform`](Self::alloc_uniform),
/// [`alloc_vertex`](Self::alloc_vertex), and
/// [`alloc_index`](Self::alloc_index) to sub-allocate from the pool.
///
/// If any buffer runs out of space, it is reallocated at 2× the needed
/// size. The old buffer is kept alive until the next `begin_frame` call
/// so that in-flight draw calls referencing it remain valid.
pub struct FrameBufferPool {
    /// Uniform ring buffer and current write offset.
    pub(crate) uniform_buffer: wgpu::Buffer,
    uniform_capacity: u64,
    uniform_offset: u64,
    /// Incremented each time the uniform buffer is replaced (growth).
    /// The renderer uses this to invalidate cached bind groups.
    uniform_generation: u64,

    /// Vertex ring buffer and current write offset.
    pub(crate) vertex_buffer: wgpu::Buffer,
    vertex_capacity: u64,
    vertex_offset: u64,

    /// Index ring buffer and current write offset.
    pub(crate) index_buffer: wgpu::Buffer,
    index_capacity: u64,
    index_offset: u64,

    /// Required alignment for uniform buffer dynamic offsets, queried
    /// from the device at construction time.
    uniform_align: u64,

    /// Old buffers kept alive until the next frame so that in-flight
    /// GPU commands referencing them remain valid.
    retired: Vec<wgpu::Buffer>,
}

/// Result of a uniform sub-allocation: the byte offset within the buffer
/// (aligned to the device's `min_uniform_buffer_offset_alignment`).
pub struct UniformAlloc {
    /// Byte offset into the uniform buffer.
    pub offset: u64,
}

/// Result of a vertex sub-allocation.
pub struct VertexAlloc {
    /// Byte offset into the vertex buffer.
    pub offset: u64,
    /// Byte length of the allocation.
    pub size: u64,
}

/// Result of an index sub-allocation.
pub struct IndexAlloc {
    /// Byte offset into the index buffer.
    pub offset: u64,
    /// Byte length of the allocation.
    pub size: u64,
}

impl FrameBufferPool {
    /// Initial capacity for the uniform buffer (64 KB).
    const INITIAL_UNIFORM: u64 = 64 * 1024;
    /// Initial capacity for the vertex buffer (512 KB).
    const INITIAL_VERTEX: u64 = 512 * 1024;
    /// Initial capacity for the index buffer (256 KB).
    const INITIAL_INDEX: u64 = 256 * 1024;

    /// Create a new pool with default initial capacities.
    ///
    /// Queries the device for `min_uniform_buffer_offset_alignment` to
    /// ensure correct alignment of uniform sub-allocations.
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform_align = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        Self {
            uniform_buffer: Self::create_uniform_buffer(device, Self::INITIAL_UNIFORM),
            uniform_capacity: Self::INITIAL_UNIFORM,
            uniform_offset: 0,
            uniform_generation: 0,

            vertex_buffer: Self::create_vertex_buffer(device, Self::INITIAL_VERTEX),
            vertex_capacity: Self::INITIAL_VERTEX,
            vertex_offset: 0,

            index_buffer: Self::create_index_buffer(device, Self::INITIAL_INDEX),
            index_capacity: Self::INITIAL_INDEX,
            index_offset: 0,

            uniform_align,
            retired: Vec::new(),
        }
    }

    /// Current uniform buffer generation. Incremented on each growth
    /// event. The renderer compares this against a cached value to know
    /// when to recreate bind groups.
    pub fn uniform_generation(&self) -> u64 {
        self.uniform_generation
    }

    /// Reset all offsets to zero and drop retired buffers from the
    /// previous frame. Call at the start of each frame.
    pub fn begin_frame(&mut self) {
        self.uniform_offset = 0;
        self.vertex_offset = 0;
        self.index_offset = 0;
        self.retired.clear();
    }

    /// Sub-allocate space for a uniform buffer entry.
    ///
    /// Writes `data` to the uniform buffer at the next aligned offset via
    /// `queue.write_buffer`. Returns the offset for use with
    /// `set_bind_group` dynamic offsets.
    ///
    /// If the buffer is too small, it is reallocated at 2× the needed
    /// size. The old buffer is retired (kept alive until the next frame).
    pub fn alloc_uniform(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
    ) -> UniformAlloc {
        let aligned_offset = align_up(self.uniform_offset, self.uniform_align);
        let end = aligned_offset + data.len() as u64;

        if end > self.uniform_capacity {
            let new_cap = (end * 2).max(self.uniform_capacity * 2);
            let old = std::mem::replace(
                &mut self.uniform_buffer,
                Self::create_uniform_buffer(device, new_cap),
            );
            self.retired.push(old);
            self.uniform_capacity = new_cap;
            self.uniform_generation += 1;
            // Start fresh on the new buffer.
            queue.write_buffer(&self.uniform_buffer, 0, data);
            self.uniform_offset = data.len() as u64;
            return UniformAlloc { offset: 0 };
        }

        queue.write_buffer(&self.uniform_buffer, aligned_offset, data);
        self.uniform_offset = end;
        UniformAlloc {
            offset: aligned_offset,
        }
    }

    /// Sub-allocate space for vertex data.
    ///
    /// Vertex data must be 4-byte aligned (satisfied by `[f32; 2]`
    /// vertices). Old buffer is retired on growth.
    pub fn alloc_vertex(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
    ) -> VertexAlloc {
        let aligned_offset = align_up(self.vertex_offset, 4);
        let size = data.len() as u64;
        let end = aligned_offset + size;

        if end > self.vertex_capacity {
            let new_cap = (end * 2).max(self.vertex_capacity * 2);
            let old = std::mem::replace(
                &mut self.vertex_buffer,
                Self::create_vertex_buffer(device, new_cap),
            );
            self.retired.push(old);
            self.vertex_capacity = new_cap;
            queue.write_buffer(&self.vertex_buffer, 0, data);
            self.vertex_offset = size;
            return VertexAlloc { offset: 0, size };
        }

        queue.write_buffer(&self.vertex_buffer, aligned_offset, data);
        self.vertex_offset = end;
        VertexAlloc {
            offset: aligned_offset,
            size,
        }
    }

    /// Sub-allocate space for index data.
    ///
    /// Index data must be 4-byte aligned (satisfied by `u32` indices).
    /// Old buffer is retired on growth.
    pub fn alloc_index(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
    ) -> IndexAlloc {
        let aligned_offset = align_up(self.index_offset, 4);
        let size = data.len() as u64;
        let end = aligned_offset + size;

        if end > self.index_capacity {
            let new_cap = (end * 2).max(self.index_capacity * 2);
            let old = std::mem::replace(
                &mut self.index_buffer,
                Self::create_index_buffer(device, new_cap),
            );
            self.retired.push(old);
            self.index_capacity = new_cap;
            queue.write_buffer(&self.index_buffer, 0, data);
            self.index_offset = size;
            return IndexAlloc { offset: 0, size };
        }

        queue.write_buffer(&self.index_buffer, aligned_offset, data);
        self.index_offset = end;
        IndexAlloc {
            offset: aligned_offset,
            size,
        }
    }

    fn create_uniform_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_uniform_buffer"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_vertex_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_vertex_buffer"),
            size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_index_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_index_buffer"),
            size,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

/// Round `offset` up to the next multiple of `alignment`.
const fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn align_up_small() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(3, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
    }
}
