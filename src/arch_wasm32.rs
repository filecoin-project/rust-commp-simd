// WASM32 production entry points
//
// No extern "C" declarations needed - WASM is pure Rust, no .S files
// SIMD enabled globally via -C target-feature=+simd128

#[path = "wasm32.rs"]
mod wasm32;
#[allow(unused)]
pub(crate) use wasm32::sha256_x4_wasm;

#[path = "reduce_wasm32.rs"]
mod reduce_wasm32;

#[path = "extrude_wasm32.rs"]
pub(crate) mod extrude_wasm32;

// pairwise_reduce_buffer_wasm(buf, byte_len, out_pair_count)
//   input:  buf[0..byte_len] pairs as [L0 R0 L1 R1 ...], byte_len = 128 x 2^N
//   output: buf[0..32] single root (when out_pair_count=0)
pub use reduce_wasm32::pairwise_reduce_buffer_wasm;

#[allow(unused)]
// extrude_quads_wasm(buf, count)
//   input:  buf[0..count*128], first count*127 bytes hold src
//   output: buf rewritten as count*128 FR32-padded bytes
pub use extrude_wasm32::extrude_quads_wasm;

#[allow(unused)]
// extrude_one_quad(quad)
//   input:  quad[0..127] holds source bytes
//   output: quad rewritten as 128 FR32-padded bytes
pub use extrude_wasm32::extrude_one_quad;

#[allow(unused)]
// hash_4_blocks(blocks) -> [[u8; 32]; 4]
//   4 independent 64-byte blocks hashed in parallel (leaf hashing)
pub use wasm32::hash_4_blocks;

#[allow(unused)]
// hash_4_pairs(nodes) -> [[u8; 32]; 4]
//   4 pairs of 32-byte nodes hashed in parallel (tree reduction)
pub use wasm32::hash_4_pairs;

#[allow(unused)]
// process_4_quads(sources, pad_bufs) -> [[u8; 32]; 8]
//   FR32 extrude + leaf hash for 4 quads batched
pub use wasm32::process_4_quads;