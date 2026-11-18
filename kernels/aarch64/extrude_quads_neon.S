/**
 * extrude_neon.S - FR32 extrusion (127B -> 128B per quad), ARM64 NEON
 *
 * Direct port of the SSE extrudinator logic. NEON 128-bit registers
 * map 1:1 to SSE XMM, but ARM `ext` is non-destructive (no copies needed).
 *
 * void extrude_quads_neon_core(uint8_t *buf, uint64_t count)
 *   x0 = reserved for writeback running ptr
 *   x1 - read ptr
 *   x2 = quad count (caller handles last-quad boundary padding)
 *
 * FR32 algorithm per quad:
 *   128B = 4 nodes × 32B = 16 × uint64 limbs (LE)
 *   Node 0 (L0-L3):   no shift, just clear bits 62-63 of L3
 *   Node 1 (L4-L7):   shift left 2, carry from L3, clear bits 62-63 of L7
 *   Node 2 (L8-L11):  shift left 4, carry from L7, clear bits 62-63 of L11
 *   Node 3 (L12-L15): shift left 6, carry from L11, clear bits 62-63 of L15
 *   Process nodes 3 -> 2 -> 1 -> 0 to avoid clobbering unread carry sources
 *
 * Register allocation (1x loop - no callee-saved regs needed):
 *   v0,v1:  Node 0 [L0,L1], [L2,L3]
 *   v2,v3:  Node 1 [L4,L5], [L6,L7]
 *   v4,v5:  Node 2 [L8,L9], [L10,L11]
 *   v6,v7:  Node 3 [L12,L13], [L14,L15]
 *   v16,v17: carry temporaries
 *   v18:    padMask = [0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF]
 *
 * Instruction mapping (SSE -> NEON):
 *   PALIGNR $8, Xsrc, Xdst  ->  ext  Vd.16b, Vn.16b, Vm.16b, #8
 *   PSLLQ $N, X             ->  shl  Vd.2d, Vn.2d, #N
 *   PSRLQ $N, X             ->  ushr Vd.2d, Vn.2d, #N
 *   POR Xsrc, Xdst          ->  orr  Vd.16b, Vn.16b, Vm.16b
 *   PAND Xsrc, Xdst         ->  and  Vd.16b, Vn.16b, Vm.16b
 */

/* ---- Cross-platform glue ---- */
#ifdef __APPLE__
#define GLBL(name)   .globl _##name
#define ENTRY(name)  _##name:
#define SIZE(name)
#define ADRP_PAD(r)  adrp r, LS_PAD_MASK@PAGE
#define ADD_PAD(r)   add  r, r, LS_PAD_MASK@PAGEOFF
#else
#define GLBL(name)   .globl name ; .type name, %function
#define ENTRY(name)  name:
#define SIZE(name)   .size name, .-name
#define ADRP_PAD(r)  adrp r, LS_PAD_MASK
#define ADD_PAD(r)   add  r, r, :lo12:LS_PAD_MASK
#endif

    .text
    .p2align 3

/* ================================================================
 * extrude_quads_neon_core - 1-quad-at-a-time loop
 * ================================================================ */
    .p2align 2
GLBL(extrude_quads_neon_core)
ENTRY(extrude_quads_neon_core)
    mov    x2, x1   /* count came in as ARG1, put in x2 */

    cbz     x2, .Ldone_1x

    /* Init: x0 = buf + (count - 1) x 128 => last quad's write position (dst) */
    sub     x3, x2, #1              /* x3  = (count - 1) */
    add     x0, x0, x3, lsl #7      /* x0 += (count - 1) x 128 */

    /* Init: x1 = x0 - (count - 1) = buf + (count - 1) * 127 = last quad's read position (src) */
    sub     x1, x0, x3

    /* Load padMask once: [all_ones, mask62] */
    ADRP_PAD(x9)
    ADD_PAD(x9)
    ldr     q18, [x9]

.Lloop_1x:
    /* ===== LOAD: 8 × 128-bit = 128 bytes from src ===== */
    ld1     {v0.16b-v3.16b}, [x1], #64     /* bytes  0-63 */
    ld1     {v4.16b-v7.16b}, [x1]          /* bytes 64-127 */
    sub     x1, x1, #191                 /* total advance = -64-127 = -191 */

    /* ===== NODE 3: shift left 6 ===== */
    /* carry from Node 2's last limb (L11, in v5 high lane) */
    ext     v16.16b, v5.16b, v6.16b, #8    /* v16 = [L11, L12] */
    ext     v17.16b, v6.16b, v7.16b, #8    /* v17 = [L13, L14] */
    shl     v6.2d, v6.2d, #6
    ushr    v16.2d, v16.2d, #58
    orr     v6.16b, v6.16b, v16.16b
    shl     v7.2d, v7.2d, #6
    ushr    v17.2d, v17.2d, #58
    orr     v7.16b, v7.16b, v17.16b
    and     v7.16b, v7.16b, v18.16b         /* clear bits 62-63 of L15 */

    /* ===== NODE 2: shift left 4 ===== */
    ext     v16.16b, v3.16b, v4.16b, #8    /* v16 = [L7, L8] */
    ext     v17.16b, v4.16b, v5.16b, #8    /* v17 = [L9, L10] */
    shl     v4.2d, v4.2d, #4
    ushr    v16.2d, v16.2d, #60
    orr     v4.16b, v4.16b, v16.16b
    shl     v5.2d, v5.2d, #4
    ushr    v17.2d, v17.2d, #60
    orr     v5.16b, v5.16b, v17.16b
    and     v5.16b, v5.16b, v18.16b         /* clear bits 62-63 of L11 */

    /* ===== NODE 1: shift left 2 ===== */
    ext     v16.16b, v1.16b, v2.16b, #8    /* v16 = [L3, L4] */
    ext     v17.16b, v2.16b, v3.16b, #8    /* v17 = [L5, L6] */
    shl     v2.2d, v2.2d, #2
    ushr    v16.2d, v16.2d, #62
    orr     v2.16b, v2.16b, v16.16b
    shl     v3.2d, v3.2d, #2
    ushr    v17.2d, v17.2d, #62
    orr     v3.16b, v3.16b, v17.16b
    and     v3.16b, v3.16b, v18.16b         /* clear bits 62-63 of L7 */

    /* ===== NODE 0: no shift, just mask ===== */
    and     v1.16b, v1.16b, v18.16b         /* clear bits 62-63 of L3 */

    /* ===== STORE: 8 × 128-bit = 128 bytes to dst ===== */
    st1     {v0.16b-v3.16b}, [x0], #64
    st1     {v4.16b-v7.16b}, [x0], #64
    sub     x0, x0, #256

    subs    x2, x2, #1
    bne     .Lloop_1x

.Ldone_1x:
    ret

SIZE(extrude_quads_neon_core)

/* ================================================================
 * extrude_quads_2x_neon_core - 2-quad unrolled loop
 *
 * count MUST be a multiple of 2.
 * Interleaves two quads to saturate load/store and ALU ports.
 *
 * Register allocation (21 of 32):
 *   Q0: v0-v7    (Node 0-3 of quad A)
 *   Q1: v8-v15   (Node 0-3 of quad B)
 *   v16: padMask
 *   v17-v20: carry temps (2 per quad, interleaved)
 * ================================================================ */
    .p2align 2
GLBL(extrude_quads_2x_neon_core)
ENTRY(extrude_quads_2x_neon_core)
    mov    x2, x1

    cbz     x2, .Ldone_2x

    sub     x3, x2, #2              /* x3 = count - 2 */
    add     x0, x0, x3, lsl #7      /* x0 = buf + (count-2)*128 */
    sub     x1, x0, x3              /* x1 = buf + (count-2)*127 */

    ADRP_PAD(x9)
    ADD_PAD(x9)
    ldr     q16, [x9]

.Lloop_2x:
    /* ===== LOAD: quad A (bytes 0-127 at src) ===== */
    ld1     {v0.16b-v3.16b}, [x1], #64
    ld1     {v4.16b-v7.16b}, [x1], #64
    /* x1 (src) now at src+128; next quad at src_orig+127 */

    /* ===== LOAD: quad B (bytes 127-254 at src_orig) ===== */
    sub     x1, x1, #1             /*  step back 1 byte to land at quad B's start (which is quad A's start + 127) */
    ld1     {v8.16b-v11.16b}, [x1], #64
    ld1     {v12.16b-v15.16b}, [x1], #64
    sub     x1, x1, #509         /* back 1 -> src_orig+254 = next 2x start (backwards)*/

    /* ===== NODE 3: shift left 6 (both quads interleaved) ===== */
    ext     v17.16b, v5.16b,  v6.16b,  #8  /* A: [L11,L12] */
    ext     v18.16b, v13.16b, v14.16b, #8  /* B: [L11,L12] */
    ext     v19.16b, v6.16b,  v7.16b,  #8  /* A: [L13,L14] */
    ext     v20.16b, v14.16b, v15.16b, #8  /* B: [L13,L14] */
    shl     v6.2d,  v6.2d,  #6
    shl     v14.2d, v14.2d, #6
    ushr    v17.2d, v17.2d, #58
    ushr    v18.2d, v18.2d, #58
    orr     v6.16b,  v6.16b,  v17.16b
    orr     v14.16b, v14.16b, v18.16b
    shl     v7.2d,  v7.2d,  #6
    shl     v15.2d, v15.2d, #6
    ushr    v19.2d, v19.2d, #58
    ushr    v20.2d, v20.2d, #58
    orr     v7.16b,  v7.16b,  v19.16b
    orr     v15.16b, v15.16b, v20.16b
    and     v7.16b,  v7.16b,  v16.16b
    and     v15.16b, v15.16b, v16.16b

    /* ===== NODE 2: shift left 4 ===== */
    ext     v17.16b, v3.16b,  v4.16b,  #8
    ext     v18.16b, v11.16b, v12.16b, #8
    ext     v19.16b, v4.16b,  v5.16b,  #8
    ext     v20.16b, v12.16b, v13.16b, #8
    shl     v4.2d,  v4.2d,  #4
    shl     v12.2d, v12.2d, #4
    ushr    v17.2d, v17.2d, #60
    ushr    v18.2d, v18.2d, #60
    orr     v4.16b,  v4.16b,  v17.16b
    orr     v12.16b, v12.16b, v18.16b
    shl     v5.2d,  v5.2d,  #4
    shl     v13.2d, v13.2d, #4
    ushr    v19.2d, v19.2d, #60
    ushr    v20.2d, v20.2d, #60
    orr     v5.16b,  v5.16b,  v19.16b
    orr     v13.16b, v13.16b, v20.16b
    and     v5.16b,  v5.16b,  v16.16b
    and     v13.16b, v13.16b, v16.16b

    /* ===== NODE 1: shift left 2 ===== */
    ext     v17.16b, v1.16b,  v2.16b,  #8
    ext     v18.16b, v9.16b,  v10.16b, #8
    ext     v19.16b, v2.16b,  v3.16b,  #8
    ext     v20.16b, v10.16b, v11.16b, #8
    shl     v2.2d,  v2.2d,  #2
    shl     v10.2d, v10.2d, #2
    ushr    v17.2d, v17.2d, #62
    ushr    v18.2d, v18.2d, #62
    orr     v2.16b,  v2.16b,  v17.16b
    orr     v10.16b, v10.16b, v18.16b
    shl     v3.2d,  v3.2d,  #2
    shl     v11.2d, v11.2d, #2
    ushr    v19.2d, v19.2d, #62
    ushr    v20.2d, v20.2d, #62
    orr     v3.16b,  v3.16b,  v19.16b
    orr     v11.16b, v11.16b, v20.16b
    and     v3.16b,  v3.16b,  v16.16b
    and     v11.16b, v11.16b, v16.16b

    /* ===== NODE 0: just mask ===== */
    and     v1.16b, v1.16b, v16.16b
    and     v9.16b, v9.16b, v16.16b

    /* ===== STORE: quad A, then quad B ===== */
    st1     {v0.16b-v3.16b}, [x0], #64
    st1     {v4.16b-v7.16b}, [x0], #64
    st1     {v8.16b-v11.16b}, [x0], #64
    st1     {v12.16b-v15.16b}, [x0], #64
    sub     x0, x0, 512

    subs    x2, x2, #2
    bne     .Lloop_2x

.Ldone_2x:
    ret

SIZE(extrude_quads_2x_neon_core)


/* ================================================================
 * Read-only data
 * ================================================================ */
#ifdef __APPLE__
    .section __TEXT,__const
#else
    .section .rodata,"a",@progbits
#endif
    .align 4

/* padMask: lane 0 = all ones (preserve), lane 1 = 0x3FFFFFFFFFFFFFFF (clear top 2 bits) */
LS_PAD_MASK:
    .quad 0xFFFFFFFFFFFFFFFF
    .quad 0x3FFFFFFFFFFFFFFF

#ifndef __APPLE__
    .section .note.GNU-stack,"",@progbits
#endif