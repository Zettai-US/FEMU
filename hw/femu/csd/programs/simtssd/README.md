# SIMT SSD hetgpu program (CSD shared library)

The SIMT SSD fused-attention program for LMCache, built as a CSD
shared-library program: FEMU's CSD module loads `libsimtssd_fused.so`
from `csd_program_dir`, resolves the caller-chosen symbol
(`simtssd_fused_main`), and calls it with a `FemuCsdArgs` whose memory
ranges (an MRS the host registered) are already resolved to host
addresses. The plugin NVRTC-compiles the embedded kernels
(`simtssd_kernels_src.h`, embedded from
`lmcache/.../simtssd/hetgpu/simtssd_hetgpu_kernels.cu`) and runs the
fused attention on a real GPU — functional execution on hardware,
NAND/channel timing in FEMU's SSD model.

## Build

    gcc -std=c11 -O2 -fPIC -shared \
        -I. \
        simtssd_csd_plugin.c -ldl -lm \
        -o libsimtssd_fused.so

Place `libsimtssd_fused.so` in the directory passed as the device's
`csd_program_dir` property. Loading inside the FEMU tree uses the real
`hw/femu/csd/csd.h` (build with `-DSIMTSSD_CSD_REAL_HEADERS
-I../..`); standalone cross-validation against the LMCache reference
uses the shipped `csd_args_shim.h`.

## Command contract

The EXECUTE payload references ONE AFDM memory range (cparam1 = its
index; the kernel NVMe passthrough cannot set the EXEC dlen/cdw4
field, so descriptors cannot travel in the EXEC payload). The plugin
resolves every descriptor as a byte offset from that range's base:

- The range holds, in order: the KV blob (token-major
  `[layers][2][heads][tokens][head_dim]`, cut into 16-token pages),
  the u64 page tables for K and V, the host staging regions
  (K_APP/V_APP/Q/POS/MASK), and O_FINAL.
- `cparam2`: byte offset of the 13 packed 64-byte MemoryRange
  descriptors within the same range, in the `fused_attn_prefill` role
  order (K_PRE, V_PRE, K_APP, V_APP, Q, POS, MASK, K_MERGED,
  V_MERGED, SCORES, PROBS, O_PART, O_FINAL; see
  `lmcache/v1/distributed/l2_adapters/simtssd/programs/fused_attn.py`).
- The score scale is derived as `1/sqrt(head_dim)` from the
  descriptors' geometry.

`PAGE_INDIRECT` descriptors address pages through a u64 page-offset
table appended to the same range (offsets relative to the range base).

Driver-ABI note (bring-up finding): CUDA memory entry points must use
the `_v2` symbols (`cuMemAlloc_v2`, `cuMemcpyHtoDAsync_v2`, ...) — the
undecorated names bind to the legacy-context entry points and fail
with `CUDA_ERROR_INVALID_CONTEXT` (the plugin's `dlsym2` handles
this).

## Compute deployment: out-of-process GPU server

cuInit inside the QEMU process conflicts with the emulator's own
context/threading (observed as CUDA_ERROR_OUT_OF_MEMORY under GPU
pressure from co-tenants and CUDA_ERROR_INVALID_CONTEXT on CSD
workers). The supported deployment therefore runs CUDA in a separate
daemon: `gpu_server.c` (same directory) NVRTC-compiles the shared
kernels at startup and serves framed fused-attention requests over a
UNIX socket (`SIMTSSD_GPU_SOCK`, default `/tmp/simtssd-gpu.sock`).
The plugin gathers operands in the emulator (no CUDA there), ships
them, and writes the returned O_final into the AFDM.

    gcc -std=c11 -O2 -Wall -I. gpu_server.c -o gpu_server -ldl -lm -lpthread
    ./gpu_server &            # before the first EXECUTE

Both deployment shapes are validated in-process by the LMCache
reference tests (test_simtssd_csd_plugin.py drives the plugin; the
gpu-server request path is exercised by the same operand layout).

The host side (LMCache `simtssd` L2 adapter) records the per-key page
tables and issues the standard CSD/CP commands (COMPUTE LOAD /
ACTIVATE on this program, EXECUTE with the descriptor payload).

## Validation

`lmcache/tests/v1/distributed/l2_adapters/test_simtssd_csd_plugin.py`
compiles this plugin with plain gcc, drives `simtssd_fused_main` the
way the CSD layer would, and asserts the output matches
`reference_fused_attention` within fp32 tolerance on a real GPU.
