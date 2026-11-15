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

`data_buffer` carries 13 packed 64-byte SIMT SSD MemoryRange
descriptors in the `fused_attn_prefill` role order
(K_PRE, V_PRE, K_APP, V_APP, Q, POS, MASK, K_MERGED, V_MERGED,
SCORES, PROBS, O_PART, O_FINAL; see
`lmcache/v1/distributed/l2_adapters/simtssd/programs/fused_attn.py`).

- `cparam1`: index of the MRS range holding the KV store. KV blobs
  are token-major (`[layers][2][heads][tokens][head_dim]`), cut into
  16-token pages; `PAGE_INDIRECT` descriptors address them through a
  u64 page-offset table appended to the same range (offsets relative
  to the range base).
- `cparam2`: the score scale `1/sqrt(d)` as float bits.
- HOST_DMA/DEV_DRAM descriptors resolve to `mr_addr[descriptor
  index]`; outputs (O_FINAL) are written in place (zero-copy from the
  plugin's perspective).

The host side (LMCache `simtssd` L2 adapter) records the per-key page
tables and issues the standard CSD/CP commands (COMPUTE LOAD /
ACTIVATE on this program, EXECUTE with the descriptor payload).

## Validation

`lmcache/tests/v1/distributed/l2_adapters/test_simtssd_csd_plugin.py`
compiles this plugin with plain gcc, drives `simtssd_fused_main` the
way the CSD layer would, and asserts the output matches
`reference_fused_attention` within fp32 tolerance on a real GPU.
