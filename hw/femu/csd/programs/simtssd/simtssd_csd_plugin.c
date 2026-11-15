/* SPDX-License-Identifier: Apache-2.0
 *
 * SIMT SSD hetgpu engine as a FEMU CSD shared-library program.
 *
 * FEMU's CSD module (hw/femu/csd/) loads this .so through
 * g_module_open and calls the exported symbol with a FemuCsdArgs: the
 * memory ranges (an MRS the host registered) are already resolved to
 * host addresses, and the EXECUTE payload carries the SIMT SSD
 * descriptors. The plugin NVRTC-compiles the embedded kernels and
 * runs the fused attention on a real GPU — functional execution on
 * hardware, NAND/channel timing in FEMU's SSD model.
 *
 * Command contract (v1):
 * - data_buffer: 13 packed 64-byte MemoryRange descriptors in the
 *   fused_attn_prefill role order (see
 *   lmcache/.../simtssd/programs/fused_attn.py).
 * - cparam1: index of the MRS range holding the KV store; PAGE_INDIRECT
 *   table offsets are relative to that range's base address.
 * - HOST_DMA / DEV_DRAM descriptors resolve to mr_addr[index].
 */

#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef SIMTSSD_CSD_REAL_HEADERS
#include "csd.h"
#else
#include "csd_args_shim.h"
#endif

#include "simtssd_kernels_src.h"

#define N_RANGES 13

/* MemoryRange wire layout (protocol.py; page_bytes precedes flags so a
 * plain C struct needs no packing). */
typedef struct ssd_range {
    uint8_t kind;
    uint8_t dir;
    uint8_t ndim;
    uint8_t dtype;
    uint8_t _pad0[4];
    uint64_t addr;
    uint64_t length;
    uint32_t dims[8];
    uint32_t page_bytes;
    uint16_t flags;
    uint8_t _pad1[2];
} ssd_range_t;

#define FLAG_PAGE_INDIRECT 1
#define RANGE_KIND_NAND 0
#define RANGE_KIND_DRAM 1
#define RANGE_KIND_HOST 2

typedef int (*pfn_culaunch)(void *, unsigned, unsigned, unsigned, unsigned,
                            unsigned, unsigned, unsigned, void *, void **,
                            void **);
typedef int (*pfn_curet)(void *, const void *, unsigned long long,
                         unsigned long long, void *);
typedef int (*pfn_cualloc)(void **, unsigned long long);
typedef int (*pfn_cufree)(void *);

typedef struct ssd_cuda {
    void *libcuda;
    void *libnvrtc;
    void *module;
    void *fn_filter_merge;
    void *fn_matmul;
    void *fn_softmax;
    void *stream;
} ssd_cuda_t;

static ssd_cuda_t g_cuda;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_init_rc = -1;

/* CUDA 4.0 split 64-bit memory APIs into _v2 entry points; the
 * undecorated names bind to the legacy-context entry point and fail
 * with CUDA_ERROR_INVALID_CONTEXT on a primary context. Prefer _v2. */
static void *dlsym2(void *handle, const char *v2, const char *v1) {
    void *p = dlsym(handle, v2);
    return p ? p : dlsym(handle, v1);
}

static int cuda_init_locked(ssd_cuda_t *c) {
    int (*cuInit)(unsigned) = dlsym(c->libcuda, "cuInit");
    int (*cuDeviceGet)(void *, int) = dlsym(c->libcuda, "cuDeviceGet");
    int (*cuCtxSetCurrent)(void *) = dlsym(c->libcuda, "cuCtxSetCurrent");
    int (*cuDevicePrimaryCtxRetain)(void **, void *) =
        dlsym(c->libcuda, "cuDevicePrimaryCtxRetain");
    int (*cuStreamCreate)(void **, unsigned) =
        dlsym(c->libcuda, "cuStreamCreate");
    int (*cuModuleLoadData)(void **, const void *) =
        dlsym(c->libcuda, "cuModuleLoadData");
    int (*cuModuleGetFunction)(void **, void *, const char *) =
        dlsym(c->libcuda, "cuModuleGetFunction");
    int (*nvrtcCreateProgram)(void **, const char *, const char *, int,
                              const char **, const char **) =
        dlsym(c->libnvrtc, "nvrtcCreateProgram");
    int (*nvrtcCompileProgram)(void *, int, const char **) =
        dlsym(c->libnvrtc, "nvrtcCompileProgram");
    int (*nvrtcGetPTXSize)(void *, unsigned long long *) =
        dlsym(c->libnvrtc, "nvrtcGetPTXSize");
    int (*nvrtcGetPTX)(void *, char *) = dlsym(c->libnvrtc, "nvrtcGetPTX");
    if (!cuInit || !cuDeviceGet || !cuCtxSetCurrent ||
        !cuDevicePrimaryCtxRetain || !cuStreamCreate || !cuModuleLoadData ||
        !cuModuleGetFunction || !nvrtcCreateProgram || !nvrtcCompileProgram ||
        !nvrtcGetPTXSize || !nvrtcGetPTX) {
        return -1;
    }
    void *dev = NULL;
    void *ctx = NULL;
    if (cuInit(0) || cuDeviceGet(&dev, 0) ||
        cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx) ||
        cuStreamCreate(&c->stream, 0)) {
        return -1;
    }
    void *prog = NULL;
    char name[] = "simtssd_hetgpu_kernels";
    char opt0[] = "--std=c++17";
    char opt1[] = "-arch=compute_80";
    char *opts[2] = {opt0, opt1};
    if (nvrtcCreateProgram(&prog, g_simtssd_kernels_src, name, 0, NULL,
                           NULL) ||
        nvrtcCompileProgram(prog, 2, (const char **)opts)) {
        return -1;
    }
    unsigned long long ptx_size = 0;
    if (nvrtcGetPTXSize(prog, &ptx_size)) {
        return -1;
    }
    char *ptx = (char *)malloc(ptx_size);
    if (!ptx || nvrtcGetPTX(prog, ptx)) {
        free(ptx);
        return -1;
    }
    int rc = cuModuleLoadData(&c->module, ptx);
    free(ptx);
    if (rc || cuModuleGetFunction(&c->fn_filter_merge, c->module,
                                  "simtssd_filter_merge") ||
        cuModuleGetFunction(&c->fn_matmul, c->module, "simtssd_matmul") ||
        cuModuleGetFunction(&c->fn_softmax, c->module,
                            "simtssd_softmax_masked")) {
        return -1;
    }
    return 0;
}

static void cuda_once(void) {
    g_cuda.libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    g_cuda.libnvrtc = dlopen("libnvrtc.so.12", RTLD_NOW | RTLD_LOCAL);
    if (!g_cuda.libnvrtc) {
        g_cuda.libnvrtc = dlopen("libnvrtc.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_cuda.libcuda || !g_cuda.libnvrtc) {
        g_init_rc = -1;
        return;
    }
    g_init_rc = cuda_init_locked(&g_cuda);
}

/* The exported CSD entry point: int64_t fn(FemuCsdArgs *args);
 * return 0 on success, negative on failure (lands in the CQE status). */
static void cuda_lazy_init(void) {
    pthread_once(&g_once, cuda_once);
}

static int launch1(ssd_cuda_t *c, void *fn, void **params,
                   unsigned long long blocks) {
    pfn_culaunch cuLaunchKernel =
        (pfn_culaunch)dlsym(c->libcuda, "cuLaunchKernel");
    if (!cuLaunchKernel) {
        return -1;
    }
    return cuLaunchKernel(fn, (unsigned)blocks, 1, 1, 256, 1, 1, 0, c->stream,
                          params, NULL)
               ? -1
               : 0;
}

static int stage_in(ssd_cuda_t *c, pfn_cualloc cuMemAlloc,
                    pfn_curet cuMemcpyHtoD, const void *host, uint64_t len,
                    void **dev) {
    if (cuMemAlloc(dev, len ? len : 1) != 0) {
        return -1;
    }
    if (len && cuMemcpyHtoD(*dev, host, len, 0, c->stream) != 0) {
        return -1;
    }
    return 0;
}

int64_t simtssd_fused_main(FemuCsdArgs *args) {
    cuda_lazy_init();
    if (g_init_rc != 0 || args->numr < N_RANGES || !args->data_buffer) {
        return -1;
    }
    ssd_cuda_t *c = &g_cuda;
    pfn_cualloc cuMemAlloc_ =
        (pfn_cualloc)dlsym2(c->libcuda, "cuMemAlloc_v2", "cuMemAlloc");
    pfn_cufree cuMemFree_ =
        (pfn_cufree)dlsym2(c->libcuda, "cuMemFree_v2", "cuMemFree");
    pfn_curet cuMemcpyHtoD_ =
        (pfn_curet)dlsym2(c->libcuda, "cuMemcpyHtoDAsync_v2",
                          "cuMemcpyHtoDAsync");
    pfn_curet cuMemcpyDtoH_ =
        (pfn_curet)dlsym2(c->libcuda, "cuMemcpyDtoHAsync_v2",
                          "cuMemcpyDtoHAsync");
    int (*cuStreamSync)(void *) =
        (int (*)(void *))dlsym(c->libcuda, "cuStreamSynchronize");
    if (!cuMemAlloc_ || !cuMemFree_ || !cuMemcpyHtoD_ || !cuMemcpyDtoH_ ||
        !cuStreamSync) {
        return -1;
    }

    if ((size_t)args->buffer_len < N_RANGES * sizeof(ssd_range_t)) {
        return -1;
    }
    ssd_range_t r[N_RANGES];
    memcpy(r, args->data_buffer, sizeof(r));
    int kv_mr = (int)args->cparam1;
    if (kv_mr < 0 || kv_mr >= args->numr || !args->mr_addr[kv_mr]) {
        return -1;
    }
    uint8_t *kv_base = (uint8_t *)args->mr_addr[kv_mr];

    /* Gather the pre K/V pages (PAGE_INDIRECT into the KV MRS, or a
     * contiguous window when the flag is absent). */
    uint8_t *kbuf = NULL;
    uint8_t *vbuf = NULL;
    for (int i = 0; i < 2; i++) {
        ssd_range_t *rp = &r[i];
        uint64_t len = rp->length;
        uint8_t *buf = (uint8_t *)malloc(len ? len : 1);
        if (!buf) {
            free(kbuf);
            free(vbuf);
            return -1;
        }
        if (rp->flags & FLAG_PAGE_INDIRECT) {
            int n = (int)((len + rp->page_bytes - 1) / rp->page_bytes);
            const uint64_t *tbl =
                (const uint64_t *)(kv_base + rp->addr);
            for (int j = 0; j < n; j++) {
                uint64_t off = tbl[j];
                size_t chunk = rp->page_bytes;
                if ((uint64_t)(j + 1) * rp->page_bytes > len) {
                    chunk = len - (uint64_t)j * rp->page_bytes;
                }
                memcpy(buf + (size_t)j * rp->page_bytes,
                       kv_base + off, chunk);
            }
        } else {
            memcpy(buf, kv_base + rp->addr, len);
        }
        if (i == 0) {
            kbuf = buf;
        } else {
            vbuf = buf;
        }
    }

    int heads = (int)r[0].dims[0];
    int n_ctx = (int)r[0].dims[1];
    int d = (int)r[0].dims[2];
    int n_app = (int)r[2].dims[1];
    int n_hkvd = (int)r[5].dims[0];
    int n_q = n_app - n_hkvd;
    int total = n_ctx + n_q;
    int q_rows = n_hkvd + n_q;
    /* The host sends the score scale (1/sqrt(d)) as float bits in
     * cparam2. */
    float scale = 0.0f;
    memcpy(&scale, &args->cparam2, sizeof(scale));
    if (scale <= 0.0f || scale > 1.0f) {
        scale = 1.0f / sqrtf((float)d);
    }

    long long *pos64 =
        (long long *)malloc((size_t)n_hkvd * sizeof(long long));
    if (!pos64) {
        free(kbuf);
        free(vbuf);
        return -1;
    }
    {
        const int *p32 = (const int *)args->mr_addr[5];
        for (int i = 0; i < n_hkvd; i++) {
            pos64[i] = p32[i];
        }
    }

    void *d_pre_k = NULL;
    void *d_pre_v = NULL;
    void *d_app_k = NULL;
    void *d_app_v = NULL;
    void *d_q = NULL;
    void *d_pos = NULL;
    void *d_mask = NULL;
    void *d_km = NULL;
    void *d_vm = NULL;
    void *d_scores = NULL;
    void *d_probs = NULL;
    void *d_opart = NULL;
    void *d_ofinal = NULL;
    int failed = 0;
    if (stage_in(c, cuMemAlloc_, cuMemcpyHtoD_, kbuf, (uint64_t)r[0].length,
                 &d_pre_k) ||
        stage_in(c, cuMemAlloc_, cuMemcpyHtoD_, vbuf, (uint64_t)r[1].length,
                 &d_pre_v) ||
        stage_in(c, cuMemAlloc_, cuMemcpyHtoD_, args->mr_addr[2],
                 r[2].length, &d_app_k) ||
        stage_in(c, cuMemAlloc_, cuMemcpyHtoD_, args->mr_addr[3],
                 r[3].length, &d_app_v) ||
        stage_in(c, cuMemAlloc_, cuMemcpyHtoD_, args->mr_addr[4],
                 r[4].length, &d_q) ||
        stage_in(c, cuMemAlloc_, cuMemcpyHtoD_, pos64,
                 (uint64_t)n_hkvd * sizeof(long long), &d_pos) ||
        stage_in(c, cuMemAlloc_, cuMemcpyHtoD_, args->mr_addr[6],
                 r[6].length, &d_mask) ||
        cuMemAlloc_(&d_km, (unsigned long long)heads * total * d * 4) ||
        cuMemAlloc_(&d_vm, (unsigned long long)heads * total * d * 4) ||
        cuMemAlloc_(&d_scores,
                    (unsigned long long)heads * q_rows * total * 4) ||
        cuMemAlloc_(&d_probs,
                    (unsigned long long)heads * q_rows * total * 4) ||
        cuMemAlloc_(&d_opart, (unsigned long long)heads * q_rows * d * 4) ||
        cuMemAlloc_(&d_ofinal, (unsigned long long)heads * q_rows * d * 4)) {
        failed = 1;
    }

    if (!failed) {
        unsigned long long blocks =
            ((unsigned long long)total * heads * d + 255) / 256;
        int hkvd = n_hkvd;
        int nqq = n_q;
        void *a = d_pre_k;
        void *o = d_km;
        void *params_k[] = {&a, &d_app_k, &d_pos, &o, &n_ctx, &hkvd, &nqq,
                            &heads, &d};
        void *a2 = d_pre_v;
        void *o2 = d_vm;
        void *params_v[] = {&a2, &d_app_v, &d_pos, &o2, &n_ctx, &hkvd, &nqq,
                            &heads, &d};
        failed |= launch1(c, c->fn_filter_merge, params_k, blocks);
        failed |= launch1(c, c->fn_filter_merge, params_v, blocks);
        unsigned long long sn = (unsigned long long)heads * q_rows * total;
        int tb = 1;
        void *params_s[] = {&d_q, &d_km, &d_scores, &heads, &q_rows, &d,
                            &total, &tb, &scale};
        failed |= launch1(c, c->fn_matmul, params_s, (sn + 255) / 256);
        void *params_m[] = {&d_scores, &d_mask, &d_probs, &heads, &q_rows,
                            &total};
        failed |= launch1(c, c->fn_softmax, params_m,
                          ((unsigned long long)heads * q_rows + 255) / 256);
        unsigned long long on = (unsigned long long)heads * q_rows * d;
        int tb0 = 0;
        float one = 1.0f;
        void *params_o[] = {&d_probs, &d_vm, &d_opart, &heads, &q_rows,
                            &total, &d, &tb0, &one};
        failed |= launch1(c, c->fn_matmul, params_o, (on + 255) / 256);
        /* LSE reduce: normalized probs make the partial sum the final
         * output (numerically identical to the LSE merge). */
        unsigned long long obytes =
            (unsigned long long)heads * q_rows * d * 4;
        uint8_t *h_ofinal = (uint8_t *)args->mr_addr[12];
        if (cuMemcpyDtoH_(h_ofinal, d_opart, obytes, 0, c->stream) != 0) {
            failed = 1;
        }
        if (cuStreamSync(c->stream) != 0) {
            failed = 1;
        }
    }

    void *devs[] = {d_pre_k, d_pre_v, d_app_k, d_app_v, d_q,  d_pos,
                    d_mask,  d_km,    d_vm,    d_scores, d_probs,
                    d_opart, d_ofinal};
    for (unsigned long long i = 0; i < sizeof(devs) / sizeof(devs[0]); i++) {
        if (devs[i]) {
            cuMemFree_(devs[i]);
        }
    }
    free(pos64);
    free(kbuf);
    free(vbuf);
    return failed ? -1 : 0;
}
