/* SPDX-License-Identifier: Apache-2.0
 *
 * SIMT SSD GPU compute server.
 *
 * A standalone daemon that owns the CUDA context for the SIMT SSD
 * hetgpu program: NVRTC-compiles the embedded kernels at startup, then
 * serves compute requests over a UNIX socket. The FEMU CSD plugin (or
 * any SIMT SSD backend) ships operand buffers to it and receives the
 * fused attention output — keeping every trace of CUDA out of the
 * emulator process (cuInit inside QEMU faults; a separate process is
 * the supported compute path).
 *
 * Request protocol (little-endian):
 *   u32 magic 'SSDX'; u32 heads, n_ctx, n_hkvd, n_q, d; u32 pad;
 *   f32 scale;
 *   then the operand blobs in order: pre_k, pre_v, app_k, app_v, q
 *   (fp32), positions (int32), mask (uint8 bool, q_rows*ntotal);
 *   then u32 mask_len followed by the mask bytes is folded into the
 *   header total instead: the request is length-framed — see below.
 * Framing: u32 payload_len, then the header + operand bytes; the reply
 * is u32 status (0 = ok) followed by the O_final fp32 bytes.
 */

#include <arpa/inet.h>
#include <dlfcn.h>
#include <math.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "gpu_server_kernels_src.h"

#define SOCK_PATH_ENV "SIMTSSD_GPU_SOCK"
#define SOCK_PATH_DEFAULT "/tmp/simtssd-gpu.sock"
#define MAX_BODY (64u * 1024 * 1024)

static void *libcuda;
static void *libnvrtc;
static void *module;
static void *fn_filter_merge;
static void *fn_matmul;
static void *fn_softmax;
static void *stream;

typedef int (*pfn_culaunch)(void *, unsigned, unsigned, unsigned, unsigned,
                            unsigned, unsigned, unsigned, void *, void **,
                            void **);
typedef int (*pfn_curet)(void *, const void *, unsigned long long,
                         unsigned long long, void *);
typedef int (*pfn_cualloc)(void **, unsigned long long);
typedef int (*pfn_cufree)(void *);

/* CUDA driver + NVRTC bindings via dlopen/dlsym (no dev headers). */
static int cuda_init(void) {
    libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    libnvrtc = dlopen("libnvrtc.so.12", RTLD_NOW | RTLD_LOCAL);
    if (!libnvrtc) {
        libnvrtc = dlopen("libnvrtc.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!libcuda || !libnvrtc) {
        fprintf(stderr, "gpu-server: dlopen failed\n");
        return -1;
    }
    int (*cuInit)(unsigned) = dlsym(libcuda, "cuInit");
    int (*cuDeviceGet)(void *, int) = dlsym(libcuda, "cuDeviceGet");
    int (*cuCtxSetCurrent)(void *) = dlsym(libcuda, "cuCtxSetCurrent");
    int (*cuDevicePrimaryCtxRetain)(void **, void *) =
        dlsym(libcuda, "cuDevicePrimaryCtxRetain");
    int (*cuStreamCreate)(void **, unsigned) =
        dlsym(libcuda, "cuStreamCreate");
    int (*cuModuleLoadData)(void **, const void *) =
        dlsym(libcuda, "cuModuleLoadData");
    int (*cuModuleGetFunction)(void **, void *, const char *) =
        dlsym(libcuda, "cuModuleGetFunction");
    int (*nvrtcCreateProgram)(void **, const char *, const char *, int,
                              const char **, const char **) =
        dlsym(libnvrtc, "nvrtcCreateProgram");
    int (*nvrtcCompileProgram)(void *, int, const char **) =
        dlsym(libnvrtc, "nvrtcCompileProgram");
    int (*nvrtcGetPTXSize)(void *, unsigned long long *) =
        dlsym(libnvrtc, "nvrtcGetPTXSize");
    int (*nvrtcGetPTX)(void *, char *) = dlsym(libnvrtc, "nvrtcGetPTX");
    if (!cuInit || !cuDeviceGet || !cuCtxSetCurrent ||
        !cuDevicePrimaryCtxRetain || !cuStreamCreate || !cuModuleLoadData ||
        !cuModuleGetFunction || !nvrtcCreateProgram || !nvrtcCompileProgram ||
        !nvrtcGetPTXSize || !nvrtcGetPTX) {
        fprintf(stderr, "gpu-server: symbols missing\n");
        return -1;
    }
    void *dev = NULL;
    void *ctx = NULL;
    if (cuInit(0) || cuDeviceGet(&dev, 0) ||
        cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx) ||
        cuStreamCreate(&stream, 0)) {
        fprintf(stderr, "gpu-server: CUDA init failed\n");
        return -1;
    }
    void *prog = NULL;
    char name[] = "simtssd_hetgpu_kernels";
    char opt0[] = "--std=c++17";
    char opt1[] = "-arch=compute_80";
    char *opts[2] = {opt0, opt1};
    if (nvrtcCreateProgram(&prog, g_src, name, 0, NULL, NULL) ||
        nvrtcCompileProgram(prog, 2, (const char **)opts)) {
        fprintf(stderr, "gpu-server: NVRTC compile failed\n");
        return -1;
    }
    unsigned long long ptx_size = 0;
    if (nvrtcGetPTXSize(prog, &ptx_size)) {
        return -1;
    }
    char *ptx = malloc(ptx_size);
    if (!ptx || nvrtcGetPTX(prog, ptx)) {
        return -1;
    }
    int rc = cuModuleLoadData(&module, ptx);
    free(ptx);
    if (rc || cuModuleGetFunction(&fn_filter_merge, module,
                                  "simtssd_filter_merge") ||
        cuModuleGetFunction(&fn_matmul, module, "simtssd_matmul") ||
        cuModuleGetFunction(&fn_softmax, module, "simtssd_softmax_masked")) {
        fprintf(stderr, "gpu-server: module load failed\n");
        return -1;
    }
    return 0;
}

static int launch1(void *fn, const char *name, void **params,
                   unsigned long long blocks) {
    pfn_culaunch cuLaunchKernel = dlsym(libcuda, "cuLaunchKernel");
    if (!cuLaunchKernel) {
        return -1;
    }
    return cuLaunchKernel(fn, (unsigned)blocks, 1, 1, 256, 1, 1, 0, stream,
                          params, NULL)
               ? -1
               : 0;
}

/* ---- framed socket IO ---- */

static int recv_all(int fd, void *buf, size_t len) {
    uint8_t *p = buf;
    size_t got = 0;
    while (got < len) {
        ssize_t n = read(fd, p + got, len - got);
        if (n <= 0) {
            return -1;
        }
        got += (size_t)n;
    }
    return 0;
}

static int send_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = write(fd, p + sent, len - sent);
        if (n <= 0) {
            return -1;
        }
        sent += (size_t)n;
    }
    return 0;
}

/* Compute one fused-attention request. Body layout:
 * [0..24)  heads, n_ctx, n_hkvd, n_q, d, pad (u32 each)
 * [24..28) scale (f32 bits)
 * then contiguous fp32/i32/bool operand bytes:
 * pre_k [H,ctx,D], pre_v [H,ctx,D], app_k [H,r,D], app_v [H,r,D],
 * q [H,r,D], positions i32 [hkvd], mask u8 [r, ctx+q]
 * where r = hkvd + q. Reply: status u32 + O_final fp32 [H,r,D]. */
static int compute(const uint8_t *body, uint32_t body_len,
                   float *out, unsigned long long *out_floats) {
    if (body_len < 28) {
        return -1;
    }
    uint32_t heads, n_ctx, n_hkvd, n_q, d, pad;
    float scale;
    memcpy(&heads, body, 4);
    memcpy(&n_ctx, body + 4, 4);
    memcpy(&n_hkvd, body + 8, 4);
    memcpy(&n_q, body + 12, 4);
    memcpy(&d, body + 16, 4);
    memcpy(&pad, body + 20, 4);
    memcpy(&scale, body + 24, 4);
    int q_rows = (int)(n_hkvd + n_q);
    int n_total = (int)(n_ctx + n_q);
    size_t rows_bytes = (size_t)heads * q_rows * d * 4;
    size_t kv_bytes = 2ull * heads * n_ctx * d * 4;
    size_t q_bytes = rows_bytes;
    size_t pos_bytes = (size_t)n_hkvd * 4;
    size_t mask_bytes = (size_t)q_rows * n_total;
    size_t off = 28;
    const uint8_t *pre = body + off; off += kv_bytes;
    const uint8_t *app = body + off; off += rows_bytes * 2;
    const uint8_t *qp = body + off; off += q_bytes;
    const int *positions = (const int *)(body + off); off += pos_bytes;
    const uint8_t *mask = body + off; off += mask_bytes;
    if (off > body_len) {
        return -1;
    }

    int (*cuMemAlloc)(void **, unsigned long long) =
        dlsym(libcuda, "cuMemAlloc");
    int (*cuMemFree)(void *) = dlsym(libcuda, "cuMemFree");
    int (*cuMemcpyHtoD)(void *, const void *, unsigned long long,
                        unsigned long long, void *) =
        dlsym(libcuda, "cuMemcpyHtoDAsync");
    int (*cuMemcpyDtoH)(void *, const void *, unsigned long long,
                        unsigned long long, void *) =
        dlsym(libcuda, "cuMemcpyDtoHAsync");
    int (*cuStreamSync)(void *) = dlsym(libcuda, "cuStreamSynchronize");
    if (!cuMemAlloc || !cuMemFree || !cuMemcpyHtoD || !cuMemcpyDtoH ||
        !cuStreamSync) {
        return -1;
    }

    void *d_pre = NULL;
    void *d_app = NULL;
    void *d_q = NULL;
    void *d_pos = NULL;
    void *d_mask = NULL;
    void *d_km = NULL;
    void *d_vm = NULL;
    void *d_scores = NULL;
    void *d_probs = NULL;
    void *d_out = NULL;
    if (cuMemAlloc(&d_pre, kv_bytes) ||
        cuMemAlloc(&d_app, rows_bytes * 2) ||
        cuMemAlloc(&d_q, q_bytes) ||
        cuMemAlloc(&d_pos, pos_bytes) ||
        cuMemAlloc(&d_mask, mask_bytes) ||
        cuMemAlloc(&d_km, rows_bytes) ||
        cuMemAlloc(&d_vm, rows_bytes) ||
        cuMemAlloc(&d_scores,
                   (unsigned long long)heads * q_rows * n_total * 4) ||
        cuMemAlloc(&d_probs,
                   (unsigned long long)heads * q_rows * n_total * 4) ||
        cuMemAlloc(&d_out, rows_bytes)) {
        fprintf(stderr, "gpu-server: alloc failed\n");
        goto fail;
    }
    if (cuMemcpyHtoD(d_pre, pre, kv_bytes, 0, stream) ||
        cuMemcpyHtoD(d_app, app, rows_bytes * 2, 0, stream) ||
        cuMemcpyHtoD(d_q, qp, q_bytes, 0, stream) ||
        cuMemcpyHtoD(d_pos, positions, pos_bytes, 0, stream) ||
        cuMemcpyHtoD(d_mask, mask, mask_bytes, 0, stream)) {
        fprintf(stderr, "gpu-server: H2D failed\n");
        goto fail;
    }

    {
        int hkvd = (int)n_hkvd;
        int nqq = (int)n_q;
        int ctx_i = (int)n_ctx;
        unsigned long long blocks =
            ((unsigned long long)heads * n_total * d + 255) / 256;
        void *a = d_pre;
        void *o = d_km;
        void *pk[] = {&a, &d_app, &d_pos, &o, &ctx_i, &hkvd, &nqq,
                      &heads, &d};
        /* V half of the app blob: [K_app rows][V_app rows] contiguous */
        void *a2 = d_pre + kv_bytes;
        void *b2 = d_app + rows_bytes;
        void *o2 = d_vm;
        void *pv[] = {&a2, &b2, &d_pos, &o2, &ctx_i, &hkvd, &nqq,
                      &heads, &d};
        fprintf(stderr, "gpu-server: filter launches\n");
        if (launch1(fn_filter_merge, "filter_k", pk, blocks) ||
            launch1(fn_filter_merge, "filter_v", pv, blocks)) {
            fprintf(stderr, "gpu-server: filter launch failed\n");
            goto fail;
        }
    }
    {
        int tb = 1;
        int ctx_i = (int)n_ctx;
        int qr = q_rows;
        unsigned long long sn =
            ((unsigned long long)heads * q_rows * n_total + 255) / 256;
        void *ps[] = {&d_q, &d_km, &d_scores, &heads, &qr, &d,
                      &ctx_i, &tb, &scale};
        fprintf(stderr, "gpu-server: scores launch\n");
        if (launch1(fn_matmul, "scores", ps, sn)) {
            fprintf(stderr, "gpu-server: scores launch failed\n");
            goto fail;
        }
        void *pm[] = {&d_scores, &d_mask, &d_probs, &heads, &qr, &ctx_i};
        unsigned long long sm =
            ((unsigned long long)heads * q_rows + 255) / 256;
        fprintf(stderr, "gpu-server: softmax launch\n");
        if (launch1(fn_softmax, "softmax", pm, sm)) {
            fprintf(stderr, "gpu-server: softmax launch failed\n");
            goto fail;
        }
        int tb0 = 0;
        unsigned long long on =
            ((unsigned long long)heads * q_rows * d + 255) / 256;
        void *po[] = {&d_probs, &d_vm, &d_out, &heads, &qr, &d,
                      &ctx_i, &tb0, &scale};
        fprintf(stderr, "gpu-server: pv launch\n");
        if (launch1(fn_matmul, "ofinal", po, on)) {
            fprintf(stderr, "gpu-server: pv launch failed\n");
            goto fail;
        }
    }
    if (cuStreamSync(stream)) {
        return -1;
    }
    if (cuMemcpyDtoH(out, d_out, rows_bytes, 0, stream)) {
        return -1;
    }
    cuStreamSync(stream);
    *out_floats = rows_bytes / 4;

    if (d_pre) cuMemFree(d_pre);
    if (d_app) cuMemFree(d_app);
    if (d_q) cuMemFree(d_q);
    if (d_pos) cuMemFree(d_pos);
    if (d_mask) cuMemFree(d_mask);
    if (d_km) cuMemFree(d_km);
    if (d_vm) cuMemFree(d_vm);
    if (d_scores) cuMemFree(d_scores);
    if (d_probs) cuMemFree(d_probs);
    if (d_out) cuMemFree(d_out);
    return 0;

fail:
    if (d_pre) cuMemFree(d_pre);
    if (d_app) cuMemFree(d_app);
    if (d_q) cuMemFree(d_q);
    if (d_pos) cuMemFree(d_pos);
    if (d_mask) cuMemFree(d_mask);
    if (d_km) cuMemFree(d_km);
    if (d_vm) cuMemFree(d_vm);
    if (d_scores) cuMemFree(d_scores);
    if (d_probs) cuMemFree(d_probs);
    if (d_out) cuMemFree(d_out);
    return -1;
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    const char *sock_path = getenv(SOCK_PATH_ENV);
    if (!sock_path || !*sock_path) {
        sock_path = SOCK_PATH_DEFAULT;
    }
    if (cuda_init()) {
        return 1;
    }
    fprintf(stderr, "gpu-server: ready on %s\n", sock_path);

    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) {
        return 1;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
    unlink(sock_path);
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) || listen(srv, 4)) {
        perror("bind/listen");
        return 1;
    }

    for (;;) {
        int fd = accept(srv, NULL, NULL);
        if (fd < 0) {
            continue;
        }
        uint32_t body_len = 0;
        if (recv_all(fd, &body_len, 4) || body_len > MAX_BODY) {
            close(fd);
            continue;
        }
        uint8_t *body = malloc(body_len ? body_len : 1);
        if (!body || recv_all(fd, body, body_len)) {
            free(body);
            close(fd);
            continue;
        }
        /* head geometry from the body prefix */
        uint32_t heads = 0, n_ctx = 0, n_hkvd = 0, n_q = 0, d = 0;
        memcpy(&heads, body, 4);
        memcpy(&n_ctx, body + 4, 4);
        memcpy(&n_hkvd, body + 8, 4);
        memcpy(&n_q, body + 12, 4);
        memcpy(&d, body + 16, 4);
        size_t out_bytes = (size_t)heads * (n_hkvd + n_q) * d * 4;
        float *out = malloc(out_bytes);
        unsigned long long out_floats = 0;
        uint32_t status = compute(body, body_len, out, &out_floats) ? 1 : 0;
        uint32_t reply_len = (uint32_t)(4 + out_floats * 4);
        uint8_t *reply = malloc(reply_len);
        memcpy(reply, &status, 4);
        memcpy(reply + 4, out, out_floats * 4);
        if (send_all(fd, reply, reply_len)) {
            fprintf(stderr, "gpu-server: reply failed\n");
        }
        free(reply);
        free(out);
        free(body);
        close(fd);
    }
}
