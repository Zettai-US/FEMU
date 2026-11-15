/* SPDX-License-Identifier: Apache-2.0
 *
 * Stand-in for hw/femu/csd/csd.h's FemuCsdArgs so the plugin compiles
 * standalone (gcc + pytest cross-validation) without QEMU headers.
 * When building inside the FEMU tree, include the real csd.h instead
 * (-DSIMTSSD_CSD_REAL_HEADERS) — the layout is identical: every field
 * is naturally aligned, so QEMU_PACKED does not change the ABI here.
 * Keep in sync with hw/femu/csd/csd.h.
 */
#ifndef SIMTSSD_CSD_ARGS_SHIM_H
#define SIMTSSD_CSD_ARGS_SHIM_H

#include <stddef.h>
#include <stdint.h>

typedef struct FemuCsdArgs {
    int numr;
    void **mr_addr;
    long long *mr_len;
    long long cparam1;
    long long cparam2;
    void *data_buffer;
    long long buffer_len;
} FemuCsdArgs;

#endif /* SIMTSSD_CSD_ARGS_SHIM_H */
