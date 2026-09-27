#define _POSIX_C_SOURCE 200809L
#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    perror("clock_gettime");
    exit(70);
  }
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int compare_double(const void *lhs, const void *rhs) {
  double a = *(const double *)lhs;
  double b = *(const double *)rhs;
  return (a > b) - (a < b);
}

static double quantile_sorted(const double *values, int count, double q) {
  if (count == 1) return values[0];
  double index = q * (double)(count - 1);
  int lower = (int)index;
  int upper = lower + 1;
  if (upper >= count) return values[lower];
  double fraction = index - (double)lower;
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

static void usage(const char *argv0) {
  fprintf(stderr,
          "usage: %s --warmup N --reps N --load-seconds S [--m M --k K "
          "--n N]\n",
          argv0);
}

int main(int argc, char **argv) {
  int M = 4;
  int K = 64;
  int N = 32;
  int warmup = -1;
  int reps = -1;
  double load_seconds = -1.0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
      M = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
      K = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      N = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
      reps = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--load-seconds") == 0 && i + 1 < argc) {
      load_seconds = atof(argv[++i]);
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (M <= 0 || K <= 0 || N <= 0 || warmup < 0 || reps <= 0 ||
      load_seconds < 0.0) {
    usage(argv[0]);
    return 2;
  }

  rknn_matmul_ctx ctx = 0;
  rknn_matmul_info info;
  rknn_matmul_io_attr io_attr;
  memset(&info, 0, sizeof(info));
  memset(&io_attr, 0, sizeof(io_attr));
  info.M = M;
  info.K = K;
  info.N = N;
  info.type = RKNN_INT8_MM_INT8_TO_INT32;
  info.B_layout = 0;
  info.AC_layout = 0;
  info.B_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;
  info.AC_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;
  info.iommu_domain_id = 0;

  int ret = rknn_matmul_create(&ctx, &info, &io_attr);
  if (ret < 0) {
    fprintf(stderr, "rknn_matmul_create failed: %d\n", ret);
    return 80;
  }

  int core_mask_ret = rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_0);
  if (core_mask_ret < 0) {
    fprintf(stderr, "rknn_matmul_set_core_mask(core0) failed: %d\n",
            core_mask_ret);
    rknn_matmul_destroy(ctx);
    return 81;
  }

  rknn_tensor_mem *a_mem = rknn_create_mem(ctx, io_attr.A.size);
  rknn_tensor_mem *b_mem = rknn_create_mem(ctx, io_attr.B.size);
  rknn_tensor_mem *c_mem = rknn_create_mem(ctx, io_attr.C.size);
  if (!a_mem || !b_mem || !c_mem) {
    fprintf(stderr, "rknn_create_mem failed\n");
    if (a_mem) rknn_destroy_mem(ctx, a_mem);
    if (b_mem) rknn_destroy_mem(ctx, b_mem);
    if (c_mem) rknn_destroy_mem(ctx, c_mem);
    rknn_matmul_destroy(ctx);
    return 82;
  }

  int8_t *a = calloc((size_t)M * (size_t)K, sizeof(*a));
  int8_t *b = calloc((size_t)K * (size_t)N, sizeof(*b));
  int32_t *reference = calloc((size_t)M * (size_t)N, sizeof(*reference));
  double *latencies_us = calloc((size_t)reps, sizeof(*latencies_us));
  if (!a || !b || !reference || !latencies_us) {
    fprintf(stderr, "host allocation failed\n");
    return 83;
  }

  for (int i = 0; i < M * K; ++i) a[i] = (int8_t)(((i * 17 + 3) % 15) - 7);
  for (int i = 0; i < K * N; ++i) b[i] = (int8_t)(((i * 29 + 5) % 15) - 7);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int32_t sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += (int32_t)a[m * K + k] * (int32_t)b[k * N + n];
      }
      reference[m * N + n] = sum;
    }
  }

  memset(a_mem->virt_addr, 0, io_attr.A.size);
  memset(b_mem->virt_addr, 0, io_attr.B.size);
  memset(c_mem->virt_addr, 0, io_attr.C.size);
  memcpy(a_mem->virt_addr, a, (size_t)M * (size_t)K);
  memcpy(b_mem->virt_addr, b, (size_t)K * (size_t)N);

  if ((ret = rknn_matmul_set_io_mem(ctx, a_mem, &io_attr.A)) < 0 ||
      (ret = rknn_matmul_set_io_mem(ctx, b_mem, &io_attr.B)) < 0 ||
      (ret = rknn_matmul_set_io_mem(ctx, c_mem, &io_attr.C)) < 0) {
    fprintf(stderr, "rknn_matmul_set_io_mem failed: %d\n", ret);
    return 84;
  }

  for (int i = 0; i < warmup; ++i) {
    if ((ret = rknn_matmul_run(ctx)) < 0) {
      fprintf(stderr, "warmup rknn_matmul_run failed: %d\n", ret);
      return 85;
    }
  }

  for (int i = 0; i < reps; ++i) {
    double start = now_seconds();
    ret = rknn_matmul_run(ctx);
    double end = now_seconds();
    if (ret < 0) {
      fprintf(stderr, "timed rknn_matmul_run failed at rep %d: %d\n", i,
              ret);
      return 86;
    }
    latencies_us[i] = (end - start) * 1e6;
  }

  uint64_t load_loop_count = 0;
  double load_start = now_seconds();
  while (now_seconds() - load_start < load_seconds) {
    if ((ret = rknn_matmul_run(ctx)) < 0) {
      fprintf(stderr, "load-window rknn_matmul_run failed: %d\n", ret);
      return 87;
    }
    ++load_loop_count;
  }
  double load_elapsed = now_seconds() - load_start;

  int mismatch_count = 0;
  int32_t max_abs_diff = 0;
  const int32_t *actual = (const int32_t *)c_mem->virt_addr;
  for (int i = 0; i < M * N; ++i) {
    int64_t diff64 = (int64_t)actual[i] - (int64_t)reference[i];
    int32_t abs_diff = (int32_t)(diff64 < 0 ? -diff64 : diff64);
    if (abs_diff != 0) ++mismatch_count;
    if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
  }

  qsort(latencies_us, (size_t)reps, sizeof(*latencies_us), compare_double);
  double p25 = quantile_sorted(latencies_us, reps, 0.25);
  double p50 = quantile_sorted(latencies_us, reps, 0.50);
  double p75 = quantile_sorted(latencies_us, reps, 0.75);

  printf("OPE0_RESULT_JSON={\"schema_version\":1,\"primitive\":"
         "\"rknn_matmul_run\",\"type\":\"int8xint8_to_int32\","
         "\"shape_mkn\":[%d,%d,%d],\"layouts\":{\"A\":\"normal\","
         "\"B\":\"normal\",\"C\":\"normal\"},\"core_mask\":"
         "\"RKNN_NPU_CORE_0\",\"core_mask_ret\":%d,"
         "\"io_sizes_bytes\":{\"A\":%u,\"B\":%u,\"C\":%u},"
         "\"warmup_reps\":%d,\"timed_reps\":%d,"
         "\"run_only_us\":{\"p25\":%.6f,\"p50\":%.6f,"
         "\"p75\":%.6f,\"iqr\":%.6f},\"correctness\":{"
         "\"oracle\":\"CPU_int32_exact\",\"elements\":%d,"
         "\"mismatch_count\":%d,\"max_abs_diff\":%d,\"pass\":%s},"
         "\"load_window\":{\"target_seconds\":%.3f,"
         "\"elapsed_seconds\":%.6f,\"loop_count\":%" PRIu64 "}}\n",
         M, K, N, core_mask_ret, io_attr.A.size, io_attr.B.size,
         io_attr.C.size, warmup, reps, p25, p50, p75, p75 - p25, M * N,
         mismatch_count, max_abs_diff, mismatch_count == 0 ? "true" : "false",
         load_seconds, load_elapsed, load_loop_count);

  free(a);
  free(b);
  free(reference);
  free(latencies_us);
  rknn_destroy_mem(ctx, a_mem);
  rknn_destroy_mem(ctx, b_mem);
  rknn_destroy_mem(ctx, c_mem);
  rknn_matmul_destroy(ctx);
  return mismatch_count == 0 ? 0 : 88;
}
