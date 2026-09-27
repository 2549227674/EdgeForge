#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#if !defined(__ARM_FEATURE_DOTPROD)
#error "K-E1 requires ARMv8.2 dot-product support"
#endif
#elif !defined(KE1_HOST_TEST)
#error "Formal K-E1 builds are AArch64-only; define KE1_HOST_TEST only for local semantic tests"
#endif

enum {
  KE1_K = 10240,
  KE1_N = 2560,
};

typedef enum {
  VARIANT_SCALAR = 0,
  VARIANT_AUTOVEC = 1,
  VARIANT_NEON_SDOT = 2,
  VARIANT_COUNT = 3,
} variant_t;

typedef enum {
  DIST_SMALL = 0,
  DIST_FULL = 1,
  DIST_ALTERNATING = 2,
} distribution_t;

typedef struct {
  int8_t *x;
  int8_t *weights;
  int32_t *outputs[VARIANT_COUNT];
  int threads;
  int first_cpu;
} bench_t;

static _Thread_local int pinned_cpu = -2;

static void die(const char *message) {
  fprintf(stderr, "%s\n", message);
  exit(70);
}

static double now_seconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    perror("clock_gettime");
    exit(70);
  }
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void ensure_thread_pinned(int first_cpu) {
  if (first_cpu < 0) return;
  int target = first_cpu + omp_get_thread_num();
  if (pinned_cpu == target) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(target, &set);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (rc != 0) {
    errno = rc;
    perror("pthread_setaffinity_np");
    exit(71);
  }
  pinned_cpu = target;
}

static uint64_t splitmix64(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

static int8_t draw_value(uint64_t *state, distribution_t distribution,
                         size_t index) {
  uint64_t r = splitmix64(state);
  if (distribution == DIST_SMALL) {
    return (int8_t)((int)(r & 15u) - 8);
  }
  if (distribution == DIST_FULL) {
    return (int8_t)((int)(r % 255u) - 127);
  }
  int magnitude = (index & 1u) ? 113 : 127;
  return (int8_t)((r & 1u) ? magnitude : -magnitude);
}

static const char *distribution_name(distribution_t distribution) {
  switch (distribution) {
    case DIST_SMALL:
      return "small_symmetric_int8";
    case DIST_FULL:
      return "full_symmetric_int8";
    case DIST_ALTERNATING:
      return "alternating_extrema_with_random_sign";
  }
  return "invalid";
}

static const char *variant_name(variant_t variant) {
  switch (variant) {
    case VARIANT_SCALAR:
      return "scalar";
    case VARIANT_AUTOVEC:
      return "autovec";
    case VARIANT_NEON_SDOT:
      return "neon_sdot";
    default:
      return "invalid";
  }
}

static variant_t parse_variant(const char *text) {
  for (int i = 0; i < VARIANT_COUNT; ++i) {
    if (strcmp(text, variant_name((variant_t)i)) == 0) return (variant_t)i;
  }
  fprintf(stderr, "invalid variant: %s\n", text);
  exit(2);
}

static uint64_t checksum_output(const int32_t *output) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (size_t i = 0; i < KE1_N; ++i) {
    uint32_t word = (uint32_t)output[i];
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= (word >> (byte * 8)) & 0xffu;
      hash *= UINT64_C(1099511628211);
    }
  }
  return hash;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline, optimize("no-tree-vectorize,no-tree-slp-vectorize")))
#else
__attribute__((noinline))
#endif
static int32_t dot_scalar(const int8_t *restrict x,
                          const int8_t *restrict weights) {
  int32_t sum = 0;
  for (size_t k = 0; k < KE1_K; ++k) {
    sum += (int32_t)x[k] * (int32_t)weights[k];
  }
  return sum;
}

__attribute__((noinline)) static int32_t
dot_autovec(const int8_t *restrict x, const int8_t *restrict weights) {
  int32_t sum = 0;
  for (size_t k = 0; k < KE1_K; ++k) {
    sum += (int32_t)x[k] * (int32_t)weights[k];
  }
  return sum;
}

__attribute__((noinline)) static int32_t
dot_neon_sdot(const int8_t *restrict x, const int8_t *restrict weights) {
#if defined(__aarch64__)
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0);
  int32x4_t acc3 = vdupq_n_s32(0);
  size_t k = 0;
  for (; k + 63 < KE1_K; k += 64) {
    const int8x16_t x0 = vld1q_s8(x + k);
    const int8x16_t x1 = vld1q_s8(x + k + 16);
    const int8x16_t x2 = vld1q_s8(x + k + 32);
    const int8x16_t x3 = vld1q_s8(x + k + 48);
    acc0 = vdotq_s32(acc0, x0, vld1q_s8(weights + k));
    acc1 = vdotq_s32(acc1, x1, vld1q_s8(weights + k + 16));
    acc2 = vdotq_s32(acc2, x2, vld1q_s8(weights + k + 32));
    acc3 = vdotq_s32(acc3, x3, vld1q_s8(weights + k + 48));
  }
  int32x4_t combined = vaddq_s32(vaddq_s32(acc0, acc1),
                                 vaddq_s32(acc2, acc3));
  int32_t sum = vaddvq_s32(combined);
  for (; k < KE1_K; ++k) {
    sum += (int32_t)x[k] * (int32_t)weights[k];
  }
  return sum;
#else
  return dot_scalar(x, weights);
#endif
}

static void run_kernel(bench_t *bench, variant_t variant) {
  int32_t *restrict output = bench->outputs[variant];
#pragma omp parallel for schedule(static)
  for (int n = 0; n < KE1_N; ++n) {
    ensure_thread_pinned(bench->first_cpu);
    const int8_t *row = bench->weights + (size_t)n * KE1_K;
    if (variant == VARIANT_SCALAR) {
      output[n] = dot_scalar(bench->x, row);
    } else if (variant == VARIANT_AUTOVEC) {
      output[n] = dot_autovec(bench->x, row);
    } else {
      output[n] = dot_neon_sdot(bench->x, row);
    }
  }
}

static void initialize_inputs(bench_t *bench, uint64_t seed,
                              distribution_t distribution) {
  uint64_t state = seed;
  for (size_t k = 0; k < KE1_K; ++k) {
    bench->x[k] = draw_value(&state, distribution, k);
  }
  const size_t weight_elements = (size_t)KE1_N * KE1_K;
  for (size_t i = 0; i < weight_elements; ++i) {
    bench->weights[i] = draw_value(&state, distribution, i);
  }
  for (int variant = 0; variant < VARIANT_COUNT; ++variant) {
    memset(bench->outputs[variant], 0, KE1_N * sizeof(int32_t));
  }
}

static void allocate_bench(bench_t *bench) {
  const size_t weight_bytes = (size_t)KE1_N * KE1_K;
  if (posix_memalign((void **)&bench->x, 64, KE1_K) != 0 ||
      posix_memalign((void **)&bench->weights, 64, weight_bytes) != 0) {
    die("failed to allocate K-E1 inputs");
  }
  for (int i = 0; i < VARIANT_COUNT; ++i) {
    if (posix_memalign((void **)&bench->outputs[i], 64,
                       KE1_N * sizeof(int32_t)) != 0) {
      die("failed to allocate K-E1 output");
    }
  }
}

static void free_bench(bench_t *bench) {
  free(bench->x);
  free(bench->weights);
  for (int i = 0; i < VARIANT_COUNT; ++i) free(bench->outputs[i]);
}

static int run_correctness(bench_t *bench) {
  const uint64_t seeds[] = {UINT64_C(20260919), UINT64_C(20260920),
                            UINT64_C(20260921)};
  const distribution_t distributions[] = {DIST_SMALL, DIST_FULL,
                                           DIST_ALTERNATING};
  printf("record_type,seed,distribution,variant,checked,mismatches,"
         "max_abs_diff,reference_checksum,actual_checksum\n");
  int any_failure = 0;
  for (size_t trial = 0; trial < 3; ++trial) {
    initialize_inputs(bench, seeds[trial], distributions[trial]);
    run_kernel(bench, VARIANT_SCALAR);
    uint64_t reference_checksum = checksum_output(bench->outputs[VARIANT_SCALAR]);
    for (variant_t variant = VARIANT_AUTOVEC; variant <= VARIANT_NEON_SDOT;
         variant = (variant_t)(variant + 1)) {
      run_kernel(bench, variant);
      size_t mismatches = 0;
      uint64_t max_abs_diff = 0;
      for (size_t n = 0; n < KE1_N; ++n) {
        int64_t diff = (int64_t)bench->outputs[variant][n] -
                       (int64_t)bench->outputs[VARIANT_SCALAR][n];
        uint64_t abs_diff = (uint64_t)(diff < 0 ? -diff : diff);
        if (abs_diff != 0) ++mismatches;
        if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
      }
      uint64_t actual_checksum = checksum_output(bench->outputs[variant]);
      printf("correctness,%" PRIu64 ",%s,%s,%d,%zu,%" PRIu64
             ",%" PRIu64 ",%" PRIu64 "\n",
             seeds[trial], distribution_name(distributions[trial]),
             variant_name(variant), KE1_N, mismatches, max_abs_diff,
             reference_checksum, actual_checksum);
      if (mismatches != 0 || reference_checksum != actual_checksum) {
        any_failure = 1;
      }
    }
  }
  return any_failure ? 80 : 0;
}

static const variant_t permutations[6][VARIANT_COUNT] = {
    {VARIANT_SCALAR, VARIANT_AUTOVEC, VARIANT_NEON_SDOT},
    {VARIANT_AUTOVEC, VARIANT_NEON_SDOT, VARIANT_SCALAR},
    {VARIANT_NEON_SDOT, VARIANT_SCALAR, VARIANT_AUTOVEC},
    {VARIANT_SCALAR, VARIANT_NEON_SDOT, VARIANT_AUTOVEC},
    {VARIANT_AUTOVEC, VARIANT_SCALAR, VARIANT_NEON_SDOT},
    {VARIANT_NEON_SDOT, VARIANT_AUTOVEC, VARIANT_SCALAR},
};

static int run_measurement(bench_t *bench, int warmup, int reps,
                           uint64_t seed) {
  initialize_inputs(bench, seed, DIST_FULL);
  for (int round = 0; round < warmup; ++round) {
    const variant_t *order = permutations[round % 6];
    for (int slot = 0; slot < VARIANT_COUNT; ++slot) {
      run_kernel(bench, order[slot]);
    }
  }

  printf("record_type,round,order_slot,variant,elapsed_us,checksum\n");
  uint64_t observed = 0;
  for (int round = 0; round < reps; ++round) {
    const variant_t *order = permutations[round % 6];
    for (int slot = 0; slot < VARIANT_COUNT; ++slot) {
      variant_t variant = order[slot];
      double start = now_seconds();
      run_kernel(bench, variant);
      double elapsed_us = (now_seconds() - start) * 1e6;
      uint64_t checksum = checksum_output(bench->outputs[variant]);
      observed ^= checksum + (uint64_t)round * UINT64_C(0x9e3779b97f4a7c15) +
                  (uint64_t)slot;
      printf("timing,%d,%d,%s,%.6f,%" PRIu64 "\n", round, slot,
             variant_name(variant), elapsed_us, checksum);
    }
  }
  fprintf(stderr, "KE1_MEASUREMENT_OBSERVED=%" PRIu64 "\n", observed);
  return 0;
}

static int run_perf(bench_t *bench, variant_t variant, int warmup, int reps,
                    uint64_t seed) {
  initialize_inputs(bench, seed, DIST_FULL);
  for (int i = 0; i < warmup; ++i) run_kernel(bench, variant);
  uint64_t observed = 0;
  double start = now_seconds();
  for (int i = 0; i < reps; ++i) {
    run_kernel(bench, variant);
    observed ^= (uint64_t)bench->outputs[variant][(size_t)i % KE1_N];
  }
  double elapsed_us = (now_seconds() - start) * 1e6;
  uint64_t checksum = checksum_output(bench->outputs[variant]);
  printf("record_type,variant,warmup,reps,elapsed_us,checksum,observed\n");
  printf("perf,%s,%d,%d,%.6f,%" PRIu64 ",%" PRIu64 "\n",
         variant_name(variant), warmup, reps, elapsed_us, checksum, observed);
  return 0;
}

static int parse_int(const char *name, const char *value) {
  char *end = NULL;
  errno = 0;
  long parsed = strtol(value, &end, 10);
  if (errno != 0 || !end || *end != '\0' || parsed < -1 ||
      parsed > INT32_MAX) {
    fprintf(stderr, "invalid %s: %s\n", name, value);
    exit(2);
  }
  return (int)parsed;
}

static uint64_t parse_u64(const char *name, const char *value) {
  char *end = NULL;
  errno = 0;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (errno != 0 || !end || *end != '\0') {
    fprintf(stderr, "invalid %s: %s\n", name, value);
    exit(2);
  }
  return (uint64_t)parsed;
}

static void usage(const char *argv0) {
  fprintf(stderr,
          "usage: %s --mode correctness|measure|perf --threads N "
          "--first-cpu N [--warmup N] [--reps N] [--seed N] "
          "[--variant scalar|autovec|neon_sdot]\n",
          argv0);
}

int main(int argc, char **argv) {
  const char *mode = NULL;
  int threads = 4;
  int first_cpu = -1;
  int warmup = 0;
  int reps = 0;
  uint64_t seed = UINT64_C(20260920);
  variant_t perf_variant = VARIANT_NEON_SDOT;
  int variant_was_set = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      mode = argv[++i];
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      threads = parse_int("threads", argv[++i]);
    } else if (strcmp(argv[i], "--first-cpu") == 0 && i + 1 < argc) {
      first_cpu = parse_int("first-cpu", argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = parse_int("warmup", argv[++i]);
    } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
      reps = parse_int("reps", argv[++i]);
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      seed = parse_u64("seed", argv[++i]);
    } else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
      perf_variant = parse_variant(argv[++i]);
      variant_was_set = 1;
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  if (!mode || threads <= 0 || threads > KE1_N || first_cpu < -1 ||
      warmup < 0 || reps < 0) {
    usage(argv[0]);
    return 2;
  }
  if (strcmp(mode, "measure") == 0 && (warmup != 50 || reps != 200)) {
    fprintf(stderr, "formal measure mode requires exactly warmup=50 reps=200\n");
    return 2;
  }
  if (strcmp(mode, "perf") == 0 && (!variant_was_set || reps <= 0)) {
    fprintf(stderr, "perf mode requires --variant and positive --reps\n");
    return 2;
  }

  omp_set_dynamic(0);
  omp_set_num_threads(threads);
  bench_t bench = {.threads = threads, .first_cpu = first_cpu};
  allocate_bench(&bench);

#if defined(__aarch64__)
  const char *candidate_backend = "aarch64_neon_sdot";
#else
  const char *candidate_backend = "host_test_scalar_fallback_not_evidence";
#endif
  fprintf(stderr,
          "KE1_METADATA mode=%s M=1 K=%d N=%d threads=%d first_cpu=%d "
          "seed=%" PRIu64 " candidate_backend=%s analytic_bytes=26234880 "
          "analytic_ops=52428800\n",
          mode, KE1_K, KE1_N, threads, first_cpu, seed, candidate_backend);

  int rc = 0;
  if (strcmp(mode, "correctness") == 0) {
    rc = run_correctness(&bench);
  } else if (strcmp(mode, "measure") == 0) {
    rc = run_measurement(&bench, warmup, reps, seed);
  } else if (strcmp(mode, "perf") == 0) {
    rc = run_perf(&bench, perf_variant, warmup, reps, seed);
  } else {
    usage(argv[0]);
    rc = 2;
  }

  free_bench(&bench);
  return rc;
}
