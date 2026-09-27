#define _GNU_SOURCE
#include <arm_neon.h>
#include <errno.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
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

static void pin_current_thread(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (rc != 0) {
    errno = rc;
    perror("pthread_setaffinity_np");
    exit(71);
  }
}

static void initialize_arrays(double *a, double *b, double *c, size_t n,
                              int first_cpu) {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    pin_current_thread(first_cpu + tid);
    size_t start = n * (size_t)tid / (size_t)omp_get_num_threads();
    size_t end = n * (size_t)(tid + 1) / (size_t)omp_get_num_threads();
    for (size_t i = start; i < end; ++i) {
      a[i] = 1.0 + (double)(i & 7u) * 0.125;
      b[i] = 0.5 + (double)(i & 3u) * 0.25;
      c[i] = 0.0;
    }
  }
}

static void triad_neon(const double *restrict a, const double *restrict b,
                       double *restrict c, size_t n, double scalar) {
  const float64x2_t vs = vdupq_n_f64(scalar);
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int threads = omp_get_num_threads();
    size_t start = n * (size_t)tid / (size_t)threads;
    size_t end = n * (size_t)(tid + 1) / (size_t)threads;
    size_t i = start;
    for (; i + 1 < end; i += 2) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      vst1q_f64(c + i, vfmaq_f64(va, vb, vs));
    }
    for (; i < end; ++i) {
      c[i] = a[i] + scalar * b[i];
    }
  }
}

static int parse_sizes(char *text, size_t **sizes_out, size_t *count_out) {
  size_t capacity = 8;
  size_t count = 0;
  size_t *sizes = calloc(capacity, sizeof(*sizes));
  if (!sizes) return -1;
  char *save = NULL;
  for (char *token = strtok_r(text, ",", &save); token;
       token = strtok_r(NULL, ",", &save)) {
    char *end = NULL;
    unsigned long value = strtoul(token, &end, 10);
    if (!end || *end != '\0' || value == 0) {
      free(sizes);
      return -1;
    }
    if (count == capacity) {
      capacity *= 2;
      size_t *grown = realloc(sizes, capacity * sizeof(*sizes));
      if (!grown) {
        free(sizes);
        return -1;
      }
      sizes = grown;
    }
    sizes[count++] = (size_t)value;
  }
  if (count == 0) {
    free(sizes);
    return -1;
  }
  *sizes_out = sizes;
  *count_out = count;
  return 0;
}

static void usage(const char *argv0) {
  fprintf(stderr,
          "usage: %s --sizes-mib LIST --warmup N --reps N --threads N "
          "--first-cpu N\n",
          argv0);
}

int main(int argc, char **argv) {
  char *sizes_text = NULL;
  int warmup = -1;
  int reps = -1;
  int threads = -1;
  int first_cpu = -1;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--sizes-mib") == 0 && i + 1 < argc) {
      sizes_text = strdup(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
      reps = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      threads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--first-cpu") == 0 && i + 1 < argc) {
      first_cpu = atoi(argv[++i]);
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (!sizes_text || warmup < 0 || reps <= 0 || threads <= 0 ||
      first_cpu < 0) {
    usage(argv[0]);
    return 2;
  }

  size_t *sizes_mib = NULL;
  size_t size_count = 0;
  if (parse_sizes(sizes_text, &sizes_mib, &size_count) != 0) {
    fprintf(stderr, "invalid --sizes-mib list\n");
    return 2;
  }

  size_t max_mib = 0;
  for (size_t i = 0; i < size_count; ++i) {
    if (sizes_mib[i] > max_mib) max_mib = sizes_mib[i];
  }
  if (max_mib > SIZE_MAX / (1024u * 1024u)) {
    fprintf(stderr, "array size overflow\n");
    return 2;
  }
  size_t max_bytes = max_mib * 1024u * 1024u;
  size_t max_n = max_bytes / sizeof(double);

  double *a = NULL;
  double *b = NULL;
  double *c = NULL;
  if (posix_memalign((void **)&a, 64, max_bytes) != 0 ||
      posix_memalign((void **)&b, 64, max_bytes) != 0 ||
      posix_memalign((void **)&c, 64, max_bytes) != 0) {
    fprintf(stderr, "failed to allocate three %zu MiB arrays\n", max_mib);
    free(a);
    free(b);
    free(c);
    return 73;
  }

  omp_set_dynamic(0);
  omp_set_num_threads(threads);
  initialize_arrays(a, b, c, max_n, first_cpu);

  fprintf(stderr,
          "OPE0_TRIAD_METADATA threads=%d first_cpu=%d max_array_mib=%zu "
          "warmup=%d reps=%d analytic_bytes_per_element=24\n",
          threads, first_cpu, max_mib, warmup, reps);
  printf("size_mib,rep,seconds,analytic_bytes,gbps,checksum\n");
  volatile double observed = 0.0;
  const double scalar = 3.0;

  for (size_t s = 0; s < size_count; ++s) {
    size_t bytes = sizes_mib[s] * 1024u * 1024u;
    size_t n = bytes / sizeof(double);
    for (int rep = 0; rep < warmup; ++rep) {
      triad_neon(a, b, c, n, scalar);
      observed += c[((size_t)rep * 104729u) % n];
    }
    for (int rep = 0; rep < reps; ++rep) {
      double start = now_seconds();
      triad_neon(a, b, c, n, scalar);
      double elapsed = now_seconds() - start;
      observed += c[((size_t)rep * 104729u + sizes_mib[s]) % n];
      double analytic_bytes = 24.0 * (double)n;
      double gbps = analytic_bytes / elapsed / 1e9;
      printf("%zu,%d,%.9f,%.0f,%.9f,%.9f\n", sizes_mib[s], rep,
             elapsed, analytic_bytes, gbps, (double)observed);
      fflush(stdout);
    }
  }

  fprintf(stderr, "OPE0_TRIAD_OBSERVED %.9f\n", (double)observed);
  free(a);
  free(b);
  free(c);
  free(sizes_mib);
  free(sizes_text);
  return 0;
}
