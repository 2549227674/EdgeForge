#define _POSIX_C_SOURCE 200809L
#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

namespace {

struct Options {
  int M = 0;
  int K = 0;
  int N = 0;
  int warmup = -1;
  int min_reps = -1;
  double min_timed_ms = -1.0;
  std::string type;
  std::string core_mask;
  std::string b_layout;
  std::string pattern = "constant";
  uint32_t seed = 1;
};

double now_seconds() {
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    std::perror("clock_gettime");
    std::exit(70);
  }
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

double quantile_sorted(const std::vector<double>& values, double q) {
  if (values.size() == 1) return values[0];
  const double index = q * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(index);
  const size_t upper = std::min(lower + 1, values.size() - 1);
  const double fraction = index - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --type int8|int4|fp16 --m M --k K --n N "
               "--core-mask core0|core01|core012 --b-layout normal|native "
               "--warmup N --min-reps N --min-timed-ms MS "
               "[--pattern constant|random] [--seed N]\n",
               argv0);
}

bool parse_int(const char* text, int* value) {
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || !end || *end != '\0' || parsed < 0 || parsed > INT32_MAX) return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool parse_double(const char* text, double* value) {
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(text, &end);
  if (errno != 0 || !end || *end != '\0' || !std::isfinite(parsed)) return false;
  *value = parsed;
  return true;
}

bool parse_args(int argc, char** argv, Options* opts) {
  for (int i = 1; i < argc; ++i) {
    if (i + 1 >= argc) return false;
    const std::string key = argv[i];
    const char* value = argv[++i];
    if (key == "--type") opts->type = value;
    else if (key == "--m" && parse_int(value, &opts->M)) {}
    else if (key == "--k" && parse_int(value, &opts->K)) {}
    else if (key == "--n" && parse_int(value, &opts->N)) {}
    else if (key == "--core-mask") opts->core_mask = value;
    else if (key == "--b-layout") opts->b_layout = value;
    else if (key == "--warmup" && parse_int(value, &opts->warmup)) {}
    else if (key == "--min-reps" && parse_int(value, &opts->min_reps)) {}
    else if (key == "--min-timed-ms" && parse_double(value, &opts->min_timed_ms)) {}
    else if (key == "--pattern") opts->pattern = value;
    else if (key == "--seed") {
      int seed = 0;
      if (!parse_int(value, &seed)) return false;
      opts->seed = static_cast<uint32_t>(seed);
    } else {
      return false;
    }
  }
  return opts->M > 0 && opts->K > 0 && opts->N > 0 && opts->warmup >= 0 &&
         opts->min_reps > 0 && opts->min_timed_ms >= 0.0 &&
         (opts->type == "int8" || opts->type == "int4" || opts->type == "fp16") &&
         (opts->core_mask == "core0" || opts->core_mask == "core01" ||
          opts->core_mask == "core012") &&
         (opts->b_layout == "normal" || opts->b_layout == "native") &&
         (opts->pattern == "constant" || opts->pattern == "random");
}

uint32_t lcg(uint32_t* state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

uint16_t half_bits_for_quarters(int value_quarters) {
  switch (value_quarters) {
    case -4: return 0xbc00u;  // -1.0
    case -2: return 0xb800u;  // -0.5
    case -1: return 0xb400u;  // -0.25
    case 0: return 0x0000u;
    case 1: return 0x3400u;   // 0.25
    case 2: return 0x3800u;   // 0.5
    case 4: return 0x3c00u;   // 1.0
    default: return 0x0000u;
  }
}

float quarters_to_float(int value_quarters) {
  return static_cast<float>(value_quarters) * 0.25f;
}

void pack_int4(const std::vector<int8_t>& logical, uint8_t* packed, size_t packed_size) {
  std::memset(packed, 0, packed_size);
  for (size_t index = 0; index < logical.size(); ++index) {
    const uint8_t nibble = static_cast<uint8_t>(logical[index]) & 0x0fu;
    if ((index & 1u) == 0) packed[index / 2] = nibble;
    else packed[index / 2] |= static_cast<uint8_t>(nibble << 4);
  }
}

const char* api_type_name(const Options& opts) {
  if (opts.type == "int8") return "RKNN_INT8_MM_INT8_TO_INT32";
  if (opts.type == "int4") return "RKNN_INT4_MM_INT4_TO_INT16";
  return "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32";
}

rknn_matmul_type api_type(const Options& opts) {
  if (opts.type == "int8") return RKNN_INT8_MM_INT8_TO_INT32;
  if (opts.type == "int4") return RKNN_INT4_MM_INT4_TO_INT16;
  return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
}

rknn_core_mask api_core_mask(const Options& opts) {
  if (opts.core_mask == "core0") return RKNN_NPU_CORE_0;
  if (opts.core_mask == "core01") return RKNN_NPU_CORE_0_1;
  return RKNN_NPU_CORE_0_1_2;
}

int core_count(const Options& opts) {
  if (opts.core_mask == "core0") return 1;
  if (opts.core_mask == "core01") return 2;
  return 3;
}

void emit_error(const Options& opts, const char* stage, int ret) {
  std::printf(
      "OPE2_RESULT_JSON={\"schema_version\":1,\"status\":\"ERROR\","
      "\"error_stage\":\"%s\",\"api_ret\":%d,\"type\":\"%s\","
      "\"api_type\":\"%s\",\"shape_mkn\":[%d,%d,%d],"
      "\"core_mask\":\"%s\",\"b_layout\":\"%s\","
      "\"pattern\":\"%s\",\"seed\":%u}\n",
      stage, ret, opts.type.c_str(), api_type_name(opts), opts.M, opts.K, opts.N,
      opts.core_mask.c_str(), opts.b_layout.c_str(), opts.pattern.c_str(), opts.seed);
  std::fflush(stdout);
}

std::vector<size_t> verification_indices(size_t elements) {
  if (elements <= 1048576u) {
    std::vector<size_t> result(elements);
    for (size_t i = 0; i < elements; ++i) result[i] = i;
    return result;
  }
  std::vector<size_t> result;
  result.reserve(4096);
  result.push_back(0);
  result.push_back(elements - 1);
  uint64_t state = 0x9e3779b97f4a7c15ull ^ static_cast<uint64_t>(elements);
  while (result.size() < 4096) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    result.push_back(static_cast<size_t>(state % elements));
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!parse_args(argc, argv, &opts)) {
    usage(argv[0]);
    return 2;
  }

  rknn_matmul_info info{};
  rknn_matmul_io_attr io_attr{};
  rknn_matmul_ctx ctx = 0;
  info.M = opts.M;
  info.K = opts.K;
  info.N = opts.N;
  info.type = api_type(opts);
  info.B_layout = opts.b_layout == "native" ? RKNN_MM_LAYOUT_NATIVE : RKNN_MM_LAYOUT_NORM;
  info.AC_layout = RKNN_MM_LAYOUT_NORM;
  info.B_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;
  info.AC_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;
  info.iommu_domain_id = 0;

  int ret = rknn_matmul_create(&ctx, &info, &io_attr);
  if (ret < 0) {
    emit_error(opts, "rknn_matmul_create", ret);
    return 80;
  }

  ret = rknn_matmul_set_core_mask(ctx, api_core_mask(opts));
  if (ret < 0) {
    emit_error(opts, "rknn_matmul_set_core_mask", ret);
    rknn_matmul_destroy(ctx);
    return 81;
  }
  const int core_mask_ret = ret;

  rknn_tensor_mem* a_mem = rknn_create_mem(ctx, io_attr.A.size);
  rknn_tensor_mem* b_mem = rknn_create_mem(ctx, io_attr.B.size);
  rknn_tensor_mem* c_mem = rknn_create_mem(ctx, io_attr.C.size);
  if (!a_mem || !b_mem || !c_mem) {
    emit_error(opts, "rknn_create_mem", -1);
    if (a_mem) rknn_destroy_mem(ctx, a_mem);
    if (b_mem) rknn_destroy_mem(ctx, b_mem);
    if (c_mem) rknn_destroy_mem(ctx, c_mem);
    rknn_matmul_destroy(ctx);
    return 82;
  }

  const size_t a_elements = static_cast<size_t>(opts.M) * opts.K;
  const size_t b_elements = static_cast<size_t>(opts.K) * opts.N;
  const size_t c_elements = static_cast<size_t>(opts.M) * opts.N;
  const size_t a_normal_bytes = opts.type == "int4" ? (a_elements + 1) / 2
                                  : opts.type == "fp16" ? a_elements * 2
                                                         : a_elements;
  const size_t b_normal_bytes = opts.type == "int4" ? (b_elements + 1) / 2
                                  : opts.type == "fp16" ? b_elements * 2
                                                         : b_elements;
  std::vector<uint8_t> b_normal(b_normal_bytes);
  std::vector<int8_t> a_logical;
  std::vector<int8_t> b_logical;
  std::vector<float> a_fp;
  std::vector<float> b_fp;
  uint32_t rng = opts.seed ? opts.seed : 1u;

  std::memset(a_mem->virt_addr, 0, io_attr.A.size);
  std::memset(b_mem->virt_addr, 0, io_attr.B.size);
  std::memset(c_mem->virt_addr, 0, io_attr.C.size);

  if (opts.type == "int8" || opts.type == "int4") {
    a_logical.resize(a_elements);
    b_logical.resize(b_elements);
    if (opts.pattern == "constant") {
      std::fill(a_logical.begin(), a_logical.end(), static_cast<int8_t>(1));
      std::fill(b_logical.begin(), b_logical.end(), static_cast<int8_t>(1));
    } else {
      for (auto& value : a_logical)
        value = static_cast<int8_t>(static_cast<int>(lcg(&rng) % 7u) - 3);
      for (auto& value : b_logical)
        value = static_cast<int8_t>(static_cast<int>(lcg(&rng) % 7u) - 3);
    }
    if (opts.type == "int4") {
      pack_int4(a_logical, static_cast<uint8_t*>(a_mem->virt_addr), io_attr.A.size);
      pack_int4(b_logical, b_normal.data(), b_normal.size());
    } else {
      std::memcpy(a_mem->virt_addr, a_logical.data(), a_normal_bytes);
      std::memcpy(b_normal.data(), b_logical.data(), b_normal_bytes);
    }
  } else {
    a_fp.resize(a_elements);
    b_fp.resize(b_elements);
    auto* a_bits = static_cast<uint16_t*>(a_mem->virt_addr);
    if (opts.pattern == "constant") {
      std::fill(a_fp.begin(), a_fp.end(), 1.0f);
      std::fill(b_fp.begin(), b_fp.end(), 0.25f);
      for (size_t i = 0; i < a_elements; ++i) a_bits[i] = half_bits_for_quarters(4);
      for (size_t i = 0; i < b_elements; ++i) {
        const uint16_t bits = half_bits_for_quarters(1);
        std::memcpy(b_normal.data() + i * sizeof(bits), &bits, sizeof(bits));
      }
    } else {
      const int values[] = {-4, -2, -1, 0, 1, 2, 4};
      for (size_t i = 0; i < a_elements; ++i) {
        const int q = values[lcg(&rng) % 7u];
        a_fp[i] = quarters_to_float(q);
        a_bits[i] = half_bits_for_quarters(q);
      }
      for (size_t i = 0; i < b_elements; ++i) {
        const int q = values[lcg(&rng) % 7u];
        b_fp[i] = quarters_to_float(q);
        const uint16_t bits = half_bits_for_quarters(q);
        std::memcpy(b_normal.data() + i * sizeof(bits), &bits, sizeof(bits));
      }
    }
  }

  const double prepare_start = now_seconds();
  if (opts.b_layout == "native") {
    void* conversion_input = opts.type == "int4"
                                 ? static_cast<void*>(b_logical.data())
                                 : static_cast<void*>(b_normal.data());
    ret = rknn_B_normal_layout_to_native_layout(
        conversion_input, b_mem->virt_addr, opts.K, opts.N, &info);
  } else {
    std::memcpy(b_mem->virt_addr, b_normal.data(), b_normal_bytes);
    ret = 0;
  }
  const double b_prepare_us = (now_seconds() - prepare_start) * 1e6;
  if (ret < 0) {
    emit_error(opts, "rknn_B_normal_layout_to_native_layout", ret);
    rknn_destroy_mem(ctx, a_mem);
    rknn_destroy_mem(ctx, b_mem);
    rknn_destroy_mem(ctx, c_mem);
    rknn_matmul_destroy(ctx);
    return 83;
  }

  if ((ret = rknn_matmul_set_io_mem(ctx, a_mem, &io_attr.A)) < 0 ||
      (ret = rknn_matmul_set_io_mem(ctx, b_mem, &io_attr.B)) < 0 ||
      (ret = rknn_matmul_set_io_mem(ctx, c_mem, &io_attr.C)) < 0) {
    emit_error(opts, "rknn_matmul_set_io_mem", ret);
    rknn_destroy_mem(ctx, a_mem);
    rknn_destroy_mem(ctx, b_mem);
    rknn_destroy_mem(ctx, c_mem);
    rknn_matmul_destroy(ctx);
    return 84;
  }

  for (int i = 0; i < opts.warmup; ++i) {
    if ((ret = rknn_matmul_run(ctx)) < 0) {
      emit_error(opts, "warmup_rknn_matmul_run", ret);
      rknn_destroy_mem(ctx, a_mem);
      rknn_destroy_mem(ctx, b_mem);
      rknn_destroy_mem(ctx, c_mem);
      rknn_matmul_destroy(ctx);
      return 85;
    }
  }

  std::vector<double> latencies_us;
  latencies_us.reserve(static_cast<size_t>(opts.min_reps) + 1024u);
  const double timed_start = now_seconds();
  do {
    const double start = now_seconds();
    ret = rknn_matmul_run(ctx);
    const double end = now_seconds();
    if (ret < 0) {
      emit_error(opts, "timed_rknn_matmul_run", ret);
      rknn_destroy_mem(ctx, a_mem);
      rknn_destroy_mem(ctx, b_mem);
      rknn_destroy_mem(ctx, c_mem);
      rknn_matmul_destroy(ctx);
      return 86;
    }
    latencies_us.push_back((end - start) * 1e6);
  } while (static_cast<int>(latencies_us.size()) < opts.min_reps ||
           (now_seconds() - timed_start) * 1e3 < opts.min_timed_ms);
  const double timed_elapsed_ms = (now_seconds() - timed_start) * 1e3;

  uint64_t mismatch_count = 0;
  double max_abs_diff = 0.0;
  double first_expected = 0.0;
  double first_actual = 0.0;
  const auto check_indices = verification_indices(c_elements);
  for (const size_t index : check_indices) {
    const int m = static_cast<int>(index / static_cast<size_t>(opts.N));
    const int n = static_cast<int>(index % static_cast<size_t>(opts.N));
    if (opts.type == "int8") {
      int32_t expected = 0;
      if (opts.pattern == "constant") expected = opts.K;
      else {
        for (int k = 0; k < opts.K; ++k)
          expected += static_cast<int32_t>(a_logical[static_cast<size_t>(m) * opts.K + k]) *
                      static_cast<int32_t>(b_logical[static_cast<size_t>(k) * opts.N + n]);
      }
      const int32_t actual = static_cast<int32_t*>(c_mem->virt_addr)[index];
      if (index == 0) {
        first_expected = expected;
        first_actual = actual;
      }
      const double diff = std::fabs(static_cast<double>(actual) - expected);
      if (diff != 0.0) ++mismatch_count;
      max_abs_diff = std::max(max_abs_diff, diff);
    } else if (opts.type == "int4") {
      int32_t expected32 = 0;
      if (opts.pattern == "constant") expected32 = opts.K;
      else {
        for (int k = 0; k < opts.K; ++k)
          expected32 += static_cast<int32_t>(a_logical[static_cast<size_t>(m) * opts.K + k]) *
                        static_cast<int32_t>(b_logical[static_cast<size_t>(k) * opts.N + n]);
      }
      const int16_t expected = static_cast<int16_t>(expected32);
      const int16_t actual = static_cast<int16_t*>(c_mem->virt_addr)[index];
      if (index == 0) {
        first_expected = expected;
        first_actual = actual;
      }
      const double diff = std::fabs(static_cast<double>(actual) - expected);
      if (diff != 0.0) ++mismatch_count;
      max_abs_diff = std::max(max_abs_diff, diff);
    } else {
      float expected = 0.0f;
      if (opts.pattern == "constant") expected = static_cast<float>(opts.K) * 0.25f;
      else {
        for (int k = 0; k < opts.K; ++k)
          expected += a_fp[static_cast<size_t>(m) * opts.K + k] *
                      b_fp[static_cast<size_t>(k) * opts.N + n];
      }
      const float actual = static_cast<float*>(c_mem->virt_addr)[index];
      if (index == 0) {
        first_expected = expected;
        first_actual = actual;
      }
      const double diff = std::fabs(static_cast<double>(actual) - expected);
      const double tolerance = 1e-3 + 1e-5 * std::fabs(static_cast<double>(expected));
      if (!std::isfinite(actual) || diff > tolerance) ++mismatch_count;
      max_abs_diff = std::max(max_abs_diff, diff);
    }
  }

  std::vector<double> sorted = latencies_us;
  std::sort(sorted.begin(), sorted.end());
  const double p25 = quantile_sorted(sorted, 0.25);
  const double p50 = quantile_sorted(sorted, 0.50);
  const double p75 = quantile_sorted(sorted, 0.75);
  const double iqr = p75 - p25;
  const uint64_t ops = 2ull * static_cast<uint64_t>(opts.M) * opts.K * opts.N;
  const uint64_t analytic_bytes = static_cast<uint64_t>(a_normal_bytes) +
                                  static_cast<uint64_t>(b_normal_bytes) +
                                  static_cast<uint64_t>(io_attr.C.size);
  const double measured_gops = static_cast<double>(ops) / (p50 * 1e3);
  const double measured_gbps = static_cast<double>(analytic_bytes) / (p50 * 1e3);
  const double tier_a_gbps = 23.1951846685;
  const bool pass = mismatch_count == 0;

  std::printf(
      "OPE2_RESULT_JSON={\"schema_version\":1,\"status\":\"%s\","
      "\"primitive\":\"rknn_matmul_run\",\"type\":\"%s\","
      "\"api_type\":\"%s\",\"shape_mkn\":[%d,%d,%d],"
      "\"core_mask\":\"%s\",\"core_count\":%d,"
      "\"core_mask_ret\":%d,\"b_layout\":\"%s\","
      "\"ac_layout\":\"normal\",\"pattern\":\"%s\","
      "\"b_conversion_input\":\"%s\","
      "\"seed\":%u,\"io_sizes_bytes\":{\"A\":%u,\"B\":%u,\"C\":%u},"
      "\"normal_analytic_bytes\":%llu,\"analytic_ops\":%llu,"
      "\"b_prepare_us\":%.6f,\"layout_conv_us\":%.6f,"
      "\"warmup_reps\":%d,\"timed_reps\":%zu,\"timed_elapsed_ms\":%.6f,"
      "\"run_only_us\":{\"p25\":%.6f,\"p50\":%.6f,\"p75\":%.6f,"
      "\"iqr\":%.6f,\"iqr_over_p50\":%.9f},"
      "\"throughput\":{\"measured_gops\":%.9f,\"measured_gbps\":%.9f,"
      "\"memory_relative_to_ope0\":%.9f},"
      "\"correctness\":{\"oracle\":\"%s\",\"checked_elements\":%zu,"
      "\"total_elements\":%zu,\"mismatch_count\":%llu,"
      "\"max_abs_diff\":%.9f,\"first_expected\":%.9f,"
      "\"first_actual\":%.9f,\"pass\":%s}}\n",
      pass ? "PASS" : "CORRECTNESS_FAIL", opts.type.c_str(), api_type_name(opts),
      opts.M, opts.K, opts.N, opts.core_mask.c_str(), core_count(opts), core_mask_ret,
      opts.b_layout.c_str(), opts.pattern.c_str(),
      opts.type == "int4" && opts.b_layout == "native"
          ? "unpacked_signed_int4_values"
          : "normal_storage_representation",
      opts.seed, io_attr.A.size,
      io_attr.B.size, io_attr.C.size,
      static_cast<unsigned long long>(analytic_bytes),
      static_cast<unsigned long long>(ops), b_prepare_us,
      opts.b_layout == "native" ? b_prepare_us : 0.0, opts.warmup,
      latencies_us.size(), timed_elapsed_ms, p25, p50, p75, iqr,
      p50 > 0.0 ? iqr / p50 : 0.0, measured_gops, measured_gbps,
      measured_gbps / tier_a_gbps,
      opts.pattern == "random" ? "deterministic_CPU_reference" : "constant_input_analytic",
      check_indices.size(), c_elements,
      static_cast<unsigned long long>(mismatch_count), max_abs_diff,
      first_expected, first_actual,
      pass ? "true" : "false");
  std::fflush(stdout);

  rknn_destroy_mem(ctx, a_mem);
  rknn_destroy_mem(ctx, b_mem);
  rknn_destroy_mem(ctx, c_mem);
  rknn_matmul_destroy(ctx);
  return pass ? 0 : 88;
}
