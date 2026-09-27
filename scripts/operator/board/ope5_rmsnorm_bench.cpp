#include <arm_neon.h>
#include <sys/prctl.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fp16 = float16_t;

constexpr float kEpsilon = 1.0e-6F;
constexpr double kOpe0CeilingGbps = 23.1951846685;
constexpr double kOpe4TpotMs = 446.1262611013072;
constexpr std::size_t kPopulationBytes = 3'840'000;
constexpr int kPopulationCalls = 302;
constexpr int kFormalRingFrames = 16;
constexpr int kFormalWarmup = 50;
constexpr int kFormalRepetitions = 200;
constexpr double kFormalPopulationMinSeconds = 3.0;

struct CaseDef {
  const char* id;
  int m;
  int n;
  bool with_scale;
  int calls_per_forward;
  std::size_t analytic_bytes_per_call;
};

constexpr std::array<CaseDef, 8> kCases{{
    {"decoder_hidden_scaled", 1, 2560, true, 211, 15360},
    {"q_norm_sliding_scaled", 8, 256, true, 35, 8704},
    {"q_norm_full_scaled", 8, 512, true, 7, 17408},
    {"k_norm_sliding_scaled", 2, 256, true, 20, 2560},
    {"v_norm_sliding_scale_free", 2, 256, false, 20, 2048},
    {"k_norm_full_scaled", 2, 512, true, 4, 5120},
    {"v_norm_full_scale_free", 2, 512, false, 4, 4096},
    {"ple_projection_norm_scaled", 42, 256, true, 1, 43520},
}};

using Kernel = void (*)(const fp16*, const fp16*, fp16*, int, int);

extern "C" __attribute__((noinline, optimize("O3,no-tree-vectorize"))) void
scalar_ref(const fp16* __restrict input, const fp16* __restrict scale,
           fp16* __restrict output, int m, int n) {
  for (int row = 0; row < m; ++row) {
    const fp16* row_input = input + static_cast<std::size_t>(row) * n;
    fp16* row_output = output + static_cast<std::size_t>(row) * n;
    float sum_squares = 0.0F;
    for (int column = 0; column < n; ++column) {
      const float value = static_cast<float>(row_input[column]);
      sum_squares += value * value;
    }
    const float reciprocal_rms =
        1.0F / std::sqrt(sum_squares / static_cast<float>(n) + kEpsilon);
    if (scale != nullptr) {
      for (int column = 0; column < n; ++column) {
        const float value = static_cast<float>(row_input[column]);
        const float weight = static_cast<float>(scale[column]);
        row_output[column] = static_cast<fp16>(value * reciprocal_rms * weight);
      }
    } else {
      for (int column = 0; column < n; ++column) {
        const float value = static_cast<float>(row_input[column]);
        row_output[column] = static_cast<fp16>(value * reciprocal_rms);
      }
    }
  }
}

extern "C" __attribute__((noinline)) void autovec_o3(
    const fp16* __restrict input, const fp16* __restrict scale,
    fp16* __restrict output, int m, int n) {
  for (int row = 0; row < m; ++row) {
    const fp16* row_input = input + static_cast<std::size_t>(row) * n;
    fp16* row_output = output + static_cast<std::size_t>(row) * n;
    float sum_squares = 0.0F;
    for (int column = 0; column < n; ++column) {
      const float value = static_cast<float>(row_input[column]);
      sum_squares += value * value;
    }
    const float reciprocal_rms =
        1.0F / std::sqrt(sum_squares / static_cast<float>(n) + kEpsilon);
    if (scale != nullptr) {
      for (int column = 0; column < n; ++column) {
        const float value = static_cast<float>(row_input[column]);
        const float weight = static_cast<float>(scale[column]);
        row_output[column] = static_cast<fp16>(value * reciprocal_rms * weight);
      }
    } else {
      for (int column = 0; column < n; ++column) {
        const float value = static_cast<float>(row_input[column]);
        row_output[column] = static_cast<fp16>(value * reciprocal_rms);
      }
    }
  }
}

extern "C" __attribute__((noinline)) void neon_fp16(
    const fp16* __restrict input, const fp16* __restrict scale,
    fp16* __restrict output, int m, int n) {
  for (int row = 0; row < m; ++row) {
    const fp16* row_input = input + static_cast<std::size_t>(row) * n;
    fp16* row_output = output + static_cast<std::size_t>(row) * n;
    float32x4_t sum0 = vdupq_n_f32(0.0F);
    float32x4_t sum1 = vdupq_n_f32(0.0F);
    for (int column = 0; column < n; column += 8) {
      const float16x8_t packed = vld1q_f16(row_input + column);
      const float32x4_t low = vcvt_f32_f16(vget_low_f16(packed));
      const float32x4_t high = vcvt_f32_f16(vget_high_f16(packed));
      sum0 = vfmaq_f32(sum0, low, low);
      sum1 = vfmaq_f32(sum1, high, high);
    }
    const float sum_squares = vaddvq_f32(vaddq_f32(sum0, sum1));
    const float reciprocal_rms =
        1.0F / std::sqrt(sum_squares / static_cast<float>(n) + kEpsilon);
    const float32x4_t reciprocal = vdupq_n_f32(reciprocal_rms);
    for (int column = 0; column < n; column += 8) {
      const float16x8_t packed = vld1q_f16(row_input + column);
      float32x4_t low = vmulq_f32(
          vcvt_f32_f16(vget_low_f16(packed)), reciprocal);
      float32x4_t high = vmulq_f32(
          vcvt_f32_f16(vget_high_f16(packed)), reciprocal);
      if (scale != nullptr) {
        const float16x8_t packed_scale = vld1q_f16(scale + column);
        low = vmulq_f32(
            low, vcvt_f32_f16(vget_low_f16(packed_scale)));
        high = vmulq_f32(
            high, vcvt_f32_f16(vget_high_f16(packed_scale)));
      }
      vst1q_f16(row_output + column,
                vcombine_f16(vcvt_f16_f32(low), vcvt_f16_f32(high)));
    }
  }
}

struct Variant {
  const char* name;
  Kernel kernel;
};

constexpr std::array<Variant, 3> kVariants{{
    {"scalar_ref", scalar_ref},
    {"autovec_o3", autovec_o3},
    {"neon_fp16", neon_fp16},
}};

constexpr std::array<std::array<int, 3>, 6> kPermutations{{
    {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
    {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}},
}};

volatile std::uint64_t g_output_sink = 0;

std::uint64_t monotonic_raw_ns() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
    throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC_RAW) failed");
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

void consume_output(const fp16* output, std::size_t elements) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, output + (elements - 1), sizeof(bits));
  g_output_sink = g_output_sink ^
                  (static_cast<std::uint64_t>(bits) + 0x9e3779b97f4a7c15ULL);
}

struct Invocation {
  const CaseDef* definition = nullptr;
  fp16* input = nullptr;
  fp16* output = nullptr;
  fp16* scale = nullptr;
};

struct Frame {
  std::vector<fp16> storage;
  std::vector<Invocation> invocations;
  std::array<std::size_t, kCases.size()> first_invocation{};
};

std::size_t case_elements(const CaseDef& definition) {
  return static_cast<std::size_t>(definition.m) * definition.n;
}

Frame build_frame(std::uint32_t seed) {
  std::size_t total_elements = 0;
  for (const CaseDef& definition : kCases) {
    const std::size_t elements = case_elements(definition);
    total_elements += static_cast<std::size_t>(definition.calls_per_forward) *
                      (2 * elements +
                       (definition.with_scale ? definition.n : 0));
  }
  if (total_elements * sizeof(fp16) != kPopulationBytes) {
    throw std::runtime_error("population storage does not equal 3,840,000 bytes");
  }

  Frame frame;
  frame.storage.resize(total_elements);
  frame.invocations.reserve(kPopulationCalls);
  std::mt19937 generator(seed);
  std::normal_distribution<float> input_distribution(0.0F, 1.0F);
  std::uniform_real_distribution<float> scale_distribution(-0.5F, 96.0F);
  std::size_t cursor = 0;
  for (std::size_t case_index = 0; case_index < kCases.size(); ++case_index) {
    const CaseDef& definition = kCases[case_index];
    frame.first_invocation[case_index] = frame.invocations.size();
    const std::size_t elements = case_elements(definition);
    for (int call = 0; call < definition.calls_per_forward; ++call) {
      Invocation invocation;
      invocation.definition = &definition;
      invocation.input = frame.storage.data() + cursor;
      cursor += elements;
      invocation.output = frame.storage.data() + cursor;
      cursor += elements;
      if (definition.with_scale) {
        invocation.scale = frame.storage.data() + cursor;
        cursor += definition.n;
      }
      for (std::size_t index = 0; index < elements; ++index) {
        invocation.input[index] =
            static_cast<fp16>(input_distribution(generator));
        invocation.output[index] = static_cast<fp16>(0.0F);
      }
      if (definition.with_scale) {
        for (int column = 0; column < definition.n; ++column) {
          invocation.scale[column] =
              static_cast<fp16>(scale_distribution(generator));
        }
      }
      frame.invocations.push_back(invocation);
    }
  }
  if (cursor != total_elements ||
      static_cast<int>(frame.invocations.size()) != kPopulationCalls) {
    throw std::runtime_error("population frame construction mismatch");
  }
  return frame;
}

std::vector<Frame> build_ring(int frame_count, std::uint32_t seed) {
  std::vector<Frame> frames;
  frames.reserve(static_cast<std::size_t>(frame_count));
  for (int frame = 0; frame < frame_count; ++frame) {
    frames.push_back(build_frame(seed + static_cast<std::uint32_t>(frame) *
                                          0x9e3779b9U));
  }
  return frames;
}

void run_invocation(Kernel kernel, Invocation& invocation) {
  const CaseDef& definition = *invocation.definition;
  kernel(invocation.input, invocation.scale, invocation.output,
         definition.m, definition.n);
  consume_output(invocation.output, case_elements(definition));
}

void run_population(Kernel kernel, Frame& frame) {
  for (Invocation& invocation : frame.invocations) {
    const CaseDef& definition = *invocation.definition;
    kernel(invocation.input, invocation.scale, invocation.output,
           definition.m, definition.n);
  }
  Invocation& final_invocation = frame.invocations.back();
  consume_output(final_invocation.output,
                 case_elements(*final_invocation.definition));
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    throw std::runtime_error("percentile of empty sample set");
  }
  std::sort(values.begin(), values.end());
  const double position = fraction * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double weight = position - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

struct Summary {
  double p25_us;
  double p50_us;
  double p75_us;
  double iqr_us;
  double iqr_over_p50;
};

Summary summarize(const std::vector<double>& samples) {
  const double p25 = percentile(samples, 0.25);
  const double p50 = percentile(samples, 0.50);
  const double p75 = percentile(samples, 0.75);
  const double iqr = p75 - p25;
  return {p25, p50, p75, iqr, iqr / p50};
}

enum class Distribution { kUnitNormal, kWideNormal, kNearZero };

const char* distribution_name(Distribution distribution) {
  switch (distribution) {
    case Distribution::kUnitNormal:
      return "unit_normal";
    case Distribution::kWideNormal:
      return "wide_normal_std32";
    case Distribution::kNearZero:
      return "near_zero_uniform_plus_minus_0.001";
  }
  throw std::runtime_error("unreachable distribution");
}

void fill_correctness_input(std::vector<fp16>& input,
                            std::vector<fp16>& scale,
                            Distribution distribution,
                            std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::normal_distribution<float> unit_normal(0.0F, 1.0F);
  std::normal_distribution<float> wide_normal(0.0F, 32.0F);
  std::uniform_real_distribution<float> near_zero(-0.001F, 0.001F);
  std::uniform_real_distribution<float> scale_distribution(-0.5F, 96.0F);
  for (fp16& value : input) {
    float generated = 0.0F;
    if (distribution == Distribution::kUnitNormal) {
      generated = unit_normal(generator);
    } else if (distribution == Distribution::kWideNormal) {
      generated = wide_normal(generator);
    } else {
      generated = near_zero(generator);
    }
    value = static_cast<fp16>(generated);
  }
  for (fp16& value : scale) {
    value = static_cast<fp16>(scale_distribution(generator));
  }
}

bool bytes_equal(const std::vector<fp16>& left,
                 const std::vector<fp16>& right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(),
                     left.size() * sizeof(fp16)) == 0;
}

int run_correctness() {
  constexpr std::array<Distribution, 3> distributions{{
      Distribution::kUnitNormal,
      Distribution::kWideNormal,
      Distribution::kNearZero,
  }};
  constexpr std::array<std::uint32_t, 3> seeds{{20260921U, 20260922U,
                                                20260923U}};
  constexpr float absolute_tolerance = 0.015625F;
  constexpr float relative_tolerance = 0.01F;
  int comparisons = 0;
  int failures = 0;
  std::cout
      << "case_id,distribution,seed,variant,elements,all_finite,"
         "input_unchanged,scale_unchanged,scale_pointer_contract,"
         "max_abs_error,max_rel_error,mismatch_count,pass\n";
  for (std::size_t case_index = 0; case_index < kCases.size(); ++case_index) {
    const CaseDef& definition = kCases[case_index];
    const std::size_t elements = case_elements(definition);
    for (Distribution distribution : distributions) {
      for (std::uint32_t seed : seeds) {
        std::vector<fp16> input(elements);
        std::vector<fp16> scale(definition.with_scale
                                    ? static_cast<std::size_t>(definition.n)
                                    : 0);
        fill_correctness_input(
            input, scale, distribution,
            seed ^ (static_cast<std::uint32_t>(case_index) * 0x85ebca6bU));
        const std::vector<fp16> input_before = input;
        const std::vector<fp16> scale_before = scale;
        const fp16* scale_pointer = definition.with_scale ? scale.data() : nullptr;
        std::vector<fp16> reference(elements);
        scalar_ref(input.data(), scale_pointer, reference.data(),
                   definition.m, definition.n);

        for (std::size_t variant_index = 1; variant_index < kVariants.size();
             ++variant_index) {
          const Variant& variant = kVariants[variant_index];
          std::vector<fp16> candidate(elements);
          variant.kernel(input.data(), scale_pointer, candidate.data(),
                         definition.m, definition.n);
          bool all_finite = true;
          float max_absolute_error = 0.0F;
          float max_relative_error = 0.0F;
          std::size_t mismatch_count = 0;
          for (std::size_t index = 0; index < elements; ++index) {
            const float expected = static_cast<float>(reference[index]);
            const float observed = static_cast<float>(candidate[index]);
            all_finite = all_finite && std::isfinite(expected) &&
                         std::isfinite(observed);
            const float absolute_error = std::fabs(observed - expected);
            const float relative_error =
                absolute_error / std::max(std::fabs(expected), 1.0e-12F);
            max_absolute_error = std::max(max_absolute_error, absolute_error);
            max_relative_error = std::max(max_relative_error, relative_error);
            if (!(absolute_error <=
                  absolute_tolerance + relative_tolerance * std::fabs(expected))) {
              ++mismatch_count;
            }
          }
          const bool input_unchanged = bytes_equal(input, input_before);
          const bool scale_unchanged = bytes_equal(scale, scale_before);
          const bool scale_pointer_contract =
              definition.with_scale ? scale_pointer != nullptr
                                    : scale_pointer == nullptr;
          const bool pass = all_finite && input_unchanged && scale_unchanged &&
                            scale_pointer_contract && mismatch_count == 0;
          std::cout << definition.id << ',' << distribution_name(distribution)
                    << ',' << seed << ',' << variant.name << ',' << elements
                    << ',' << (all_finite ? 1 : 0) << ','
                    << (input_unchanged ? 1 : 0) << ','
                    << (scale_unchanged ? 1 : 0) << ','
                    << (scale_pointer_contract ? 1 : 0) << ','
                    << std::setprecision(12) << max_absolute_error << ','
                    << max_relative_error << ',' << mismatch_count << ','
                    << (pass ? 1 : 0) << '\n';
          ++comparisons;
          if (!pass) {
            ++failures;
          }
        }
      }
    }
  }
  std::cerr << "OPE5_CORRECTNESS_SUMMARY comparisons=" << comparisons
            << " failures=" << failures << '\n';
  return comparisons == 144 && failures == 0 ? 0 : 3;
}

double time_invocation(Kernel kernel, Invocation& invocation) {
  const std::uint64_t begin = monotonic_raw_ns();
  run_invocation(kernel, invocation);
  const std::uint64_t end = monotonic_raw_ns();
  return static_cast<double>(end - begin) / 1000.0;
}

double time_population(Kernel kernel, Frame& frame) {
  const std::uint64_t begin = monotonic_raw_ns();
  run_population(kernel, frame);
  const std::uint64_t end = monotonic_raw_ns();
  return static_cast<double>(end - begin) / 1000.0;
}

void print_timing_header() {
  std::cout
      << "scope,case_id,variant,m,n,with_scale,calls_per_forward,"
         "repetitions,analytic_bytes,p25_us,p50_us,p75_us,iqr_us,"
         "iqr_over_p50,measured_gbps,memory_relative_to_ope0,"
         "fraction_of_ope4_tpot,permutations_used,ring_frames\n";
}

void print_timing_row(const char* scope, const char* case_id,
                      const Variant& variant, int m, int n, bool with_scale,
                      int calls_per_forward, std::size_t analytic_bytes,
                      const std::vector<double>& samples, int ring_frames,
                      bool report_ope4_fraction) {
  const Summary summary = summarize(samples);
  const double measured_gbps =
      static_cast<double>(analytic_bytes) / (summary.p50_us * 1000.0);
  const double memory_relative = measured_gbps / kOpe0CeilingGbps;
  const double ope4_fraction =
      report_ope4_fraction ? (summary.p50_us / 1000.0) / kOpe4TpotMs
                           : -1.0;
  std::cout << scope << ',' << case_id << ',' << variant.name << ',' << m
            << ',' << n << ',' << (with_scale ? 1 : 0) << ','
            << calls_per_forward << ',' << samples.size() << ','
            << analytic_bytes << ',' << std::setprecision(12)
            << summary.p25_us << ',' << summary.p50_us << ','
            << summary.p75_us << ',' << summary.iqr_us << ','
            << summary.iqr_over_p50 << ',' << measured_gbps << ','
            << memory_relative << ',' << ope4_fraction << ",6,"
            << ring_frames << '\n';
}

int run_measure(int warmup, int repetitions, double population_min_seconds,
                int ring_frames, std::uint32_t seed) {
  if (warmup != kFormalWarmup || repetitions != kFormalRepetitions ||
      population_min_seconds != kFormalPopulationMinSeconds ||
      ring_frames != kFormalRingFrames) {
    throw std::runtime_error("formal measurement constants were changed");
  }
  std::vector<Frame> frames = build_ring(ring_frames, seed);
  std::size_t frame_cursor = 0;
  print_timing_header();

  for (std::size_t case_index = 0; case_index < kCases.size(); ++case_index) {
    const CaseDef& definition = kCases[case_index];
    for (int round = 0; round < warmup; ++round) {
      const auto& order = kPermutations[static_cast<std::size_t>(round) %
                                        kPermutations.size()];
      for (int variant_index : order) {
        Frame& frame = frames[frame_cursor++ % frames.size()];
        Invocation& invocation =
            frame.invocations[frame.first_invocation[case_index]];
        run_invocation(kVariants[variant_index].kernel, invocation);
      }
    }
    std::array<std::vector<double>, kVariants.size()> samples;
    for (int round = 0; round < repetitions; ++round) {
      const auto& order = kPermutations[static_cast<std::size_t>(round) %
                                        kPermutations.size()];
      for (int variant_index : order) {
        Frame& frame = frames[frame_cursor++ % frames.size()];
        Invocation& invocation =
            frame.invocations[frame.first_invocation[case_index]];
        samples[variant_index].push_back(
            time_invocation(kVariants[variant_index].kernel, invocation));
      }
    }
    for (std::size_t variant_index = 0; variant_index < kVariants.size();
         ++variant_index) {
      print_timing_row("per_case", definition.id, kVariants[variant_index],
                       definition.m, definition.n, definition.with_scale,
                       definition.calls_per_forward,
                       definition.analytic_bytes_per_call,
                       samples[variant_index], ring_frames, false);
    }
  }

  for (int round = 0; round < warmup; ++round) {
    const auto& order = kPermutations[static_cast<std::size_t>(round) %
                                      kPermutations.size()];
    for (int variant_index : order) {
      Frame& frame = frames[frame_cursor++ % frames.size()];
      run_population(kVariants[variant_index].kernel, frame);
    }
  }

  std::array<std::vector<double>, kVariants.size()> population_samples;
  std::array<double, kVariants.size()> accumulated_us{{0.0, 0.0, 0.0}};
  int round = 0;
  while (true) {
    bool complete = true;
    for (std::size_t variant_index = 0; variant_index < kVariants.size();
         ++variant_index) {
      complete = complete &&
                 population_samples[variant_index].size() >=
                     static_cast<std::size_t>(repetitions) &&
                 accumulated_us[variant_index] >=
                     population_min_seconds * 1.0e6;
    }
    if (complete) {
      break;
    }
    if (round >= 200000) {
      throw std::runtime_error("population timing exceeded safety bound");
    }
    const auto& order = kPermutations[static_cast<std::size_t>(round) %
                                      kPermutations.size()];
    for (int variant_index : order) {
      Frame& frame = frames[frame_cursor++ % frames.size()];
      const double elapsed_us =
          time_population(kVariants[variant_index].kernel, frame);
      population_samples[variant_index].push_back(elapsed_us);
      accumulated_us[variant_index] += elapsed_us;
    }
    ++round;
  }

  bool stable = true;
  for (std::size_t variant_index = 0; variant_index < kVariants.size();
       ++variant_index) {
    print_timing_row("population", "full_decode_population",
                     kVariants[variant_index], 0, 0, false,
                     kPopulationCalls, kPopulationBytes,
                     population_samples[variant_index], ring_frames, true);
    const Summary summary = summarize(population_samples[variant_index]);
    stable = stable && summary.iqr_over_p50 <= 0.30;
  }
  std::cerr << "OPE5_MEASURE_SUMMARY rounds=" << round
            << " stable=" << (stable ? 1 : 0)
            << " sink=" << g_output_sink << '\n';
  return stable ? 0 : 4;
}

const Variant& find_variant(const std::string& name) {
  for (const Variant& variant : kVariants) {
    if (name == variant.name) {
      return variant;
    }
  }
  throw std::runtime_error("unknown variant: " + name);
}

int run_perf(const std::string& variant_name, int warmup, int repetitions,
             double population_min_seconds, int ring_frames,
             std::uint32_t seed) {
  if (warmup != kFormalWarmup || repetitions != kFormalRepetitions ||
      population_min_seconds != kFormalPopulationMinSeconds ||
      ring_frames != kFormalRingFrames) {
    throw std::runtime_error("formal perf constants were changed");
  }
  const Variant& variant = find_variant(variant_name);
  if (prctl(PR_TASK_PERF_EVENTS_DISABLE, 0, 0, 0, 0) != 0) {
    throw std::runtime_error("PR_TASK_PERF_EVENTS_DISABLE failed before setup");
  }
  std::vector<Frame> frames = build_ring(ring_frames, seed);
  std::size_t frame_cursor = 0;
  for (int round = 0; round < warmup; ++round) {
    run_population(variant.kernel, frames[frame_cursor++ % frames.size()]);
  }
  if (prctl(PR_TASK_PERF_EVENTS_ENABLE, 0, 0, 0, 0) != 0) {
    throw std::runtime_error("PR_TASK_PERF_EVENTS_ENABLE failed");
  }
  const std::uint64_t begin = monotonic_raw_ns();
  int completed = 0;
  do {
    run_population(variant.kernel, frames[frame_cursor++ % frames.size()]);
    ++completed;
  } while (completed < repetitions ||
           static_cast<double>(monotonic_raw_ns() - begin) <
               population_min_seconds * 1.0e9);
  const std::uint64_t end = monotonic_raw_ns();
  if (prctl(PR_TASK_PERF_EVENTS_DISABLE, 0, 0, 0, 0) != 0) {
    throw std::runtime_error("PR_TASK_PERF_EVENTS_DISABLE failed after workload");
  }
  const double elapsed_ms = static_cast<double>(end - begin) / 1.0e6;
  std::cout << "scope,variant,population_sequences,population_calls,"
               "analytic_bytes_per_sequence,elapsed_ms,ring_frames,sink\n";
  std::cout << "perf_population," << variant.name << ',' << completed << ','
            << static_cast<long long>(completed) * kPopulationCalls << ','
            << kPopulationBytes << ',' << std::setprecision(12) << elapsed_ms
            << ',' << ring_frames << ',' << g_output_sink << '\n';
  return 0;
}

struct Options {
  std::string mode;
  std::string variant;
  int warmup = kFormalWarmup;
  int repetitions = kFormalRepetitions;
  double population_min_seconds = kFormalPopulationMinSeconds;
  int ring_frames = kFormalRingFrames;
  std::uint32_t seed = 20260921U;
};

int parse_int(const char* value, const char* name) {
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  }
  return static_cast<int>(parsed);
}

double parse_double(const char* value, const char* name) {
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(value, &end);
  if (errno != 0 || end == value || *end != '\0' || !(parsed > 0.0)) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  }
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value for " + argument);
    }
    const char* value = argv[++index];
    if (argument == "--mode") {
      options.mode = value;
    } else if (argument == "--variant") {
      options.variant = value;
    } else if (argument == "--warmup") {
      options.warmup = parse_int(value, "warmup");
    } else if (argument == "--reps") {
      options.repetitions = parse_int(value, "reps");
    } else if (argument == "--population-min-seconds") {
      options.population_min_seconds =
          parse_double(value, "population-min-seconds");
    } else if (argument == "--ring-frames") {
      options.ring_frames = parse_int(value, "ring-frames");
    } else if (argument == "--seed") {
      const int parsed_seed = parse_int(value, "seed");
      options.seed = static_cast<std::uint32_t>(parsed_seed);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.mode != "correctness" && options.mode != "measure" &&
      options.mode != "perf") {
    throw std::runtime_error("--mode must be correctness, measure, or perf");
  }
  if (options.mode == "perf" && options.variant.empty()) {
    throw std::runtime_error("--variant is required for perf mode");
  }
  return options;
}

void validate_frozen_population() {
  std::size_t bytes = 0;
  int calls = 0;
  for (const CaseDef& definition : kCases) {
    if (definition.n % 8 != 0) {
      throw std::runtime_error("NEON formal width is not divisible by eight");
    }
    const std::size_t expected_bytes =
        4ULL * static_cast<std::size_t>(definition.m) * definition.n +
        (definition.with_scale ? 2ULL * definition.n : 0ULL);
    if (definition.analytic_bytes_per_call != expected_bytes) {
      throw std::runtime_error("case analytic byte formula mismatch");
    }
    calls += definition.calls_per_forward;
    bytes += definition.analytic_bytes_per_call *
             static_cast<std::size_t>(definition.calls_per_forward);
  }
  if (calls != kPopulationCalls || bytes != kPopulationBytes) {
    throw std::runtime_error("frozen population identity mismatch");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    validate_frozen_population();
    const Options options = parse_options(argc, argv);
    if (options.mode == "correctness") {
      return run_correctness();
    }
    if (options.mode == "measure") {
      return run_measure(options.warmup, options.repetitions,
                         options.population_min_seconds,
                         options.ring_frames, options.seed);
    }
    return run_perf(options.variant, options.warmup, options.repetitions,
                    options.population_min_seconds, options.ring_frames,
                    options.seed);
  } catch (const std::exception& error) {
    std::cerr << "OPE5_FATAL: " << error.what() << '\n';
    return 2;
  }
}
