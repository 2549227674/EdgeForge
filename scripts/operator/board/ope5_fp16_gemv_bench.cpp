#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <sys/prctl.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using fp16 = float16_t;
using RowKernel = void (*)(const fp16*, const fp16*, float*, int, int);

constexpr int kM = 1;
constexpr int kK = 10240;
constexpr int kN = 2560;
constexpr int kThreads = 4;
constexpr int kRowsPerThread = kN / kThreads;
constexpr int kFirstCpu = 4;
constexpr int kFormalWarmup = 50;
constexpr int kFormalRepetitions = 200;
constexpr double kFormalMinSeconds = 3.0;
constexpr std::size_t kActivationBytes = 20'480;
constexpr std::size_t kWeightBytes = 52'428'800;
constexpr std::size_t kOutputBytes = 10'240;
constexpr std::size_t kAnalyticBytes = 52'459'520;
constexpr std::uint64_t kAnalyticOps = 52'428'800;
constexpr int kCallsPerForward = 42;
constexpr double kOpe0CeilingGbps = 23.1951846685;
constexpr double kOpe4TpotMs = 446.1262611013072;

static_assert(kM == 1);
static_assert(kN % kThreads == 0);
static_assert(kK % 32 == 0);
static_assert(sizeof(fp16) == 2);
static_assert(kActivationBytes == static_cast<std::size_t>(kK) * sizeof(fp16));
static_assert(kWeightBytes ==
              static_cast<std::size_t>(kN) * kK * sizeof(fp16));
static_assert(kOutputBytes == static_cast<std::size_t>(kN) * sizeof(float));

extern "C" __attribute__((noinline, optimize("O3,no-tree-vectorize"))) void
scalar_fp32_rows(const fp16* __restrict activation,
                 const fp16* __restrict weights,
                 float* __restrict output, int row_begin, int row_end) {
  for (int row = row_begin; row < row_end; ++row) {
    const fp16* weight_row =
        weights + static_cast<std::size_t>(row) * kK;
    float sum = 0.0F;
    for (int column = 0; column < kK; ++column) {
      sum += static_cast<float>(activation[column]) *
             static_cast<float>(weight_row[column]);
    }
    output[row] = sum;
  }
}

extern "C" __attribute__((noinline)) void autovec_o3_rows(
    const fp16* __restrict activation, const fp16* __restrict weights,
    float* __restrict output, int row_begin, int row_end) {
  for (int row = row_begin; row < row_end; ++row) {
    const fp16* weight_row =
        weights + static_cast<std::size_t>(row) * kK;
    float sum = 0.0F;
    for (int column = 0; column < kK; ++column) {
      sum += static_cast<float>(activation[column]) *
             static_cast<float>(weight_row[column]);
    }
    output[row] = sum;
  }
}

extern "C" __attribute__((noinline)) void neon_fp16_f32acc_rows(
    const fp16* __restrict activation, const fp16* __restrict weights,
    float* __restrict output, int row_begin, int row_end) {
  for (int row = row_begin; row < row_end; ++row) {
    const fp16* weight_row =
        weights + static_cast<std::size_t>(row) * kK;
    float32x4_t acc0 = vdupq_n_f32(0.0F);
    float32x4_t acc1 = vdupq_n_f32(0.0F);
    float32x4_t acc2 = vdupq_n_f32(0.0F);
    float32x4_t acc3 = vdupq_n_f32(0.0F);
    float32x4_t acc4 = vdupq_n_f32(0.0F);
    float32x4_t acc5 = vdupq_n_f32(0.0F);
    float32x4_t acc6 = vdupq_n_f32(0.0F);
    float32x4_t acc7 = vdupq_n_f32(0.0F);
    for (int column = 0; column < kK; column += 32) {
      const float16x8_t x0 = vld1q_f16(activation + column);
      const float16x8_t x1 = vld1q_f16(activation + column + 8);
      const float16x8_t x2 = vld1q_f16(activation + column + 16);
      const float16x8_t x3 = vld1q_f16(activation + column + 24);
      const float16x8_t w0 = vld1q_f16(weight_row + column);
      const float16x8_t w1 = vld1q_f16(weight_row + column + 8);
      const float16x8_t w2 = vld1q_f16(weight_row + column + 16);
      const float16x8_t w3 = vld1q_f16(weight_row + column + 24);
      acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(x0)),
                       vcvt_f32_f16(vget_low_f16(w0)));
      acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(x0)),
                       vcvt_f32_f16(vget_high_f16(w0)));
      acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(x1)),
                       vcvt_f32_f16(vget_low_f16(w1)));
      acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(x1)),
                       vcvt_f32_f16(vget_high_f16(w1)));
      acc4 = vfmaq_f32(acc4, vcvt_f32_f16(vget_low_f16(x2)),
                       vcvt_f32_f16(vget_low_f16(w2)));
      acc5 = vfmaq_f32(acc5, vcvt_f32_f16(vget_high_f16(x2)),
                       vcvt_f32_f16(vget_high_f16(w2)));
      acc6 = vfmaq_f32(acc6, vcvt_f32_f16(vget_low_f16(x3)),
                       vcvt_f32_f16(vget_low_f16(w3)));
      acc7 = vfmaq_f32(acc7, vcvt_f32_f16(vget_high_f16(x3)),
                       vcvt_f32_f16(vget_high_f16(w3)));
    }
    const float32x4_t sum01 = vaddq_f32(acc0, acc1);
    const float32x4_t sum23 = vaddq_f32(acc2, acc3);
    const float32x4_t sum45 = vaddq_f32(acc4, acc5);
    const float32x4_t sum67 = vaddq_f32(acc6, acc7);
    output[row] = vaddvq_f32(
        vaddq_f32(vaddq_f32(sum01, sum23), vaddq_f32(sum45, sum67)));
  }
}

struct Variant {
  const char* name;
  RowKernel kernel;
};

constexpr std::array<Variant, 3> kVariants{{
    {"scalar_fp32", scalar_fp32_rows},
    {"autovec_o3", autovec_o3_rows},
    {"neon_fp16_f32acc", neon_fp16_f32acc_rows},
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

bool pin_current_thread(int cpu) {
  cpu_set_t requested;
  CPU_ZERO(&requested);
  CPU_SET(cpu, &requested);
  if (pthread_setaffinity_np(pthread_self(), sizeof(requested), &requested) !=
      0) {
    return false;
  }
  cpu_set_t observed;
  CPU_ZERO(&observed);
  if (pthread_getaffinity_np(pthread_self(), sizeof(observed), &observed) !=
      0) {
    return false;
  }
  return CPU_COUNT(&observed) == 1 && CPU_ISSET(cpu, &observed);
}

class WorkerPool {
 public:
  WorkerPool() {
    if (!pin_current_thread(kFirstCpu)) {
      throw std::runtime_error("failed to pin main worker to CPU4");
    }
    affinity_pass_[0].store(true, std::memory_order_relaxed);
    if (pthread_barrier_init(&start_barrier_, nullptr, kThreads) != 0 ||
        pthread_barrier_init(&finish_barrier_, nullptr, kThreads) != 0) {
      throw std::runtime_error("pthread_barrier_init failed");
    }
    for (int index = 0; index < kThreads - 1; ++index) {
      args_[index] = {this, index + 1};
      const int rc =
          pthread_create(&threads_[index], nullptr, &WorkerPool::worker_entry,
                         &args_[index]);
      if (rc != 0) {
        throw std::runtime_error("pthread_create failed");
      }
    }
    while (ready_.load(std::memory_order_acquire) != kThreads - 1) {
      std::this_thread::yield();
    }
    if (initialization_error_.load(std::memory_order_acquire) != 0 ||
        !affinity_pass()) {
      stop_workers();
      throw std::runtime_error("worker affinity initialization failed");
    }
    active_ = true;
  }

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  ~WorkerPool() {
    if (active_) {
      stop_workers();
    }
    pthread_barrier_destroy(&finish_barrier_);
    pthread_barrier_destroy(&start_barrier_);
  }

  void run(RowKernel kernel, const fp16* activation, const fp16* weights,
           float* output) {
    kernel_ = kernel;
    activation_ = activation;
    weights_ = weights;
    output_ = output;
    job_ = Job::kKernel;
    wait_barrier(start_barrier_);
    kernel(activation, weights, output, 0, kRowsPerThread);
    ++invocations_[0];
    wait_barrier(finish_barrier_);
  }

  void perf_control(bool enable) {
    control_error_.store(0, std::memory_order_relaxed);
    job_ = enable ? Job::kEnablePerf : Job::kDisablePerf;
    wait_barrier(start_barrier_);
    const int operation =
        enable ? PR_TASK_PERF_EVENTS_ENABLE : PR_TASK_PERF_EVENTS_DISABLE;
    if (prctl(operation, 0, 0, 0, 0) != 0) {
      control_error_.store(errno == 0 ? 1 : errno, std::memory_order_relaxed);
    }
    wait_barrier(finish_barrier_);
    if (control_error_.load(std::memory_order_relaxed) != 0) {
      throw std::runtime_error("per-thread perf prctl control failed");
    }
  }

  bool affinity_pass() const {
    for (const std::atomic<bool>& value : affinity_pass_) {
      if (!value.load(std::memory_order_relaxed)) {
        return false;
      }
    }
    return true;
  }

  bool partition_coverage_pass() const {
    int cursor = 0;
    for (int worker = 0; worker < kThreads; ++worker) {
      const int begin = worker * kRowsPerThread;
      const int end = begin + kRowsPerThread;
      if (begin != cursor || end <= begin) {
        return false;
      }
      cursor = end;
    }
    return cursor == kN;
  }

 private:
  enum class Job { kKernel, kEnablePerf, kDisablePerf, kStop };

  struct WorkerArg {
    WorkerPool* pool;
    int worker_id;
  };

  static void wait_barrier(pthread_barrier_t& barrier) {
    const int rc = pthread_barrier_wait(&barrier);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
      throw std::runtime_error("pthread_barrier_wait failed");
    }
  }

  static void* worker_entry(void* opaque) {
    WorkerArg* argument = static_cast<WorkerArg*>(opaque);
    argument->pool->worker_loop(argument->worker_id);
    return nullptr;
  }

  void worker_loop(int worker_id) {
    const bool pinned = pin_current_thread(kFirstCpu + worker_id);
    affinity_pass_[worker_id].store(pinned, std::memory_order_relaxed);
    if (!pinned) {
      initialization_error_.store(1, std::memory_order_release);
    }
    ready_.fetch_add(1, std::memory_order_release);
    while (true) {
      wait_barrier(start_barrier_);
      if (job_ == Job::kStop) {
        break;
      }
      if (job_ == Job::kKernel) {
        const int begin = worker_id * kRowsPerThread;
        kernel_(activation_, weights_, output_, begin, begin + kRowsPerThread);
        ++invocations_[worker_id];
      } else {
        const int operation = job_ == Job::kEnablePerf
                                  ? PR_TASK_PERF_EVENTS_ENABLE
                                  : PR_TASK_PERF_EVENTS_DISABLE;
        if (prctl(operation, 0, 0, 0, 0) != 0) {
          control_error_.store(errno == 0 ? 1 : errno,
                               std::memory_order_relaxed);
        }
      }
      wait_barrier(finish_barrier_);
    }
  }

  void stop_workers() {
    job_ = Job::kStop;
    wait_barrier(start_barrier_);
    for (pthread_t thread : threads_) {
      pthread_join(thread, nullptr);
    }
    active_ = false;
  }

  pthread_barrier_t start_barrier_{};
  pthread_barrier_t finish_barrier_{};
  std::array<pthread_t, kThreads - 1> threads_{};
  std::array<WorkerArg, kThreads - 1> args_{};
  std::array<std::atomic<bool>, kThreads> affinity_pass_{};
  std::array<std::uint64_t, kThreads> invocations_{};
  std::atomic<int> ready_{0};
  std::atomic<int> initialization_error_{0};
  std::atomic<int> control_error_{0};
  Job job_{Job::kStop};
  RowKernel kernel_{nullptr};
  const fp16* activation_{nullptr};
  const fp16* weights_{nullptr};
  float* output_{nullptr};
  bool active_{false};
};

void consume_output(const std::vector<float>& output) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &output.back(), sizeof(bits));
  g_output_sink = g_output_sink ^
                  (static_cast<std::uint64_t>(bits) + 0x9e3779b97f4a7c15ULL);
}

enum class Distribution {
  kQuarterGridExact,
  kUnitNormalClipped3,
  kWideNormalStd4Clipped8,
};

const char* distribution_name(Distribution distribution) {
  switch (distribution) {
    case Distribution::kQuarterGridExact:
      return "quarter_grid_exact";
    case Distribution::kUnitNormalClipped3:
      return "unit_normal_clipped_3";
    case Distribution::kWideNormalStd4Clipped8:
      return "wide_normal_std4_clipped_8";
  }
  throw std::runtime_error("unreachable distribution");
}

struct Problem {
  std::vector<fp16> activation;
  std::vector<fp16> weights;
};

Problem build_problem(Distribution distribution, std::uint32_t seed) {
  Problem problem;
  problem.activation.resize(kK);
  problem.weights.resize(static_cast<std::size_t>(kN) * kK);
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> quarter_grid(-8, 8);
  std::normal_distribution<float> unit_normal(0.0F, 1.0F);
  std::normal_distribution<float> wide_normal(0.0F, 4.0F);
  auto generate = [&]() -> fp16 {
    float value = 0.0F;
    if (distribution == Distribution::kQuarterGridExact) {
      value = static_cast<float>(quarter_grid(generator)) * 0.25F;
    } else if (distribution == Distribution::kUnitNormalClipped3) {
      value = std::clamp(unit_normal(generator), -3.0F, 3.0F);
    } else {
      value = std::clamp(wide_normal(generator), -8.0F, 8.0F);
    }
    return static_cast<fp16>(value);
  };
  for (fp16& value : problem.activation) {
    value = generate();
  }
  for (fp16& value : problem.weights) {
    value = generate();
  }
  return problem;
}

std::vector<double> fp64_oracle(const Problem& problem) {
  std::vector<double> output(kN, 0.0);
  for (int row = 0; row < kN; ++row) {
    const fp16* weight_row =
        problem.weights.data() + static_cast<std::size_t>(row) * kK;
    double sum = 0.0;
    for (int column = 0; column < kK; ++column) {
      sum += static_cast<double>(problem.activation[column]) *
             static_cast<double>(weight_row[column]);
    }
    output[row] = sum;
  }
  return output;
}

bool bytes_equal(const std::vector<fp16>& left,
                 const std::vector<fp16>& right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(),
                     left.size() * sizeof(fp16)) == 0;
}

void run_variant(WorkerPool& pool, const Variant& variant,
                 const Problem& problem, std::vector<float>& output) {
  pool.run(variant.kernel, problem.activation.data(), problem.weights.data(),
           output.data());
  consume_output(output);
}

int run_correctness() {
  constexpr std::array<Distribution, 3> distributions{{
      Distribution::kQuarterGridExact,
      Distribution::kUnitNormalClipped3,
      Distribution::kWideNormalStd4Clipped8,
  }};
  constexpr std::array<std::uint32_t, 3> seeds{{
      20260924U, 20260925U, 20260926U,
  }};
  WorkerPool pool;
  std::uint64_t element_comparisons = 0;
  int aggregate_rows = 0;
  int failures = 0;
  std::cout
      << "distribution,seed,variant,elements,atol,rtol,all_finite,"
         "input_unchanged,weight_unchanged,output_shape_pass,"
         "partition_coverage_pass,affinity_pass,max_abs_error,max_rel_error,"
         "mismatch_count,pass\n";
  for (Distribution distribution : distributions) {
    const double absolute_tolerance =
        distribution == Distribution::kQuarterGridExact ? 0.0 : 0.05;
    const double relative_tolerance =
        distribution == Distribution::kQuarterGridExact ? 0.0 : 0.001;
    for (std::uint32_t seed : seeds) {
      const Problem problem = build_problem(distribution, seed);
      const std::vector<fp16> activation_before = problem.activation;
      const std::vector<fp16> weights_before = problem.weights;
      const std::vector<double> reference = fp64_oracle(problem);
      std::array<std::vector<float>, kVariants.size()> outputs;
      for (std::vector<float>& output : outputs) {
        output.resize(kN);
      }
      for (std::size_t variant_index = 0; variant_index < kVariants.size();
           ++variant_index) {
        const Variant& variant = kVariants[variant_index];
        run_variant(pool, variant, problem, outputs[variant_index]);
        bool all_finite = true;
        double max_absolute_error = 0.0;
        double max_relative_error = 0.0;
        std::size_t mismatches = 0;
        for (int row = 0; row < kN; ++row) {
          const double expected = reference[row];
          const double observed = outputs[variant_index][row];
          all_finite = all_finite && std::isfinite(expected) &&
                       std::isfinite(observed);
          const double absolute_error = std::fabs(observed - expected);
          const double relative_error =
              absolute_error / std::max(std::fabs(expected), 1.0e-300);
          max_absolute_error = std::max(max_absolute_error, absolute_error);
          max_relative_error = std::max(max_relative_error, relative_error);
          if (!(absolute_error <= absolute_tolerance +
                                      relative_tolerance *
                                          std::fabs(expected))) {
            ++mismatches;
          }
        }
        const bool input_unchanged =
            bytes_equal(problem.activation, activation_before);
        const bool weight_unchanged =
            bytes_equal(problem.weights, weights_before);
        const bool output_shape_pass = outputs[variant_index].size() == kN;
        const bool partition_pass = pool.partition_coverage_pass();
        const bool affinity_pass = pool.affinity_pass();
        const bool pass = all_finite && input_unchanged && weight_unchanged &&
                          output_shape_pass && partition_pass && affinity_pass &&
                          mismatches == 0;
        std::cout << distribution_name(distribution) << ',' << seed << ','
                  << variant.name << ',' << kN << ',' << std::setprecision(12)
                  << absolute_tolerance << ',' << relative_tolerance << ','
                  << (all_finite ? 1 : 0) << ','
                  << (input_unchanged ? 1 : 0) << ','
                  << (weight_unchanged ? 1 : 0) << ','
                  << (output_shape_pass ? 1 : 0) << ','
                  << (partition_pass ? 1 : 0) << ','
                  << (affinity_pass ? 1 : 0) << ',' << max_absolute_error << ','
                  << max_relative_error << ',' << mismatches << ','
                  << (pass ? 1 : 0) << '\n';
        element_comparisons += kN;
        ++aggregate_rows;
        if (!pass) {
          ++failures;
        }
      }
    }
  }
  std::cerr << "OPE5_FP16_GEMV_CORRECTNESS rows=" << aggregate_rows
            << " element_comparisons=" << element_comparisons
            << " failures=" << failures << " sink=" << g_output_sink << '\n';
  return aggregate_rows == 27 && element_comparisons == 69'120 &&
                 failures == 0
             ? 0
             : 3;
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

double time_variant(WorkerPool& pool, const Variant& variant,
                    const Problem& problem, std::vector<float>& output) {
  const std::uint64_t begin = monotonic_raw_ns();
  run_variant(pool, variant, problem, output);
  const std::uint64_t end = monotonic_raw_ns();
  return static_cast<double>(end - begin) / 1000.0;
}

void validate_formal_arguments(int warmup, int repetitions,
                               double minimum_seconds) {
  if (warmup != kFormalWarmup || repetitions != kFormalRepetitions ||
      minimum_seconds != kFormalMinSeconds) {
    throw std::runtime_error("formal measurement constants were changed");
  }
}

int run_measure(int warmup, int repetitions, double minimum_seconds,
                std::uint32_t seed) {
  validate_formal_arguments(warmup, repetitions, minimum_seconds);
  const Problem problem =
      build_problem(Distribution::kUnitNormalClipped3, seed);
  std::array<std::vector<float>, kVariants.size()> outputs;
  for (std::vector<float>& output : outputs) {
    output.resize(kN);
  }
  WorkerPool pool;
  for (int round = 0; round < warmup; ++round) {
    const auto& order =
        kPermutations[static_cast<std::size_t>(round) % kPermutations.size()];
    for (int variant_index : order) {
      run_variant(pool, kVariants[variant_index], problem,
                  outputs[variant_index]);
    }
  }

  std::array<std::vector<double>, kVariants.size()> samples;
  std::array<double, kVariants.size()> accumulated_us{{0.0, 0.0, 0.0}};
  int round = 0;
  while (true) {
    bool complete = true;
    for (std::size_t index = 0; index < kVariants.size(); ++index) {
      complete = complete &&
                 samples[index].size() >=
                     static_cast<std::size_t>(repetitions) &&
                 accumulated_us[index] >= minimum_seconds * 1.0e6;
    }
    if (complete) {
      break;
    }
    if (round >= 200'000) {
      throw std::runtime_error("formal timing exceeded safety bound");
    }
    const auto& order =
        kPermutations[static_cast<std::size_t>(round) % kPermutations.size()];
    for (int variant_index : order) {
      const double elapsed = time_variant(
          pool, kVariants[variant_index], problem, outputs[variant_index]);
      samples[variant_index].push_back(elapsed);
      accumulated_us[variant_index] += elapsed;
    }
    ++round;
  }

  std::cout
      << "scope,variant,m,k,n,threads,repetitions,analytic_bytes,ops,p25_us,"
         "p50_us,p75_us,iqr_us,iqr_over_p50,measured_gops,measured_gbps,"
         "memory_relative_to_ope0,projection_42_calls_ms,"
         "projection_fraction_of_ope4,permutations_used,"
         "working_set_weight_bytes,affinity_pass,partition_coverage_pass,sink\n";
  bool stable = true;
  for (std::size_t variant_index = 0; variant_index < kVariants.size();
       ++variant_index) {
    const Summary summary = summarize(samples[variant_index]);
    const double measured_gops =
        static_cast<double>(kAnalyticOps) / (summary.p50_us * 1000.0);
    const double measured_gbps =
        static_cast<double>(kAnalyticBytes) / (summary.p50_us * 1000.0);
    const double memory_relative = measured_gbps / kOpe0CeilingGbps;
    const bool candidate = variant_index == 2;
    const double projection_ms =
        candidate ? kCallsPerForward * summary.p50_us / 1000.0 : -1.0;
    const double projection_fraction =
        candidate ? projection_ms / kOpe4TpotMs : -1.0;
    std::cout << "formal_invocation," << kVariants[variant_index].name << ','
              << kM << ',' << kK << ',' << kN << ',' << kThreads << ','
              << samples[variant_index].size() << ',' << kAnalyticBytes << ','
              << kAnalyticOps << ',' << std::setprecision(12)
              << summary.p25_us << ',' << summary.p50_us << ','
              << summary.p75_us << ',' << summary.iqr_us << ','
              << summary.iqr_over_p50 << ',' << measured_gops << ','
              << measured_gbps << ',' << memory_relative << ','
              << projection_ms << ',' << projection_fraction << ",6,"
              << kWeightBytes << ',' << (pool.affinity_pass() ? 1 : 0) << ','
              << (pool.partition_coverage_pass() ? 1 : 0) << ','
              << g_output_sink << '\n';
    stable = stable && summary.iqr_over_p50 <= 0.30;
  }
  std::cerr << "OPE5_FP16_GEMV_MEASURE rounds=" << round
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
             double minimum_seconds, std::uint32_t seed) {
  validate_formal_arguments(warmup, repetitions, minimum_seconds);
  if (prctl(PR_TASK_PERF_EVENTS_DISABLE, 0, 0, 0, 0) != 0) {
    throw std::runtime_error("PR_TASK_PERF_EVENTS_DISABLE failed before setup");
  }
  const Variant& variant = find_variant(variant_name);
  const Problem problem =
      build_problem(Distribution::kUnitNormalClipped3, seed);
  std::vector<float> output(kN);
  WorkerPool pool;
  for (int round = 0; round < warmup; ++round) {
    run_variant(pool, variant, problem, output);
  }
  pool.perf_control(true);
  const std::uint64_t begin = monotonic_raw_ns();
  int completed = 0;
  do {
    run_variant(pool, variant, problem, output);
    ++completed;
  } while (completed < repetitions ||
           static_cast<double>(monotonic_raw_ns() - begin) <
               minimum_seconds * 1.0e9);
  const std::uint64_t end = monotonic_raw_ns();
  pool.perf_control(false);
  const double elapsed_ms = static_cast<double>(end - begin) / 1.0e6;
  std::cout
      << "scope,variant,invocations,analytic_bytes_per_invocation,"
         "ops_per_invocation,elapsed_ms,threads,working_set_weight_bytes,"
         "affinity_pass,partition_coverage_pass,sink\n";
  std::cout << "perf_invocation," << variant.name << ',' << completed << ','
            << kAnalyticBytes << ',' << kAnalyticOps << ','
            << std::setprecision(12) << elapsed_ms << ',' << kThreads << ','
            << kWeightBytes << ',' << (pool.affinity_pass() ? 1 : 0) << ','
            << (pool.partition_coverage_pass() ? 1 : 0) << ','
            << g_output_sink << '\n';
  return 0;
}

struct Options {
  std::string mode;
  std::string variant;
  int warmup = kFormalWarmup;
  int repetitions = kFormalRepetitions;
  double minimum_seconds = kFormalMinSeconds;
  std::uint32_t seed = 20260924U;
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
    } else if (argument == "--min-seconds") {
      options.minimum_seconds = parse_double(value, "min-seconds");
    } else if (argument == "--seed") {
      options.seed = static_cast<std::uint32_t>(parse_int(value, "seed"));
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

void validate_frozen_identity() {
  if (kRowsPerThread != 640 ||
      kActivationBytes + kWeightBytes + kOutputBytes != kAnalyticBytes ||
      kCallsPerForward != 42) {
    throw std::runtime_error("frozen shape or analytic account mismatch");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    validate_frozen_identity();
    const Options options = parse_options(argc, argv);
    if (options.mode == "correctness") {
      return run_correctness();
    }
    if (options.mode == "measure") {
      return run_measure(options.warmup, options.repetitions,
                         options.minimum_seconds, options.seed);
    }
    return run_perf(options.variant, options.warmup, options.repetitions,
                    options.minimum_seconds, options.seed);
  } catch (const std::exception& error) {
    std::cerr << "OPE5_FP16_GEMV_FATAL: " << error.what() << '\n';
    return 2;
  }
}
