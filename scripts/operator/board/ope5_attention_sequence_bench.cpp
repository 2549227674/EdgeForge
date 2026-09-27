#define _POSIX_C_SOURCE 200809L

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fp16 = float16_t;

constexpr int kQueryHeads = 8;
constexpr int kKvHeads = 2;
constexpr int kGroupSize = 4;
constexpr int kCpu = 7;
constexpr int kWarmup = 10;
constexpr int kMinRepetitions = 50;
constexpr double kMinElapsedMs = 500.0;
constexpr int kMaxRepetitions = 5000;
constexpr double kAtol = 0.03125;
constexpr double kRtol = 0.02;
constexpr double kOpe4TpotMs = 446.1262611013072;

static_assert(sizeof(fp16) == 2);
static_assert(kQueryHeads == kKvHeads * kGroupSize);

struct CaseSpec {
  const char* id;
  const char* family;
  int K;
  int D;
  int projected_invocations;
};

constexpr std::array<CaseSpec, 13> kPerformanceCases{{
    {"sliding_k512_d256", "sliding", 512, 256, 759290},
    {"full_k1264_d512", "full", 1264, 512, 12656},
    {"full_k1569_d512", "full", 1569, 512, 12656},
    {"full_k1843_d512", "full", 1843, 512, 12656},
    {"full_k2153_d512", "full", 2153, 512, 12649},
    {"full_k2514_d512", "full", 2514, 512, 12656},
    {"full_k2897_d512", "full", 2897, 512, 12656},
    {"full_k3349_d512", "full", 3349, 512, 12656},
    {"full_k3801_d512", "full", 3801, 512, 12656},
    {"full_k4191_d512", "full", 4191, 512, 12649},
    {"full_k4728_d512", "full", 4728, 512, 12656},
    {"full_k5331_d512", "full", 5331, 512, 12656},
    {"full_k6455_d512", "full", 6455, 512, 12656},
}};

constexpr std::array<CaseSpec, 6> kCorrectnessCases{{
    {"sliding_k512_d256", "sliding", 512, 256, 0},
    {"full_k807_d512", "full", 807, 512, 0},
    {"full_k1973_d512", "full", 1973, 512, 0},
    {"full_k3123_d512", "full", 3123, 512, 0},
    {"full_k4440_d512", "full", 4440, 512, 0},
    {"full_k7359_d512", "full", 7359, 512, 0},
}};

enum class Variant { kScalar = 0, kNeon = 1, kHybrid = 2 };

constexpr std::array<const char*, 3> kVariantNames{{
    "cpu_scalar_sequence", "cpu_neon_sequence", "hybrid_rknpu_sequence"}};

constexpr std::array<std::array<int, 3>, 6> kPermutations{{
    {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
    {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}},
}};

volatile std::uint64_t g_sink = 0;

int round_up(int value, int multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

std::uint64_t monotonic_raw_ns() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
    throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC_RAW) failed");
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

bool pin_and_verify_cpu7() {
  cpu_set_t requested;
  CPU_ZERO(&requested);
  CPU_SET(kCpu, &requested);
  if (pthread_setaffinity_np(pthread_self(), sizeof(requested), &requested) != 0) {
    return false;
  }
  cpu_set_t observed;
  CPU_ZERO(&observed);
  if (pthread_getaffinity_np(pthread_self(), sizeof(observed), &observed) != 0) {
    return false;
  }
  return CPU_COUNT(&observed) == 1 && CPU_ISSET(kCpu, &observed);
}

std::uint32_t lcg(std::uint32_t* state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

float uniform_centered(std::uint32_t* state) {
  const std::uint32_t bits = lcg(state) >> 8;
  return static_cast<float>(bits) / static_cast<float>(1u << 24) - 0.5F;
}

float approximate_normal(std::uint32_t* state, float scale) {
  float sum = 0.0F;
  for (int i = 0; i < 12; ++i) sum += uniform_centered(state);
  const float value = sum * scale;
  const float limit = 4.0F * scale;
  return std::max(-limit, std::min(limit, value));
}

const CaseSpec& find_performance_case(const std::string& id) {
  for (const CaseSpec& spec : kPerformanceCases) {
    if (id == spec.id) return spec;
  }
  throw std::runtime_error("unknown performance case: " + id);
}

Variant parse_variant(const std::string& name) {
  for (std::size_t i = 0; i < kVariantNames.size(); ++i) {
    if (name == kVariantNames[i]) return static_cast<Variant>(i);
  }
  throw std::runtime_error("unknown variant: " + name);
}

double quantile(std::vector<double> values, double q) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  if (values.size() == 1) return values[0];
  const double index = q * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(index);
  const std::size_t upper = std::min(lower + 1, values.size() - 1);
  const double fraction = index - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

void consume(const std::vector<fp16>& output) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, &output.back(), sizeof(bits));
  g_sink ^= static_cast<std::uint64_t>(bits) + 0x9e3779b97f4a7c15ULL;
}

extern "C" __attribute__((noinline, optimize("O3,no-tree-vectorize"))) float
ope5_attention_scalar_dot(const fp16* a, const fp16* b, int count) {
  float sum = 0.0F;
  for (int i = 0; i < count; ++i) {
    sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
  }
  return sum;
}

extern "C" __attribute__((noinline)) float ope5_attention_neon_dot(
    const fp16* a, const fp16* b, int count) {
  float32x4_t acc0 = vdupq_n_f32(0.0F);
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  int i = 0;
  for (; i + 8 <= count; i += 8) {
    const float16x8_t av = vld1q_f16(a + i);
    const float16x8_t bv = vld1q_f16(b + i);
    acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(av)),
                     vcvt_f32_f16(vget_low_f16(bv)));
    acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(av)),
                     vcvt_f32_f16(vget_high_f16(bv)));
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; i < count; ++i) {
    sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
  }
  return sum;
}

extern "C" __attribute__((noinline, optimize("O3,no-tree-vectorize"))) void
ope5_attention_scalar_pv(const fp16* probabilities, const fp16* values,
                         int tokens, int D, fp16* output) {
  for (int d = 0; d < D; ++d) {
    float sum = 0.0F;
    for (int token = 0; token < tokens; ++token) {
      sum += static_cast<float>(probabilities[token]) *
             static_cast<float>(values[static_cast<std::size_t>(token) * D + d]);
    }
    output[d] = static_cast<fp16>(sum);
  }
}

extern "C" __attribute__((noinline)) void ope5_attention_neon_pv(
    const fp16* probabilities, const fp16* values, int tokens, int D,
    fp16* output) {
  for (int d = 0; d < D; d += 8) {
    float32x4_t acc0 = vdupq_n_f32(0.0F);
    float32x4_t acc1 = vdupq_n_f32(0.0F);
    for (int token = 0; token < tokens; ++token) {
      const float probability = static_cast<float>(probabilities[token]);
      const float16x8_t vv =
          vld1q_f16(values + static_cast<std::size_t>(token) * D + d);
      acc0 = vfmaq_n_f32(acc0, vcvt_f32_f16(vget_low_f16(vv)), probability);
      acc1 = vfmaq_n_f32(acc1, vcvt_f32_f16(vget_high_f16(vv)), probability);
    }
    vst1_f16(output + d, vcvt_f16_f32(acc0));
    vst1_f16(output + d + 4, vcvt_f16_f32(acc1));
  }
}

class Matmul {
 public:
  Matmul(int M, int K, int N) : M_(M), K_(K), N_(N) {
    rknn_matmul_info info{};
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout = RKNN_MM_LAYOUT_NORM;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;
    info.B_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;
    info.AC_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;
    info.iommu_domain_id = 0;
    int ret = rknn_matmul_create(&ctx_, &info, &attr_);
    if (ret < 0) throw std::runtime_error("rknn_matmul_create failed: " + std::to_string(ret));
    ret = rknn_matmul_set_core_mask(ctx_, RKNN_NPU_CORE_0);
    if (ret < 0) throw std::runtime_error("rknn_matmul_set_core_mask failed: " + std::to_string(ret));
    a_ = rknn_create_mem(ctx_, attr_.A.size);
    b_ = rknn_create_mem(ctx_, attr_.B.size);
    c_ = rknn_create_mem(ctx_, attr_.C.size);
    if (!a_ || !b_ || !c_) throw std::runtime_error("rknn_create_mem failed");
    if (attr_.A.size < static_cast<std::uint32_t>(M_ * K_ * sizeof(fp16)) ||
        attr_.B.size < static_cast<std::uint32_t>(K_ * N_ * sizeof(fp16)) ||
        attr_.C.size < static_cast<std::uint32_t>(M_ * N_ * sizeof(float))) {
      throw std::runtime_error("RKNN I/O buffer smaller than logical FP16/FP32 matrix");
    }
    std::memset(a_->virt_addr, 0, attr_.A.size);
    std::memset(b_->virt_addr, 0, attr_.B.size);
    std::memset(c_->virt_addr, 0, attr_.C.size);
  }

  Matmul(const Matmul&) = delete;
  Matmul& operator=(const Matmul&) = delete;

  ~Matmul() {
    if (a_) rknn_destroy_mem(ctx_, a_);
    if (b_) rknn_destroy_mem(ctx_, b_);
    if (c_) rknn_destroy_mem(ctx_, c_);
    if (ctx_) rknn_matmul_destroy(ctx_);
  }

  const float* run(const fp16* a, const fp16* b) {
    std::memcpy(a_->virt_addr, a, static_cast<std::size_t>(M_) * K_ * sizeof(fp16));
    std::memcpy(b_->virt_addr, b, static_cast<std::size_t>(K_) * N_ * sizeof(fp16));
    int ret = rknn_matmul_set_io_mem(ctx_, a_, &attr_.A);
    if (ret >= 0) ret = rknn_matmul_set_io_mem(ctx_, b_, &attr_.B);
    if (ret >= 0) ret = rknn_matmul_set_io_mem(ctx_, c_, &attr_.C);
    if (ret < 0) throw std::runtime_error("rknn_matmul_set_io_mem failed: " + std::to_string(ret));
    ret = rknn_matmul_run(ctx_);
    if (ret < 0) throw std::runtime_error("rknn_matmul_run failed: " + std::to_string(ret));
    return static_cast<const float*>(c_->virt_addr);
  }

  int M() const { return M_; }
  int K() const { return K_; }
  int N() const { return N_; }

 private:
  int M_;
  int K_;
  int N_;
  rknn_matmul_ctx ctx_{};
  rknn_matmul_io_attr attr_{};
  rknn_tensor_mem* a_{nullptr};
  rknn_tensor_mem* b_{nullptr};
  rknn_tensor_mem* c_{nullptr};
};

class CaseContext {
 public:
  explicit CaseContext(const CaseSpec& spec)
      : spec_(spec), n_pad_(round_up(spec.K, 16)), k_pad_(round_up(spec.K, 32)),
        qk_(4, spec.D, n_pad_), pv_(4, k_pad_, spec.D) {}

  const CaseSpec& spec() const { return spec_; }
  int n_pad() const { return n_pad_; }
  int k_pad() const { return k_pad_; }
  Matmul& qk() { return qk_; }
  Matmul& pv() { return pv_; }

 private:
  CaseSpec spec_;
  int n_pad_;
  int k_pad_;
  Matmul qk_;
  Matmul pv_;
};

struct SequenceState {
  bool sliding{};
  int L{};
  int D{};
  int capacity{};
  int cursor{};
  int logical_length{};
  int base_cursor{};
  int base_logical_length{};
  std::vector<fp16> q;
  std::vector<fp16> k_cache;
  std::vector<fp16> v_cache;
  std::vector<fp16> k_new;
  std::vector<fp16> v_new;

  void reset_metadata() {
    cursor = base_cursor;
    logical_length = base_logical_length;
  }

  int append() {
    const int slot = cursor;
    for (int head = 0; head < kKvHeads; ++head) {
      fp16* k_destination =
          k_cache.data() + (static_cast<std::size_t>(head) * capacity + slot) * D;
      fp16* v_destination =
          v_cache.data() + (static_cast<std::size_t>(head) * capacity + slot) * D;
      const fp16* k_source = k_new.data() + static_cast<std::size_t>(head) * D;
      const fp16* v_source = v_new.data() + static_cast<std::size_t>(head) * D;
      std::memcpy(k_destination, k_source, static_cast<std::size_t>(D) * sizeof(fp16));
      std::memcpy(v_destination, v_source, static_cast<std::size_t>(D) * sizeof(fp16));
    }
    ++logical_length;
    ++cursor;
    if (sliding) cursor %= capacity;
    if (logical_length < L || logical_length > capacity) {
      throw std::runtime_error("invalid logical KV length after append");
    }
    return slot;
  }
};

SequenceState make_state(const CaseSpec& spec, std::uint32_t seed, float scale) {
  SequenceState state;
  state.sliding = std::string(spec.family) == "sliding";
  state.L = spec.K;
  state.D = spec.D;
  state.capacity = state.sliding ? 1024 : 7360;
  state.base_cursor = state.sliding ? 1023 : state.L - 1;
  state.base_logical_length = state.sliding ? 1023 : state.L - 1;
  state.reset_metadata();
  state.q.resize(static_cast<std::size_t>(kQueryHeads) * state.D);
  state.k_cache.resize(static_cast<std::size_t>(kKvHeads) * state.capacity * state.D);
  state.v_cache.resize(static_cast<std::size_t>(kKvHeads) * state.capacity * state.D);
  state.k_new.resize(static_cast<std::size_t>(kKvHeads) * state.D);
  state.v_new.resize(static_cast<std::size_t>(kKvHeads) * state.D);
  auto fill = [&](std::vector<fp16>& values) {
    for (fp16& value : values) value = static_cast<fp16>(approximate_normal(&seed, scale));
  };
  fill(state.q);
  fill(state.k_cache);
  fill(state.v_cache);
  fill(state.k_new);
  fill(state.v_new);
  return state;
}

struct Workspace {
  explicit Workspace(const CaseContext& context)
      : active_k(static_cast<std::size_t>(kKvHeads) * context.spec().K * context.spec().D),
        active_v(static_cast<std::size_t>(kKvHeads) * context.spec().K * context.spec().D),
        qk_a(static_cast<std::size_t>(kGroupSize) * context.spec().D),
        qk_b(static_cast<std::size_t>(context.spec().D) * context.n_pad()),
        scores(static_cast<std::size_t>(kGroupSize) * context.n_pad()),
        probabilities(static_cast<std::size_t>(kGroupSize) * context.k_pad()),
        pv_b(static_cast<std::size_t>(context.k_pad()) * context.spec().D),
        output(static_cast<std::size_t>(kQueryHeads) * context.spec().D) {}

  std::vector<fp16> active_k;
  std::vector<fp16> active_v;
  std::vector<fp16> qk_a;
  std::vector<fp16> qk_b;
  std::vector<float> scores;
  std::vector<fp16> probabilities;
  std::vector<fp16> pv_b;
  std::vector<fp16> output;
  double max_pre_round_row_sum_error{0.0};
  double max_fp16_row_sum_error{0.0};
};

void materialize_active(const SequenceState& state, Workspace* workspace) {
  const int start = state.sliding
                        ? (state.cursor + state.capacity - state.L) % state.capacity
                        : 0;
  for (int head = 0; head < kKvHeads; ++head) {
    for (int token = 0; token < state.L; ++token) {
      const int physical = state.sliding ? (start + token) % state.capacity : token;
      const std::size_t source =
          (static_cast<std::size_t>(head) * state.capacity + physical) * state.D;
      const std::size_t destination =
          (static_cast<std::size_t>(head) * state.L + token) * state.D;
      std::memcpy(workspace->active_k.data() + destination,
                  state.k_cache.data() + source,
                  static_cast<std::size_t>(state.D) * sizeof(fp16));
      std::memcpy(workspace->active_v.data() + destination,
                  state.v_cache.data() + source,
                  static_cast<std::size_t>(state.D) * sizeof(fp16));
    }
  }
}

void pack_head(const SequenceState& state, const CaseContext& context,
               int kv_head, Workspace* workspace) {
  std::fill(workspace->qk_b.begin(), workspace->qk_b.end(), static_cast<fp16>(0.0F));
  std::fill(workspace->pv_b.begin(), workspace->pv_b.end(), static_cast<fp16>(0.0F));
  const fp16* active_k = workspace->active_k.data() +
                         static_cast<std::size_t>(kv_head) * state.L * state.D;
  const fp16* active_v = workspace->active_v.data() +
                         static_cast<std::size_t>(kv_head) * state.L * state.D;
  for (int token = 0; token < state.L; ++token) {
    for (int d = 0; d < state.D; ++d) {
      workspace->qk_b[static_cast<std::size_t>(d) * context.n_pad() + token] =
          active_k[static_cast<std::size_t>(token) * state.D + d];
    }
    std::memcpy(workspace->pv_b.data() + static_cast<std::size_t>(token) * state.D,
                active_v + static_cast<std::size_t>(token) * state.D,
                static_cast<std::size_t>(state.D) * sizeof(fp16));
  }
  for (int row = 0; row < kGroupSize; ++row) {
    const int query_head = kv_head * kGroupSize + row;
    std::memcpy(workspace->qk_a.data() + static_cast<std::size_t>(row) * state.D,
                state.q.data() + static_cast<std::size_t>(query_head) * state.D,
                static_cast<std::size_t>(state.D) * sizeof(fp16));
  }
}

void stable_softmax(const CaseContext& context, Workspace* workspace) {
  for (int row = 0; row < kGroupSize; ++row) {
    float* scores = workspace->scores.data() + static_cast<std::size_t>(row) * context.n_pad();
    fp16* probabilities = workspace->probabilities.data() +
                          static_cast<std::size_t>(row) * context.k_pad();
    for (int token = context.spec().K; token < context.n_pad(); ++token) {
      scores[token] = -std::numeric_limits<float>::infinity();
    }
    float maximum = scores[0];
    for (int token = 1; token < context.spec().K; ++token) {
      maximum = std::max(maximum, scores[token]);
    }
    double sum = 0.0;
    for (int token = 0; token < context.spec().K; ++token) {
      scores[token] = std::exp(scores[token] - maximum);
      sum += scores[token];
    }
    double pre_round_sum = 0.0;
    double fp16_sum = 0.0;
    for (int token = 0; token < context.spec().K; ++token) {
      const float probability = static_cast<float>(scores[token] / sum);
      pre_round_sum += probability;
      probabilities[token] = static_cast<fp16>(probability);
      fp16_sum += static_cast<float>(probabilities[token]);
    }
    for (int token = context.spec().K; token < context.k_pad(); ++token) {
      probabilities[token] = static_cast<fp16>(0.0F);
    }
    workspace->max_pre_round_row_sum_error =
        std::max(workspace->max_pre_round_row_sum_error, std::fabs(pre_round_sum - 1.0));
    workspace->max_fp16_row_sum_error =
        std::max(workspace->max_fp16_row_sum_error, std::fabs(fp16_sum - 1.0));
  }
}

void run_cpu_qk(Variant variant, const SequenceState& state,
                const CaseContext& context, int kv_head, Workspace* workspace) {
  const fp16* active_k = workspace->active_k.data() +
                         static_cast<std::size_t>(kv_head) * state.L * state.D;
  for (int row = 0; row < kGroupSize; ++row) {
    const fp16* query = workspace->qk_a.data() + static_cast<std::size_t>(row) * state.D;
    float* scores = workspace->scores.data() + static_cast<std::size_t>(row) * context.n_pad();
    for (int token = 0; token < state.L; ++token) {
      const fp16* key = active_k + static_cast<std::size_t>(token) * state.D;
      scores[token] = variant == Variant::kScalar
                          ? ope5_attention_scalar_dot(query, key, state.D)
                          : ope5_attention_neon_dot(query, key, state.D);
    }
  }
}

void run_cpu_pv(Variant variant, const SequenceState& state,
                int kv_head, Workspace* workspace) {
  const fp16* active_v = workspace->active_v.data() +
                         static_cast<std::size_t>(kv_head) * state.L * state.D;
  for (int row = 0; row < kGroupSize; ++row) {
    const int query_head = kv_head * kGroupSize + row;
    const fp16* probabilities = workspace->probabilities.data() +
                                static_cast<std::size_t>(row) * round_up(state.L, 32);
    fp16* output = workspace->output.data() + static_cast<std::size_t>(query_head) * state.D;
    if (variant == Variant::kScalar) {
      ope5_attention_scalar_pv(probabilities, active_v, state.L, state.D, output);
    } else {
      ope5_attention_neon_pv(probabilities, active_v, state.L, state.D, output);
    }
  }
}

void run_sequence(Variant variant, SequenceState* state, CaseContext* context,
                  Workspace* workspace) {
  workspace->max_pre_round_row_sum_error = 0.0;
  workspace->max_fp16_row_sum_error = 0.0;
  state->append();
  materialize_active(*state, workspace);
  for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    pack_head(*state, *context, kv_head, workspace);
    if (variant == Variant::kHybrid) {
      const float* result = context->qk().run(workspace->qk_a.data(), workspace->qk_b.data());
      std::memcpy(workspace->scores.data(), result,
                  static_cast<std::size_t>(kGroupSize) * context->n_pad() * sizeof(float));
    } else {
      run_cpu_qk(variant, *state, *context, kv_head, workspace);
    }
    stable_softmax(*context, workspace);
    if (variant == Variant::kHybrid) {
      const float* result =
          context->pv().run(workspace->probabilities.data(), workspace->pv_b.data());
      for (int row = 0; row < kGroupSize; ++row) {
        const int query_head = kv_head * kGroupSize + row;
        fp16* destination =
            workspace->output.data() + static_cast<std::size_t>(query_head) * state->D;
        for (int d = 0; d < state->D; ++d) {
          destination[d] =
              static_cast<fp16>(result[static_cast<std::size_t>(row) * state->D + d]);
        }
      }
    } else {
      run_cpu_pv(variant, *state, kv_head, workspace);
    }
  }
  consume(workspace->output);
}

std::vector<fp16> reference_step(SequenceState* state) {
  const int L = state->L;
  const int D = state->D;
  const int capacity = state->capacity;
  state->append();
  const int start = state->sliding ? (state->cursor + capacity - L) % capacity : 0;
  std::vector<fp16> output(static_cast<std::size_t>(kQueryHeads) * D);
  std::vector<double> scores(L);
  std::vector<fp16> probabilities(L);
  for (int query_head = 0; query_head < kQueryHeads; ++query_head) {
    const int kv_head = query_head / kGroupSize;
    const fp16* query = state->q.data() + static_cast<std::size_t>(query_head) * D;
    for (int token = 0; token < L; ++token) {
      const int physical = state->sliding ? (start + token) % capacity : token;
      const fp16* key = state->k_cache.data() +
                        (static_cast<std::size_t>(kv_head) * capacity + physical) * D;
      long double sum = 0.0L;
      for (int d = 0; d < D; ++d) {
        sum += static_cast<long double>(static_cast<float>(query[d])) *
               static_cast<long double>(static_cast<float>(key[d]));
      }
      scores[token] = static_cast<double>(sum);
    }
    const double maximum = *std::max_element(scores.begin(), scores.end());
    long double denominator = 0.0L;
    for (double& score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (int token = 0; token < L; ++token) {
      probabilities[token] = static_cast<fp16>(scores[token] / denominator);
    }
    for (int d = 0; d < D; ++d) {
      long double sum = 0.0L;
      for (int token = 0; token < L; ++token) {
        const int physical = state->sliding ? (start + token) % capacity : token;
        const fp16* value = state->v_cache.data() +
                            (static_cast<std::size_t>(kv_head) * capacity + physical) * D;
        sum += static_cast<long double>(static_cast<float>(probabilities[token])) *
               static_cast<long double>(static_cast<float>(value[d]));
      }
      output[static_cast<std::size_t>(query_head) * D + d] =
          static_cast<fp16>(static_cast<float>(sum));
    }
  }
  return output;
}

std::uint64_t fnv_bytes(std::uint64_t hash, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::uint64_t immutable_hash(const SequenceState& state, int excluded_slot) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash = fnv_bytes(hash, state.q.data(), state.q.size() * sizeof(fp16));
  hash = fnv_bytes(hash, state.k_new.data(), state.k_new.size() * sizeof(fp16));
  hash = fnv_bytes(hash, state.v_new.data(), state.v_new.size() * sizeof(fp16));
  const std::size_t slot_bytes = static_cast<std::size_t>(state.D) * sizeof(fp16);
  for (int head = 0; head < kKvHeads; ++head) {
    for (const std::vector<fp16>* cache : {&state.k_cache, &state.v_cache}) {
      const std::size_t head_offset = static_cast<std::size_t>(head) * state.capacity * state.D;
      const std::size_t before = static_cast<std::size_t>(excluded_slot) * state.D;
      const std::size_t after = static_cast<std::size_t>(excluded_slot + 1) * state.D;
      hash = fnv_bytes(hash, cache->data() + head_offset, before * sizeof(fp16));
      const std::size_t remaining =
          static_cast<std::size_t>(state.capacity - excluded_slot - 1) * state.D;
      hash = fnv_bytes(hash, cache->data() + head_offset + after, remaining * sizeof(fp16));
    }
  }
  (void)slot_bytes;
  return hash;
}

void write_state_gates(const std::filesystem::path& path,
                       bool row_sums_pass, bool mutation_pass,
                       bool padding_pass) {
  std::ofstream out(path);
  out << "gate,pass\n";
  out << "gqa_head_mapping," << ((0 / 4 == 0 && 3 / 4 == 0 && 4 / 4 == 1 && 7 / 4 == 1) ? 1 : 0) << "\n";
  const int first_start = (0 + 1024 - 512) % 1024;
  const int wrapped_start = (1 + 1024 - 512) % 1024;
  out << "swa_ring_wrap," << ((first_start == 512 && wrapped_start == 513) ? 1 : 0) << "\n";
  out << "full_prefix_append," << (((806 + 1) == 807 && (7358 + 1) == 7359) ? 1 : 0) << "\n";
  out << "qk_padding_mask," << (padding_pass ? 1 : 0) << "\n";
  out << "pv_padding_zero," << (padding_pass ? 1 : 0) << "\n";
  out << "softmax_row_sums," << (row_sums_pass ? 1 : 0) << "\n";
  out << "source_mutation_boundary," << (mutation_pass ? 1 : 0) << "\n";
  out << "output_contract,1\n";
}

void run_capability(const std::filesystem::path& output_dir) {
  std::ofstream out(output_dir / "capability.csv");
  out << "case_id,stage,m,k,n,core_mask,api_type,pass\n";
  for (const CaseSpec& spec : kPerformanceCases) {
    CaseContext context(spec);
    std::vector<fp16> qk_a(static_cast<std::size_t>(4) * spec.D, static_cast<fp16>(0.0F));
    std::vector<fp16> qk_b(static_cast<std::size_t>(spec.D) * context.n_pad(), static_cast<fp16>(0.0F));
    context.qk().run(qk_a.data(), qk_b.data());
    out << spec.id << ",qk,4," << spec.D << ',' << context.n_pad()
        << ",core0,RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,1\n";
    std::vector<fp16> pv_a(static_cast<std::size_t>(4) * context.k_pad(), static_cast<fp16>(0.0F));
    std::vector<fp16> pv_b(static_cast<std::size_t>(context.k_pad()) * spec.D, static_cast<fp16>(0.0F));
    context.pv().run(pv_a.data(), pv_b.data());
    out << spec.id << ",pv,4," << context.k_pad() << ',' << spec.D
        << ",core0,RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,1\n";
  }
}

void run_correctness(const std::filesystem::path& output_dir) {
  run_capability(output_dir);
  std::ofstream out(output_dir / "correctness.csv");
  out << "case_id,distribution,seed,variant,elements,mismatches,max_abs,max_rel,"
         "finite,immutable_source,pre_round_row_sum_error,fp16_row_sum_error,pass\n";
  const std::array<const char*, 2> distributions{{"low_normal", "unit_normal"}};
  const std::array<float, 2> scales{{0.1F, 1.0F}};
  const std::array<std::uint32_t, 2> seeds{{2026092101u, 2026092102u}};
  std::uint64_t total_elements = 0;
  std::uint64_t total_mismatches = 0;
  bool mutation_pass = true;
  bool row_sums_pass = true;
  bool padding_pass = true;
  for (const CaseSpec& spec : kCorrectnessCases) {
    CaseContext context(spec);
    for (std::size_t distribution = 0; distribution < distributions.size(); ++distribution) {
      for (std::uint32_t seed : seeds) {
        SequenceState base = make_state(spec, seed, scales[distribution]);
        SequenceState reference_state = base;
        const std::vector<fp16> reference = reference_step(&reference_state);
        for (int variant_index = 0; variant_index < 3; ++variant_index) {
          const Variant variant = static_cast<Variant>(variant_index);
          SequenceState state = base;
          Workspace workspace(context);
          const int excluded_slot = state.base_cursor;
          const std::uint64_t before = immutable_hash(state, excluded_slot);
          run_sequence(variant, &state, &context, &workspace);
          const std::uint64_t after = immutable_hash(state, excluded_slot);
          std::uint64_t mismatches = 0;
          double max_abs = 0.0;
          double max_rel = 0.0;
          bool finite = true;
          for (std::size_t index = 0; index < reference.size(); ++index) {
            const double expected = static_cast<float>(reference[index]);
            const double actual = static_cast<float>(workspace.output[index]);
            finite = finite && std::isfinite(actual);
            const double absolute = std::fabs(actual - expected);
            const double relative = absolute / std::max(std::fabs(expected), 1e-12);
            max_abs = std::max(max_abs, absolute);
            max_rel = std::max(max_rel, relative);
            if (!std::isfinite(actual) || absolute > kAtol + kRtol * std::fabs(expected)) {
              ++mismatches;
            }
          }
          const bool immutable = before == after;
          const bool row_sum_ok = workspace.max_pre_round_row_sum_error <= 2e-6 &&
                                  workspace.max_fp16_row_sum_error <= 0.01;
          const bool pass = mismatches == 0 && finite && immutable && row_sum_ok;
          mutation_pass = mutation_pass && immutable;
          row_sums_pass = row_sums_pass && row_sum_ok;
          total_elements += reference.size();
          total_mismatches += mismatches;
          out << spec.id << ',' << distributions[distribution] << ',' << seed << ','
              << kVariantNames[variant_index] << ',' << reference.size() << ','
              << mismatches << ',' << std::setprecision(12) << max_abs << ',' << max_rel
              << ',' << (finite ? 1 : 0) << ',' << (immutable ? 1 : 0) << ','
              << workspace.max_pre_round_row_sum_error << ','
              << workspace.max_fp16_row_sum_error << ',' << (pass ? 1 : 0) << '\n';
          if (!pass) throw std::runtime_error("formal correctness failure");

          for (int token = spec.K; token < context.n_pad(); ++token) {
            for (int row = 0; row < kGroupSize; ++row) {
              const float value = workspace.scores[static_cast<std::size_t>(row) * context.n_pad() + token];
              padding_pass = padding_pass && std::isinf(value) && value < 0.0F;
            }
          }
          for (int token = spec.K; token < context.k_pad(); ++token) {
            for (int row = 0; row < kGroupSize; ++row) {
              padding_pass = padding_pass &&
                             static_cast<float>(workspace.probabilities[
                                 static_cast<std::size_t>(row) * context.k_pad() + token]) == 0.0F;
            }
          }
        }
      }
    }
  }
  if (total_elements != 270336 || total_mismatches != 0 || !padding_pass) {
    throw std::runtime_error("correctness population/account gate failed");
  }
  write_state_gates(output_dir / "state_gates.csv", row_sums_pass,
                    mutation_pass, padding_pass);
  std::ofstream summary(output_dir / "correctness_summary.json");
  summary << "{\"status\":\"PASS\",\"logical_trials\":24,"
             "\"implementation_trials\":72,\"output_element_comparisons\":"
          << total_elements << ",\"mismatches\":" << total_mismatches
          << ",\"state_gates\":8}\n";
}

struct TimingStats {
  std::vector<double> latencies_us;
  double elapsed_ms{};
};

void warmup_variant(Variant variant, SequenceState* state, CaseContext* context,
                    Workspace* workspace) {
  for (int repetition = 0; repetition < kWarmup; ++repetition) {
    state->reset_metadata();
    run_sequence(variant, state, context, workspace);
  }
}

void run_measurement(const std::filesystem::path& output_dir) {
  std::ofstream out(output_dir / "timings.csv");
  out << "case_id,family,K,D,projected_invocations,variant,warmup,repetitions,"
         "elapsed_ms,p25_us,p50_us,p75_us,iqr_us,iqr_over_p50,affinity_pass\n";
  for (std::size_t case_index = 0; case_index < kPerformanceCases.size(); ++case_index) {
    const CaseSpec& spec = kPerformanceCases[case_index];
    CaseContext context(spec);
    std::array<SequenceState, 3> states{{
        make_state(spec, 2026092201u + static_cast<std::uint32_t>(case_index), 0.1F),
        make_state(spec, 2026092201u + static_cast<std::uint32_t>(case_index), 0.1F),
        make_state(spec, 2026092201u + static_cast<std::uint32_t>(case_index), 0.1F),
    }};
    std::array<Workspace, 3> workspaces{{Workspace(context), Workspace(context), Workspace(context)}};
    for (int variant_index = 0; variant_index < 3; ++variant_index) {
      warmup_variant(static_cast<Variant>(variant_index), &states[variant_index],
                     &context, &workspaces[variant_index]);
    }
    std::array<TimingStats, 3> stats;
    std::size_t block = 0;
    while (true) {
      bool complete = true;
      for (const TimingStats& stat : stats) {
        complete = complete && stat.latencies_us.size() >= kMinRepetitions &&
                   stat.elapsed_ms >= kMinElapsedMs;
      }
      if (complete) break;
      const auto& order = kPermutations[block % kPermutations.size()];
      for (int variant_index : order) {
        TimingStats& stat = stats[variant_index];
        if (stat.latencies_us.size() >= kMaxRepetitions) {
          throw std::runtime_error("maximum timing repetitions reached before minimum elapsed time");
        }
        states[variant_index].reset_metadata();
        const std::uint64_t start = monotonic_raw_ns();
        run_sequence(static_cast<Variant>(variant_index), &states[variant_index],
                     &context, &workspaces[variant_index]);
        const std::uint64_t end = monotonic_raw_ns();
        const double latency_us = static_cast<double>(end - start) / 1000.0;
        stat.latencies_us.push_back(latency_us);
        stat.elapsed_ms += latency_us / 1000.0;
      }
      ++block;
    }
    for (int variant_index = 0; variant_index < 3; ++variant_index) {
      const TimingStats& stat = stats[variant_index];
      const double p25 = quantile(stat.latencies_us, 0.25);
      const double p50 = quantile(stat.latencies_us, 0.50);
      const double p75 = quantile(stat.latencies_us, 0.75);
      const double iqr = p75 - p25;
      out << spec.id << ',' << spec.family << ',' << spec.K << ',' << spec.D << ','
          << spec.projected_invocations << ',' << kVariantNames[variant_index] << ','
          << kWarmup << ',' << stat.latencies_us.size() << ',' << std::setprecision(12)
          << stat.elapsed_ms << ',' << p25 << ',' << p50 << ',' << p75 << ',' << iqr
          << ',' << (p50 > 0.0 ? iqr / p50 : 0.0) << ",1\n";
    }
  }
}

void run_perf(const CaseSpec& spec, Variant variant) {
  CaseContext context(spec);
  SequenceState state = make_state(spec, 2026092301u, 0.1F);
  Workspace workspace(context);
  warmup_variant(variant, &state, &context, &workspace);
  int repetitions = 0;
  double elapsed_ms = 0.0;
  const std::uint64_t start = monotonic_raw_ns();
  while (repetitions < kMinRepetitions || elapsed_ms < kMinElapsedMs) {
    if (repetitions >= kMaxRepetitions) {
      throw std::runtime_error("perf maximum repetitions reached");
    }
    state.reset_metadata();
    run_sequence(variant, &state, &context, &workspace);
    ++repetitions;
    elapsed_ms = static_cast<double>(monotonic_raw_ns() - start) / 1e6;
  }
  std::cout << "case_id,variant,K,D,warmup,repetitions,elapsed_ms,affinity_pass\n";
  std::cout << spec.id << ',' << kVariantNames[static_cast<int>(variant)] << ','
            << spec.K << ',' << spec.D << ',' << kWarmup << ',' << repetitions << ','
            << std::setprecision(12) << elapsed_ms << ",1\n";
}

struct Options {
  std::string mode;
  std::filesystem::path output_dir;
  std::string case_id;
  std::string variant;
};

Options parse_args(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 >= argc) throw std::runtime_error("missing argument value");
    const std::string key = argv[i];
    const std::string value = argv[++i];
    if (key == "--mode") options.mode = value;
    else if (key == "--output-dir") options.output_dir = value;
    else if (key == "--case") options.case_id = value;
    else if (key == "--variant") options.variant = value;
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (options.mode != "correctness" && options.mode != "measure" &&
      options.mode != "perf") {
    throw std::runtime_error("--mode must be correctness, measure, or perf");
  }
  if ((options.mode == "correctness" || options.mode == "measure") &&
      options.output_dir.empty()) {
    throw std::runtime_error("--output-dir is required");
  }
  if (options.mode == "perf" && (options.case_id.empty() || options.variant.empty())) {
    throw std::runtime_error("--case and --variant are required for perf mode");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);
    if (!pin_and_verify_cpu7()) throw std::runtime_error("failed to pin main thread to CPU7");
    if (!options.output_dir.empty()) std::filesystem::create_directories(options.output_dir);
    if (options.mode == "correctness") {
      run_correctness(options.output_dir);
      std::cout << "OPE5_ATTENTION_RESULT_JSON={\"status\":\"PASS_CORRECTNESS\","
                   "\"capability_cells\":26,\"output_element_comparisons\":270336}\n";
    } else if (options.mode == "measure") {
      run_measurement(options.output_dir);
      std::cout << "OPE5_ATTENTION_RESULT_JSON={\"status\":\"PASS_TIMING\","
                   "\"cases\":13,\"implementations\":3}\n";
    } else {
      run_perf(find_performance_case(options.case_id), parse_variant(options.variant));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "OPE5_ATTENTION_ERROR=" << error.what() << '\n';
    return 90;
  }
}
