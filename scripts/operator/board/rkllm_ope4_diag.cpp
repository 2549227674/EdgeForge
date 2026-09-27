// OPE4 board-diagnostic-1core runner.
// Captures callback timing and RKLLMPerfStat without interpreting transcript semantics.

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "rkllm.h"

namespace {

LLMHandle g_handle = nullptr;

uint64_t monotonic_raw_ns() {
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        std::perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
        std::exit(90);
    }
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(ts.tv_nsec);
}

void destroy_handle() {
    if (g_handle != nullptr) {
        LLMHandle handle = g_handle;
        g_handle = nullptr;
        rkllm_destroy(handle);
    }
}

void signal_handler(int signal_number) {
    destroy_handle();
    std::_Exit(128 + signal_number);
}

const char* state_name(LLMCallState state) {
    switch (state) {
        case RKLLM_RUN_NORMAL:
            return "NORMAL";
        case RKLLM_RUN_WAITING:
            return "WAITING";
        case RKLLM_RUN_FINISH:
            return "FINISH";
        case RKLLM_RUN_ERROR:
            return "ERROR";
    }
    return "UNKNOWN";
}

struct RunContext {
    const char* carrier = nullptr;
    int prompt_id = 0;
    FILE* events = nullptr;
    FILE* transcript = nullptr;
    uint64_t run_start_ns = 0;
    uint64_t first_normal_ns = 0;
    uint64_t last_normal_ns = 0;
    uint64_t finish_ns = 0;
    uint64_t event_index = 0;
    uint64_t normal_callbacks = 0;
    uint64_t waiting_callbacks = 0;
    uint64_t finish_callbacks = 0;
    uint64_t output_bytes = 0;
    bool callback_error = false;
    bool perf_seen = false;
    RKLLMPerfStat perf{};
};

int result_callback(RKLLMResult* result, void* userdata, LLMCallState state) {
    auto* context = static_cast<RunContext*>(userdata);
    if (context == nullptr || context->events == nullptr) {
        return 0;
    }

    const uint64_t now_ns = monotonic_raw_ns();
    const int token_id = result == nullptr ? -1 : result->token_id;
    const char* text =
        (result != nullptr && result->text != nullptr) ? result->text : "";
    const size_t text_bytes = std::strlen(text);
    RKLLMPerfStat perf{};
    if (result != nullptr) {
        perf = result->perf;
        context->perf = perf;
        context->perf_seen = true;
    }

    if (state == RKLLM_RUN_NORMAL) {
        if (context->first_normal_ns == 0) {
            context->first_normal_ns = now_ns;
        }
        context->last_normal_ns = now_ns;
        ++context->normal_callbacks;
        if (text_bytes != 0) {
            const size_t written =
                std::fwrite(text, 1, text_bytes, context->transcript);
            if (written != text_bytes) {
                context->callback_error = true;
            }
            context->output_bytes += written;
            std::fflush(context->transcript);
        }
    } else if (state == RKLLM_RUN_WAITING) {
        ++context->waiting_callbacks;
    } else if (state == RKLLM_RUN_FINISH) {
        context->finish_ns = now_ns;
        ++context->finish_callbacks;
    } else if (state == RKLLM_RUN_ERROR) {
        context->callback_error = true;
    }

    const double since_start_ms =
        static_cast<double>(now_ns - context->run_start_ns) / 1.0e6;
    std::fprintf(
        context->events,
        "%s,%d,%llu,%s,%llu,%.6f,%d,%zu,%.6f,%d,%.6f,%d,%.6f\n",
        context->carrier, context->prompt_id,
        static_cast<unsigned long long>(context->event_index),
        state_name(state), static_cast<unsigned long long>(now_ns),
        since_start_ms, token_id, text_bytes, perf.prefill_time_ms,
        perf.prefill_tokens, perf.generate_time_ms, perf.generate_tokens,
        perf.memory_usage_mb);
    std::fflush(context->events);
    ++context->event_index;
    return 0;
}

bool valid_carrier(const std::string& carrier) {
    if (carrier.empty()) {
        return false;
    }
    for (char ch : carrier) {
        const bool ok = (ch >= 'a' && ch <= 'z') ||
                        (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
        if (!ok) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> read_prompts(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot open prompt payload: %s\n", path);
        std::exit(2);
    }
    std::vector<std::string> prompts;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            std::fputs("empty prompt is forbidden\n", stderr);
            std::exit(2);
        }
        prompts.push_back(line);
    }
    if (prompts.size() != 5) {
        std::fprintf(stderr, "expected exactly five prompts, got %zu\n",
                     prompts.size());
        std::exit(2);
    }
    return prompts;
}

FILE* open_output(const std::string& path, const char* mode) {
    FILE* file = std::fopen(path.c_str(), mode);
    if (file == nullptr) {
        std::fprintf(stderr, "cannot open %s: %s\n", path.c_str(),
                     std::strerror(errno));
        std::exit(2);
    }
    return file;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
                     "usage: %s MODEL CARRIER PROMPTS OUTPUT_DIR "
                     "MAX_NEW_TOKENS MAX_CONTEXT_LEN\n",
                     argv[0]);
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string carrier = argv[2];
    const char* prompts_path = argv[3];
    const std::string output_dir = argv[4];
    const int max_new_tokens = std::atoi(argv[5]);
    const int max_context_len = std::atoi(argv[6]);
    if (!valid_carrier(carrier) || max_new_tokens != 96 ||
        max_context_len != 16384) {
        std::fputs("formal OPE4 requires a valid carrier, 96 new tokens, and "
                   "16384 context\n",
                   stderr);
        return 2;
    }

    struct stat model_stat {};
    struct stat output_stat {};
    if (stat(model_path.c_str(), &model_stat) != 0 ||
        !S_ISREG(model_stat.st_mode)) {
        std::fprintf(stderr, "model is not a regular file: %s\n",
                     model_path.c_str());
        return 2;
    }
    if (stat(output_dir.c_str(), &output_stat) != 0 ||
        !S_ISDIR(output_stat.st_mode)) {
        std::fprintf(stderr, "output directory is unavailable: %s\n",
                     output_dir.c_str());
        return 2;
    }
    const std::vector<std::string> prompts = read_prompts(prompts_path);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    FILE* events = open_output(output_dir + "/callback_events.csv", "wb");
    FILE* summary = open_output(output_dir + "/prompt_summary.csv", "wb");
    FILE* meta = open_output(output_dir + "/runner_meta.env", "wb");
    std::fputs("carrier,prompt_id,event_index,state,monotonic_raw_ns,"
               "since_run_start_ms,token_id,text_bytes,prefill_time_ms,"
               "prefill_tokens,generate_time_ms,generate_tokens,"
               "memory_usage_mb\n",
               events);
    std::fputs("carrier,prompt_id,run_start_ns,run_return_ns,first_normal_ns,"
               "last_normal_ns,finish_ns,rkllm_run_return,callback_error,"
               "normal_callbacks,waiting_callbacks,finish_callbacks,"
               "output_bytes,perf_seen,prefill_time_ms,prefill_tokens,"
               "generate_time_ms,generate_tokens,memory_usage_mb\n",
               summary);

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = model_path.c_str();
    param.top_k = 1;
    param.top_p = 0.95F;
    param.temperature = 0.8F;
    param.repeat_penalty = 1.1F;
    param.frequency_penalty = 0.0F;
    param.presence_penalty = 0.0F;
    param.max_new_tokens = max_new_tokens;
    param.max_context_len = max_context_len;
    param.skip_special_token = true;
    param.ignore_eos_token = false;
    param.is_async = false;
    param.extend_param.base_domain_id = 0;
    param.extend_param.embed_flash = 1;
    param.extend_param.enabled_cpus_num = 4;
    param.extend_param.enabled_cpus_mask = CPU4 | CPU5 | CPU6 | CPU7;
    param.extend_param.n_batch = 1;

    RKLLMCallback callbacks{};
    callbacks.result_callback = result_callback;

    const uint64_t init_start_ns = monotonic_raw_ns();
    std::puts("rkllm init start");
    const int init_ret = rkllm_init(&g_handle, &param, &callbacks);
    const uint64_t init_end_ns = monotonic_raw_ns();
    std::fprintf(meta, "carrier=%s\n", carrier.c_str());
    std::fprintf(meta, "model_path=%s\n", model_path.c_str());
    std::fprintf(meta, "model_size_bytes=%lld\n",
                 static_cast<long long>(model_stat.st_size));
    std::fprintf(meta, "prompt_payload=%s\n", prompts_path);
    std::fprintf(meta, "max_new_tokens=%d\n", max_new_tokens);
    std::fprintf(meta, "max_context_len=%d\n", max_context_len);
    std::fprintf(meta, "enabled_cpus_mask=0x%02x\n",
                 CPU4 | CPU5 | CPU6 | CPU7);
    std::fprintf(meta, "init_start_monotonic_raw_ns=%llu\n",
                 static_cast<unsigned long long>(init_start_ns));
    std::fprintf(meta, "init_end_monotonic_raw_ns=%llu\n",
                 static_cast<unsigned long long>(init_end_ns));
    std::fprintf(meta, "init_return=%d\n", init_ret);
    std::fflush(meta);
    if (init_ret != 0) {
        std::fprintf(stderr, "rkllm_init failed: %d\n", init_ret);
        std::fclose(events);
        std::fclose(summary);
        std::fclose(meta);
        destroy_handle();
        return 3;
    }
    std::puts("rkllm init success");

    const int template_ret = rkllm_set_chat_template(
        g_handle, "", "<bos><|turn>user\n", "<turn|>\n<|turn>model\n");
    std::fprintf(meta, "manual_template_return=%d\n", template_ret);
    std::fflush(meta);
    if (template_ret != 0) {
        std::fprintf(stderr, "rkllm_set_chat_template failed: %d\n",
                     template_ret);
        std::fclose(events);
        std::fclose(summary);
        std::fclose(meta);
        destroy_handle();
        return 4;
    }

    RKLLMInferParam infer_param{};
    infer_param.mode = RKLLM_INFER_GENERATE;
    infer_param.keep_history = 0;
    infer_param.max_new_tokens = max_new_tokens;

    int overall_ret = 0;
    for (size_t index = 0; index < prompts.size(); ++index) {
        char transcript_name[64];
        std::snprintf(transcript_name, sizeof(transcript_name),
                      "/prompt_%02zu.utf8", index + 1);
        FILE* transcript =
            open_output(output_dir + transcript_name, "wb");

        RunContext context{};
        context.carrier = carrier.c_str();
        context.prompt_id = static_cast<int>(index + 1);
        context.events = events;
        context.transcript = transcript;
        context.run_start_ns = monotonic_raw_ns();

        RKLLMInput input{};
        input.role = "user";
        input.enable_thinking = false;
        input.input_type = RKLLM_INPUT_PROMPT;
        input.prompt_input = prompts[index].c_str();

        std::printf("prompt_start=%zu\n", index + 1);
        std::fflush(stdout);
        const int run_ret =
            rkllm_run(g_handle, &input, &infer_param, &context);
        const uint64_t run_return_ns = monotonic_raw_ns();
        std::fclose(transcript);

        const RKLLMPerfStat perf = context.perf;
        std::fprintf(
            summary,
            "%s,%zu,%llu,%llu,%llu,%llu,%llu,%d,%d,%llu,%llu,%llu,%llu,"
            "%d,%.6f,%d,%.6f,%d,%.6f\n",
            carrier.c_str(), index + 1,
            static_cast<unsigned long long>(context.run_start_ns),
            static_cast<unsigned long long>(run_return_ns),
            static_cast<unsigned long long>(context.first_normal_ns),
            static_cast<unsigned long long>(context.last_normal_ns),
            static_cast<unsigned long long>(context.finish_ns), run_ret,
            context.callback_error ? 1 : 0,
            static_cast<unsigned long long>(context.normal_callbacks),
            static_cast<unsigned long long>(context.waiting_callbacks),
            static_cast<unsigned long long>(context.finish_callbacks),
            static_cast<unsigned long long>(context.output_bytes),
            context.perf_seen ? 1 : 0, perf.prefill_time_ms,
            perf.prefill_tokens, perf.generate_time_ms, perf.generate_tokens,
            perf.memory_usage_mb);
        std::fflush(summary);
        std::printf("prompt_finish=%zu run_ret=%d callbacks=%llu bytes=%llu\n",
                    index + 1, run_ret,
                    static_cast<unsigned long long>(context.normal_callbacks),
                    static_cast<unsigned long long>(context.output_bytes));
        std::fflush(stdout);
        if (run_ret != 0 || context.callback_error ||
            context.finish_callbacks != 1 || context.normal_callbacks == 0 ||
            context.output_bytes == 0) {
            overall_ret = 5;
            break;
        }
    }

    const uint64_t destroy_start_ns = monotonic_raw_ns();
    const int destroy_ret = rkllm_destroy(g_handle);
    g_handle = nullptr;
    const uint64_t destroy_end_ns = monotonic_raw_ns();
    std::fprintf(meta, "destroy_start_monotonic_raw_ns=%llu\n",
                 static_cast<unsigned long long>(destroy_start_ns));
    std::fprintf(meta, "destroy_end_monotonic_raw_ns=%llu\n",
                 static_cast<unsigned long long>(destroy_end_ns));
    std::fprintf(meta, "destroy_return=%d\n", destroy_ret);
    std::fprintf(meta, "runner_return=%d\n", overall_ret);
    std::fclose(events);
    std::fclose(summary);
    std::fclose(meta);
    return overall_ret;
}
