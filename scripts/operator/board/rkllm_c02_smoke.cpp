// C0-5 single-turn RKLLM smoke runner for the frozen Gemma 4 manual template.

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "rkllm.h"

static LLMHandle g_handle = nullptr;
static bool g_run_error = false;

static void destroy_handle() {
    if (g_handle != nullptr) {
        LLMHandle handle = g_handle;
        g_handle = nullptr;
        rkllm_destroy(handle);
    }
}

static void signal_handler(int signal_number) {
    destroy_handle();
    std::_Exit(128 + signal_number);
}

static int callback(RKLLMResult* result, void*, LLMCallState state) {
    if (state == RKLLM_RUN_NORMAL && result != nullptr && result->text != nullptr) {
        std::fputs(result->text, stdout);
        std::fflush(stdout);
    } else if (state == RKLLM_RUN_FINISH) {
        std::fputc('\n', stdout);
        std::fflush(stdout);
    } else if (state == RKLLM_RUN_ERROR) {
        g_run_error = true;
        std::fputs("rkllm run error\n", stderr);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s MODEL_PATH MAX_NEW_TOKENS MAX_CONTEXT_LEN\n", argv[0]);
        return 2;
    }

    const int max_new_tokens = std::atoi(argv[2]);
    const int max_context_len = std::atoi(argv[3]);
    if (max_new_tokens < 1 || max_new_tokens > 64 || max_context_len < 32 ||
        max_context_len > 16384 || max_context_len % 32 != 0) {
        std::fputs("invalid smoke limits\n", stderr);
        return 2;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];
    param.top_k = 1;
    param.top_p = 0.95;
    param.temperature = 0.8;
    param.repeat_penalty = 1.1;
    param.frequency_penalty = 0.0;
    param.presence_penalty = 0.0;
    param.max_new_tokens = max_new_tokens;
    param.max_context_len = max_context_len;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;
    param.extend_param.embed_flash = 1;

    RKLLMCallback rkllm_callback = {};
    rkllm_callback.result_callback = callback;

    std::puts("rkllm init start");
    int ret = rkllm_init(&g_handle, &param, &rkllm_callback);
    if (ret != 0) {
        std::fprintf(stderr, "rkllm init failed: %d\n", ret);
        destroy_handle();
        return 3;
    }
    std::puts("rkllm init success");

    // Exact no-system/no-tool/single-turn rendering of the frozen Gemma 4
    // template. The RKLLM 1.3.0 automatic parser cannot parse its full Jinja.
    ret = rkllm_set_chat_template(
        g_handle, "", "<bos><|turn>user\n", "<turn|>\n<|turn>model\n");
    std::printf("manual_gemma4_template_ret=%d\n", ret);
    if (ret != 0) {
        destroy_handle();
        return 4;
    }

    RKLLMInferParam infer_param = {};
    infer_param.mode = RKLLM_INFER_GENERATE;
    infer_param.keep_history = 0;

    std::string prompt;
    while (std::getline(std::cin, prompt)) {
        if (prompt == "exit") {
            break;
        }
        RKLLMInput input = {};
        input.input_type = RKLLM_INPUT_PROMPT;
        input.role = "user";
        input.prompt_input = const_cast<char*>(prompt.c_str());
        std::fputs("robot: ", stdout);
        std::fflush(stdout);
        g_run_error = false;
        ret = rkllm_run(g_handle, &input, &infer_param, nullptr);
        if (ret != 0 || g_run_error) {
            std::fprintf(stderr, "rkllm_run failed: ret=%d callback_error=%d\n", ret,
                         g_run_error ? 1 : 0);
            destroy_handle();
            return 5;
        }
    }

    destroy_handle();
    return 0;
}
