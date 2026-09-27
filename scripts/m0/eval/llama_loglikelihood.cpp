#include "ggml-backend.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

std::vector<llama_token> tokenize(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special) {
    int32_t count = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0,
        add_special, true);
    if (count == 0) {
        return {};
    }

    const int32_t required = count < 0 ? -count : count;
    std::vector<llama_token> tokens(required);
    count = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
        static_cast<int32_t>(tokens.size()), add_special, true);
    if (count < 0) {
        throw std::runtime_error("tokenization buffer was unexpectedly too small");
    }
    tokens.resize(count);
    return tokens;
}

json score(
        llama_context * ctx,
        const llama_vocab * vocab,
        const std::vector<llama_token> & context,
        const std::vector<llama_token> & continuation) {
    if (context.empty()) {
        throw std::runtime_error("context_tokens must not be empty");
    }
    if (continuation.empty()) {
        throw std::runtime_error("continuation_tokens must not be empty");
    }

    std::vector<llama_token> tokens = context;
    tokens.insert(tokens.end(), continuation.begin(), continuation.end());
    if (tokens.size() > llama_n_batch(ctx)) {
        throw std::runtime_error(
            "request has " + std::to_string(tokens.size()) +
            " tokens, exceeding scorer n_batch=" +
            std::to_string(llama_n_batch(ctx)));
    }
    if (tokens.size() > llama_n_ctx(ctx)) {
        throw std::runtime_error(
            "request exceeds scorer context window " +
            std::to_string(llama_n_ctx(ctx)));
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(
        static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());

    const int32_t first_output = static_cast<int32_t>(context.size()) - 1;
    const int32_t last_output = static_cast<int32_t>(tokens.size()) - 2;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i >= first_output && i <= last_output;
    }

    const int decode_status = llama_decode(ctx, batch);
    if (decode_status != 0) {
        llama_batch_free(batch);
        throw std::runtime_error(
            "llama_decode failed with status " + std::to_string(decode_status));
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    double total_logprob = 0.0;
    bool is_greedy = true;

    for (int32_t j = 0; j < static_cast<int32_t>(continuation.size()); ++j) {
        const int32_t input_index = first_output + j;
        const float * logits = llama_get_logits_ith(ctx, input_index);
        if (logits == nullptr) {
            llama_batch_free(batch);
            throw std::runtime_error("requested logits row was not produced");
        }

        const llama_token target = continuation[j];
        if (target < 0 || target >= n_vocab) {
            llama_batch_free(batch);
            throw std::runtime_error("continuation contains an invalid token id");
        }

        const float max_logit = *std::max_element(logits, logits + n_vocab);
        double exp_sum = 0.0;
        for (int32_t token = 0; token < n_vocab; ++token) {
            exp_sum += std::exp(static_cast<double>(logits[token] - max_logit));
        }
        total_logprob += static_cast<double>(logits[target] - max_logit) -
                         std::log(exp_sum);

        const llama_token greedy = static_cast<llama_token>(
            std::max_element(logits, logits + n_vocab) - logits);
        is_greedy = is_greedy && greedy == target;
    }

    llama_batch_free(batch);
    return {
        {"loglikelihood", total_logprob},
        {"is_greedy", is_greedy},
    };
}

json score_options(
        llama_context * ctx,
        const llama_vocab * vocab,
        const std::vector<llama_token> & context,
        const std::vector<std::vector<llama_token>> & continuations) {
    if (context.empty()) {
        throw std::runtime_error("context_tokens must not be empty");
    }
    if (continuations.empty()) {
        throw std::runtime_error("continuations must not be empty");
    }
    for (const auto & continuation : continuations) {
        if (continuation.size() != 1) {
            throw std::runtime_error(
                "score_options only accepts one-token continuations");
        }
    }
    if (context.size() > llama_n_batch(ctx)) {
        throw std::runtime_error(
            "request has " + std::to_string(context.size()) +
            " context tokens, exceeding scorer n_batch=" +
            std::to_string(llama_n_batch(ctx)));
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(
        static_cast<int32_t>(context.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(context.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = context[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == batch.n_tokens - 1;
    }
    const int decode_status = llama_decode(ctx, batch);
    if (decode_status != 0) {
        llama_batch_free(batch);
        throw std::runtime_error(
            "llama_decode failed with status " + std::to_string(decode_status));
    }

    const float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    if (logits == nullptr) {
        llama_batch_free(batch);
        throw std::runtime_error("requested logits row was not produced");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float max_logit = *std::max_element(logits, logits + n_vocab);
    double exp_sum = 0.0;
    for (int32_t token = 0; token < n_vocab; ++token) {
        exp_sum += std::exp(static_cast<double>(logits[token] - max_logit));
    }
    const double log_normalizer = static_cast<double>(max_logit) + std::log(exp_sum);
    const llama_token greedy = static_cast<llama_token>(
        std::max_element(logits, logits + n_vocab) - logits);

    json scores = json::array();
    for (const auto & continuation : continuations) {
        const llama_token target = continuation[0];
        if (target < 0 || target >= n_vocab) {
            llama_batch_free(batch);
            throw std::runtime_error("continuation contains an invalid token id");
        }
        scores.push_back({
            {"loglikelihood", static_cast<double>(logits[target]) - log_normalizer},
            {"is_greedy", target == greedy},
        });
    }
    llama_batch_free(batch);
    return {{"scores", scores}};
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " MODEL.gguf\n";
        return 2;
    }

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::cerr << "failed to load model: " << argv[1] << "\n";
        llama_backend_free();
        return 1;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 131072;
    context_params.n_batch = 16384;
    context_params.n_ubatch = 512;
    context_params.n_seq_max = 1;
    context_params.n_outputs_max = 64;
    context_params.type_k = GGML_TYPE_Q8_0;
    context_params.type_v = GGML_TYPE_Q8_0;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    llama_context * ctx = llama_init_from_model(model, context_params);
    if (ctx == nullptr) {
        std::cerr << "failed to create llama context\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string line;
    while (std::getline(std::cin, line)) {
        json response;
        try {
            const json request = json::parse(line);
            const std::string op = request.value("op", "score");
            response["id"] = request.value("id", 0);
            response["ok"] = true;

            if (op == "ping") {
                response["n_ctx"] = llama_n_ctx(ctx);
                response["n_batch"] = llama_n_batch(ctx);
                response["n_vocab"] = llama_vocab_n_tokens(vocab);
            } else if (op == "tokenize") {
                response["tokens"] = tokenize(
                    vocab,
                    request.at("text").get<std::string>(),
                    request.value("add_special", false));
            } else if (op == "score") {
                const auto context =
                    request.at("context_tokens").get<std::vector<llama_token>>();
                const auto continuation =
                    request.at("continuation_tokens").get<std::vector<llama_token>>();
                response.update(score(ctx, vocab, context, continuation));
            } else if (op == "score_options") {
                const auto context =
                    request.at("context_tokens").get<std::vector<llama_token>>();
                const auto continuations = request.at("continuations").get<
                    std::vector<std::vector<llama_token>>>();
                response.update(score_options(ctx, vocab, context, continuations));
            } else {
                throw std::runtime_error("unknown operation: " + op);
            }
        } catch (const std::exception & error) {
            response = {{"ok", false}, {"error", error.what()}};
        }

        std::cout << response.dump() << '\n' << std::flush;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
