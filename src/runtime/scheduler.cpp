#include "tiny_llm/runtime/scheduler.h"

#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/kv_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tiny_llm {

namespace {

int to_output_key(uint64_t request_id)
{
    if (request_id > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("Scheduler::update_from_output: request_id exceeds output key range.");
    }
    return static_cast<int>(request_id);
}

int32_t prompt_token_count(const Request& request)
{
    return static_cast<int32_t>(request.prompt_token_ids.size());
}

Request* find_request(std::map<int64_t, Request>& requests, uint64_t request_id)
{
    const auto it = requests.find(static_cast<int64_t>(request_id));
    if (it == requests.end())
    {
        return nullptr;
    }
    return &it->second;
}

void remove_from_queue(std::deque<uint64_t>& queue, uint64_t request_id)
{
    queue.erase(
        std::remove(queue.begin(), queue.end(), request_id),
        queue.end());
}

void push_to_running_if_absent(std::deque<uint64_t>& running, uint64_t request_id)
{
    if (std::find(running.begin(), running.end(), request_id) == running.end())
    {
        running.push_back(request_id);
    }
}

void push_unique_id(std::vector<uint64_t>& ids, uint64_t id)
{
    if (std::find(ids.begin(), ids.end(), id) == ids.end())
    {
        ids.push_back(id);
    }
}

} // namespace

KVCacheManager::KVCacheManager(KVCache* kv)
    : kv_(kv)
{
}

KVCacheManager::KVCacheManager(int32_t kv_num_layers,
                               int32_t kv_block_size_tokens,
                               size_t kv_num_blocks,
                               size_t kv_block_size_bytes,
                               void* kv_memory_pool)
{
    init_owned(
        kv_num_layers,
        kv_block_size_tokens,
        kv_num_blocks,
        kv_block_size_bytes,
        kv_memory_pool);
}

KVCacheManager::~KVCacheManager() = default;

void KVCacheManager::bind(KVCache* kv)
{
    owned_kv_.reset();
    kv_ = kv;
}

void KVCacheManager::init_owned(int32_t kv_num_layers,
                                int32_t kv_block_size_tokens,
                                size_t kv_num_blocks,
                                size_t kv_block_size_bytes,
                                void* kv_memory_pool)
{
    if (kv_num_layers <= 0)
    {
        throw std::runtime_error("KVCacheManager: kv_num_layers must be positive.");
    }
    if (kv_block_size_tokens <= 0)
    {
        throw std::runtime_error("KVCacheManager: kv_block_size_tokens must be positive.");
    }
    if (kv_num_blocks == 0)
    {
        throw std::runtime_error("KVCacheManager: kv_num_blocks must be positive.");
    }
    if (kv_block_size_bytes == 0)
    {
        throw std::runtime_error("KVCacheManager: kv_block_size_bytes must be positive.");
    }
    if (kv_memory_pool == nullptr)
    {
        throw std::runtime_error("KVCacheManager: kv_memory_pool must be non-null.");
    }

    KVCache::Config kv_cfg;
    kv_cfg.num_layers = kv_num_layers;
    kv_cfg.block_size_tokens = kv_block_size_tokens;

    owned_kv_ = std::make_unique<KVCache>(
        kv_cfg,
        kv_num_blocks,
        kv_block_size_bytes,
        kv_memory_pool);
    kv_ = owned_kv_.get();
}

size_t KVCacheManager::free_block_count() const
{
    if (kv_ == nullptr)
    {
        return 0;
    }
    return kv_->free_block_count();
}

int32_t KVCacheManager::num_layers() const
{
    if (kv_ == nullptr)
    {
        return 0;
    }
    return kv_->num_layers();
}

void KVCacheManager::start_sequence(int32_t core_seq_id) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("KVCacheManager::start_sequence: kv must be non-null.");
    }
    kv_->start_sequence(core_seq_id);
}

void KVCacheManager::end_sequence(int32_t core_seq_id) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("KVCacheManager::end_sequence: kv must be non-null.");
    }
    kv_->end_sequence(core_seq_id);
}

size_t KVCacheManager::estimate_append_new_blocks(
    int32_t core_seq_id,
    bool started,
    int32_t num_computed) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("KVCacheManager::estimate_append_new_blocks: kv must be non-null.");
    }

    if (!started)
    {
        return 0;
    }

    const int32_t next_position = num_computed;
    const int32_t required_blocks = (next_position / kv_->block_size_tokens()) + 1;

    size_t additional_blocks = 0;
    for (int32_t layer_id = 0; layer_id < kv_->num_layers(); ++layer_id)
    {
        size_t current_blocks = 0;
        try
        {
            current_blocks = kv_->page_table(core_seq_id, layer_id).size();
        }
        catch (const std::exception&)
        {
            return std::numeric_limits<size_t>::max();
        }

        if (static_cast<size_t>(required_blocks) > current_blocks)
        {
            additional_blocks += static_cast<size_t>(required_blocks) - current_blocks;
        }
    }

    return additional_blocks;
}

size_t KVCacheManager::estimate_prefill_new_blocks(
    int32_t core_seq_id,
    bool started,
    int32_t prompt_tokens,
    int32_t num_computed,
    int32_t prefill_tokens) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("KVCacheManager::estimate_prefill_new_blocks: kv must be non-null.");
    }

    if (prefill_tokens <= 0)
    {
        return 0;
    }

    if (prompt_tokens <= 0 || num_computed >= prompt_tokens)
    {
        return 0;
    }

    const int32_t target_computed = std::min(prompt_tokens, num_computed + prefill_tokens);
    const int32_t target_position = target_computed - 1;
    if (target_position < 0)
    {
        return 0;
    }

    const int32_t required_blocks = (target_position / kv_->block_size_tokens()) + 1;
    size_t additional_blocks = 0;
    for (int32_t layer_id = 0; layer_id < kv_->num_layers(); ++layer_id)
    {
        size_t current_blocks = 0;
        if (started)
        {
            try
            {
                current_blocks = kv_->page_table(core_seq_id, layer_id).size();
            }
            catch (const std::exception&)
            {
                return std::numeric_limits<size_t>::max();
            }
        }

        if (static_cast<size_t>(required_blocks) > current_blocks)
        {
            additional_blocks += static_cast<size_t>(required_blocks) - current_blocks;
        }
    }

    return additional_blocks;
}

void KVCacheManager::refresh_block_table(
    int32_t core_seq_id,
    bool started,
    std::vector<int32_t>& block_table) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("KVCacheManager::refresh_block_table: kv must be non-null.");
    }

    block_table.clear();
    if (!started || kv_->num_layers() <= 0)
    {
        return;
    }

    const std::vector<int32_t>& page_table = kv_->page_table(core_seq_id, 0);
    block_table.insert(block_table.end(), page_table.begin(), page_table.end());
}

bool KVCacheManager::allocate_slots(
    int32_t core_seq_id,
    bool started,
    int32_t num_computed_tokens,
    int32_t num_new_tokens) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("KVCacheManager::allocate_slots: kv must be non-null.");
    }

    if (num_new_tokens <= 0)
    {
        return true;
    }

    const int32_t target_position = num_computed_tokens + num_new_tokens - 1;
    if (target_position < 0)
    {
        return true;
    }

    const int32_t required_blocks = (target_position / kv_->block_size_tokens()) + 1;
    size_t needed_blocks = 0;
    for (int32_t layer_id = 0; layer_id < kv_->num_layers(); ++layer_id)
    {
        size_t current_blocks = 0;
        if (started)
        {
            try
            {
                current_blocks = kv_->page_table(core_seq_id, layer_id).size();
            }
            catch (const std::exception&)
            {
                return false;
            }
        }

        if (static_cast<size_t>(required_blocks) > current_blocks)
        {
            needed_blocks += static_cast<size_t>(required_blocks) - current_blocks;
        }
    }

    if (needed_blocks > kv_->free_block_count())
    {
        return false;
    }

    try
    {
        if (!started)
        {
            kv_->start_sequence(core_seq_id);
        }

        for (int32_t layer_id = 0; layer_id < kv_->num_layers(); ++layer_id)
        {
            kv_->ensure_capacity(core_seq_id, layer_id, target_position);
        }
    }
    catch (const std::exception&)
    {
        return false;
    }

    return true;
}

Scheduler::Scheduler(SchedulerConfig config)
    : policy(config.policy),
      max_num_scheduled_tokens(
          std::max<int64_t>(1, static_cast<int64_t>(config.max_prefill_tokens_per_step)))
{
}

Scheduler::Scheduler(const EngineArgs& args)
    : Scheduler(args.scheduler_config)
{
    if (args.kv != nullptr)
    {
        kvcache_manager.bind(args.kv);
        return;
    }

    kvcache_manager.init_owned(
        args.kv_num_layers,
        args.kv_block_size_tokens,
        args.kv_num_blocks,
        args.kv_block_size_bytes,
        args.kv_memory_pool);
}

Scheduler::Scheduler(KVCache* kv, SchedulerConfig config)
    : Scheduler(config)
{
    kvcache_manager.bind(kv);
}

void Scheduler::add_request(Request request)
{
    if (request.request_id == 0)
    {
        throw std::runtime_error("Scheduler::add_request: request_id must be non-zero.");
    }
    if (request.prompt_token_ids.empty())
    {
        throw std::runtime_error("Scheduler::add_request: prompt_token_ids must be non-empty.");
    }
    if (request.request_id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
    {
        throw std::runtime_error("Scheduler::add_request: request_id exceeds supported core_seq_id range.");
    }

    for (int32_t token_id : request.prompt_token_ids)
    {
        if (token_id < 0)
        {
            throw std::runtime_error("Scheduler::add_request: token id must be non-negative.");
        }
    }

    const int64_t key = static_cast<int64_t>(request.request_id);
    if (requests.find(key) != requests.end())
    {
        throw std::runtime_error("Scheduler::add_request: duplicated request_id.");
    }

    request.reset_generated_tokens();
    if (!request.has_valid_token_layout())
    {
        throw std::runtime_error("Scheduler::add_request: invalid token layout.");
    }

    request.status = RequestStatus::WAITING;
    request.num_computed = 0;

    requests[key] = request;
    waiting.push_back(request.request_id);
}

int Scheduler::get_num_unfinished_requests()
{
    int unfinished = 0;
    for (const auto& item : requests)
    {
        if (item.second.status != RequestStatus::FINISHED)
        {
            ++unfinished;
        }
    }
    return unfinished;
}

bool Scheduler::has_unfinished_requests()
{
    return get_num_unfinished_requests() > 0;
}

void Scheduler::_preempt_request(Request request)
{
    Request* req = find_request(requests, request.request_id);
    if (req == nullptr || req->status == RequestStatus::FINISHED)
    {
        return;
    }

    const int32_t core_seq_id = static_cast<int32_t>(req->request_id);
    if (req->status == RequestStatus::RUNNING)
    {
        try
        {
            kvcache_manager.end_sequence(core_seq_id);
        }
        catch (const std::exception&)
        {
            // Keep preemption best-effort and preserve scheduler forward progress.
        }
    }

    req->status = RequestStatus::WAITING;
    req->num_computed = 0;

    remove_from_queue(running, req->request_id);
    remove_from_queue(waiting, req->request_id);
    waiting.push_front(req->request_id);
}

SchedulerOutput Scheduler::schedule()
{
    if (kvcache_manager.num_layers() <= 0)
    {
        throw std::runtime_error("Scheduler::schedule: KVCacheManager is not initialized.");
    }

    SchedulerOutput scheduler_output;
    int64_t remaining_token_budget = std::max<int64_t>(1, max_num_scheduled_tokens);
    bool preempted_during_running = false;

    auto try_allocate_with_tail_preempt = [&](Request& req, int32_t num_new_tokens, bool is_running) -> bool {
        if (num_new_tokens <= 0)
        {
            return false;
        }

        const int32_t core_seq_id = static_cast<int32_t>(req.request_id);
        while (true)
        {
            const bool started = req.status == RequestStatus::RUNNING;
            if (kvcache_manager.allocate_slots(core_seq_id, started, req.num_computed, num_new_tokens))
            {
                req.status = RequestStatus::RUNNING;
                return true;
            }
            
            // 从队尾找到第一个合法的 Request，将其 preempt
            Request* victim = nullptr;
            for (auto it = running.rbegin(); it != running.rend(); ++it)
            {
                Request* candidate = find_request(requests, *it);
                if (candidate == nullptr || candidate->status != RequestStatus::RUNNING)
                {
                    continue;
                }
                victim = candidate;
                break;
            }

            if (victim == nullptr)
            {
                return false;
            }

            const uint64_t victim_id = victim->request_id;
            _preempt_request(*victim);
            push_unique_id(scheduler_output.preempted_req_ids, victim_id);
            if (is_running)
            {
                preempted_during_running = true;
            }

            if (victim_id == req.request_id)
            {
                return false;
            }
        }
    };

    const std::deque<uint64_t> running_snapshot = running;
    for (uint64_t request_id : running_snapshot)
    {
        if (remaining_token_budget <= 0)
        {
            break;
        }

        // 检测该 Request 是否已经被调度
        if (scheduler_output.num_scheduled_tokens.find(request_id) != scheduler_output.num_scheduled_tokens.end())
        {
            continue;
        }

        Request* req = find_request(requests, request_id);
        if (req == nullptr || req->status != RequestStatus::RUNNING)
        {
            continue;
        }

        const int32_t core_seq_id = static_cast<int32_t>(request_id);
        const int32_t prompt_tokens = prompt_token_count(*req);

        if (req->num_computed < prompt_tokens)
        {// 处于 prefilling 阶段
            const int32_t remaining_prefill = prompt_tokens - req->num_computed;
            const int32_t chunk = std::min(remaining_prefill, static_cast<int32_t>(remaining_token_budget)); // 本轮调度的 token 数量
            if (!try_allocate_with_tail_preempt(*req, chunk, true))
            {
                continue;
            }

            std::vector<int32_t> block_table;
            kvcache_manager.refresh_block_table(core_seq_id, true, block_table);

            NewRequestData new_req;
            new_req.req_id = request_id;
            new_req.core_seq_id = core_seq_id;
            new_req.prompt_token_ids = req->prompt_token_ids;
            new_req.block_ids = std::move(block_table);
            new_req.num_computed_tokens = req->num_computed;
            new_req.sampling_params = req->sampling_params;

            scheduler_output.num_scheduled_tokens[new_req.req_id] = chunk;
            scheduler_output.scheduled_new_reqs.push_back(std::move(new_req));
            remaining_token_budget -= chunk;
            continue;
        }

        if (req->_all_token_ids.empty())
        {
            req->status = RequestStatus::FINISHED;
            continue;
        }

        if (!try_allocate_with_tail_preempt(*req, 1, true))
        {
            continue;
        }

        scheduler_output.scheduled_cached_reqs.req_ids.push_back(request_id);
        scheduler_output.scheduled_cached_reqs.core_seq_ids.push_back(core_seq_id);
        scheduler_output.scheduled_cached_reqs.input_token_ids.push_back(req->_all_token_ids.back());
        scheduler_output.scheduled_cached_reqs.num_computed_tokens.push_back(req->num_computed);
        scheduler_output.scheduled_cached_reqs.new_block_ids.push_back(std::nullopt);
        scheduler_output.num_scheduled_tokens[request_id] = 1;
        remaining_token_budget -= 1;
    }

    if (!preempted_during_running)
    {
        const std::deque<uint64_t> waiting_snapshot = waiting;
        for (uint64_t request_id : waiting_snapshot)
        {
            if (remaining_token_budget <= 0)
            {
                break;
            }

            if (scheduler_output.num_scheduled_tokens.find(request_id) != scheduler_output.num_scheduled_tokens.end())
            {
                continue;
            }

            Request* req = find_request(requests, request_id);
            if (req == nullptr || req->status == RequestStatus::FINISHED)
            {
                continue;
            }

            const int32_t core_seq_id = static_cast<int32_t>(request_id);
            const int32_t prompt_tokens = prompt_token_count(*req);
            const int32_t remaining_prefill = prompt_tokens - req->num_computed;

            if (remaining_prefill <= 0)
            {
                if (req->_all_token_ids.empty())
                {
                    req->status = RequestStatus::FINISHED;
                    continue;
                }

                if (!try_allocate_with_tail_preempt(*req, 1, false))
                {
                    break;
                }

                scheduler_output.scheduled_cached_reqs.req_ids.push_back(request_id);
                scheduler_output.scheduled_cached_reqs.core_seq_ids.push_back(core_seq_id);
                scheduler_output.scheduled_cached_reqs.input_token_ids.push_back(req->_all_token_ids.back());
                scheduler_output.scheduled_cached_reqs.num_computed_tokens.push_back(req->num_computed);
                scheduler_output.scheduled_cached_reqs.new_block_ids.push_back(std::nullopt);
                scheduler_output.num_scheduled_tokens[request_id] = 1;
                remaining_token_budget -= 1;
                continue;
            }

            const int32_t chunk = std::min(remaining_prefill, static_cast<int32_t>(remaining_token_budget));
            if (!try_allocate_with_tail_preempt(*req, chunk, false))
            {
                break;
            }

            std::vector<int32_t> block_table;
            kvcache_manager.refresh_block_table(core_seq_id, true, block_table);

            NewRequestData new_req;
            new_req.req_id = request_id;
            new_req.core_seq_id = core_seq_id;
            new_req.prompt_token_ids = req->prompt_token_ids;
            new_req.block_ids = std::move(block_table);
            new_req.num_computed_tokens = req->num_computed;
            new_req.sampling_params = req->sampling_params;

            scheduler_output.num_scheduled_tokens[new_req.req_id] = chunk;
            scheduler_output.scheduled_new_reqs.push_back(std::move(new_req));
            remaining_token_budget -= chunk;
        }
    }

    for (const auto& item : requests)
    {
        if (item.second.status == RequestStatus::FINISHED)
        {
            scheduler_output.finished_req_ids.push_back(item.second.request_id);
        }
    }

    for (const auto& item : scheduler_output.num_scheduled_tokens)
    {
        scheduler_output.total_num_scheduled_tokens += item.second;
    }

    return scheduler_output;
}

std::map<int, EngineCoreOutput> Scheduler::update_from_output(
    SchedulerOutput scheduler_output,
    ModelRunnerOutput model_runner_output)
{
    std::map<int, EngineCoreOutput> results;

    auto mark_running = [&](Request& req) {
        req.status = RequestStatus::RUNNING;
        remove_from_queue(waiting, req.request_id);
        push_to_running_if_absent(running, req.request_id);
    };

    auto cleanup_finished = [&]() {
        for (auto it = requests.begin(); it != requests.end();)
        {
            Request& req = it->second;
            if (req.status != RequestStatus::FINISHED)
            {
                ++it;
                continue;
            }

            remove_from_queue(waiting, req.request_id);
            remove_from_queue(running, req.request_id);
            it = requests.erase(it);
        }
    };

    auto finish_request = [&](Request& req) {
        if (req.status == RequestStatus::RUNNING)
        {
            try
            {
                kvcache_manager.end_sequence(static_cast<int32_t>(req.request_id));
            }
            catch (const std::exception&)
            {
                // Keep finish cleanup best-effort.
            }
        }

        req.status = RequestStatus::FINISHED;
    };

    cleanup_finished();

    for (const ModelTaskOutput& task_output : model_runner_output.tasks)
    {
        Request* req = find_request(requests, task_output.internal_id);
        if (req == nullptr || req->status == RequestStatus::FINISHED)
        {
            continue;
        }

        if (task_output.has_error)
        {
            finish_request(*req);

            EngineCoreOutput result;
            result.internal_id = task_output.internal_id;
            result.sequence = nullptr;
            result.has_error = true;
            result.error_message = task_output.error_message;
            results[to_output_key(task_output.internal_id)] = std::move(result);
            continue;
        }

        if (task_output.is_prefill)
        {
            if (task_output.processed_tokens > 0)
            {
                mark_running(*req);
                const int32_t prompt_tokens = prompt_token_count(*req);
                const int32_t target_computed = std::min(prompt_tokens, req->num_computed + task_output.processed_tokens);
                req->num_computed = target_computed;
            }
            continue;
        }

        if (task_output.processed_tokens <= 0)
        {
            continue;
        }

        mark_running(*req);
        req->_all_token_ids.push_back(task_output.sampled_token_id);
        req->num_computed += task_output.processed_tokens;

        EngineCoreOutput result;
        result.internal_id = req->request_id;
        result.new_token_id = task_output.sampled_token_id;
        result.generated_tokens = req->generated_tokens();
        result.sequence = nullptr;
        results[to_output_key(req->request_id)] = std::move(result);

        const bool stop_by_token =
            std::find(
                req->sampling_params.stop_token_ids.begin(),
                req->sampling_params.stop_token_ids.end(),
                task_output.sampled_token_id)
            != req->sampling_params.stop_token_ids.end();
        const bool stop_by_length =
            req->generated_tokens() >= req->sampling_params.max_tokens;
        if (stop_by_token || stop_by_length)
        {
            finish_request(*req);
        }
    }

    cleanup_finished();
    return results;
}

} // namespace tiny_llm
