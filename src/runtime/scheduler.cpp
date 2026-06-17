#include "tiny_llm/runtime/scheduler.h"

#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/kv_cache_manager.h"
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

int32_t context_token_count(const Request& request)
{
    return static_cast<int32_t>(request._all_token_ids.size());
}

int32_t generated_token_count(const Request& request)
{
    const size_t prompt_tokens = request.prompt_token_ids.size();
    if (request._all_token_ids.size() < prompt_tokens)
    {
        return 0;
    }
    return static_cast<int32_t>(request._all_token_ids.size() - prompt_tokens);
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

} // namespace

KVCacheManager::KVCacheManager(KVCache* kv)
    : kv_(kv)
{
    if (kv_ != nullptr)
    {
        kv_->parallel_config().validate();
    }
}

KVCacheManager::~KVCacheManager() = default;

KVCacheManager::KVCacheManager(KVCacheManager&&) noexcept = default;

KVCacheManager& KVCacheManager::operator=(KVCacheManager&&) noexcept = default;

void KVCacheManager::bind(KVCache* kv)
{
    if (kv != nullptr)
    {
        kv->parallel_config().validate();
    }
    owned_kv_.reset();
    kv_ = kv;
}

void KVCacheManager::init_owned(int32_t kv_num_layers,
                                int32_t kv_block_size_tokens,
                                size_t kv_num_blocks,
                                size_t kv_block_size_bytes,
                                void* kv_memory_pool,
                                ParallelConfig parallel_config)
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
    parallel_config.validate();

    KVCache::Config kv_cfg;
    kv_cfg.num_layers = kv_num_layers;
    kv_cfg.block_size_tokens = kv_block_size_tokens;

    owned_kv_ = std::make_unique<KVCache>(
        kv_cfg,
        kv_num_blocks,
        kv_block_size_bytes,
        kv_memory_pool,
        parallel_config);
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

void KVCacheManager::refresh_block_tables(
    int32_t core_seq_id,
    bool started,
    std::vector<std::vector<int32_t>>& block_tables) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("KVCacheManager::refresh_block_tables: kv must be non-null.");
    }

    block_tables.clear();
    if (!started || kv_->num_layers() <= 0)
    {
        return;
    }

    block_tables.resize(static_cast<size_t>(kv_->num_layers()));
    for (int32_t layer_id = 0; layer_id < kv_->num_layers(); ++layer_id)
    {
        // Collect physical blocks allocated for each layer of this sequence.
        const std::vector<int32_t>& page_table = kv_->page_table(core_seq_id, layer_id);
        block_tables[static_cast<size_t>(layer_id)].insert(
            block_tables[static_cast<size_t>(layer_id)].end(),
            page_table.begin(),
            page_table.end());
    }
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
    // blocks that need to be allocated to satisfy the current request
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
            kv_->start_sequence(core_seq_id); // register the new seq to the kvcache
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
    : kvcache_manager(std::make_unique<KVCacheManager>()),
      max_num_scheduled_tokens(
          std::max<int64_t>(1, static_cast<int64_t>(config.max_prefill_tokens_per_step))),
      max_running_requests(config.max_running_requests),
      enable_preemption(config.enable_preemption)
{
}

Scheduler::Scheduler(const EngineArgs& args)
    : Scheduler(args.scheduler_config)
{
    args.parallel_config.validate();
    if (args.kv != nullptr)
    {
        if (args.kv->parallel_config() != args.parallel_config)
        {
            throw std::runtime_error("Scheduler: KV cache device does not match EngineArgs parallel_config.");
        }
        kvcache_manager->bind(args.kv);
        return;
    }

    kvcache_manager->init_owned(
        args.kv_num_layers,
        args.kv_block_size_tokens,
        args.kv_num_blocks,
        args.kv_block_size_bytes,
        args.kv_memory_pool,
        args.parallel_config);
}

Scheduler::Scheduler(KVCache* kv, SchedulerConfig config)
    : Scheduler(config)
{
    kvcache_manager->bind(kv);
}

Scheduler::~Scheduler() = default;

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

    request._all_token_ids = request.prompt_token_ids;
    request.status = RequestStatus::WAITING;
    request.num_computed = 0;

    requests[key] = request;
    waiting.push_back(request.request_id);
}

bool Scheduler::has_unfinished_requests() const
{
    for (const auto& item : requests)
    {
        if (item.second.status != RequestStatus::FINISHED)
        {
            return true;
        }
    }
    return false;
}

KVCache* Scheduler::kv_cache() const
{
    return kvcache_manager->kv_cache();
}

void Scheduler::preempt_request(uint64_t request_id)
{
    Request* req = find_request(requests, request_id);
    if (req == nullptr || req->status == RequestStatus::FINISHED)
    {
        return;
    }

    const int32_t core_seq_id = static_cast<int32_t>(req->request_id);
    if (req->status == RequestStatus::RUNNING)
    {
        try
        {
            kvcache_manager->end_sequence(core_seq_id);
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
    if (kvcache_manager->num_layers() <= 0)
    {
        throw std::runtime_error("Scheduler::schedule: KVCacheManager is not initialized.");
    }

    SchedulerOutput scheduler_output;
    int64_t remaining_token_budget = std::max<int64_t>(1, max_num_scheduled_tokens);
    bool preempted_during_running = false;
    auto count_running = [&]() -> size_t {
        size_t count = 0;
        for (const auto& item : requests)
        {
            if (item.second.status == RequestStatus::RUNNING)
            {
                ++count;
            }
        }
        return count;
    };
    auto can_admit_waiting = [&]() -> bool {
        return max_running_requests == 0 || count_running() < max_running_requests;
    };
    auto count_prefill_candidates = [&](const std::deque<uint64_t>& snapshot,
                                        size_t start_index,
                                        bool waiting_phase) -> int32_t {
        int32_t count = 0;
        size_t admitted_running = count_running();
        for (size_t index = start_index; index < snapshot.size(); ++index)
        {
            const uint64_t candidate_id = snapshot[index];
            if (scheduler_output.num_scheduled_tokens.find(candidate_id)
                != scheduler_output.num_scheduled_tokens.end())
            {
                continue;
            }
            Request* candidate = find_request(requests, candidate_id);
            if (candidate == nullptr || candidate->status == RequestStatus::FINISHED)
            {
                continue;
            }
            if (waiting_phase)
            {
                if (candidate->status == RequestStatus::WAITING)
                {
                    if (max_running_requests != 0 && admitted_running >= max_running_requests)
                    {
                        break;
                    }
                    ++admitted_running;
                }
            }
            else if (candidate->status != RequestStatus::RUNNING)
            {
                continue;
            }
            if (candidate->num_computed < context_token_count(*candidate))
            {
                ++count;
            }
        }
        return std::max<int32_t>(1, count);
    };
    auto fair_prefill_chunk = [&](int32_t remaining_prefill, int32_t candidate_count) -> int32_t {
        const int64_t fair_budget =
            (remaining_token_budget + static_cast<int64_t>(candidate_count) - 1)
            / static_cast<int64_t>(candidate_count);
        return std::min<int32_t>(
            remaining_prefill,
            static_cast<int32_t>(std::max<int64_t>(1, fair_budget)));
    };
    // try to allocate `num_new_tokens` for req, whose current state is is_running.
    auto try_allocate_with_tail_preempt = [&](Request& req, int32_t num_new_tokens, bool is_running) -> bool {
        if (num_new_tokens <= 0)
        {
            return false;
        }

        const int32_t core_seq_id = static_cast<int32_t>(req.request_id);
        while (true)
        {
            const bool started = req.status == RequestStatus::RUNNING;
            if (kvcache_manager->allocate_slots(core_seq_id, started, req.num_computed, num_new_tokens))
            {
                req.status = RequestStatus::RUNNING;
                return true;
            }
            if (!enable_preemption)
            {
                return false;
            }
            // No enough Space for a the current request
            // Find a legal Request from the back of the queue and preempt it
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
            preempt_request(victim_id);
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
    for (size_t running_index = 0; running_index < running_snapshot.size(); ++running_index)
    {
        const uint64_t request_id = running_snapshot[running_index];
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
        const int32_t context_tokens = context_token_count(*req);

        if (req->num_computed < context_tokens)
        {// 处于 prefilling 阶段
            const int32_t remaining_prefill = context_tokens - req->num_computed;
            const int32_t chunk = fair_prefill_chunk(
                remaining_prefill,
                count_prefill_candidates(running_snapshot, running_index, false));
            if (!try_allocate_with_tail_preempt(*req, chunk, true))
            {
                continue;
            }

            std::vector<std::vector<int32_t>> block_tables;
            kvcache_manager->refresh_block_tables(core_seq_id, true, block_tables);

            RequestData req_data;
            req_data.req_id = request_id;
            req_data.num_computed_tokens = req->num_computed;
            req_data.prompt_token_count = static_cast<int32_t>(req->prompt_token_ids.size());
            req_data.sampling_params = req->sampling_params;
            req_data.all_token_ids = req->_all_token_ids;
            req_data.block_tables = std::move(block_tables);
            req_data.new_token_ids.reserve(static_cast<size_t>(chunk));
            for (int32_t i = 0; i < chunk; ++i)
            {
                const int32_t prompt_index = req->num_computed + i;
                req_data.new_token_ids.push_back(req->_all_token_ids[static_cast<size_t>(prompt_index)]);
            }

            scheduler_output.num_scheduled_tokens[req_data.req_id] = chunk;
            scheduler_output.scheduled_reqs.push_back(std::move(req_data));
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

        std::vector<std::vector<int32_t>> block_tables;
        kvcache_manager->refresh_block_tables(core_seq_id, true, block_tables);

        RequestData req_data;
        req_data.req_id = request_id;
        req_data.num_computed_tokens = req->num_computed;
        req_data.prompt_token_count = static_cast<int32_t>(req->prompt_token_ids.size());
        req_data.sampling_params = req->sampling_params;
        req_data.all_token_ids = req->_all_token_ids;
        req_data.block_tables = std::move(block_tables);
        req_data.new_token_ids.push_back(req->_all_token_ids.back());

        scheduler_output.num_scheduled_tokens[request_id] = 1;
        scheduler_output.scheduled_reqs.push_back(std::move(req_data));
        remaining_token_budget -= 1;
    }

    if (!preempted_during_running)
    {
        const std::deque<uint64_t> waiting_snapshot = waiting;
        for (size_t waiting_index = 0; waiting_index < waiting_snapshot.size(); ++waiting_index)
        {
            const uint64_t request_id = waiting_snapshot[waiting_index];
            if (remaining_token_budget <= 0)
            {
                break;
            }
            // If this request has been scheduled before, skip it.
            if (scheduler_output.num_scheduled_tokens.find(request_id) != scheduler_output.num_scheduled_tokens.end())
            {
                continue;
            }

            Request* req = find_request(requests, request_id);
            if (req == nullptr || req->status == RequestStatus::FINISHED)
            {
                continue;
            }
            if (req->status == RequestStatus::WAITING && !can_admit_waiting())
            {
                break;
            }

            const int32_t core_seq_id = static_cast<int32_t>(request_id);
            const int32_t context_tokens = context_token_count(*req);
            const int32_t remaining_prefill = context_tokens - req->num_computed;

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

                std::vector<std::vector<int32_t>> block_tables;
                kvcache_manager->refresh_block_tables(core_seq_id, true, block_tables);

                RequestData req_data;
                req_data.req_id = request_id;
                req_data.num_computed_tokens = req->num_computed;
                req_data.prompt_token_count = static_cast<int32_t>(req->prompt_token_ids.size());
                req_data.sampling_params = req->sampling_params;
                req_data.all_token_ids = req->_all_token_ids;
                req_data.block_tables = std::move(block_tables);
                req_data.new_token_ids.push_back(req->_all_token_ids.back());

                scheduler_output.num_scheduled_tokens[request_id] = 1;
                scheduler_output.scheduled_reqs.push_back(std::move(req_data));
                remaining_token_budget -= 1;
                continue;
            }
            // 本轮对该请求调度的 token 数量
            const int32_t chunk = fair_prefill_chunk(
                remaining_prefill,
                count_prefill_candidates(waiting_snapshot, waiting_index, true));
            if (!try_allocate_with_tail_preempt(*req, chunk, false))
            {
                break;
            }

            std::vector<std::vector<int32_t>> block_tables;
            kvcache_manager->refresh_block_tables(core_seq_id, true, block_tables);

            RequestData req_data;
            req_data.req_id = request_id;
            req_data.num_computed_tokens = req->num_computed;
            req_data.prompt_token_count = static_cast<int32_t>(req->prompt_token_ids.size());
            req_data.sampling_params = req->sampling_params;
            req_data.all_token_ids = req->_all_token_ids;
            req_data.block_tables = std::move(block_tables);
            req_data.new_token_ids.reserve(static_cast<size_t>(chunk));
            for (int32_t i = 0; i < chunk; ++i)
            {
                const int32_t prompt_index = req->num_computed + i;
                req_data.new_token_ids.push_back(req->_all_token_ids[static_cast<size_t>(prompt_index)]);
            }

            scheduler_output.num_scheduled_tokens[req_data.req_id] = chunk;
            scheduler_output.scheduled_reqs.push_back(std::move(req_data));
            remaining_token_budget -= chunk;
        }
    }

    for (const auto& item : scheduler_output.num_scheduled_tokens)
    {
        scheduler_output.total_num_scheduled_tokens += item.second;
    }

    return scheduler_output;
}

std::map<int, EngineCoreOutput> Scheduler::update_from_output(
    const SchedulerOutput& scheduler_output,
    const ModelRunnerOutput& model_runner_output)
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
                kvcache_manager->end_sequence(static_cast<int32_t>(req.request_id));
            }
            catch (const std::exception&)
            {
                // Keep finish cleanup best-effort.
            }
        }

        req.status = RequestStatus::FINISHED;
    };

    cleanup_finished();

    for (const RequestData& req_data : scheduler_output.scheduled_reqs)
    {
        Request* req = find_request(requests, req_data.req_id);
        if (req == nullptr || req->status == RequestStatus::FINISHED)
        {
            continue;
        }

        const auto scheduled_it = scheduler_output.num_scheduled_tokens.find(req_data.req_id);
        if (scheduled_it == scheduler_output.num_scheduled_tokens.end())
        {
            continue;
        }

        const int32_t processed_tokens = scheduled_it->second;
        if (processed_tokens <= 0)
        {
            continue;
        }

        const int32_t context_tokens = context_token_count(*req);
        mark_running(*req);

        const bool was_prefilling = req->num_computed < context_tokens;
        if (was_prefilling)
        {
            const int32_t target_computed = std::min(context_tokens, req->num_computed + processed_tokens);
            req->num_computed = target_computed;
            if (req->num_computed < context_tokens)
            {
                continue;
            }
        }

        const auto index_it = model_runner_output.req_id_to_index.find(req_data.req_id);
        if (index_it == model_runner_output.req_id_to_index.end())
        {
            continue;
        }

        const int32_t index = index_it->second;
        if (index < 0 || static_cast<size_t>(index) >= model_runner_output.sampled_token_ids.size())
        {
            continue;
        }

        const int32_t sampled_token_id = model_runner_output.sampled_token_ids[static_cast<size_t>(index)];
        if (sampled_token_id < 0)
        {
            continue;
        }

        req->_all_token_ids.push_back(sampled_token_id);
        if (!was_prefilling)
        {
            req->num_computed += processed_tokens;
        }

        EngineCoreOutput result;
        result.internal_id = req->request_id;
        result.new_token_id = sampled_token_id;
        results[to_output_key(req->request_id)] = std::move(result);

        const bool stop_by_token =
            std::find(
                req->sampling_params.stop_token_ids.begin(),
                req->sampling_params.stop_token_ids.end(),
                sampled_token_id)
            != req->sampling_params.stop_token_ids.end();
        const bool stop_by_length =
            generated_token_count(*req) >= req->sampling_params.max_tokens;
        if (stop_by_token || stop_by_length)
        {
            finish_request(*req);
        }
    }

    cleanup_finished();
    return results;
}

} // namespace tiny_llm
