#include "tiny_llm/runtime/scheduler.h"

#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/kv_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace tiny_llm {

namespace {

const SchedulerRequestMeta& checked_meta(
    const std::unordered_map<uint64_t, SchedulerRequestMeta>& requests,
    uint64_t internal_id)
{
    const auto it = requests.find(internal_id);
    if (it == requests.end())
    {
        throw std::runtime_error("Scheduler: request metadata is missing for internal_id.");
    }
    return it->second;
}

void push_unique(std::vector<uint64_t>& ids, uint64_t id)
{
    if (std::find(ids.begin(), ids.end(), id) == ids.end())
    {
        ids.push_back(id);
    }
}

int32_t prefill_chunk_tokens(const SchedulerConfig& config, const SchedulerRequestMeta& meta)
{
    const int32_t remaining = meta.prompt_tokens - meta.num_computed;
    if (remaining <= 0)
    {
        return 0;
    }

    const int32_t chunk_limit = (config.max_prefill_tokens_per_step > 0)
        ? config.max_prefill_tokens_per_step
        : 1;
    return std::min(remaining, chunk_limit);
}

template <typename T>
void erase_all(std::deque<T>& q, const T& value)
{
    q.erase(std::remove(q.begin(), q.end(), value), q.end());
}

int32_t prompt_token_count(const CoreSequence& seq)
{
    const int32_t total_tokens = static_cast<int32_t>(seq.token_ids.size());
    return total_tokens - seq.generated_tokens;
}

} // namespace

BlockManager::BlockManager(KVCache* kv)
    : kv_(kv)
{
}

BlockManager::~BlockManager() = default;

BlockManager::BlockManager(int32_t kv_num_layers,
                           int32_t kv_block_size_tokens,
                           size_t kv_num_blocks,
                           size_t kv_block_size_bytes,
                           void* kv_memory_pool)
{
    if (kv_num_layers <= 0)
    {
        throw std::runtime_error("BlockManager: kv_num_layers must be positive.");
    }
    if (kv_block_size_tokens <= 0)
    {
        throw std::runtime_error("BlockManager: kv_block_size_tokens must be positive.");
    }
    if (kv_num_blocks == 0)
    {
        throw std::runtime_error("BlockManager: kv_num_blocks must be positive.");
    }
    if (kv_block_size_bytes == 0)
    {
        throw std::runtime_error("BlockManager: kv_block_size_bytes must be positive.");
    }
    if (kv_memory_pool == nullptr)
    {
        throw std::runtime_error("BlockManager: kv_memory_pool must be non-null.");
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

size_t BlockManager::free_block_count() const
{
    if (kv_ == nullptr)
    {
        return 0;
    }
    return kv_->free_block_count();
}

int32_t BlockManager::num_layers() const
{
    if (kv_ == nullptr)
    {
        return 0;
    }
    return kv_->num_layers();
}

void BlockManager::start_sequence(int32_t core_seq_id) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("BlockManager::start_sequence: kv must be non-null.");
    }
    kv_->start_sequence(core_seq_id);
}

void BlockManager::end_sequence(int32_t core_seq_id) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("BlockManager::end_sequence: kv must be non-null.");
    }
    kv_->end_sequence(core_seq_id);
}

size_t BlockManager::estimate_append_new_blocks(
    int32_t core_seq_id,
    bool kv_started,
    int32_t num_computed) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("BlockManager::estimate_append_new_blocks: kv must be non-null.");
    }

    if (!kv_started)
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

size_t BlockManager::estimate_prefill_new_blocks(
    int32_t core_seq_id,
    bool kv_started,
    int32_t prompt_tokens,
    int32_t num_computed,
    int32_t prefill_tokens) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("BlockManager::estimate_prefill_new_blocks: kv must be non-null.");
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
        if (kv_started)
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

void BlockManager::refresh_block_table(
    int32_t core_seq_id,
    bool kv_started,
    std::vector<int32_t>& block_table) const
{
    if (kv_ == nullptr)
    {
        throw std::runtime_error("BlockManager::refresh_block_table: kv must be non-null.");
    }

    block_table.clear();
    if (!kv_started || kv_->num_layers() <= 0)
    {
        return;
    }

    const std::vector<int32_t>& page_table = kv_->page_table(core_seq_id, 0);
    block_table.insert(block_table.end(), page_table.begin(), page_table.end());
}

void FcfsSchedulerStrategy::sort_waiting(
    std::deque<uint64_t>& waiting_queue,
    const std::unordered_map<uint64_t, SchedulerRequestMeta>& requests) const
{
    std::stable_sort(
        waiting_queue.begin(),
        waiting_queue.end(),
        [&](uint64_t lhs, uint64_t rhs) {
            const auto& left_meta = checked_meta(requests, lhs);
            const auto& right_meta = checked_meta(requests, rhs);
            if (left_meta.arrival_order != right_meta.arrival_order)
            {
                return left_meta.arrival_order < right_meta.arrival_order;
            }
            return left_meta.internal_id < right_meta.internal_id;
        });
}

uint64_t FcfsSchedulerStrategy::select_preempt_candidate(
    const std::deque<uint64_t>& running_queue,
    const std::unordered_map<uint64_t, SchedulerRequestMeta>& requests,
    uint64_t current_request_id) const
{
    (void)running_queue;
    (void)requests;
    // Minimal FCFS implementation: preempt current request on append failure.
    return current_request_id;
}

Scheduler::Scheduler(SchedulerConfig config)
    : config_(config), strategy_(build_strategy(config.policy))
{
}

Scheduler::Scheduler(const EngineArgs& args)
    : config_(args.scheduler_config),
      strategy_(build_strategy(args.scheduler_config.policy))
{
    if (args.kv != nullptr)
    {
        block_manager_ = std::make_unique<BlockManager>(args.kv);
        return;
    }

    block_manager_ = std::make_unique<BlockManager>(
        args.kv_num_layers,
        args.kv_block_size_tokens,
        args.kv_num_blocks,
        args.kv_block_size_bytes,
        args.kv_memory_pool);
}

Scheduler::Scheduler(KVCache* kv, SchedulerConfig config)
    : config_(config),
      strategy_(build_strategy(config.policy)),
      block_manager_(std::make_unique<BlockManager>(kv))
{
}

KVCache* Scheduler::kv_cache() const
{
    if (!block_manager_)
    {
        return nullptr;
    }
    return block_manager_->kv_cache();
}

int32_t Scheduler::assign_core_seq_id()
{
    if (next_core_seq_id_ <= 0)
    {
        throw std::runtime_error("Scheduler::assign_core_seq_id: invalid core sequence id state.");
    }

    if (next_core_seq_id_ == std::numeric_limits<int32_t>::max())
    {
        throw std::runtime_error("Scheduler::assign_core_seq_id: exhausted core sequence id space.");
    }

    const int32_t assigned = next_core_seq_id_;
    ++next_core_seq_id_;
    return assigned;
}

void Scheduler::remove_from_waiting_queue(uint64_t internal_id)
{
    erase_all(waiting_queue_, internal_id);
}

void Scheduler::remove_from_running_queue(uint64_t internal_id)
{
    erase_all(running_queue_, internal_id);
}

void Scheduler::sync_queues()
{
    const auto alive = [&](uint64_t internal_id) {
        const auto it = seqs_.find(internal_id);
        return it != seqs_.end() && !it->second.finished;
    };

    waiting_queue_.erase(
        std::remove_if(
            waiting_queue_.begin(),
            waiting_queue_.end(),
            [&](uint64_t internal_id) { return !alive(internal_id); }),
        waiting_queue_.end());

    running_queue_.erase(
        std::remove_if(
            running_queue_.begin(),
            running_queue_.end(),
            [&](uint64_t internal_id) { return !alive(internal_id); }),
        running_queue_.end());

    std::unordered_set<uint64_t> seen_waiting;
    waiting_queue_.erase(
        std::remove_if(
            waiting_queue_.begin(),
            waiting_queue_.end(),
            [&](uint64_t internal_id) {
                if (seen_waiting.find(internal_id) != seen_waiting.end())
                {
                    return true;
                }
                seen_waiting.insert(internal_id);
                return false;
            }),
        waiting_queue_.end());

    std::unordered_set<uint64_t> seen_running;
    running_queue_.erase(
        std::remove_if(
            running_queue_.begin(),
            running_queue_.end(),
            [&](uint64_t internal_id) {
                if (seen_running.find(internal_id) != seen_running.end())
                {
                    return true;
                }
                if (seen_waiting.find(internal_id) != seen_waiting.end())
                {
                    return true;
                }
                seen_running.insert(internal_id);
                return false;
            }),
        running_queue_.end());
}

std::unordered_map<uint64_t, SchedulerRequestMeta> Scheduler::build_scheduler_meta() const
{
    std::unordered_map<uint64_t, SchedulerRequestMeta> meta;
    meta.reserve(seqs_.size());
    for (const auto& item : seqs_)
    {
        const CoreSequence& seq = item.second;
        if (seq.finished)
        {
            continue;
        }

        SchedulerRequestMeta request_meta;
        request_meta.internal_id = seq.internal_id;
        request_meta.arrival_order = seq.arrival_order;
        request_meta.prompt_tokens = prompt_token_count(seq);
        request_meta.num_computed = seq.num_computed;
        meta[seq.internal_id] = request_meta;
    }
    return meta;
}

void Scheduler::validate_schedule_contracts() const
{
    if (!strategy_)
    {
        throw std::runtime_error("Scheduler::schedule: strategy is not configured.");
    }

    if (!block_manager_)
    {
        throw std::runtime_error("Scheduler::schedule: block_manager must be configured for runtime state.");
    }
}

CoreSequence* Scheduler::find_sequence(uint64_t internal_id)
{
    const auto it = seqs_.find(internal_id);
    if (it == seqs_.end())
    {
        return nullptr;
    }
    return &it->second;
}

void Scheduler::add_request(const EngineCoreRequest& request, int32_t vocab_size)
{
    if (vocab_size <= 0)
    {
        throw std::runtime_error("Scheduler::add_request: vocab size must be positive.");
    }

    if (request.internal_id == 0)
    {
        throw std::runtime_error("Scheduler::add_request: internal_id must be non-zero.");
    }

    if (request.prompt_token_ids.empty())
    {
        throw std::runtime_error("Scheduler::add_request: prompt tokens must be non-empty.");
    }

    for (int32_t token : request.prompt_token_ids)
    {
        if (token < 0 || token >= vocab_size)
        {
            throw std::runtime_error("Scheduler::add_request: prompt token is out of model vocab range.");
        }
    }

    if (seqs_.find(request.internal_id) != seqs_.end())
    {
        throw std::runtime_error("Scheduler::add_request: duplicated internal_id.");
    }

    const int32_t core_seq_id = assign_core_seq_id();
    for (const auto& item : seqs_)
    {
        if (item.second.core_seq_id == core_seq_id)
        {
            throw std::runtime_error("Scheduler::add_request: duplicated core_seq_id.");
        }
    }

    CoreSequence seq;
    seq.internal_id = request.internal_id;
    seq.core_seq_id = core_seq_id;
    seq.arrival_order = next_arrival_order_++;
    seq.token_ids = request.prompt_token_ids;
    seq.sampling_params = request.sampling_params;
    seq.generated_tokens = 0;
    seq.num_computed = 0;
    seq.kv_started = false;
    seq.finished = false;

    seqs_[request.internal_id] = std::move(seq);
    waiting_queue_.push_back(request.internal_id);
    sync_queues();
}

SchedulerOutput Scheduler::schedule()
{
    validate_schedule_contracts();
    sync_queues();

    SchedulerOutput scheduler_output;

    const std::unordered_map<uint64_t, SchedulerRequestMeta> scheduler_meta = build_scheduler_meta();

    size_t available_blocks = block_manager_->free_block_count();
    auto can_append_one_token = [&](uint64_t internal_id) -> bool {
        CoreSequence* seq = find_sequence(internal_id);
        if (seq == nullptr || seq->finished)
        {
            return false;
        }

        const size_t needed = block_manager_->estimate_append_new_blocks(
            seq->core_seq_id,
            seq->kv_started,
            seq->num_computed);
        if (needed == std::numeric_limits<size_t>::max())
        {
            return false;
        }
        if (needed > available_blocks)
        {
            return false;
        }
        available_blocks -= needed;
        return true;
    };

    auto can_allocate_prefill = [&](uint64_t internal_id, int32_t prefill_tokens) -> bool {
        CoreSequence* seq = find_sequence(internal_id);
        if (seq == nullptr || seq->finished)
        {
            return false;
        }

        const int32_t prompt_tokens = prompt_token_count(*seq);
        const size_t needed = block_manager_->estimate_prefill_new_blocks(
            seq->core_seq_id,
            seq->kv_started,
            prompt_tokens,
            seq->num_computed,
            prefill_tokens);
        if (needed == std::numeric_limits<size_t>::max())
        {
            return false;
        }
        if (needed > available_blocks)
        {
            return false;
        }
        available_blocks -= needed;
        return true;
    };

    const SchedulerResult schedule_result = schedule(
        running_queue_,
        waiting_queue_,
        scheduler_meta,
        can_append_one_token,
        can_allocate_prefill);

    scheduler_output.preempted_ids = schedule_result.preempted_ids;

    sync_queues();
    scheduler_output.tasks.reserve(schedule_result.tasks.size());

    for (const ScheduleTask& task : schedule_result.tasks)
    {
        CoreSequence* seq = find_sequence(task.internal_id);
        if (seq == nullptr || seq->finished)
        {
            continue;
        }

        SchedulerTaskDescriptor descriptor;
        descriptor.internal_id = seq->internal_id;
        descriptor.core_seq_id = seq->core_seq_id;
        descriptor.is_prefill = task.is_prefill;

        if (task.is_prefill)
        {
            const int32_t prompt_tokens = prompt_token_count(*seq);
            if (prompt_tokens <= 0 || seq->num_computed >= prompt_tokens)
            {
                continue;
            }

            if (!seq->kv_started)
            {
                block_manager_->start_sequence(seq->core_seq_id);
                seq->kv_started = true;
            }

            descriptor.start_position = seq->num_computed;
            const int32_t target_computed = std::min(prompt_tokens, seq->num_computed + task.num_tokens_to_process);
            descriptor.token_ids.reserve(static_cast<size_t>(target_computed - descriptor.start_position));
            for (int32_t pos = descriptor.start_position; pos < target_computed; ++pos)
            {
                descriptor.token_ids.push_back(seq->token_ids[static_cast<size_t>(pos)]);
            }

            if (descriptor.token_ids.empty())
            {
                continue;
            }
            scheduler_output.tasks.push_back(std::move(descriptor));
            continue;
        }

        const int32_t prompt_tokens = prompt_token_count(*seq);
        if (seq->num_computed < prompt_tokens)
        {
            continue;
        }

        if (!seq->kv_started)
        {
            block_manager_->start_sequence(seq->core_seq_id);
            seq->kv_started = true;
        }

        descriptor.start_position = seq->num_computed;
        if (!seq->token_ids.empty())
        {
            descriptor.token_ids.push_back(seq->token_ids.back());
        }
        scheduler_output.tasks.push_back(std::move(descriptor));
    }

    sync_queues();
    return scheduler_output;
}

std::map<uint64_t, EngineCoreOutput> Scheduler::update_from_output(
    const SchedulerOutput& scheduler_output,
    const ModelOutput& model_output)
{
    validate_schedule_contracts();

    std::map<uint64_t, EngineCoreOutput> results;

    for (uint64_t internal_id : scheduler_output.preempted_ids)
    {
        const std::string error_message = "preempted by FCFS scheduler due to KV capacity.";

        CoreSequence* seq = find_sequence(internal_id);
        if (seq != nullptr)
        {
            seq->finished = true;
        }

        EngineCoreOutput result;
        result.internal_id = internal_id;
        result.sequence = seq;
        result.has_error = true;
        result.error_message = error_message;
        results[internal_id] = std::move(result);
    }

    for (const ModelTaskOutput& task_output : model_output.tasks)
    {
        CoreSequence* seq = find_sequence(task_output.internal_id);
        if (seq == nullptr || seq->finished)
        {
            continue;
        }

        if (task_output.has_error)
        {
            seq->finished = true;

            EngineCoreOutput result;
            result.internal_id = task_output.internal_id;
            result.sequence = seq;
            result.has_error = true;
            result.error_message = task_output.error_message;
            results[task_output.internal_id] = std::move(result);
            continue;
        }

        if (task_output.is_prefill)
        {
            if (task_output.processed_tokens <= 0)
            {
                continue;
            }

            const int32_t prompt_tokens = prompt_token_count(*seq);
            const int32_t target_computed = std::min(prompt_tokens, seq->num_computed + task_output.processed_tokens);
            seq->num_computed = target_computed;
            block_manager_->refresh_block_table(seq->core_seq_id, seq->kv_started, seq->block_table);
            continue;
        }

        if (task_output.processed_tokens <= 0)
        {
            continue;
        }

        seq->token_ids.push_back(task_output.sampled_token_id);
        seq->generated_tokens += task_output.processed_tokens;
        seq->num_computed += task_output.processed_tokens;
        block_manager_->refresh_block_table(seq->core_seq_id, seq->kv_started, seq->block_table);

        EngineCoreOutput result;
        result.internal_id = seq->internal_id;
        result.new_token_id = task_output.sampled_token_id;
        result.generated_tokens = seq->generated_tokens;
        result.sequence = seq;
        results[seq->internal_id] = std::move(result);
    }

    sync_queues();
    return results;
}

void Scheduler::abort_request(uint64_t internal_id)
{
    CoreSequence* seq = find_sequence(internal_id);
    if (seq == nullptr)
    {
        return;
    }

    if (seq->kv_started)
    {
        block_manager_->end_sequence(seq->core_seq_id);
        seq->kv_started = false;
    }
    seq->finished = true;

    seq->block_table.clear();
    remove_from_waiting_queue(internal_id);
    remove_from_running_queue(internal_id);
    seqs_.erase(internal_id);
}

void Scheduler::post_step()
{
    std::vector<uint64_t> finished_ids;
    finished_ids.reserve(seqs_.size());

    for (const auto& item : seqs_)
    {
        if (item.second.finished)
        {
            finished_ids.push_back(item.first);
        }
    }

    for (uint64_t internal_id : finished_ids)
    {
        abort_request(internal_id);
    }
}

SchedulerResult Scheduler::schedule(
    std::deque<uint64_t>& running_queue,
    std::deque<uint64_t>& waiting_queue,
    const std::unordered_map<uint64_t, SchedulerRequestMeta>& requests,
    const std::function<bool(uint64_t)>& can_append_one_token,
    const std::function<bool(uint64_t, int32_t)>& can_allocate_prefill) const
{
    if (!strategy_)
    {
        throw std::runtime_error("Scheduler::schedule: strategy is not configured.");
    }

    SchedulerResult result;

    // Step A: build execution tasks for running requests.
    std::deque<uint64_t> next_running;
    for (uint64_t internal_id : running_queue)
    {
        const SchedulerRequestMeta& meta = checked_meta(requests, internal_id);
        const bool in_prefill = meta.num_computed < meta.prompt_tokens;

        if (in_prefill)
        {
            const int32_t chunk = prefill_chunk_tokens(config_, meta);
            if (chunk <= 0)
            {
                next_running.push_back(internal_id);
                continue;
            }

            if (!can_allocate_prefill(internal_id, chunk))
            {
                if (config_.enable_preemption)
                {
                    const uint64_t victim =
                        strategy_->select_preempt_candidate(running_queue, requests, internal_id);
                    push_unique(result.preempted_ids, victim);
                    if (victim != internal_id)
                    {
                        push_unique(result.preempted_ids, internal_id);
                    }
                    continue;
                }

                next_running.push_back(internal_id);
                continue;
            }

            ScheduleTask task;
            task.internal_id = internal_id;
            task.is_prefill = true;
            task.num_tokens_to_process = chunk;
            result.tasks.push_back(task);
            next_running.push_back(internal_id);
            continue;
        }

        if (!can_append_one_token(internal_id))
        {
            if (config_.enable_preemption)
            {
                const uint64_t victim =
                    strategy_->select_preempt_candidate(running_queue, requests, internal_id);
                push_unique(result.preempted_ids, victim);
                if (victim != internal_id)
                {
                    push_unique(result.preempted_ids, internal_id);
                }
                continue;
            }

            next_running.push_back(internal_id);
            continue;
        }

        ScheduleTask task;
        task.internal_id = internal_id;
        task.is_prefill = false;
        task.num_tokens_to_process = 1;
        result.tasks.push_back(task);
        next_running.push_back(internal_id);
    }
    running_queue = std::move(next_running);

    // Step B: sort waiting by strategy before trying admissions.
    strategy_->sort_waiting(waiting_queue, requests);

    const auto under_running_limit = [&]() -> bool {
        return config_.max_running_requests == 0
            || running_queue.size() < config_.max_running_requests;
    };

    while (!waiting_queue.empty() && under_running_limit())
    {
        const uint64_t internal_id = waiting_queue.front();
        const SchedulerRequestMeta& meta = checked_meta(requests, internal_id);
        const int32_t chunk = prefill_chunk_tokens(config_, meta);
        if (chunk <= 0)
        {
            waiting_queue.pop_front();
            running_queue.push_back(internal_id);
            continue;
        }

        if (!can_allocate_prefill(internal_id, chunk))
        {
            break;
        }

        waiting_queue.pop_front();
        running_queue.push_back(internal_id);

        ScheduleTask task;
        task.internal_id = internal_id;
        task.is_prefill = true;
        task.num_tokens_to_process = chunk;
        result.tasks.push_back(task);
    }

    return result;
}

std::unique_ptr<SchedulerStrategy> Scheduler::build_strategy(SchedulerPolicy policy) const
{
    switch (policy)
    {
        case SchedulerPolicy::kFcfs:
            return std::make_unique<FcfsSchedulerStrategy>();
    }

    throw std::runtime_error("Scheduler: unknown scheduler policy.");
}

} // namespace tiny_llm
