#include "tiny_llm/models/mini_llama.h"

namespace tiny_llm {

void MiniLLaMA::forward_step(const Tensor& input_ids,
							 const Tensor& positions,
							 Tensor& logits,
							 ExecutionContext& ctx) {
	(void)input_ids;
	(void)positions;
	(void)logits;
	(void)ctx;
}

} // namespace tiny_llm
