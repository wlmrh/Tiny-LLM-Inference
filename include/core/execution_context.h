#include <cuda_runtime.h>

namespace tiny_llm {
    class StackAllocator;
    class KVCacheManager;

    class ExecutionContext {
    public:
        cudaStream_t stream;
        StackAllocator* workspace;
        KVCacheManager* kv_manager;

        void begin_step() {
            // Reset workspace logic here
        }
    };
}