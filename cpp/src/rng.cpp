#include "rng.h"

#include <random>

namespace cudaoplib
{
    uint32_t _global_seed = 42;

    std::mt19937& get_random_engine()
    {
        static std::mt19937 engine(_global_seed); 
        return engine;
    }

    uint32_t get_global_seed() { return _global_seed; }
    void set_global_seed(uint32_t seed) { 
        _global_seed = seed; 
        get_random_engine().seed(seed); 
    }
}; // namespace cudaoplib