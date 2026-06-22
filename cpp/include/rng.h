#pragma once

#include "tensor.h"
#include <random>

namespace cudaoplib
{
    std::mt19937& get_random_engine();
    uint32_t get_global_seed();
    void set_global_seed(uint32_t seed);
}; // namespace cudaoplib