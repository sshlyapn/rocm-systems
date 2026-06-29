#!/bin/bash
# Build the Vulkan inter-kernel gap microbenchmark.
set -e
cd "$(dirname "$0")"
glslc gap.comp -o gap.comp.spv
g++ -O2 -std=c++17 main.cpp -lvulkan -o vk_gap_test
echo "built: $(pwd)/vk_gap_test  (+ gap.comp.spv)"
