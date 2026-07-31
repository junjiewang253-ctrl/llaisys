#include "llaisys.h"
#include "llaisys/error.h"
#include "llaisys/runtime.h"
#include "llaisys/tensor.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

int main() {
    for (int repeat = 0; repeat < 256; ++repeat) {
        std::array<std::size_t, 3> shape{2, 3, 4};
        auto tensor = tensorCreate(shape.data(), shape.size(), LLAISYS_DTYPE_F32,
                                   LLAISYS_DEVICE_CPU, 0);
        assert(tensor != nullptr);
        std::array<float, 24> values{};
        tensorLoad(tensor, values.data());
        assert(llaisysGetLastErrorCode() == LLAISYS_STATUS_SUCCESS);
        std::array<std::size_t, 3> order{1, 0, 2};
        auto view = tensorPermute(tensor, order.data());
        auto dense = tensorContiguous(view);
        tensorDestroy(tensor);
        tensorDestroy(view);
        tensorDestroy(dense);
        assert(llaisysGetLastErrorCode() == LLAISYS_STATUS_SUCCESS);
    }

    std::thread survivor([] {
        std::array<std::size_t, 1> shape{4096};
        auto tensor = tensorCreate(shape.data(), shape.size(), LLAISYS_DTYPE_U8,
                                   LLAISYS_DEVICE_CPU, 0);
        std::vector<std::uint8_t> values(shape[0], 7);
        tensorLoad(tensor, values.data());
        tensorDestroy(tensor);
    });
    survivor.join();

    auto api = llaisysGetRuntimeAPI(LLAISYS_DEVICE_CPU);
    assert(api != nullptr);
    for (int repeat = 0; repeat < 256; ++repeat) {
        void *memory = api->malloc_device(4096);
        assert(memory != nullptr);
        api->free_device(memory);
    }
    api->free_device(nullptr);
    tensorDestroy(nullptr);
    return 0;
}
