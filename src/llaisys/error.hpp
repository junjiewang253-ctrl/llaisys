#pragma once

#include "llaisys/error.h"

#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaisys::capi {

class NotSupportedError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void clearError() noexcept;
void setError(llaisysStatus_t code, const char *message) noexcept;

inline void captureCurrentException() noexcept {
    try {
        throw;
    } catch (const std::invalid_argument &error) {
        setError(LLAISYS_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::out_of_range &error) {
        setError(LLAISYS_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::bad_alloc &error) {
        setError(LLAISYS_STATUS_OUT_OF_MEMORY, error.what());
    } catch (const NotSupportedError &error) {
        setError(LLAISYS_STATUS_NOT_SUPPORTED, error.what());
    } catch (const std::exception &error) {
        setError(LLAISYS_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        setError(LLAISYS_STATUS_INTERNAL_ERROR, "unknown C++ exception");
    }
}

template <typename Result, typename Function>
Result guard(Result sentinel, Function &&function) noexcept {
    clearError();
    try {
        return std::forward<Function>(function)();
    } catch (...) {
        captureCurrentException();
        return sentinel;
    }
}

template <typename Function>
void guardVoid(Function &&function) noexcept {
    clearError();
    try {
        std::forward<Function>(function)();
    } catch (...) {
        captureCurrentException();
    }
}

} // namespace llaisys::capi
