#include "error.hpp"

namespace {
thread_local llaisysStatus_t last_error_code = LLAISYS_STATUS_SUCCESS;
thread_local std::string last_error_message;
} // namespace

namespace llaisys::capi {

void clearError() noexcept {
    last_error_code = LLAISYS_STATUS_SUCCESS;
    last_error_message.clear();
}

void setError(llaisysStatus_t code, const char *message) noexcept {
    last_error_code = code;
    try {
        last_error_message = message == nullptr ? "" : message;
    } catch (...) {
        last_error_message = "failed to preserve error message";
        last_error_code = LLAISYS_STATUS_INTERNAL_ERROR;
    }
}

} // namespace llaisys::capi

__C llaisysStatus_t llaisysGetLastErrorCode(void) {
    return last_error_code;
}

__C const char *llaisysGetLastErrorMessage(void) {
    return last_error_message.c_str();
}

__C void llaisysClearLastError(void) {
    llaisys::capi::clearError();
}
