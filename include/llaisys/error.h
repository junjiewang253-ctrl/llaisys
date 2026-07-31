#ifndef LLAISYS_ERROR_H
#define LLAISYS_ERROR_H

#include "../llaisys.h"

__C {
    typedef enum {
        LLAISYS_STATUS_SUCCESS = 0,
        LLAISYS_STATUS_INVALID_ARGUMENT = 1,
        LLAISYS_STATUS_OUT_OF_MEMORY = 2,
        LLAISYS_STATUS_NOT_SUPPORTED = 3,
        LLAISYS_STATUS_INTERNAL_ERROR = 4
    } llaisysStatus_t;

    __export llaisysStatus_t llaisysGetLastErrorCode(void);
    __export const char *llaisysGetLastErrorMessage(void);
    __export void llaisysClearLastError(void);
}

#endif
