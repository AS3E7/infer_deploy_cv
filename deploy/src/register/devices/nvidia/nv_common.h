#pragma once

#define CHECK_AND_RET(status, ret)                                                                          \
    do                                                                                                      \
    {                                                                                                       \
        if (status != 0)                                                                                    \
        {                                                                                                   \
            GDDEPLOY_ERROR("[{}] [{}] [{}]Cuda <<failure: \n", __FILE__, __FUNCTION__, __LINE__, int(ret)); \
            return ret;                                                                                     \
        }                                                                                                   \
    } while (0)