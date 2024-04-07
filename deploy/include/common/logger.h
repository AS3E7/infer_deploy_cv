#pragma once

#include <spdlog/spdlog.h>
#include "spdlog/sinks/stdout_color_sinks.h"

namespace gddeploy
{

    spdlog::logger *GetLogger();

    void SetLogger(spdlog::logger *logger);

}

#define GDDEPLOY_LEVEL_TRACE 0
#define GDDEPLOY_LEVEL_DEBUG 1
#define GDDEPLOY_LEVEL_INFO 2
#define GDDEPLOY_LEVEL_WARN 3
#define GDDEPLOY_LEVEL_ERROR 4
#define GDDEPLOY_LEVEL_CRITICAL 5
#define GDDEPLOY_LEVEL_OFF 6

#define GDDEPLOY_LOG(level, ...) GetLogger()->log(level, __VA_ARGS__)

#if !defined(GDDEPLOY_ACTIVE_LEVEL)
#define GDDEPLOY_ACTIVE_LEVEL GDDEPLOY_LEVEL_INFO
#endif

#if GDDEPLOY_ACTIVE_LEVEL <= GDDEPLOY_LEVEL_TRACE
#define GDDEPLOY_TRACE(...) GDDEPLOY_LOG(spdlog::level::trace, __VA_ARGS__)
#else
#define GDDEPLOY_TRACE(...) (void)0;
#endif

#if GDDEPLOY_ACTIVE_LEVEL <= GDDEPLOY_LEVEL_DEBUG
#define GDDEPLOY_DEBUG(...) GDDEPLOY_LOG(spdlog::level::debug, __VA_ARGS__)
#else
#define GDDEPLOY_DEBUG(...) (void)0;
#endif

#if GDDEPLOY_ACTIVE_LEVEL <= GDDEPLOY_LEVEL_INFO
#define GDDEPLOY_INFO(...) GDDEPLOY_LOG(spdlog::level::info, __VA_ARGS__)
#else
#define GDDEPLOY_INFO(...) (void)0;
#endif

#if GDDEPLOY_ACTIVE_LEVEL <= GDDEPLOY_LEVEL_WARN
#define GDDEPLOY_WARN(...) GDDEPLOY_LOG(spdlog::level::warn, __VA_ARGS__)
#else
#define GDDEPLOY_WARN(...) (void)0;
#endif

#if GDDEPLOY_ACTIVE_LEVEL <= GDDEPLOY_LEVEL_ERROR
#define GDDEPLOY_ERROR(...) GDDEPLOY_LOG(spdlog::level::err, __VA_ARGS__)
#else
#define GDDEPLOY_ERROR(...) (void)0;
#endif

#if GDDEPLOY_ACTIVE_LEVEL <= GDDEPLOY_LEVEL_CRITICAL
#define GDDEPLOY_CRITICAL(...) GDDEPLOY_LOG(spdlog::level::critical, __VA_ARGS__)
#else
#define GDDEPLOY_CRITICAL(...) (void)0;
#endif

#define CHECK(a)                                                                                         \
    if (!(a))                                                                                            \
    {                                                                                                    \
        GDDEPLOY_ERROR("[{}] [{}] [{}]CHECK failed {} is nullptr", __FILE__, __FUNCTION__, __LINE__, a); \
    }

#define CHECK_NOTNULL(a)                                                                                 \
    if (NULL == (a))                                                                                     \
    {                                                                                                    \
        GDDEPLOY_ERROR("[{}] [{}] [{}]CHECK failed {} is nullptr", __FILE__, __FUNCTION__, __LINE__, a); \
    }

#define CHECK_NULL(a)                                                                                        \
    if (NULL != (a))                                                                                         \
    {                                                                                                        \
        GDDEPLOY_ERROR("[{}] [{}] [{}]CHECK failed {} is not nullptr", __FILE__, __FUNCTION__, __LINE__, a); \
    }

#define CHECK_EQ(a, b, comment, ...)          \
    if (!((a) == (b)))                        \
    {                                         \
        GDDEPLOY_ERROR(comment, __VA_ARGS__); \
    }

#define CHECK_NE(a, b, comment, ...)          \
    if (!((a) != (b)))                        \
    {                                         \
        GDDEPLOY_ERROR(comment, __VA_ARGS__); \
    }

#define CHECK_LT(a, b, comment, ...)          \
    if (!((a) < (b)))                         \
    {                                         \
        GDDEPLOY_ERROR(comment, __VA_ARGS__); \
    }

#define CHECK_GT(a, b, comment, ...)          \
    if (!((a) > (b)))                         \
    {                                         \
        GDDEPLOY_ERROR(comment, __VA_ARGS__); \
    }

// #define CHECK_LE(a, b, comment, ...)                        \
//    if(!((a) <= (b))) {                                      \
//         GDDEPLOY_ERROR(comment, __VA_ARGS__);                        \
//     }

#define CHECK_GE(a, b, comment, ...)          \
    if (!((a) >= (b)))                        \
    {                                         \
        GDDEPLOY_ERROR(comment, __VA_ARGS__); \
    }
