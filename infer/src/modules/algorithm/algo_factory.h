#ifndef __ALGO_FACTORY_H__
#define __ALGO_FACTORY_H__

#if defined(WITH_BM1684)
#include "bm_inference.h"
#elif defined(WITH_MLU220) || defined(WITH_MLU270) || defined(WITH_MLU370)
#include "mlu_inference.h"
#elif defined(WITH_NVIDIA)
#include "nv_inference.h"
#elif defined(WITH_JETSON)
#include "jet_inference.h"
#elif defined(WITH_INTEL)
#include "intel_inference.h"
#elif defined(WITH_RV1126)
#include "rv_inference.h"
#elif defined(WITH_TX5368)
#include "tsing_inference.h"
#endif

namespace gddi {
namespace algo {

static std::unique_ptr<AbstractAlgo> make_algo_impl() {
#if defined(WITH_BM1684)
    return std::make_unique<BmInference>();
#elif defined(WITH_MLU220) || defined(WITH_MLU270) || defined(WITH_MLU370)
    return std::make_unique<MluInference>();
#elif defined(WITH_NVIDIA)
    return std::make_unique<NvInference>();
#elif defined(WITH_JETSON)
    return std::make_unique<JetInference>();
#elif defined(WITH_INTEL)
    return std::make_unique<IntelInference>();
#elif defined(WITH_RV1126)
    return std::make_unique<RvInference>();
#elif defined(WITH_TX5368)
    return std::make_unique<TsingInference>();
#else
    return {};
#endif
}
}// namespace algo
}// namespace gddi

#endif