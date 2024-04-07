//
// Created by cc on 2021/11/5.
//

#include "message_ref_time.hpp"
#include "nodes/ffvcodec/demuxer_node_v1.hpp"
#include "nodes/ffvcodec/decoder_node_v1.hpp"
#include "nodes/ffvcodec/demuxer_node_v1_1.hpp"
#include "nodes/ffvcodec/decoder_node_v1_1.hpp"

namespace gddi {
namespace nodes {
namespace lib_av {
namespace msgs {

AVCodecParameters *avcodec_parameters_clone_(const AVCodecParameters *src) {
    auto codecpar = avcodec_parameters_alloc();
    avcodec_parameters_copy(codecpar, src);
    return codecpar;
}

}
}
}
}