//
// Created by cc on 2021/11/1.
//
#include <utility>
#include "runnable_node.hpp"
#include "debug_tools.hpp"
#include "../../src/modules/endecode/demux_stream.hpp"
#include "utils.hpp"
#include "debug_tools.hpp"
#include "nodes/message_templates.hpp"
#include "nodes/ffvcodec/demuxer_node_v1.hpp"
#include "nodes/node_any_basic.hpp"
#include "../../src/modules/endecode/decode_video_impl.hpp"
#include "OpenCV/cv_tools.hpp"
#include "nodes/ffvcodec/decoder_node_v1.hpp"

int main(int argc, char *argv[]) {
    auto runner = std::make_shared<gddi::ngraph::Runner>("default");

    auto demuxer = std::make_shared<gddi::nodes::lib_av::Demuxer_v1>("Demuxer_v1");
    auto decoder = std::make_shared<gddi::nodes::Decoder_v1>("Decoder_v1");
//    auto cv_show = std::make_shared<gddi::nodes::cv_nodes::CvImShow_v1>("show");

    demuxer->bind_runner(runner);
    decoder->bind_runner(runner);

    auto url_231 = "rtsp://admin:gddi1234@192.168.1.231:554/h264/ch1/main/av_stream";
    auto url_233 = "rtsp://admin:gddi1234@192.168.1.233:554/h264/ch1/main/av_stream";

    demuxer->connect_to(decoder, 0, 0);
    demuxer->connect_to(decoder, 1, 1);
//    decoder->connect_to(cv_show, 0, 0);
//    decoder->connect_to(cv_show, 1, 1);

    demuxer->properties().try_set_property("stream_url", url_233);

    demuxer->print_info(std::cout);
    decoder->print_info(std::cout);

    runner->start_in_local();
    return 0;
}