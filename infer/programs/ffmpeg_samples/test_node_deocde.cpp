//
// Created by cc on 2021/11/2.
//
#include <utility>
#include "runnable_node.hpp"
#include "nodes/message_templates.hpp"
#include "nodes/ffvcodec/demuxer_node_v1.hpp"
#include "nodes/ffvcodec/decoder_node_v1.hpp"

int main(int argc, char *argv[]) {
    auto runner = std::make_shared<gddi::ngraph::Runner>("default");
    auto demuxer = std::make_shared<gddi::nodes::lib_av::Demuxer_v1>("Demuxer_v1");
    auto decoder = std::make_shared<gddi::nodes::Decoder_v1>("Decoder_v1");
    auto bridge = std::make_shared<gddi::ngraph::Bridge>("");

    demuxer->bind_runner(runner);
    decoder->bind_runner(runner);
    bridge->bind_runner(runner);

    auto url_231 = "rtsp://admin:gddi1234@192.168.1.231:554/h264/ch1/main/av_stream";

    demuxer->connect_to(decoder, 0, 0);
    demuxer->connect_to(decoder, 1, 1);
    decoder->connect_to(bridge);

    demuxer->print_info(std::cout);
    decoder->print_info(std::cout);
    bridge->print_info(std::cout);

    decoder->properties().try_set_property("disable_hw_acc", true);
    decoder->properties().try_set_property("prefer_hw", 2);
    demuxer->properties().try_set_property("stream_url", url_231);

    runner->start();

    std::thread another_conn([&] {

        for (int i = 0; i < 70; i++) {
            std::stringstream oss;
            oss << "tmp_" << i;
            auto tmp_bridge = std::make_shared<gddi::ngraph::Bridge>(oss.str());
            tmp_bridge->bind_runner(runner);
            decoder->connect_to(tmp_bridge);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(3500));

    std::cout<<"==========================================================";
    decoder->debug_print_output_links(std::cout);
    another_conn.join();
    std::cout<<"==========================================================";
    decoder->debug_print_output_links(std::cout);
    // runner->stop();
    return 0;
}
