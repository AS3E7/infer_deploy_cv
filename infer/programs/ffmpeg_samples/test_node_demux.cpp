//
// Created by cc on 2021/11/2.
//
#include "runnable_node.hpp"
#include "debug_tools.hpp"
#include "../src/modules/endecode/demux_stream.hpp"
#include "utils.hpp"
#include "debug_tools.hpp"
#include "nodes/message_templates.hpp"
#include "node_manager/node_manager.hpp"

#include "nodes/ffvcodec/demuxer_node_v1.hpp"

int main(int argc, char *argv[]) {
    gddi::NodeManager::get_instance().print_std_term_style(std::cout);

    auto runner = std::make_shared<gddi::ngraph::Runner>("default");
    auto demuxer = std::make_shared<gddi::nodes::lib_av::Demuxer_v1>();
    auto bridge_packet = std::make_shared<gddi::ngraph::Bridge>("packet");
    auto bridge_open = std::make_shared<gddi::ngraph::Bridge>("open");

    demuxer->bind_runner(runner);
    bridge_packet->bind_runner(runner);
    bridge_open->bind_runner(runner);

    auto url_231 = "rtsp://admin:gddi1234@192.168.1.231:554/h264/ch1/main/av_stream";
    auto url_233 = "rtsp://admin:gddi1234@192.168.1.233:554/h264/ch1/main/av_stream";

    demuxer->connect_to(bridge_packet, 0);
    demuxer->connect_to(bridge_open, 1);

    runner->start();

    demuxer->open_url(url_231);
    demuxer->open_url(url_233);
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));


//    demuxer->open_url(url_233);
//    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
//    runner->stop();
    return 0;
}
