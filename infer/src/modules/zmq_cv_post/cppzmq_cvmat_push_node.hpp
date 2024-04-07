//
// Created by cc on 2021/12/1.
//

#ifndef FFMPEG_WRAPPER_SRC_MODULES_ZMQ_CV_POST_CPPZMQ_CVMAT_PUSH_NODE_HPP_
#define FFMPEG_WRAPPER_SRC_MODULES_ZMQ_CV_POST_CPPZMQ_CVMAT_PUSH_NODE_HPP_
#include "cppzmq_instance.hpp"
#include "nodes/node_any_basic.hpp"
#include "OpenCV/cv_to_cvmat.hpp"

namespace gddi {
namespace nodes {

class CppZmqCvMatPusher_v1 : public node_any_basic<CppZmqCvMatPusher_v1> {
private:
    std::unique_ptr<zmq::socket_t> sock_;
    std::string push_addr_;
    int32_t push_count_{0};

public:
    explicit CppZmqCvMatPusher_v1(std::string name)
        : node_any_basic(std::move(name)) {
        bind_simple_property("push_addr", push_addr_, "推流地址");
        register_input_message_handler_<msgs::cv_mat>([=](const std::shared_ptr<msgs::cv_mat> &cv_mat) {
            zmq::multipart_t mp;

            push_count_++;
            mp.pushtyp(push_count_);
            mp.pushtyp(cv_mat->message.cols);
            mp.pushtyp(cv_mat->message.rows);
            mp.pushtyp(cv_mat->message.type());
            // std::cout << "push size: " << cv_mat->message.size().area() << ", "
            //           << cv_mat->message.elemSize() << std::endl;
            mp.pushmem(cv_mat->message.data, cv_mat->message.size().area() * cv_mat->message.elemSize());
            mp.send(*sock_);
        });
    }

protected:
    void on_setup() override {
        try {
            sock_ = std::make_unique<zmq::socket_t>(cppzmq::instance().get(), zmq::socket_type::pub);
            sock_->bind(push_addr_);
        } catch (std::exception &exception) {
            std::cout << "sock error: " << exception.what() << std::endl;
            quit_runner_(TaskErrorCode::kZmq);
        }
    }
};

}//nodes
}//gddi

#endif //FFMPEG_WRAPPER_SRC_MODULES_ZMQ_CV_POST_CPPZMQ_CVMAT_PUSH_NODE_HPP_
