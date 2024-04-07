#include "app/runner_video.h"
#include "api/global_config.h"

#include <iostream>
#include <unistd.h>
#include <iostream>

#include <boost/program_options.hpp>
using namespace boost::program_options;

int main(int argc, char **argv)
{
    std::vector<std::string> model_paths;
    std::vector<std::string> license_paths;
    std::string video_path;
    int_fast32_t multi_stream;
    bool is_save;
    std::string save_path;

    options_description desc{"Options"};
    desc.add_options()
            ("help,h", "Help screen")
            ("model",  value<std::string>()->default_value(""), "model file path")
            ("license",  value<std::string>()->default_value(""), "model license file path")
            ("video-path",  value<std::string>()->default_value(""), "video file path")
            ("multi-stream",  value<int>()->default_value(1), "multi stream")
            ("is-save",  value<bool>()->default_value(true), "is save result pic")
            ("save-pic",  value<std::string>()->default_value(""), "model file path");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    } 
    if (vm.count("model")) {
        // 判断字符串中是否有字符“;”，如果有就分割字符串赋值
        std::string tmp = vm["model"].as<std::string>();
        if (tmp.find(",") != std::string::npos) {
            std::stringstream ss(tmp);
            std::string item;
            while(std::getline(ss, item, ',')) {
                model_paths.push_back(item);
            }
        } else {
            model_paths.push_back(tmp);
        }
    }

    if (vm.count("license")) {
        // 判断字符串中是否有字符“;”，如果有就分割字符串赋值
        std::string tmp = vm["license"].as<std::string>();
        if (tmp.find(",") != std::string::npos) {
            std::stringstream ss(tmp);
            std::string item;
            while(std::getline(ss, item, ',')) {
                license_paths.push_back(item);
            }
        } else {
            license_paths.push_back(tmp);
        }
    } 

    if (vm.count("video-path")) {
        video_path = vm["video-path"].as<std::string>();
    } 
    if (vm.count("multi-stream")) {
        multi_stream = vm["multi-stream"].as<int>();
    } 
    if (vm.count("is-save")) {
        is_save = vm["is-save"].as<bool>();
    } 
    if (vm.count("save-pic")) {
        save_path = vm["save-pic"].as<std::string>();
    } 

    gddeploy::VideoRunner runner[multi_stream];

    for (int j = 0; j < multi_stream; j++) {
        runner[j].Init("", model_paths, license_paths);
        runner[j].OpenStream(video_path, save_path, is_save);
    }

    for (int j = 0; j < multi_stream; j++) {
        runner[j].Join();
    }

    // runner.OpencvOpen(video_path, save_path, true);

    return 0;
}