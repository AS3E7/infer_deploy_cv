#include "app/runner_dataset.h"
#include "app/runner_pic.h"
#include "api/global_config.h"
#include "core/infer_server.h"

#include <unistd.h>
#include <iostream>

#include <boost/program_options.hpp>
using namespace boost::program_options;

int main(int argc, char **argv)
{
    std::vector<std::string> model_paths;
    std::vector<std::string> license_paths;
    std::string anno_path;
    std::string pic_path;
    std::string result_path;
    std::string save_path;

    options_description desc{"Options"};
    desc.add_options()
            ("help,h", "Help screen")
            ("model",  value<std::string>()->default_value(""), "model file path")
            ("license",  value<std::string>()->default_value(""), "model license file path")
            ("anno-file",  value<std::string>()->default_value(""), "anno file path")
            ("pic-path",  value<std::string>()->default_value(""), "dataset pic file path")
            ("result-path",  value<std::string>()->default_value(""), "result file save path")
            ("save-pic",  value<std::string>()->default_value(""), "draw result on pic and save path");

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
    if (vm.count("anno-file")) {
        anno_path = vm["anno-file"].as<std::string>();
    } 
    if (vm.count("pic-path")) {
        pic_path = vm["pic-path"].as<std::string>();
    } 
    if (vm.count("result-path")) {
        result_path = vm["result-path"].as<std::string>();
    } 
    if (vm.count("save-pic")) {
        save_path = vm["save-pic"].as<std::string>();
    } 

    gddeploy::DatasetRunner runner;
    runner.Init("", model_paths, license_paths);

    runner.InferSync(anno_path, pic_path, result_path, save_path, true);
    return 0;
}