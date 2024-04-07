#pragma once

#include <string>
#include <memory>
#include <vector>

namespace gddeploy {

class VideoRunnerPriv;
class VideoRunner{
public:
    VideoRunner();

    int Init(const std::string config, std::string model_path, std::string license_path = "");
    int Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license_paths);

    int OpencvOpen(std::string video_path, std::string save_path="", bool is_draw=false);  

    int OpenStream(std::string video_path, std::string save_path="", bool is_draw=false);  

    int Join();
private:
    std::shared_ptr<VideoRunnerPriv> priv_;
};

} // namespace gddeploy

