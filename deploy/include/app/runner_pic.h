#pragma once

#include <string>
#include <memory>
#include <vector>

namespace gddeploy {

class PicRunnerPriv;
class PicRunner{
public:
    PicRunner();

    int Init(const std::string config, std::string model_path, std::string license = "");
    int Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license);
    int InferAsync(std::string pic_path, std::string save_path, bool is_draw);
    int InferSync(std::string pic_path, std::string save_path, bool is_draw);
private:
    std::shared_ptr<PicRunnerPriv> priv_;
};

} // namespace gddeploy

