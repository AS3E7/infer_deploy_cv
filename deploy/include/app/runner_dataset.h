#pragma once

#include <string>
#include <memory>
#include <vector>

namespace gddeploy {

class DatasetRunnerPriv;
class DatasetRunner{
public:
    DatasetRunner();

    int Init(const std::string config, std::string model_path, std::string license_path = "");
    int Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license_paths);

    int InferSync(std::string anno_path, std::string pic_path, std::string result_path, std::string save_path, bool is_drowtrue);
private:
    std::shared_ptr<DatasetRunnerPriv> priv_;
};

} // namespace gddeploy

