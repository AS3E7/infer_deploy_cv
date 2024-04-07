#include "core/infer_server.h"
#include <stdio.h>
#include "core/model.h"

int main()
{
    const std::string model_path = "./data/models/gddi_model.onnx";
    const std::string properties_path = "./data/models/gddi_model.properties";
    
    gddeploy::ModelManager::Instance()->Load(model_path, properties_path, "");

    printf("over\n");

    return 0;
}