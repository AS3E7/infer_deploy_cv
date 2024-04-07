#include "core/register.h"

#include "core/model.h"
#include "core/device.h"
#include "core/preprocess.h"
#include "core/predictor.h"
#include "core/postprocess.h"
#include "core/alg.h"
#include "core/model.h"
#include "core/mem/buf_surface_impl.h"
#include "core/cv.h"

using namespace gddeploy;

int gddeploy::register_all_module()
{
    register_device_module();
    
    register_model_module();

    register_preproc_module();

    register_predictor_module();

    register_postproc_module();

    register_alg_module();  // 算法注册

    register_mem_module();

    register_cv_module();

    return 0;
}