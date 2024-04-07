#include "core/cv.h"
#include "common/logger.h"

#ifdef WITH_BM1684
#include "bmnn/bmnn_cv.h"
#endif

namespace gddeploy
{
CV *CV::pInstance_ = nullptr;

int register_cv_module()
{
    CV *cv = CV::Instance();

#ifdef WITH_BM1684
    GDDEPLOY_INFO("[Register] register cv bmnn module");
    BmnnCV *bmnn_cv = new BmnnCV();
    cv->register_cv("SOPHGO", "SE5", bmnn_cv);
#endif
    return 0;
}
}