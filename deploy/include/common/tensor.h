#ifndef TENSOR_H
#define TENSOR_H

#include "ulti.h"

namespace gddeploy {

class Tensor{
public:

private:

    VoidPtr priv_[4];    //私有数据，一般存储推理后的结果，结果形式参考api/Result类，其中的空间申请和释放自行管理

}; 

}

#endif