对外提供API

供给对象
客户：高度封装，以session、request、model等作为接口
同事：提供灵活多变，考虑多路解码，也可以高度封装

层级：
最高层：只有Init、LoadModel、InferSync、InferAsync接口，参考infer_api.cpp
中间层：有session，可配参数，可调资源，参考session_api.cpp
最底层：只有pre/infer/post接口，参考processor_api.cpp


