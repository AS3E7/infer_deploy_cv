包括以下：
（不同层级的调用demo）
pic/video/dataset的runner
infer_server
旧接口


demo：包含最基础的输入图片、推理，得出结果，画图
opencv：使用ffmpeg解码
ffmpeg：使用ffmpeg解码
multi_stream：多路流
post：画框画xx保存图片，编码输出
multi_model：多个模型

目录说明：


bin目录包含pic/video/dataset三个可执行文件，可以做一些验证类工作

图片推理：
./bin/sample_runner_pic /volume1/gddi-data/xuyuhao/gddi_model.ts /volume1/gddi-data/xuyuhao/plat_test_files/det/gddi_data/0/images/resize_valid/0a2fe35bae7ca1a5a0bcac08
bec3980362db7cae.jpg .

数据集验证
./bin/sample_runner_dataset /volume1/gddi-data/xuyuhao/gddi_model.ts /volume1/gddi-data/xuyuhao/plat_test_files/det/gddi_data/0/annotation/resize_valid/anno.json /volum
e1/gddi-data/xuyuhao/plat_test_files/det/gddi_data/0/images/resize_valid/ ./mask_prediction.json ./