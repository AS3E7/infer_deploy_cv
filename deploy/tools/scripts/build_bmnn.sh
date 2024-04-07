install_path=$PWD/build_bmnn/release_bmnn_aarch64_v0.1

mkdir build_bmnn ${install_path} -p && cd build_bmnn

cmake .. -DCMAKE_INSTALL_PREFIX=${install_path} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin//aarch64-linux-gnu-g++

make -j$(nproc) && make install

cp -rf ../thirdparty/aarch64-bm1684-soc ${install_path}/thirdparty

#dockerfile
mkdir ${install_path}/dockerfile
cp ../docker/dockerfile/bmnn.Dockerfile ${install_path}/dockerfile

cp ../release/README.md ${install_path}
cp -rf ../release/script ${install_path}

#打包
zip -r release_bmnn_aarch64_v0.1.zip release_bmnn_aarch64_v0.1
cp release_bmnn_aarch64_v0.1.zip /volume1/gddi-data/lgy/gddeploy/sdk/
cd .. && rm -rf build_bmnn