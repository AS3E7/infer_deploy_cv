docker build -f docker/bmnn_build.Dockerfile -t registry.gddi.com/lgy/test/gddeploy/bmnn/build/3.0:v0.1 .

docker run -it --rm -v $PWD:/workspace registry.gddi.com/lgy/test/gddeploy/bmnn/build/3.0:v0.1
