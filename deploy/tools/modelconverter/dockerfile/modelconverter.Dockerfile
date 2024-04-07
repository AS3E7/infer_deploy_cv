FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04

# 安装brew
RUN apt update -y && apt-get install -y wget build-essential curl git ruby libbz2-dev libcurl4-openssl-dev libexpat-dev libncurses-dev zlib1g-dev vim libgl1-mesa-glx libglib2.0-0 

# RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# # RUN test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv) 
# RUN test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv) 
# RUN echo "eval $($(brew --prefix)/bin/brew shellenv)" >>~/.profile
# ENV PATH=/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin


# # 安装pyenv和virtualenv
# # rm -rf $(brew --repo homebrew/core)
# RUN brew update && brew install pyenv 
# RUN brew install pyenv-virtualenv

# RUN export PATH="$HOME/.pyenv/bin:$PATH" && \
#     eval "$(pyenv init -)" && \
#     eval "$(pyenv virtualenv-init -)"

RUN wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P /tmp 
RUN chmod +x /tmp/Miniconda3-latest-Linux-x86_64.sh && /bin/bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b
RUN echo "export PATH=/root/miniconda3/bin:$PATH" >> ~/.bashrc

RUN /root/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN /root/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
RUN /root/miniconda3/bin/conda config --set show_channel_urls yes
RUn /root/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

RUN /root/miniconda3/bin/conda create -n pytorch1.12_python3.9 python=3.9 -y
RUN /root/miniconda3/bin/conda create -n pytorch1.6_python3.7.5 python=3.7.5 -y

COPY /volume1/gddi-data/lgy/cambricon/thirdparty/ /root

RUN /root/miniconda3/envs/pytorch1.6_python3.7.5/bin/pip3 install -r /root/thirdparty/yolov5


WORKDIR /root