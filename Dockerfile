FROM ubuntu:rolling

MAINTAINER Jens Renè Suckert <jens.suckert@gmail.com>

RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python \
        git \
        cmake \
        qt5-default \
        libqt5charts5-dev \
        libqt5opengl5-dev \
        --reinstall ca-certificates 

RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    chown ${uid}:${gid} -R /home/developer

#    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
#    chmod 0440 /etc/sudoers.d/developer && \

USER developer
ENV HOME /home/developer

RUN cd $HOME && \
    git clone https://github.com/spirit-code/spirit.git

RUN cd $HOME/spirit && \
    git checkout develop

RUN cd $HOME/spirit && \
    ./cmake.sh

RUN cd $HOME/spirit && \
    ./make.sh -j2

CMD cd $HOME/spirit && ./spirit 