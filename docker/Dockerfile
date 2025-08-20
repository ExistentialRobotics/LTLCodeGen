ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

ARG TZ=America/Los_Angeles
ARG LOCALE=en_US.UTF-8
ARG APT_MIRROR=""

ENV TZ=${TZ}
ENV LANG=${LOCALE}
ENV LANGUAGE=${LOCALE}
ENV LC_ALL=${LOCALE}

ARG APT_MIRROR=""
RUN cp /etc/apt/sources.list /etc/apt/sources.list.backup &&\
    if [ -n "$APT_MIRROR" ]; then \
    sed -i "s|http://archive.ubuntu.com/ubuntu/|$APT_MIRROR|g" /etc/apt/sources.list; \
    fi
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    DEBIAN_FRONTEND=noninteractive apt update && apt install -y locales && locale-gen en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt install -y ubuntu-desktop-minimal
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    sudo \
    apt-utils \
    software-properties-common  \
    lsb-release \
    ca-certificates gnupg \
    curl  \
    iproute2 \
    vim zsh git \
    build-essential

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | DEBIAN_FRONTEND=noninteractive apt-key add -
RUN DEBIAN_FRONTEND=noninteractive apt update -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    python3-pip \
    python3-numpy \
    python3-toml \
    python3-scipy \
    python3-tqdm \
    python3-opencv
RUN rosdep init && rosdep fix-permissions
# Gazebo Fortress installation
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    DEBIAN_FRONTEND=noninteractive apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y ignition-fortress

# Project setup
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    ros-noetic-teleop-twist-keyboard \
    ros-noetic-octomap \
    ros-noetic-octomap-msgs \
    ros-noetic-octomap-ros \
    ros-noetic-octomap-rviz-plugins \
    ros-noetic-octovis \
    ros-noetic-vision-msgs \
    nlohmann-json3-dev
RUN pip3 install ultralytics scikit-image

RUN git clone --recursive https://github.com/ExistentialRobotics/LTLCodeGen.git /opt/ros_ws/src
RUN git clone --recursive https://github.com/gazebosim/ros_gz.git -b noetic /opt/ros_gz
WORKDIR /opt/ros_ws/src
RUN ln -s /opt/ros_gz/ros_ign_gazebo ros_ign_gazebo && ln -s /opt/ros_gz/ros_ign_bridge ros_ign_bridge && cd .. && \
    rosdep update --include-eol-distros && rosdep install -r --from-paths src -i -y --rosdistro noetic

WORKDIR /root
COPY spot-2.12.2.tar.gz spot-2.12.2.tar.gz
RUN tar -xvf spot-2.12.2.tar.gz && rm spot-2.12.2.tar.gz && cd spot-2.12.2 && \
    ./configure --prefix=/usr --with-pythondir=/usr/lib/python3/dist-packages && make -j`nproc` install &&  \
    cd /root && rm -rf spot-2.12.2

WORKDIR /opt/ros_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin build --verbose --cmake-args -DCMAKE_BUILD_TYPE=Release"

ENV GOMP_SPINCOUNT=1

RUN apt-get clean  && rm -rf /var/lib/apt/lists/*
RUN if [ -n "$APT_MIRROR" ]; then \
    mv /etc/apt/sources.list.backup /etc/apt/sources.list; \
    fi
