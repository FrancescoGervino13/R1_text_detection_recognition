FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

ARG RELEASE
ARG LAUNCHPAD_BUILD_ARCH
ENV USER=user1
ARG PASSWORD=user1

# Timezone and core tools
RUN ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime && \
    apt update && apt install --no-install-recommends -y -qq sudo git cmake && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN addgroup ${USER} \
    && useradd -ms /bin/bash ${USER} -g ${USER} \
    && echo "${USER}:${PASSWORD}" | chpasswd \
    && usermod -a -G sudo ${USER} \
    && sed -i.bak -e 's/%sudo\s\+ALL=(ALL(:ALL)?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers

USER ${USER}
WORKDIR /home/${USER}
SHELL ["/bin/bash", "-c"]

# Install Python, pip, venv, etc.
RUN sudo apt update && sudo apt install -y python3-pip python3-venv ffmpeg libsm6 libxext6 python3-opencv python3-shapely

# Create and activate virtual environment
RUN python3 -m venv /home/${USER}/work_env
ENV VIRTUAL_ENV=/home/${USER}/work_env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip & setuptools
RUN pip install --upgrade pip setuptools

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# PaddlePaddle (CUDA 11.8 build, compatible with driver but not CUDA 12.9 runtime)
RUN pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Core requirements
RUN pip install \
    numpy==1.26.4 \
    opencv-python \
    open3d \
    pyransac3d \
    message-filters \
    paddleocr \
    sentence-transformers \
    python-Levenshtein \
    openai \
    platformdirs

# ROS2: install base + useful packages
ENV ROS_DISTRO=jazzy
RUN sudo apt update && sudo apt install -y locales && sudo locale-gen en_US en_US.UTF-8 \
    && sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && export LANG=en_US.UTF-8

RUN sudo apt install software-properties-common curl -y && sudo add-apt-repository universe
RUN sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN sudo apt update && sudo apt install -y \
    ros-$ROS_DISTRO-ros-base \
    ros-$ROS_DISTRO-sensor-msgs \
    ros-$ROS_DISTRO-geometry-msgs \
    ros-$ROS_DISTRO-action-msgs \
    ros-$ROS_DISTRO-actionlib-msgs \
    ros-$ROS_DISTRO-nav2-msgs \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-nav2-simple-commander \
    ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
    ros-dev-tools

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# VS Code setup
RUN wget -O- https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor | sudo tee /usr/share/keyrings/vscode.gpg
RUN echo deb [arch=amd64 signed-by=/usr/share/keyrings/vscode.gpg] https://packages.microsoft.com/repos/vscode stable main | sudo tee /etc/apt/sources.list.d/vscode.list
RUN sudo apt update && sudo apt install -y code
RUN code --install-extension ms-vscode.cpptools \
		--install-extension ms-vscode.cpptools-themes \
		--install-extension ms-vscode.cmake-tools \
        --install-extension ms-python.python \
        --install-extension eamodio.gitlens

# Clone your repo and install Python parts
RUN git clone --recurse-submodules https://github.com/FrancescoGervino13/R1_text_detection_recognition.git
RUN pip install git+https://github.com/FrancescoGervino13/imgaug.git

# Install your package in editable mode
RUN cd R1_text_detection_recognition/src/paddleocr_ros/paddleocr_ros/paddleocr && pip install -v -e .

# Build ROS2 workspace
RUN cd /home/${USER}/R1_text_detection_recognition && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install

# Final environment setup
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USER}/.bashrc
RUN echo "source /home/${USER}/R1_text_detection_recognition/install/setup.bash" >> /home/${USER}/.bashrc
RUN echo "source /home/${USER}/work_env/bin/activate" >> /home/${USER}/.bashrc

ENV PYTHONPATH=$PYTHONPATH:/home/${USER}/R1_text_detection_recognition/src/paddleocr_ros/paddleocr_ros/paddleocr

# Final utilities
RUN sudo apt update && sudo apt install -y iproute2 unzip terminator bash-completion gedit locate && sudo apt clean && sudo rm -rf /var/lib/apt/lists/* && sudo updatedb
