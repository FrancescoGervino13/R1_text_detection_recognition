# Start from the base image
FROM leogori/r1images:tourCore2_ubuntu24.04_jazzy_stable_cuda12.1

# Set a non-root user (we'll call it 'user')
RUN sudo useradd -ms /bin/bash user

# Set the working directory inside the container
WORKDIR /workspace

# Clone the repository (as non-root)
RUN git clone --recurse-submodules https://github.com/FrancescoGervino13/R1_text_detection_recognition.git

# Set the working directory to the cloned repo
WORKDIR /workspace/R1_text_detection_recognition

# Create and activate the virtual environment, and install the necessary packages
RUN python3 -m venv /workspace/ros_env && \
    /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir torch==2.6 torchvision"
    #/bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir torch==2.4 torchvision --index-url https://download.pytorch.org/whl/cu121"

RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir pyyaml"
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir git+https://github.com/FrancescoGervino13/imgaug.git"
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir lmdb matplotlib opencv-python pyclipper pycocotools rapidfuzz scikit-image"

# Install openmim and required packages
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir -U openmim"

# Upgrade pip and setuptools inside the virtual environment
#RUN /bin/bash -c "source /workspace/ros_env/bin/activate && python -m ensurepip --upgrade && python -m pip install --upgrade setuptools"

# Install additional Python modules (replace <module> with your actual module)
#RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir <module>"

# Upgrade pip and setuptools inside the virtual environment
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && python -m ensurepip --upgrade && python -m pip install --upgrade setuptools"

# Install MMOCR dependencies only by mmdet
#RUN /bin/bash -c "source /workspace/ros_env/bin/activate && mim install mmengine"
#RUN /bin/bash -c "source /workspace/ros_env/bin/activate && mim install mmcv"
ENV TORCH_CUDA_ARCH_LIST="8.6"
# Test for compiling mmcv mmdet with cuda
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1
RUN sudo apt update && \
    sudo apt install -y gcc-11 g++-11 &&\
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && export TORCH_CUDA_ARCH_LIST='8.6' && mim install mmdet"

# Install additional Python modules
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir lark catkin_pkg empy==3.3.4 platformdirs openai"

# Set the working directory to the mmocr directory
WORKDIR /workspace/R1_text_detection_recognition/src/mmocr_ros/mmocr_ros/mmocr

# Install the mmocr package in development mode
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install -v -e ."

# Set the working directory to the cloned repo
WORKDIR /workspace/R1_text_detection_recognition

# Build the workspace with colcon
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build"

# Create and configure the CycloneDDS settings file
RUN echo '<CycloneDDS>\n\
    <Domain>\n\
        <General>\n\
            <NetworkInterfaceAddress>172.17.0.1</NetworkInterfaceAddress>\n\
            <AllowMulticast>true</AllowMulticast>\n\
        </General>\n\
    </Domain>\n\
</CycloneDDS>' > /home/user1/custom_cyclone_dds_settings.xml

# Ensure proper ownership of the file (only if needed)
RUN chmod 644 /home/user1/custom_cyclone_dds_settings.xml

# Source the workspace, activate the virtual environment, and set environment variables
RUN echo 'source /workspace/ros_env/bin/activate' >> ~/.bashrc && \
    echo 'source /workspace/R1_text_detection_recognition/install/setup.bash' >> ~/.bashrc && \
    echo 'export PYTHONPATH=$PYTHONPATH:/workspace/R1_text_detection_recognition/src/mmocr_ros/mmocr_ros/mmocr' >> ~/.bashrc && \
    echo 'export CYCLONEDDS_URI="file:///home/user1/custom_cyclone_dds_settings.xml"' >> ~/.bashrc


# Ensure the environment is set up when a new shell starts
SHELL ["/bin/bash", "-c"]

# Default command to run when the container starts
CMD ["bash"]

