# Start from the base image
FROM leogori/r1images:tourCore2_ubuntu24.04_jazzy_stable_cuda12.1

# Set a non-root user (we'll call it 'user')
RUN useradd -ms /bin/bash user

# Set the working directory inside the container
# Start from the base image
FROM leogori/r1images:tourCore2_ubuntu24.04_jazzy_stable_cuda12.1

# Set the working directory inside the container
WORKDIR /workspace

# Clone the repository (as non-root)
RUN git clone --recurse-submodules https://github.com/FrancescoGervino13/R1_text_detection_recognition.git

# Set the working directory to the cloned repo
WORKDIR /workspace/R1_text_detection_recognition

# Create and activate the virtual environment, and install the necessary packages
RUN python3 -m venv /workspace/ros_env && \
    /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir torch torchvision torchaudio"

RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir pyyaml"
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir git+https://github.com/FrancescoGervino13/imgaug.git"
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir lmdb matplotlib opencv-python pyclipper pycocotools rapidfuzz scikit-image"

# Upgrade pip and setuptools inside the virtual environment
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && python -m ensurepip --upgrade && python -m pip install --upgrade setuptools"

# Install additional Python modules (replace <module> with your actual module)
#RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir <module>"

# Install openmim and required packages
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir -U openmim"

# Upgrade pip and setuptools inside the virtual environment
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && python -m ensurepip --upgrade && python -m pip install --upgrade setuptools"

RUN /bin/bash -c "source /workspace/ros_env/bin/activate && mim install mmengine"
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && mim install mmcv"
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && mim install mmdet"

# Install additional Python modules
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install --no-cache-dir lark catkin_pkg empy==3.3.4 platformdirs"

# Set the working directory to the mmocr directory
WORKDIR /workspace/R1_text_detection_recognition/src/mmocr_ros/mmocr_ros/mmocr

# Install the mmocr package in development mode
RUN /bin/bash -c "source /workspace/ros_env/bin/activate && pip install -v -e ."

# Default command to run when the container starts
CMD ["bash"]

