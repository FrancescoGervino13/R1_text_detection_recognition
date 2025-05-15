# R1_text_detection_recognition
ROS2 Jazzy nodes to perform text detection and recognition utilising mmocr library

# Build Docker
Instructions on how to build the docker environment:
Use the `build_docker.sh` script inside the folder `/docker` to build your image.
Change variables `BASE_NAME` with `@yourdockerhunname/@imagebasename` and the `TAG` accordingly.
## Prerequisites
Check that your computer Nvidia driver supports the Docker cuda image version by checking this table: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions

Install Docker and Nvidia container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html follow also the configuration section.

Have set up docker to have at runtime the `nvidia-container-runtime` by edit/create the `/etc/docker/daemon.json` with content:
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
```
And restart the docker daemon: `sudo systemctl restart docker`

### Check if MMCV has been properly installed:
Inside the container run a python3 script from terminal:
```
from mmcv.utils import IS_CUDA_AVAILABLE
print(f"CUDA extensions available: {IS_CUDA_AVAILABLE}")
```
and
```
from mmcv.ops import ModulatedDeformConv2d
print("CUDA ops are working.")
```
If no errors arise it means that everything works fine.

## Run the docker
Run the script `run_docker.sh` inside `/docker` subfolder.
The script handles automatically the `ROS_DOMAIN_ID`, porting it in the container from your machine. 
If you haven't set it, it's good pratice to isolate your ROS2 network from the rest if you need to do some testing.

Inside the docker there is ROS2 using cyclonedds. The variable `CYCLONEDDS_URI` is not set in the container, nor is the file. If you need to do so, just change the `run_docker.sh` script by exporting the file by `-v your_file_config_path:image_file_config_path` and setting properly the environment variable by `-e CYCLONEDDS_URI=image_file_config_path`.

If you are not familiar with cyclonedds follow this guide: https://cyclonedds.io/docs/cyclonedds/latest/config/index.html
