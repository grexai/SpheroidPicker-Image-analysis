# Image Analysis

This repository provides tools and resources for creating and analyzing datasets using Mask R-CNN with GPU support.

## Dataset Download

Download the datasets required for training and evaluation:
1. Primary Dataset: [https://doi.org/10.5281/zenodo.14679303](https://doi.org/10.5281/zenodo.14679303)
2. Docker Compose Configuration Dataset: [https://doi.org/10.5281/zenodo.14675683](https://doi.org/10.5281/zenodo.14675683)

## Update Paths in Docker Compose

Before building the Docker container, update the dataset paths in the `docker-compose.yml` file as follows:

```yaml
volumes:
  - ../mrcnngit/:/mnt/mrcnngit
  - ../Mask_RCNN:/mnt/MASK_RCNN
  - ../models:/mnt/models
  - ../deeplearning:/mnt/deeplearning
  - ../preselected_dataset:/mnt/preselected_dataset/
```

## Installation

### Prerequisites

Ensure that Docker is installed on your system, preferably with NVIDIA GPU support for accelerated training. Follow the instructions for setting up [Docker with NVIDIA GPU support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Build the Docker Container

Run the following command to build the Docker container:

```bash
docker compose build
```

This will create a container with a Mask R-CNN GPU-compatible runtime environment.

## Usage

### Starting the Container

Start the container using the following command:

```bash
docker compose up -d
```

The container name is `matterport_mrcnn_container`.

### Running Scripts Inside the Container

1. Enter the container:

   ```bash
   docker exec -it matterport_mrcnn_container bash
   ```

2. Run the training or inference scripts using the appropriate JSON configuration file:

   - For training:
     ```bash
     ./train.sh <configuration_file.json>
     ```

   - For evaluation or other tasks:
     ```bash
     ./g_s_run.sh <configuration_file.json>
     ```


Replace `<configuration_file.json>` with the path to your specific configuration file.

## Notes

- Ensure that all dataset paths are correctly specified in the configuration files before running any scripts.
- For optimal performance, use a system with an NVIDIA GPU and the appropriate drivers installed.

