FROM continuumio/miniconda3
ADD environment_pointcloud.yml /tmp/environment_pointcloud.yml
RUN conda env create -f /tmp/environment_pointcloud.yml

# Pull the environment name out of the environment_pointcloud.yml
# Make RUN commands use the new environment:
RUN echo "$(head -1 /tmp/environment_pointcloud.yml | cut -d' ' -f2)" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ARG ENVIRONMENTNAME="$(head -1 /tmp/environment_pointcloud.yml | cut -d' ' -f2)"

# TODO: make target folder in container variable via ARG, so different models can be loaded
# Copy the model results of the Bundestags-protocols
COPY ./checkpoints /checkpoints/

# Export port for TensorBoard
EXPOSE 6006

# TODO: read env "tensorboard" out of yml-file line 7
# Start running tensorboard when container is launched
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tensorboard", "python", "/opt/conda/envs/tensorboard/lib/python3.7/site-packages/tensorboard/main.py", "--logdir=checkpoints"]
