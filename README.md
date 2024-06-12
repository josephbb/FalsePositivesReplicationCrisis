# Reappraising the replication crisis without false positives

## System Requirements

### Hardware
Our data and analysis were conducted on an Ubuntu (22.04.4 LTS) server with 64GB of RAM, an AMD Ryzen 9 3900X 12-Core Processor, and a still-hanging-on 1080 Ti. Our code makes extensive use of parallel processing when possible, and can exert memory pressures in the low Gigabytes. We suspect our code should run fine on most modern machines, but may encounter resource errors on on some machines. If so, we recommend deploying the code via a cloud provider such as [AWS](https://aws.amazon.com/getting-started/hands-on/deploy-docker-containers/) or [Azure](https://azure.microsoft.com/en-us/products/kubernetes-service/docker?ef_id=_k_CjwKCAjw34qzBhBmEiwAOUQcFx_0B2-UFILo27RHLdv0FoDhYtqzPqgJY0YcWtep1RndzVOyYekrJxoCFpcQAvD_BwE_k_&OCID=AIDcmm5edswduu_SEM__k_CjwKCAjw34qzBhBmEiwAOUQcFx_0B2-UFILo27RHLdv0FoDhYtqzPqgJY0YcWtep1RndzVOyYekrJxoCFpcQAvD_BwE_k_&gad_source=1&gclid=CjwKCAjw34qzBhBmEiwAOUQcFx_0B2-UFILo27RHLdv0FoDhYtqzPqgJY0YcWtep1RndzVOyYekrJxoCFpcQAvD_BwE). 

## Software 

### Cloning our Repository
Cloning our repository is simple, in your command line (assuming you have git CLI installed) simply type: 

```
git clone https://github.com/josephbb/ReplicationSurveys
```

You can also download the repository directly from [https://github.com/josephbb/ReplicationSurveys](https://github.com/josephbb/ReplicationSurveys).  
 

### Docker
We highly recommend reproducing our analysis using [Docker](https://www.docker.com/). Docker provides containers that emulate full system state, ensuring no differences as software is run across machines with varying operating systems and configuration. For an introduction to Docker's use in facilitating reproduciblity, please see [this excellent tutorial](https://kordinglab.com/2022/10/28/LabTeaching-Docker-for-Science.html) by Konrad Kording. 

You'll need to [install docker](https://www.docker.com/products/docker-desktop/) following instructions for your machine. You should also install the [command line interface](https://www.docker.com/products/cli/). Our instructions will assume you're on a linux-like terminal (OS X is fine). We don't have a windows machine handy, but the docker documentation for the CLI should provide anything you need to adjust this README accordingly. 

Our docker image contains all necessary data, code, and software in a single package to run our analysis exactly in a single step. You can either pull our image from dockerhub 

CODE HERE

or create the image locally using our Dockerfile. Navigate into the root directory of our git repository and type: 

```
docker build -t repefforts .
```
This creates an image, repefforts, that you can run to reproduce our code as described below. 

### Alternative: Python
Docker has a little learning curve and you may want to avoid all of that. If so, you can run our code on your machine rather than a docker container. We recommend against this, but it should work fine if you're feeling bold. 

We recommend installing [Anaconda](https://www.anaconda.com/) and [creating a virtual environment](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084) for Python 3.10 using 

```
conda create --name repsurveys python=3.10
```

From the project directory, install requirements using:

```
pip install -r requirements.txt
```

Ensure you have ipykernel installed:

```
conda install -c anaconda ipykernel
```

then create the kernel:

```
python -m ipykernel install --user --name=repsurveys
```


## Reproducing our analysis

### Docker
The simplest way to reproduce our analysis is via the unmodified ```docker run``` command. This will produce all figures and tables, running inside of the terminal until all analyses have been completed. Resultant files will be stored on your host machine in a folder called output (which must exist):

```
docker run -it -v $(pwd)/output:/replicationsurveys/output repefforts run-reproduce
```

The ```-it``` creates an interactive run of the docker container which will show terminal output as our primary python script runs through the ipython notebooks. The next bit of our code pulls the output from inside our container and places it in our output folder. ```repefforts``` indicates which container we're running. Finally, ```run-reproduce``` is a flag telling our container to run all of the code. If you simply wish to start the container up and run the code some other way, you an omit this bit.

You may wish to avoid running everything and simply analyze the code at your leisure in a jupyter notebook. Start up a jupyter notebook

```
 docker run -it -p 8888:8888 -v $(pwd)/output:/replicationsurveys/output repefforts jupyter
 ```

This links the standard output port for jupyter from the container to your host machine. You can open any web-browser and navigate to [http://localhost:8888/](http://localhost:8888/). It will request a token, which can be found in the output where you ran the docker run command. You'll be greeted in the root directory of the container and you can navigate to the repsurveys folder containing the code and run jupyter notebooks as normal. Because we have the ```-v $(pwd)/output:/replicationsurveys/output ``` command, output should be stored on your local machine but note that any modifications to the code will *only* modify the code within the container and will not modify code locally. 

### Python

Assuming you've cloned our respository and are in the root directory, activate your virtual environment. 

```
conda activate repsurveys
```

Then run: 

```
python reproduce.py 
```

Alternatively, you can run the code piecemeal by simply [opening jupyter](https://docs.jupyter.org/en/latest/running.html). 

```
jupyter notebook
```

<h2>Adjusting Parameters</h2>
Many global parameters and stylistic choices can be adjusted in the ``src/parameters.py`` file. You're free to modify them there to avoid changing things throughout the code, let us know if you find anything interesting. 


