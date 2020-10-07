# [Geometry-Aware Gradient Algorithms for Neural Architecture Search](https://arxiv.org/pdf/2004.07802.pdf)
This repository contains the code required to run the experiments for the DARTS search space over CIFAR-10 and the NAS-Bench-201 search space over CIFAR-10, CIFAR-100, and ImageNet16-120.  Code to run the experiments on the DARTS search space over ImageNet and the NAS-Bench-1Shot1 search spaces will be made available in forked repos subsequently.  

First build the docker image using the provided docker file:
`docker build -t [name] -f docker/config.dockerfile .`

Then run a container with the image, e.g.:
`docker run -it --gpus all --rm [name]`

Then run the commands below from within the container.  The [scripts](scripts) provided may be helpful.

## DARTS Search Space on CIFAR-10
Search using GAEA PC-DARTS by running
~~~
python train_search.py 
  mode=search_pcdarts 
  nas_algo=eedarts 
  search_config=method_eedarts_space_pcdarts 
  run.seed=[int] 
  run.epochs=50
  run.dataset=cifar10
  search.single_level=false
  search.exclude_zero=false
~~~

Evaluate architecture found in search phase by running
~~~
python train_aws.py
  train.arch=[archname which must be specified in cnn/search_spaces/darts/genotypes.py]
  run.seed=[int]
  train.drop_path_prob=0.3
~~~
  
## NAS-Bench-201 Search Space
Search using GAEA DARTS by running
~~~
python train_search.py
  mode=search_nasbench201
  nas_algo=edarts
  search_config=method_edarts_space_nasbench201
  run.seed=[int]
  run.epochs=25
  run.dataset=[one of cifar10, cifar100, or ImageNet16-120]
  search.single_level=[true for ERM and false for bilevel]
  search.exclude_zero=true
~~~

