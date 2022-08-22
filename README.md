# AAS-project


## Introduction
Meta Reinforcement Learning, in short, is to do meta-learning in the field of reinforcement learning. Usually the train and test tasks are different but drawn from the same family of problems.

*Deep reinforcement learning (deep RL) has been successful in learning sophisticated behaviors automatically; however, the learning process requires a huge number of trials. In contrast, animals can learn new tasks in just a few trials, benefiting from their prior knowledge about the world. The RL2 algorithm is encoded in the weights of the RNN, which are learned slowly through a general-purpose ("slow") RL algorithm. The RNN receives all information a typical RL algorithm would receive, including observations, actions, rewards, and termination flags; and it retains its state across episodes in a given Markov Decision Process (MDP). The activations of the RNN store the state of the "fast" RL algorithm on the current (previously unseen) MDP.*
(source: https://arxiv.org/abs/1611.02779)

### Documentation
You can find the report of the project [here](./report.pdf).


## Getting Started
The weights of all the trained models are available at the following [link](https://drive.google.com/file/d/1n7UWrFqtV7kDPFTmXopX99uZsP4ddxM8/view?usp=sharing). 

Once you have downloaded it, you have to place the unzipped folders `history` and `saved_models` 
inside the `/tmp` folder (at the root level of the project).


## Installation
Below you can find all the scripts for installing based on your OS/processor
```
$ make
    > "+------------------------------------------------------+"
    > "|         OS         |  Hardware  |    Setup Command   |"
    > "+------------------------------------------------------+"
    > "|   Windows/Linux    |   - GPU    |  'make setup.CPU'  |"
    > "|   Windows/Linux    |   + GPU    |  'make setup.GPU'  |"
    > "|    Apple macOS     |    + M1    |  'make setup.M1'   |"
    > "|    Apple macOS     |    - M1    |  'make setup.CPU'  |"
    > "+------------------------------------------------------+"
```
for instance, if you have MacOS with Intel chip you have to run:
```
$ make setup.CPU
```
or alternatively you can find all the different version of the `requirements` inside the `/tools/requirements` folder.

### Apple M1
If you are using the new Apple M1 chip please be sure to have installed `hdf5` by running:
```
$ brew install hdf5
```


## Running the Tests
### Usage
```
$ python src/run.py -h
    usage: run.py [-h] --config CONFIG

    MetaRL-RL2

    optional arguments:
        -h, --help       show this help message and exit
        --config CONFIG  path of the configurations json file.
```

### Configurations
The base structure of the json configurations file can be found inside [this file](./configs/default.json). Look at the [configs](./configs/) folder for additional configuration examples.

### Training
In order to train all the models from scratch, you can uncomment all the rows inside [this file](./scripts/train.all.sh). Once you have done that, you have to run:
```
$ sh scripts/train.all.sh
```
or you can simply execute once configuration at time by running:
```
$ python src/run.py "<path_of_your_json_config>"
```
where the json file has the parameter `mode="training"`.

### Inference
In order to run the inference procedure on all the models, you can uncomment all the rows inside [this file](./scripts/test.all.sh). Once you have done that, you have to run:
```
$ sh scripts/test.all.sh
```
or you can simply execute once configuration at time by running:
```
$ python src/run.py "<path_of_your_json_config>"
```
where the json file has the parameter `mode="inference"`.

### Rendering
In order to run the rendering procedure on all the models, you can uncomment all the rows inside [this file](./scripts/render.all.sh). Once you have done that, you have to run:
```
$ sh scripts/render.all.sh
```
or you can simply execute once configuration at time by running:
```
$ python src/run.py "<path_of_your_json_config>"
```
where the json file has the parameter `mode="render"`.


## References
Papers
 - [*Reinforcement Learning, Fast and Slow*](https://www.cell.com/action/showPdf?pii=S1364-6613%2819%2930061-0)
 - [*Learning to Reinforcement Learn*](https://arxiv.org/abs/1611.05763)
 - [*RL2: Fast Reinforcement Learning via Slow Reinforcement Learning*](https://arxiv.org/abs/1611.02779)

Environments:
 - [*MetaGym*](https://github.com/PaddlePaddle/MetaGym)
 - [*openai/gym*](https://github.com/openai/gym)


## Authors
- **Matteo Conti** - [GitHub](https://github.com/contimatteo)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details