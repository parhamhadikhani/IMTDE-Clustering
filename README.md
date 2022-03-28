# Improved data clustering using multi-trial vector-based differential evolution with Gaussian crossover
An Improved clustering using multi-trial vector-based differential evolution with Gaussian crossover

## Introduction


Multi-Trial Vector-based Differential Evolution (MTDE) is an improved differential evolution (DE) algorithm that is done by combining three strategies and distributing the population between these strategies to avoid getting stuck at a local optimum. In addition, it records inferior solutions to share information of visited regions with solutions of the next generations. This repository contains, contains the implementation of the [Improved version of the Multi-Trial Vector-based Differential Evolution (IMTDE)](https://arxiv.org/...) algorithm for clustering data [the full version of this work is available on Arxiv](https://arxiv.org/...). The purpose IMTDE is to enhance the balance between the exploration and exploitation mechanisms in MTDE by employing Gaussian crossover and modifying the sub-population distribution between the strategies.

![arch](/img/DIAGRAM.jpg)


Please cite our paper when you use this code in your research.
```
@misc{hadikhani2022,
      title={Improved data clustering using multi-trial vector-based differential evolution with Gaussian crossover}, 
      author={Parham Hadikhani },
      year={2022},
      eprint={..},
      archivePrefix={...},
      primaryClass={...}
}
```
