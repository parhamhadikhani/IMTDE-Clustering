# Improved data clustering using multi-trial vector-based differential evolution with Gaussian crossover

## Introduction


Multi-Trial Vector-based Differential Evolution (MTDE) is an improved differential evolution (DE) algorithm that is done by combining three strategies and distributing the population between these strategies to avoid getting stuck at a local optimum. In addition, it records inferior solutions to share information of visited regions with solutions of the next generations. This repository contains, contains the implementation of the [Improved version of the Multi-Trial Vector-based Differential Evolution (IMTDE)](https://arxiv.org/...) algorithm for clustering data ([the full version of this work is available on Arxiv](https://arxiv.org/...)). The purpose IMTDE is to enhance the balance between the exploration and exploitation mechanisms in MTDE by employing Gaussian crossover and modifying the sub-population distribution between the strategies.

![arch](/img/DIAGRAM.jpg)


Please cite our paper if you use this code in your research.
```
@article{Hadikhani2022,
author = "Parham Hadikhani and Daphne Teck Ching Lai and Wee-Hong Ong and Mohammad H.Nadimi-Shahraki",
title = "{Improved data clustering using multi-trial vector-based differential evolution with Gaussian crossover}",
year = "2022",
month = "4",
url = "https://www.techrxiv.org/articles/preprint/Improved_data_clustering_using_multi-trial_vector-based_differential_evolution_with_Gaussian_crossover/19604527",
doi = "10.36227/techrxiv.19604527.v1"
}
```
