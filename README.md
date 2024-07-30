<div align="center">    
 
# Let Them Drop: Scalable and Efficient Secure Federated Learning Solutions Agnostic to Client Stragglers
![Inria](https://img.shields.io/badge/-INRIA-red) 
![Eurecom](https://img.shields.io/badge/-EURECOM-blue) <br> 
[![Conference](https://img.shields.io/badge/ARES-2024-red)](https://www.ares-conference.eu/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
<br>
![Image Results](github_images/logo.png)
</div>

## Description

This repository contains the official code of the research paper [Let Them Drop: Scalable and Efficient Secure Federated Learning Solutions Agnostic to Client Stragglers](https://dl.acm.org/doi/abs/10.1145/3664476.3664488) pusblished at [ARES 2024](https://www.ares-conference.eu/).<br>


## Abstract
> Secure Aggregation (SA) stands as a crucial component in modern Federated Learning (FL) systems, facilitating collaborative training of a global machine learning model while protecting the privacy of individual clients' local datasets. Many existing SA protocols described in the FL literature operate synchronously, leading to notable runtime slowdowns due to the presence of *stragglers* (i.e. late-arriving clients).
To address this challenge, one common approach is to consider stragglers as client failures and use SA solutions that are robust against dropouts. While this approach indeed seems to work, it unfortunately affects the performance of the protocol as its cost strongly depends on the dropout ratio and this ratio has increased significantly when taking stragglers into account. Another approach explored in the literature to address stragglers is to introduce asynchronicity into the FL system. Very few SA solutions exist in this setting and currently suffer from high overhead.
In this paper, similar to related work, we propose to handle stragglers as client failures but design SA solutions that do not depend on the dropout ratio so that an unavoidable increase on this metric does not affect the performance of the solution. We first introduce *Eagle*, a synchronous SA scheme designed not to depend on the client failures but on the online users' inputs only. This approach offers better computation and communication costs compared to existing solutions under realistic settings where the number of stragglers is high. We then propose *Owl*, the first SA solution that is suitable for the asynchronous setting and once again considers online clients' contributions only.
We implement both solutions and show that: _i)_ in a synchronous FL with realistic dropout rates (that takes potential stragglers into account), *Eagle* outperforms the best SA solution, namely *Flamingo*, by $\times 4$; _ii)_ In the asynchronous setting, *Owl* exhibits the best performance compared to the state-of-the-art solution *LightSecAgg*.

## How to run
### Dependecies
You'll need a working Python environment to run the code. 
The recommended way to set up your environment is through the [Anaconda Python distribution](https://www.anaconda.com/products/distribution)
which provides the `conda` package manager. 
Anaconda can be installed in your user directory and does not interfere with the system Python installation.
### Configuration
- Download the repository: `git clone https://github.com/rtaiello/let_them_drop`
- Create the environment: `conda create -n ltd python=3.7`
- Activate the environment: `conda activate ltd`
- Install the dependencies: `pip install -r requirements.txt`

#### Run 🚀
- `PYTHONPATH=. pytest tests/`
## Authors
* **Riccardo Taiello**  - [github](https://github.com/rtaiello) - [website](https://rtaiello.github.io)
* **Melek Önen**  - [website](https://www.eurecom.fr/en/people/onen-melek)
* **Clémentine Gritti**  - [LinkedIn](https://www.linkedin.com/in/clementine-gritti/?originalSubdomain=fr)
* **Marco Lorenzi**  - [website](https://marcolorenzi.github.io/)

## Code contributions
* FTSA repository - [github](https://github.com/MohamadMansouri/fault-tolerant-secure-agg)
* Flamingo repository - [github](https://github.com/eniac/flamingo/tree/main)
## Cite this work
```
@inproceedings{10.1145/3664476.3664488,
author = {Taiello, Riccardo and \"{O}nen, Melek and Gritti, Cl\'{e}mentine and Lorenzi, Marco},
title = {Let Them Drop: Scalable and Efficient Federated Learning Solutions Agnostic to Stragglers},
year = {2024},
isbn = {9798400717185},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664476.3664488},
doi = {10.1145/3664476.3664488},
booktitle = {Proceedings of the 19th International Conference on Availability, Reliability and Security},
articleno = {13},
numpages = {12},
keywords = {Secure Aggregation, Synchronous and Asynchronous Federated Learning},
location = {Vienna, Austria},
series = {ARES '24}
}
```
