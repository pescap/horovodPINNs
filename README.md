# HorovodPINNs

Data-based parallel acceleration for physics-informed neural networks (PINNs) via Horovod. Its code comes with the preprint entitled:

$h$-analysis and data-parallel physics-informed neural networks, P. Escapil-Inchauspé and G. A. Ruz.

We apply data-based Horovod acceleration to pioneer [PINNs](https://github.com/maziarraissi/PINNs/) code by Raissi, Perdikaris and Karniadakis. Horovod acceleration is inspired by Xihui Meng [Distributed-training-Horovod](https://github.com/XuhuiM/Distributed-training-Horovod) code.

This repository is intended to help users understand how to apply Horovod based data-parallel acceleration to PINNs or physics-informed machine learning schemes. It is also meant to enable users to fully replicate and reproduce the results in our manuscript, including the figures generation.       

Backend: `tensorflow.compat.v1`.

## Content

This repository contains the following folders:
- [PINNs](https://github.com/maziarraissi/PINNs/) submodule;
- [Distributed-training-Horovod](https://github.com/XuhuiM/Distributed-training-Horovod) submodule;
- Laplace: An improved version of [pinn_hvd_data.py](https://github.com/XuhuiM/Distributed-training-Horovod/blob/master/pinn_hvd_data.py);
- Schrodinger: Data-parallel [Schrodinger.py](https://github.com/maziarraissi/PINNs/blob/master/main/continuous_time_inference%20\(Schrodinger\)/Schrodinger.py) forward problem;
- NavierStokes: Data-parallel [NavierStokes.py](https://github.com/maziarraissi/PINNs/blob/master/main/continuous_time_identification%20\(Navier-Stokes\)/NavierStokes.py) inverse problem.

## Clone

To clone this repository along with its submodules, run:

```
git clone https://github.com/pescap/horovodPINNs.git --recurse-submodules 
```

## Run the experiments

To run the experiments, a Docker image with `horovod==0.26.1` can be downloaded and run throughout command:

```
nvidia-docker run -p 8888:8888 pescap/dist-training-horovod-master
```

