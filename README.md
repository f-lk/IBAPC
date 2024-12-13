# Invisible Backdoor Attack against 3D Point Cloud Classifier in Graph Spectral Domain
This repository provides the pytorch implementatin of our work: [Invisible Backdoor Attack against 3D Point Cloud Classifier in Graph Spectral Domain](https://ojs.aaai.org/index.php/AAAI/article/view/30099).

## Abstract

3D point cloud has been wildly used in security crucial domains, such as self-driving and 3D face recognition. Backdoor attack is a serious threat that usually destroy Deep Neural Networks (DNN) in the training stage. Though a few 3D backdoor attacks are designed to achieve guaranteed attack efficiency, their deformation will alarm human inspection. To obtain invisible backdoored point cloud, this paper proposes a novel 3D backdoor attack, named IBAPC, which generates backdoor trigger in the graph spectral domain. The effectiveness is grounded by the advantage of graph spectral signal that it can induce both global structure and local points to be responsible for the caused deformation in spatial domain. In detail, a new backdoor implanting function is proposed whose aim is to transform point cloud to graph spectral signal for conducting backdoor trigger. Then, we design a backdoor training procedure which updates the parameter of backdoor implanting function and victim 3D DNN alternately. Finally, the backdoored 3D DNN and its associated backdoor implanting function is obtained by finishing the backdoor training procedure. Experiment results suggest that IBAPC achieves SOTA attack stealthiness from three aspects including objective distance measurement, subjective human evaluation, graph spectral signal residual. At the same time, it obtains competitive attack efficiency. The code is available at https://github.com/f-lk/IBAPC.



## Installation

The code is based on pytorch version [DGCNN](https://github.com/WangYueFt/dgcnn/tree/master/pytorch)


## Data Preparation

Please download ModelNet40 dataset (modelnet40_ply_hdf5_2048) ./data folder.

## Create Frequency Data

Create 3D point cloud frequency data

```shell
python Create_Data.py
```
The training\inference data and their frequency siganl will be saved in ./Created_Data. 
The following backdoor attack is conducted on these data. 

## Inference

You can firstly visualize the backdoored sample, and obtain the ASR using our pretrained parameters in ./pretrained-results
```shell
python IBAPC.py --eval True
```
open the funciton show_pl() to visualize the backdoored sample.
## Backdoor Attack

Implant backdoor trigger into the victim DGCNN

```shell
python IBAPC.py --eval False
```

By completing the backdoor attack, we can obtain the optimal frequency backdoor trigger suffixed with .npy in ./record_one folder, and the corresponding checkpoints in ./checkpoints folder.

## Citation

```
@inproceedings{fan2024invisible,
  title={Invisible Backdoor Attack against 3D Point Cloud Classifier in Graph Spectral Domain},
  author={Fan, Linkun and He, Fazhi and Si, Tongzhen and Tang, Wei and Li, Bing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={19},
  pages={21072--21080},
  year={2024}
}
```

## Acknowledgements

This respository is mainly based on [DGCNN](https://github.com/WangYueFt/dgcnn/tree/master/pytorch). Thanks for the wonderful work!
