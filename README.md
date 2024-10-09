# Interdependency Matters: Graph Alignment for Multivariate Time Series Anomaly Detection (ICDM 2024)
This repository provides a PyTorch implementation of MADGA, which transforms the unsupervised anomaly detection to graph alignment problem.

## Framework
![Framework](./asset/framework.png)

## Main results
![Results](./asset/results.png)

## Data
We test our method for five publicly processed datasets, e.g., ```SWaT```, ```WADI```, ```PSM```, ```MSL```, and ```SMD```.

- [`SWaT`](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat)
- [`WADI`](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#wadi)
- [`PSM`](https://github.com/eBay/RANSynCoders/tree/main/data) is released in [`RANSynCoders`](https://github.com/eBay/RANSynCoders/tree/main).
- [`MSL`](https://github.com/d-ailin/GDN/tree/main/data/msl) is released in [`GDN`](https://github.com/d-ailin/GDN/tree/main).
- [`SMD`](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset) is released in [`OmniAnomaly`](https://github.com/NetManAIOps/OmniAnomaly).

```sh
mkdir Dataset
cd Dataset
mkdir input
```
Download the dataset in ```Data/input```.

## Train
- train for MITGFlow
For example, training for WADI
```sh
sh runners/run_WADI.sh
```
- train for ```DeepSVDD```, ```DeepSAD```, ```DROCC```, and ```ALOCC```. 
```sh
python3 train_other_model.py --name SWaT --model DeepSVDD
```
- train for ```USAD``` and ```DAGMM```
We report the results by the implementations in the following links: 
[`USAD`](https://github.com/manigalati/usad) and [`DAGMM`](https://github.com/danieltan07/dagmm/)

## Test
We provide the pretained model of MTGFlow.

For example, testing for WADI 
```sh
sh runners/run_WADI_test.sh
```
## BibTex Citation

If you find this paper or repository helpful, please cite our paper. Thanks a lot.
