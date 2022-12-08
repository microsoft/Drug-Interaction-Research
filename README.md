# DSN-DDI: an accurate and generalized framework for drug-drug interaction prediction by dual-view representation learning

**Authors**: Zimeng Li#, Shichao Zhu#, Bin Shao*, Xiangxiang Zeng*, Tong Wang* and Tie-Yan Liu  
#Equal contribution  
*Corresponding authors: binshao@microsoft.com (B. S.); xzeng@hnu.edu.cn (X. Z.); watong@microsoft.com (T. W., Lead contact)  

**Article**:

# Introduction
Drug-drug interaction (DDI) prediction identifies interactions of drug combinations in which the adverse side effects caused by the physicochemical incompatibility have attracted much attention. Previous studies usually model drug information from single or dual views of the whole drug molecules but ignore the detailed interactions among atoms, which leads to incomplete and noisy information and limits the accuracy of DDI prediction. In this work, we propose a novel dual-view drug representation learning network for DDI prediction (“DSN-DDI’’), which employs local and global representation learning modules iteratively and learns drug substructures from the single drug (“intra-view”) and the drug pair (“inter-view”) simultaneously. Comprehensive evaluations demonstrate that DSN-DDI significantly improved performance on DDI prediction for the existing drugs by achieving a relatively improved accuracy of 13.01% and an over 99% accuracy under the transductive setting. More importantly, DSN-DDI achieves a relatively improved accuracy of 7.07% to unseen drugs and shows the usefulness for real-world DDI applications. Finally, DSN-DDI exhibits good transferability on synergistic drug combination prediction and thus can serve as a generalized framework in the drug discovery field.
![image](https://github.com/microsoft/IGT-Intermolecular-Graph-Transformer/blob/DSN-DDI-for-DDI-Prediction/DSN-DDI.jpg)

# Requirement
To run the code, you need the following dependencies:
* Python == 3.7
* PyTorch == 1.9.0
* PyTorch Geometry == 2.0.3
* rdkit == 2020.09.2

# Installation
You can create a virtual environment using conda 
```bash
conda create -n DSN-DDI python=3.7
conda activate DSN-DDI
conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric==2.0.3
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
conda install -c rdkit rdkit
```

# Dataset
We provide the preprocessed DrugBank dataset for transductive setting and inductive setting, and Twosides dataset for transductive setting. 
All datasets can be downloaded via the commands
```bash
wget https://msracb.blob.core.windows.net/pub/DSN-DDI-dataset.zip
unzip DSN-DDI-dataset.zip
rm DSN-DDI-dataset.zip
cp -r dataset/drugbank  drugbank_test
cp -r dataset/inductive drugbank_test
cp -r dataset/twosides twosides_test
```

# Usage
Below is an example for evaluting the model on the test dataset of drugbank in transductive setting and inductive setting, and twosides in transductive setting
The output is the metrics of our model evaluting on test dataset.

for evaluating the performance of our model on drugbank dataset in transductive setting, you need to run:
```bash
python -u drugbank_test/transductive_test.py
```
for evaluating the performance of our model on drugbank dataset in inductive setting, you need to run:
```bash
python -u drugbank_test/inductive_test.py
```
for evaluating the performance of our model on twosides dataset in transductive setting, you need to run:
```bash
python -u twosides_test/test.py
```
If you want to train the model, you can use the commands below:

for training the model on drugbank dataset in transductive setting, you need to run:
```bash
python -u drugbank_test/transductive_train.py
```
for training the model on drugbank dataset in inductive setting, you need to run:
```bash
python -u drugbank_test/inductive_train.py
```
for training the model on twosides dataset in transductive setting, you need to run:
```bash
python -u twosides_test/train_script.py
```

The default command is
 ```bash
python -u drugbank_test/transductive_test.py
```
If you need to build the docker, it spends about 10 min to run the code. If you don't need to build the docker, it spends about 5 min to run the code.

# Dataset preparation on your data
You need to provide datasets defined as below:

DDI dataset file:
The DDI dataset file stores all the ddi, the form is drugA,drugB,type

Drug smiles file:
The drug smiles file stores the smiles of drugs, the form is drug,smiles

Train DDI file and test DDI file:
The train DDI file and test DDI file store the positive train dataset and test dataset respectively, the negative ddis will be random generated, the form is drugA,drugB,type.



