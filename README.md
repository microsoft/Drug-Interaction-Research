Improved Drug-target Interaction Prediction with Intermolecular Graph Transformer
=================================================================================

**Authors**: Siyuan Liu*, Yusong Wang*, Yifan Deng, Liang He, Bin Shao, Jian Yin, Nanning Zheng, Tie-Yan Liu, Tong Wang#
(*equal contribution, #corresponding author)

# Introduction
The identification of active binding drugs for target proteins (referred to as drug-target interaction prediction) is the key challenge in virtual screening, which plays an essential role in drug discovery. Although recent deep learningbased approaches achieve better performance than molecular docking, existing models often neglect topological or spatial of intermolecular information, hindering prediction performance. We recognize this problem and propose a novel approach called the Intermolecular Graph Transformer (IGT) that employs a dedicated attention mechanism to model intermolecular information with a three-way Transformer-based architecture. IGT outperforms state-of-the-art (SoTA) approaches by 9.1% and 20.5% over the second best option for binding activity and binding pose prediction respectively, and exhibits superior generalization ability to unseen receptor proteins than SoTA approaches. Furthermore, IGT exhibits promising drug screening ability against SARS-CoV-2 by identifying 83.1% active drugs that have been validated by wetlab experiments with near-native predicted binding poses.

![image](https://user-images.githubusercontent.com/29945329/163564297-4e651e96-d76d-4e6a-ab62-2212e07322b2.png)


# Prerequisites

Please first download the data and model checkpoint using the `prerequisites.sh`.

```bash
bash prerequisites.sh
```

The following Python packages are needed for running the code:

```
numpy
scipy
pandas
scikit-learn
torch
rdkit
oddt
hydra-core
```


# Run evaluation

Below is an example command line for evaluting the model on the test set of LIT-PCBA (assuming two GPUs):

```bash
cd code
python -m torch.distributed.launch \
    --nnode=1 --node_rank=0 --nproc_per_node=2 --use_env evaluate.py \
    hydra.run.dir=$PWD \
    local_world_size=2 \
    data_fpath=../data/lit-pcba dataset=lit-pcba batch_size=1 \
    +model=igt \
    load_from=../ckpt/igt.pt output_path=lit-pcba.eval.csv data_keys=keys/lit-pcba.pkl
```

To run evaluation on MUV, change all occurences of "lit-pcba" in the command line to "muv".

