# C-HMCNN

Code and data for the paper "[Coherent Hierarchical Multi-label Classification Networks](https://proceedings.neurips.cc//paper/2020/file/6dd4e10e3296fa63738371ec0d5df818-Paper.pdf)". 

## Evaluate C-HMCNN

In order to evaluate the model for a single seed run:

```
  python main.py --dataset <dataset_name> --seed <seed_num> --device <device_num>
```

Example:

```
  python main.py --dataset cellcycle_FUN --seed 0 --device 0
```

**Note:** the parameter passed to "dataset" must end with: '_FUN', '_GO', or '_others'.

If you want to execute the model for 10 seeds you can modify the script ```main_script.sh``` and execute it.

The results will be written in the folder ```results/``` in the file ```<dataset_name>.csv```.

## Hyperparameters search

If you want to execute again the hyperparameters search you can modify the script ```script.sh```according to your necessity and execute it. 

## Architecture

The code was run on a Titan Xp with 12GB memory. A description of the environment used and its dependencies is given in ```c-hmcnn_enc.yml```.

By running the script ```main_script.sh``` we obtain the following results (average over the 10 runs):

| Dataset          | Result |
| ---------------- | ------ |
| Cellcycle_FUN    | 0.255  |
| Derisi_FUN       | 0.195  |
| Eisen_FUN        | 0.306  |
| Expr_FUN         | 0.302  |
| Gasch1_FUN       | 0.286  |
| Gasch2_FUN       | 0.258  |
| Seq_FUN          | 0.292  |
| Spo_FUN          | 0.215  |
| Cellcycle_GO     | 0.413  |
| Derisi_GO        | 0.370  |
| Eisen_GO         | 0.455  |
| Expr_GO          | 0.447  |
| Gasch1_GO        | 0.436  |
| Gasch2_GO        | 0.414  |
| Seq_GO           | 0.446  |
| Spo_GO           | 0.382  |
| Diatoms_others   | 0.758  |
| Enron_others     | 0.756  |
| Imclef07a_others | 0.956  |
| Imclef07d_others | 0.927  |

## Reference

```
@inproceedings{giunchiglia2020neurips,
    title     = {Coherent Hierarchical Multi-label Classification Networks},
    author    = {Eleonora Giunchiglia and
               Thomas Lukasiewicz},
    booktitle = {34th Conference on Neural Information Processing Systems (NeurIPS 2020)},
    address = {Vancouver, Canada},
    month = {December},
    year = {2020}
}
```

## GCP VM Instance Setup Guide

Use the following command in GCP Cloud Shell to create a VM instance

```
export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="us-west1-b"
export INSTANCE_NAME="cs7643-fp"
gcloud compute instances create $INSTANCE_NAME   --zone=$ZONE   --image-family=$IMAGE_FAMILY   --image-project=deeplearning-platform-release   --maintenance-policy=TERMINATE --machine-type=n1-standard-4  --accelerator="type=nvidia-tesla-t4,count=1" --boot-disk-size=100GB  --metadata="install-nvidia-driver=True"
```

## To run Transformer/RNN model on the Enron dataset:

for fc model

```
python train.py --model fc --dataset enrontext_others --batch_size 4 --lr 1e-5 --dropout 0.7 --hidden_dim 1000 --num_layers 3 --weight_decay 1e-5 --non_lin 'relu' --num_epochs 10 
```

for transformer model

```
python train.py --model transformer --dataset enrontext_others --batch_size 4 --lr 1e-5 --dropout 0.7 --hidden_dim 1000 --num_layers 3 --weight_decay 1e-5 --non_lin 'relu' --num_epochs 10
```

for RNN model

```
python train.py --model lstm --dataset enrontext_others --batch_size 4 --lr 1e-5 --dropout 0.7 --hidden_dim 1000 --num_layers 3 --weight_decay 1e-5 --non_lin 'relu' --num_epochs 10
```

## To run CNN model on the Diatom dataset:

1. Download the Diatom dataset: [Diatom_Data_cs7643fp_20230504.tgz - Google Drive](https://drive.google.com/file/d/1oiDl6j4_IqyIdZEb6HqJkV1_zuqGKWO6/view?usp=sharing)

2. Untar it in a folder **In parallel** to the C-HMCNN folder.

3. Go to the C-HMCNN folder

4. Run the command: `python main_awcnn.py --dataset diatoms_others --seed 0 --device 0 --model CNN`