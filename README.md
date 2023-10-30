# Protein Secondary Structure Prediction Using Contact Maps
-------------------------------------------------------

#### In this study, we tackle the challenging task of predicting secondary structures from protein primary sequences, a pivotal initial stride towards predicting tertiary structures, while yielding crucial insights into protein activity, relationships, and functions. Existing methods often utilize extensive sets of unlabeled amino acid sequences. However, these approaches neither explicitly capture nor harness the accessible protein 3D structural data, which is recognized as a decisive factor in dictating protein functions. To address this, we utilize protein residue graphs and introduce various forms of sequential or structural connections to capture enhanced spatial information. We adeptly combine Graph Neural Networks (GNNs) and Language Models (LMs), specifically utilizing a pre-trained transformer-based protein language model to encode amino acid sequences and employing message-passing mechanisms like GCN and R-GCN to capture geometric characteristics of protein structures. Employing convolution within a specific node's nearby region, including relations, we stack multiple convolutional layers to efficiently learn combined insights from the protein's spatial graph, revealing intricate interconnections and dependencies in its structural arrangement. To assess our model's performance, we employed the training dataset provided by NetSurfP-2.0, which outlines secondary structure in 3- and 8-states. Extensive experiments show that our proposed model, SSRGNet surpasses the baseline on Accuracy and F1-scores.

<!-- display the image -->
<p align="center">
  <img src="assets\architecture.jpg" width="400" height="300">

## Installing

### Data Preprocessing Part
Soon we will release a drive link which will contain all the necessary files to run the code. For now, we will explain how to generate the files required for training.

1. Fetch the PDB files
    ```
    python scripts\preprocessing_scripts\preprocess.py
    ```
2. Calculation the distances between c_alpha items in the pdb files
    ```
    python scripts\preprocessing_scripts\distance.py
    ```

3. For language model training we need files with pdbids so we can map the sequence with the respective adjacency matrix. To generate the `df_final.xlsx, cb513_final.xlsx`, etc... files run the following `notebooks/data_prepare.ipynb` notebook and also change the path accordingly.

4. After generating the excel file generate the text file containing primary, secondary sequences either in q8 or q3 depending on the task and pdbids. To get the .txt file run `notebooks/pdbfile_process.ipynb`notebook.

5. After generating the distance file and .txt files, we can generate the graph file containing the adjacency matrix by running the following command.

    5.1. For relational data run

    ```
    python scripts/preprocessing_scripts/adjacency_relational.py
    ```

    5.2 For non-relational data run

    ```
    python scripts/preprocessing_scripts/adjacency.py
    ```

6. Last peice of input we required is sequence ids which will be required by the language model in the huggingface format i.e. input_ids, attention_mask, labels, token_type_ids, to generate this file run `notebooks/prepare_raw_tokenization_file.ipynb` notebook and it will generate all the necessary pickle files.

**Note** : To switch from q3 to q8 or vice-versa uncomment the commented code for q3 or q8 respectively.

### Training Files

* Make sure to check the configuration before running any training script and adjust it accordingly.

* Make sure to modify the relational or non-relational dataset as well as the model in the training script.

* To tune and search for the best hyperparameters using wandb's sweep run the following command:

    ```
    CUDA_VISIBLE_DEVICES=0 python scripts/train/MultiModal/train_with_sweep.py
    ```

* To train the baseline model run the following command:

    ```
    CUDA_VISIBLE_DEVICES=0 python scripts/train/MultiModal/train_baseline.py
    ```

* To train the model without wandb run the following command:

    ```
    CUDA_VISIBLE_DEVICES=0 python scripts/train/MultiModal/train_without_wandb.py
    ```

* To train the baseline model using trainer run `scripts/train/MultiModal/supplement_training_files/ProtBert-BFD-FineTune-SS3.py` file.

## Results
-----------------
![Results Table](assets/result.png)