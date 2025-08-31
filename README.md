# A GNN implementation for ASD diagnosys 🧠

### <ins>Final Project for Sose25 class GNN at Potsdam Universität </ins>
Note: The ABIDE parser script (used in some preprocessing functions) and fetch_data.py (used to download the dataset)  is from [Population_GCN](https://github.com/parisots/population-gcn). 

## How to use 

### 1. Install Required packeges
Required packages  can be installed by running the following command
```text
```
pip install -r requirements.txt
```
```

### 2. Configuration file editing

 Inside the ABIDE_config folder edit the config_abide.yaml file by specifying the project root folder (it must be a absolute path).

### 3. Data downloading and preprocessing

Using the download_preprocess.py script, download the [Preprocessed ABIDE Dataset](http://preprocessed-connectomes-project.org/abide/)  with the following command
```text
```
python fetch_data.py
```
```

## Train and Configuration Mode

The GCN model (and the baseline CNN model used to compare results) can be trained by using the following command.
```text
```
python main.py --train
```
```
Graph Creation, Model and Training parameters can be changed by editing the file config.yaml. Default settings can be printed by using the following command.

```text
```
python main.py --train
```
```

More infos on the model architecture and preprocessing can be found in the report.pdf file.


