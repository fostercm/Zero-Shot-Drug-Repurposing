# **Knowledge Graph-Based Drug Repurposing**

This repository contains the implementation of a pipeline for knowledge graph-based drug discovery, focusing on drug-disease relationships such as indications and contraindications. The pipeline processes data, trains models, and evaluates their performance, providing meaningful insights for drug repurposing.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
    - [Generate BERT Embeddings](#generate-bert-embeddings)
    - [Process Data](#process-data)
    - [Train Models](#train-models)
    - [Drug Search](#drug-search)
4. [Acknowledgments](#acknowledgements)
5. [License](#license)

---

## **Overview**
This project uses a heterogeneous knowledge graph and transformer-based embeddings to predict drug-disease relationships. It supports:
- Processing raw data into graph objects for use in machine learning tasks.
- Integrating BERT embeddings to enrich text-based features.
- Training and fine-tuning a Knowledge Graph Link Predictor model.
- Visualizing performance through ROC curves and confusion matrices.
- Identifying potential drug candidates for specific diseases.

---

## **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/knowledge-graph-drug-discovery.git
   cd knowledge-graph-drug-discovery
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure GPU support on a CUDA-compatible device

## **Usage**

To run the pipeline, follow these steps:

### **Generate BERT Embeddings**:

* Utility
    * Improve model initialization
    * Faster convergence
    * Higher few-shot accuracy

* Config file template
  ```json
  {
  "BERT_model": "dmis-lab/biobert-v1.1",
  "feature_table_path" : "path.csv",
  "features" : ["feature1","feature2","feature3"],
  "max_sentence_length" : 500,
  "device" : "cuda",
  "batch_size" : 128,
  "output_path" : "path.pt"
  }
  ```

* Execution
  ```bash
  python BERT_Embedder.py BERT_config.json
  ```

### **Process Data**:

* Utility
    * Process knowledge graph into a PyG data object
    * Create dataloaders for pretraining and finetuning
    * Save dataloaders for future use

* Config file template
     ```json
     {
      "graph_file" : "raw_graph_path.csv",
      "BERT_files" : {
          "disease_names" : "disease_names.csv",
          "disease_embeddings" : "disease_embeddings.pt",
          "drug_names" : "drug_names.csv",
          "drug_embeddings" : "drug_embeddings.pt"
      },
      "embedding_dim" : 768,
      "device" : "cuda",
      "batch_size" : 256,
      "k" : 10,
      "output_paths" : {
          "pretrain" : "pretrain_path.pt",
          "train" : "train_path.pt",
          "val" : "val_path.pt",
          "test" : "test_path.pt",
          "data_obj" : "data_path.pt",
          "graph" : "processed_graph_path.pt"
      }
     }
     ```

* Execution
     ```bash
     python Process_Data.py data_config.json
     ```

### **Train Models**:

* Utility
    * Bulk training
    * Hyperparameter tuning
    * Pretraining + finetuning
    * Train and val plots, confusion matrices, and ROC curve + AUC

* Config file template
     ```json
     {
      "model_path" : "models",
      "model_name" : "MIGs-iEndos",
      "plot_path" : "plots",
      "data_paths" : {
          "pretrain" : "processed_data/ptloader.pt",
          "train" : "processed_data/trloader.pt",
          "val" : "processed_data/vloader.pt",
          "data_obj" : "processed_data/data.pt"
      },
      "params" : {
          "learning_rate" : 0.001,
          "hidden_dim" : [128,256],
          "num_heads" : [1],
          "num_layers" : [2,4,6]
      },
      "batch_size" : 4096,
      "epochs" : 100,
      "device" : "cuda"
     }
     ```
 
 * Execution
     ```bash
     python Train_Model.py model_config.json
     ```
     
### **Drug Search**:

* Utility
    * Prediction of drug indication/contraindication
    * Return topk candidates
    * Visible score distributions

* Config file template
     ```json
     {
      "test_diseases" : ["iEndos"],
      "data_paths" : {
          "data_obj" : "processed_data/data.pt",
          "kg" : "processed_data/kg.pt"
      },
      "model_params" : {
          "hidden_dim" : 128,
          "num_heads" : 1,
          "num_layers" : 6
      },
      "model_path" : "models/D128_H1_L6.pt",
      "output_path" : "predictions/D128_H1_L6/iEndos.txt",
      "plot_path" : "predictions/D128_H1_L6/iEndos.png",
      "k" : 30
     }
     ```

* Execution
     ```bash
     python Drug_Search.py search_config.json
     ```

## **Acknowledgements**

I would like to express my gratitude to the following  organizations for their contributions to this project:

  - **Hugging Face**: For the BERT models and tokenization libraries used in embedding generation
  - **PyTorch Geometric (PyG)**: For providing an excellent library for working with graph data and enabling efficient GAT computations
  - **The Zitnik Lab**: For their TxGNN work that inspired this project

## **License**

This project is licensed under the MIT License. See the full license details below:
