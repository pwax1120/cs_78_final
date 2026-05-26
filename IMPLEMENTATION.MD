# Implementation Plan

## Data Set and Model Acquistion
- [ ] Choose between (Standard Images: CIFAR-100, Oxford Flowers, Stanford Cars), (Medical Images: PatchCamelyon, EuroSAT) 
- [ ] Import Datasets
- [ ] Choose which version of SimCLR's pretrained models to use
- [ ] Import SimCLR model

## Set Up Data Labeling and Training Pipeline
- [ ] Establish the Fraction Values to use, exponential and based on the amount of label from the datasets
- [ ] Determine set LR's, epoch values etc. to ensure continuity
- [ ] Write seperate programs for fine-tune and linear-evaluations
    - [ ] Handle seperate seed cases
    - [ ] Parameters that track how many labels from each dataset to take for the training, set which data you will use for all of those, so for example if there is 5, 10, 15 so on then set the 5 dataset, set the 10 dataset and so on
    - [ ] Linear-Evaluations
        - [ ] Add the linear-eval layer and to the SimCLR model
    - [ ] Fine-tuning Use Case (not sure entirely how to go about this but will figure out)

## Write a Testing Pipeline
- [ ] Use a set amount of random images from each dataseet to test the accuracy of the model
- [ ] Save the output values to csv's or data

## Write an Overall Expirement Loop
- [ ] Allow the system to run testing cases for each dataset, each type of finetuning and each dataset
- [ ] Find a way to automate this so that it is running in the background


# Claude's Implementation Plan

# Implementation Plan

## 1. Dataset and Model Acquisition
- [ ] Pick datasets: one standard (CIFAR-100 recommended) + one domain-shifted (PatchCamelyon recommended)
- [ ] Import datasets via torchvision/HF datasets
- [X] Download SimCLR pretrained checkpoint (ResNet-50 1x, 1000 epochs — via lightly or solo-learn for PyTorch)
- [ ] Verify checkpoint loads and produces sensible embeddings on a few sample images (sanity check)

## 2. Data Pipeline
- [ ] Define label fractions (e.g., 5, 10, 25, 50, 100, 250 per class — log-spaced)
- [ ] Write label-sampling function: given (dataset, N_per_class, seed) → return fixed labeled subset
- [ ] Critical: same labeled subset used across all three conditions within a seed (paired comparison)
- [ ] Define held-out test set (fixed across all experiments) and validation split for LR tuning
- [ ] Decide on input resolution — pretrained encoder expects 224x224; resize accordingly

## 3. Training Conditions (three, not two)
- [ ] Linear evaluation: freeze encoder, train linear head on top of h
- [ ] Fine-tuning: encoder + new head trained end-to-end with small LR
- [ ] From-scratch baseline: same architecture, random init (quantifies pretraining value)
- [ ] Brief LR tuning on validation split for each condition; commit and document

## 4. Experiment Loop
- [ ] Nested loop: dataset × label_fraction × condition × seed (5 seeds)
- [ ] Log per-run: final test acc, best val acc, training curves, hyperparameters, seed
- [ ] Save checkpoints + CSV logs (or Weights & Biases)
- [ ] Run small pilot first (1 dataset, 2 fractions, 2 seeds) end-to-end before scaling up

## 5. Analysis
- [ ] Aggregate: mean ± std across seeds per (dataset, fraction, condition) cell
- [ ] Plot: accuracy vs. label count, one panel per dataset, three lines per panel
- [ ] Identify crossover points; compare across datasets