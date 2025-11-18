# Predicting Optimal Intervention Timing in Mixed Reality Collaboration

EECS 215 — Data Mining & Modeling Project

## Team Members

* Zhijie Liu (Lilith) — zhijiel9@uci.edu
* Yudong Wan — yudongw4@uci.edu
* Kai Yao — kyao12@uci.edu
* Yancheng Chen — yanchec2@uci.edu

## Overview

Mixed Reality (MR) environments enable users to collaborate through speech, movement, and shared attention.
However, when these behaviors become unbalanced—such as one member dominating or others disengaging—overall collaboration efficiency decreases.

This project investigates whether behavioral cues in MR can predict when system-guided interventions should occur to improve teamwork efficiency.

We aim to build interpretable, data-driven models that:

* Detect early signs of imbalance or poor coordination
* Predict upcoming low-efficiency collaboration periods
* Suggest optimal timing for system interventions

Applications include education, remote collaboration, and team training in MR.

## Dataset

### Pairwise Features (per second)

* speaking_entropy – unpredictability of turn-taking
* dominance_ratio – speaking-time imbalance
* material_diversity – number of unique jointly attended virtual objects
* joint_att_count – shared gaze fixations
* dist_mean – average physical distance
* prox_binary – time within 1.5 ft
* approach_rate – movement speed toward/away

### Network Metrics (session-level & temporal)

* Density
* Reciprocity
* Eigenvector centrality

### Task Performance

* Completion time
* Object interaction logs (timestamps, participant actions)

## Technical Approach
1. Feature Engineering

We aggregate pairwise and network-level metrics to capture:

* Micro-level behaviors (speech balance, proximity, gaze synchrony)
* Macro-level coordination patterns (cohesion, dominance, engagement)
* Temporal evolution of group dynamics across the task

2. Modeling Approaches

We compare multiple families of models to balance interpretability and predictive power:

🔹 Hidden Markov Models (HMMs)

* Reveal latent coordination states
* Interpret transitions from “balanced” to “imbalanced” teamwork

🔹 Long Short-Term Memory Networks (LSTMs)

* Capture long-range temporal dependencies
* Detect gradual shifts toward low-efficiency collaboration

🔹 Random Forest & XGBoost

* Supervised classification of time windows as __stable vs. intervention-needed__
* Robust to noise, provide feature importance

🔹 Network Analytics

* Identify how changes in density, reciprocity, and centrality correlate with performance drops
* Enables interpretable, sociometric insights

## Evaluation

Model evaluation will follow:

* Accuracy / F1 score for classification-based approaches
* Window-level prediction precision for intervention timing
* Correlation with task performance (completion time, interaction diversity)
* Interpretability analysis (feature importance, state transitions, network shifts)

## Data Privacy Notice

This repository does not contain the original MR dataset due to privacy and research-use restrictions.

Data files are:

* Stored locally for each collaborator
* Shared only through the private internal folder
* Automatically excluded via "".gitignore""

## Future Work

* Real-time intervention prediction model
* Visualization of collaboration state transitions
* Graph-based temporal modeling
* Extensions toward causal inference for collaboration outcomes
* Preparing results for potential workshop or poster submission
