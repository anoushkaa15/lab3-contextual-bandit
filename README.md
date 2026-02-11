# Lab 3 – Contextual Bandit Based News Recommendation System

## Overview

This project implements a Contextual Multi-Armed Bandit (CMAB) framework for personalized news recommendation.  
The system treats user categories as contexts and news categories as arms. A reinforcement learning agent learns optimal category recommendations for each user type by interacting with a simulated reward environment.

The complete pipeline includes:

1. User classification
2. Contextual bandit learning
3. Hyperparameter tuning
4. End-to-end recommendation engine
5. Evaluation and analysis

---

## Datasets

- **train_users.csv / test_users.csv** – User feature data labeled as User1, User2, or User3.
- **news_articles.csv** – News articles with category labels and headlines.

---

## Methodology

### 1. Data Preprocessing

- Removed missing values.
- Applied numeric and categorical encoding using `ColumnTransformer`.
- Split user data into 80% training and 20% validation sets.

---

### 2. User Classification

An XGBoost classifier was trained to predict user context (User1, User2, User3).

Evaluation was performed on a validation split using `classification_report`.

The classifier serves as the context detector for the recommendation engine.

---

### 3. Contextual Bandit Algorithms

Three strategies were implemented:

#### Epsilon-Greedy
- Tested ε ∈ {0, 0.01, 0.1}
- Maintained separate bandits for each user context.

#### Upper Confidence Bound (UCB)
- Tested C ∈ {0.5, 1.0, 2.0}
- Uses optimism-based exploration.

#### SoftMax
- Fixed temperature τ = 1
- Probabilistic arm selection.

Each algorithm was trained for **T = 10,000** steps using the provided sampler.

Expected reward distributions were computed for every (User, Category) pair.

---

### 4. Recommendation Engine

The final CMAB pipeline performs:

1. **Classify** user using trained classifier.
2. **Select Category** using learned bandit policy.
3. **Recommend Article** by sampling from `news_articles.csv`.
4. **Output** predicted user type, recommended category, and article headline.

Synthetic bandit categories were mapped to dataset categories as follows:

- Entertainment → ENTERTAINMENT  
- Education → EDUCATION  
- Tech → BUSINESS  
- Crime → CRIME  

---

## Evaluation

### Classification Performance

The classifier achieved strong validation accuracy, indicating reliable detection of user contexts.

---

### Reinforcement Learning Results

Each bandit algorithm was evaluated over 10,000 timesteps.

Plots include:

- Average Reward vs Time (per user context)
- Hyperparameter comparison for ε (Epsilon-Greedy)
- Hyperparameter comparison for C (UCB)

All plots contain labeled axes, legends, and descriptive titles.

---

## Results and Analysis

### Algorithm Comparison

- **UCB** achieved the highest overall average reward.
- **Epsilon-Greedy** performed competitively when ε was small.
- **SoftMax** showed stable but slightly lower performance.

### Hyperparameter Sensitivity

- Increasing ε reduced average reward due to excessive exploration.
- Intermediate C values produced best UCB performance.
- SoftMax with τ = 1 provided smooth exploration but limited adaptability.

### Contextual Behavior

The learned policies converged to:

- User1 → Education
- User2 → Tech
- User3 → Education

demonstrating successful context-aware learning.

---

## Strengths and Limitations

### Epsilon-Greedy
- Simple and efficient
- Requires manual tuning
- Random exploration can be inefficient

### UCB
- Best empirical performance
- Adaptive exploration
- Slightly higher computational cost

### SoftMax
- Smooth probabilistic exploration
- Sensitive to temperature selection

---

## Conclusion

This project demonstrates a complete Contextual Multi-Armed Bandit recommendation system integrating:

- Supervised user classification
- Context-aware reinforcement learning
- Hyperparameter tuning
- Real-world article recommendation

Among the evaluated strategies, UCB provided the strongest performance.  
The system successfully learns personalized category preferences and improves recommendations over time.
