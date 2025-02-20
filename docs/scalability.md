## Scalability Proposal


### Idea
Graph embedding suffers from scalability issues, and hierarchical methods (e.g., graph coarsening) can help by reducing the graph size while preserving important structures. However, choosing the best coarsening strategy is non-trivial.

As a result, we aim to use deep learning to automate or assist in selecting the best hierarchical method.

- Training a model to **predict the effectiveness** of different coarsening methods.
- Using features from the original graph (or intermediate representations) to guide selection.
- Possibly integrating reinforcement learning (RL) or meta-learning for adaptive selection.

### Feasibility:
- **Graph Neural Networks (GNNs)** can learn embeddings that capture structural patterns.
- **Meta-learning** or **Reinforcement Learning (RL)** could be useful in dynamically selecting the best method.


### Potential Benefits
- might learn **non-obvious heuristics** that improve graph compression without excessive loss of structure.  
- could generalize across different graph types (social networks, molecular graphs, citation networks, etc.).  
- Reduces manual tuning of hierarchical strategies, making large-scale graph embeddings more practical.

### Challenges
1. **How to Define "Best" Coarsening? And the loss function?** 
2. **What input features should the deep learning model use?** 
3. **If no labeled dataset exists, should we create a new one?**
4. **How well will your model generalize to unseen graphs?**

### Possible Approaches
- Train a model on a dataset where each graph is labeled with the best coarsening method.  
- Use graph properties and/or precomputed embeddings as input features.  
- Formulate **coarsening selection** as an RL problem:
  - **State**: Graph features  
  - **Action**: Choose a coarsening method  
  - **Reward**: Measure quality of embedding after coarsening (e.g., performance on a downstream task).

### Key
- define a **good evaluation metric**