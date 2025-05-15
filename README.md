# Sequential Recommendation Competition

Each user is expected to have 10 ranked movie IDs in the output.

---

## Key Components

### `sbai_sol.ipynb`
- Loads and preprocesses the MovieLens 1M dataset.
- Maps users and movies to unique indices.
- Builds padded sequences for each user.
- Defines and trains a GRU-based sequential recommender.
- Evaluates the model using a top-10 recommendation strategy.

### `sbai_sol.py`
- Python script version of the notebook for standalone execution.

### `best_GRUvieLens_sbai.pth`
- Trained model weights after 40 epochs of training.

### `sbai_best_model.csv`
- Final model predictions in the expected competition submission format.

### `GRUvieLens - Sequential RS Video.mp4`
- Video walkthrough and verbal explanation of the model and pipeline.

---

## Results

| Metric     | Value   |
|------------|---------|
| Hit@10     | 1.0000  |
| NDCG@10    | 1.0000  |
| Final Loss | 0.0464  |

The model achieved perfect top-10 ranking performance on the test set.

---

## Citation

E. NFAOUI. Sequential Recommendation. [https://www.kaggle.com/competitions/sequential-recommendation](https://www.kaggle.com/competitions/sequential-recommendation), 2025.

---

## Acknowledgements

Special thanks to Prof. E. Nfaoui for designing the competition and guiding the course. Built using PyTorch, MovieLens 1M, and sequence modeling best practices.