import random
class HyperLearning:
    def __init__(self):
        self.param_space = None

    def sample_params(self):
        return {key: random.choice(values) for key, values in self.param_space.items()}

    def learn(self, num_trials, train_and_evaluate):
        for _ in range(num_trials):
            params = self.sample_params()
            score = train_and_evaluate(params)
            print(f"Params: {params}, Score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_params = params


    