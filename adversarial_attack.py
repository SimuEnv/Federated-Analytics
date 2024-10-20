import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

class MembershipInferenceAttack:
    def __init__(self, client_data_loaders, dp_results, epsilon, summarize_on='mean'):
        self.client_data_loaders = client_data_loaders
        self.dp_results = dp_results
        self.epsilon = epsilon
        self.summarize_on = summarize_on  # Specify the statistical function to apply
        self.complex_attack_success = []

    def get_true_statistic(self, client_data):
        if self.summarize_on == 'mean':
            return np.mean(client_data)
        elif self.summarize_on == 'sum':
            return np.sum(client_data)
        elif self.summarize_on == 'variance':
            return np.var(client_data)
        elif self.summarize_on == 'quantile':
            return np.quantile(client_data, 0.5)
        else:
            raise ValueError(f"Unknown summarization function: {self.summarize_on}")

    def simple_attack(self, client_data, dp_result):
        true_statistic = self.get_true_statistic(client_data)
        difference = abs(dp_result - true_statistic)
        return difference < 2 / self.epsilon

    def advanced_attack(self, dp_results, client_data_loaders):
        X, y = [], []
        for i, client_loader in enumerate(client_data_loaders):
            client_data = np.concatenate([x.numpy() for x, _ in client_loader])
            true_statistic = self.get_true_statistic(client_data)
            for dp_result in dp_results:
                dp_result_flat = dp_result.flatten() if len(dp_result.shape) > 1 else dp_result
                X.append(dp_result_flat)
                difference = abs(np.mean(dp_result_flat) - true_statistic)
                threshold = 1.5 / self.epsilon
                y.append(1 if difference < threshold else 0)

        X, y = np.array(X).reshape(len(X), -1), np.array(y)

        # Shuffle to avoid any order bias
        X, y = shuffle(X, y)

        if len(set(y)) < 2:
            logging.error("Not enough class diversity in labels y. Adjust the threshold.")
            return None

        model = LogisticRegression()
        try:
            model.fit(X, y)
            attack_success = model.score(X, y)
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            return None

        logging.info(f"Complex Membership Inference Attack Success Rate: {attack_success}")
        self.complex_attack_success.append(attack_success)
        return attack_success

    def simulate_attack(self):
        success_count = 0
        for i, client_loader in enumerate(self.client_data_loaders):
            client_data = np.concatenate([x.numpy() for x, _ in client_loader])
            dp_result = self.dp_results[i]
            if self.simple_attack(client_data, dp_result):
                success_count += 1

        attack_success_rate = success_count / len(self.client_data_loaders)
        logging.info(f"Simple Membership Inference Attack Success Rate: {attack_success_rate}")

        advanced_attack_success_rate = self.advanced_attack(self.dp_results, self.client_data_loaders)
        return attack_success_rate, advanced_attack_success_rate
