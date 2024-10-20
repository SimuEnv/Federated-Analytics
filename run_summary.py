import logging
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import get_dataloader
from dp_summarization import DPMean, DPSum, DPVariance, DPQuantile
from adversarial_attack import MembershipInferenceAttack
from tabulate import tabulate

# Plotting helper functions
def plot_results(epsilon_values, metrics, noise_models, title, ylabel):
    plt.style.use('ggplot')  # Improved plot style
    for metric, noise_model in zip(metrics, noise_models):
        plt.plot(epsilon_values, metric, label=noise_model)
    plt.title(title)
    plt.xlabel('Epsilon Values')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to run DP summarization method
def run_summarization_method(method, client_loaders, epsilon=None, summarize_on='features', use_dp=True, noise_type='laplace'):
    results = []
    if use_dp:
        dp_summarizer = method(epsilon=epsilon, noise_type=noise_type)
    else:
        dp_summarizer = method(epsilon=float('inf'), noise_type=noise_type)  # No DP
    for client_loader in client_loaders:
        client_data = np.concatenate([x.numpy() for x, _ in client_loader])
        summary = dp_summarizer.summarize(client_data)
        results.append(summary)
    return results

# Function to run adversarial attack
def run_adversarial_attack(client_loaders, dp_results, epsilon, summarize_on):
    attack_simulation = MembershipInferenceAttack(client_loaders, dp_results, epsilon, summarize_on=summarize_on)
    simple_success_rate, advanced_success_rate = attack_simulation.simulate_attack()
    return simple_success_rate, advanced_success_rate

# Function to compute accuracy
def compute_accuracy(non_dp_results, dp_results):
    accuracy = []
    for non_dp, dp in zip(non_dp_results, dp_results):
        if non_dp != 0:
            acc = 1 - (np.abs(non_dp - dp) / np.abs(non_dp))
        else:
            acc = 1.0  # Assume perfect accuracy if non-DP result is zero
        accuracy.append(acc)
    return np.mean(accuracy)

# Function to format the results as a table
def print_dp_results_as_table(epsilon, noise, simple_sr_mean, advanced_sr_mean, accuracy_mean, 
                              simple_sr_sum, advanced_sr_sum, accuracy_sum, 
                              simple_sr_var, advanced_sr_var, accuracy_var, 
                              simple_sr_quant, advanced_sr_quant, accuracy_quant):
    table = [
        ['Mean', simple_sr_mean, advanced_sr_mean if advanced_sr_mean is not None else 'N/A', accuracy_mean],
        ['Sum', simple_sr_sum, advanced_sr_sum if advanced_sr_sum is not None else 'N/A', accuracy_sum],
        ['Variance', simple_sr_var, advanced_sr_var if advanced_sr_var is not None else 'N/A', accuracy_var],
        ['Quantile', simple_sr_quant, advanced_sr_quant if advanced_sr_quant is not None else 'N/A', accuracy_quant]
    ]
    
    print(f"\nDP Summarization Results for Epsilon: {epsilon}, Noise: {noise}")
    print(tabulate(table, headers=['Function', 'Simple Attack', 'Complex Attack', 'Accuracy'], tablefmt='grid'))

def main():
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    dataset_names = ['har']  # Dataset for testing
    num_clients = 50
    epsilon_values = [0.1, 1.0, 5, 10]  # Epsilon values to test DP strength
    noise_models = ['laplace', 'gaussian', 'exponential', 'ldp']

    # Initialize containers for metrics
    attack_success_metrics = {'mean': [], 'sum': [], 'variance': [], 'quantile': []}
    accuracy_metrics = {'mean': [], 'sum': [], 'variance': [], 'quantile': []}

    for dataset_name in dataset_names:
        logging.info(f"\nProcessing dataset: {dataset_name}")

        client_loaders, test_loader, unlabeled_loader, _ = get_dataloader(dataset_name, batch_size=32, num_clients=num_clients)

        if client_loaders is None:
            logging.error(f"Failed to load {dataset_name} dataset. Skipping...")

        # Run through each noise model and summarize results for each
        for noise_type in noise_models:
            logging.info(f"\nTesting with {noise_type} noise")

            mean_simple_results, mean_advanced_results, mean_accuracy_results = [], [], []
            sum_simple_results, sum_advanced_results, sum_accuracy_results = [], [], []
            variance_simple_results, variance_advanced_results, variance_accuracy_results = [], [], []
            quantile_simple_results, quantile_advanced_results, quantile_accuracy_results = [], [], []

            # Run summarization without DP for baseline comparison
            dp_mean_results_no_dp = run_summarization_method(DPMean, client_loaders, use_dp=False)
            dp_sum_results_no_dp = run_summarization_method(DPSum, client_loaders, use_dp=False)
            dp_variance_results_no_dp = run_summarization_method(DPVariance, client_loaders, use_dp=False)
            dp_quantile_results_no_dp = run_summarization_method(DPQuantile, client_loaders, use_dp=False)

            for epsilon in epsilon_values:
                logging.info(f"\nRunning DP summarization with epsilon: {epsilon} and noise: {noise_type}")

                # DP Mean
                dp_mean_results = run_summarization_method(DPMean, client_loaders, epsilon, noise_type=noise_type)
                simple_sr, advanced_sr = run_adversarial_attack(client_loaders, dp_mean_results, epsilon, summarize_on='mean')
                mean_simple_results.append(simple_sr)
                mean_advanced_results.append(advanced_sr)
                mean_accuracy_results.append(compute_accuracy(dp_mean_results_no_dp, dp_mean_results))

                # DP Sum
                dp_sum_results = run_summarization_method(DPSum, client_loaders, epsilon, noise_type=noise_type)
                simple_sr, advanced_sr = run_adversarial_attack(client_loaders, dp_sum_results, epsilon, summarize_on='sum')
                sum_simple_results.append(simple_sr)
                sum_advanced_results.append(advanced_sr)
                sum_accuracy_results.append(compute_accuracy(dp_sum_results_no_dp, dp_sum_results))

                # DP Variance
                dp_variance_results = run_summarization_method(DPVariance, client_loaders, epsilon, noise_type=noise_type)
                simple_sr, advanced_sr = run_adversarial_attack(client_loaders, dp_variance_results, epsilon, summarize_on='variance')
                variance_simple_results.append(simple_sr)
                variance_advanced_results.append(advanced_sr)
                variance_accuracy_results.append(compute_accuracy(dp_variance_results_no_dp, dp_variance_results))

                # DP Quantile
                dp_quantile_results = run_summarization_method(DPQuantile, client_loaders, epsilon, noise_type=noise_type)
                simple_sr, advanced_sr = run_adversarial_attack(client_loaders, dp_quantile_results, epsilon, summarize_on='quantile')
                quantile_simple_results.append(simple_sr)
                quantile_advanced_results.append(advanced_sr)
                quantile_accuracy_results.append(compute_accuracy(dp_quantile_results_no_dp, dp_quantile_results))

                # Print the summarized results for this epsilon and noise model as a table
                print_dp_results_as_table(epsilon, noise_type, 
                                          mean_simple_results[-1], mean_advanced_results[-1], mean_accuracy_results[-1], 
                                          sum_simple_results[-1], sum_advanced_results[-1], sum_accuracy_results[-1], 
                                          variance_simple_results[-1], variance_advanced_results[-1], variance_accuracy_results[-1], 
                                          quantile_simple_results[-1], quantile_advanced_results[-1], quantile_accuracy_results[-1])

            # Append results for plotting later
            attack_success_metrics['mean'].append(mean_simple_results)
            attack_success_metrics['sum'].append(sum_simple_results)
            attack_success_metrics['variance'].append(variance_simple_results)
            attack_success_metrics['quantile'].append(quantile_simple_results)

            accuracy_metrics['mean'].append(mean_accuracy_results)
            accuracy_metrics['sum'].append(sum_accuracy_results)
            accuracy_metrics['variance'].append(variance_accuracy_results)
            accuracy_metrics['quantile'].append(quantile_accuracy_results)




        plot_results(epsilon_values, attack_success_metrics['mean'], noise_models, 'Mean - Attack Success Rate', 'Success Rate')
        plot_results(epsilon_values, accuracy_metrics['mean'], noise_models, 'Mean - Accuracy', 'Accuracy')

        plot_results(epsilon_values, attack_success_metrics['sum'], noise_models, 'Sum - Attack Success Rate', 'Success Rate')
        plot_results(epsilon_values, accuracy_metrics['sum'], noise_models, 'Sum - Accuracy', 'Accuracy')

    

        plot_results(epsilon_values, attack_success_metrics['variance'], noise_models, 'Variance - Attack Success Rate', 'Success Rate')
        plot_results(epsilon_values, accuracy_metrics['variance'], noise_models, 'Variance - Accuracy', 'Accuracy')

        plot_results(epsilon_values, attack_success_metrics['quantile'], noise_models, 'Quantile - Attack Success Rate', 'Success Rate')
        plot_results(epsilon_values, accuracy_metrics['quantile'], noise_models, 'Quantile - Accuracy', 'Accuracy')

if __name__ == "__main__":
    main()