# run_quantile_clustering_comparison.py

import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import get_dataloader
from clustering.dp_quantile_clustering import DPQuantileClustering
from clustering.non_dp_clustering import NonDPQuantileClustering
from sklearn.decomposition import PCA
import os
import csv
import logging
import itertools
from multiprocessing import Pool
from functools import partial

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_clusters(data, assignments, title):
    if len(np.unique(assignments)) < 2:
        logging.warning(f"Not enough clusters for visualization: {title}")
        return
    
    plt.figure(figsize=(10, 7))
    
    if data.shape[1] == 1:
        plt.scatter(data.flatten(), np.zeros_like(data.flatten()), c=assignments, cmap='viridis')
        plt.yticks([])
        plt.xlabel('Data Value')
    else:
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
        else:
            data_2d = data
        
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=assignments, cmap='viridis')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
    
    plt.title(title)
    plt.colorbar(label='Cluster')
    plt.savefig(f"results/{title.replace(' ', '_')}.png")
    plt.close()

def print_quality_metrics(metrics, method):
    logging.info(f"\nClustering Quality Metrics for {method}:")
    for metric, value in metrics.items():
        if value is not None:
            logging.info(f"{metric}: {value:.4f}")
        else:
            logging.info(f"{metric}: N/A")

def evaluate_params(epsilon, num_quantiles, client_loaders, dataset_name, cluster_on):
    clustering = DPQuantileClustering(
        max_clusters=10, epsilon=epsilon, num_quantiles=num_quantiles,
        dataset_name=dataset_name, cluster_on=cluster_on
    )
    _, quality = clustering.cluster_clients(client_loaders)
    return epsilon, num_quantiles, quality['silhouette_score']

def grid_search_parameters(client_loaders, dataset_name, cluster_on):
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    num_quantiles_list = [5, 10, 20, 30] if dataset_name == 'cifar100' else [3, 5, 7, 9]
    
    param_combinations = list(itertools.product(epsilons, num_quantiles_list))
    
    with Pool() as pool:
        results = pool.starmap(
            partial(evaluate_params, client_loaders=client_loaders, 
                    dataset_name=dataset_name, cluster_on=cluster_on),
            param_combinations
        )
    
    best_result = max(results, key=lambda x: x[2])
    return best_result[0], best_result[1]  # best_epsilon, best_num_quantiles

def run_dp_quantile_clustering(client_loaders, epsilon, num_quantiles, dataset_name, cluster_on):
    clustering = DPQuantileClustering(
        max_clusters=10, epsilon=epsilon, num_quantiles=num_quantiles,
        dataset_name=dataset_name, cluster_on=cluster_on
    )
    client_stats = clustering.compute_client_statistics(client_loaders)
    assignments, quality = clustering.cluster_clients(client_loaders)
    return {
        'epsilon': epsilon,
        'num_quantiles': num_quantiles,
        'num_clusters': clustering.num_clusters,
        'quality': quality,
        'assignments': assignments,
        'client_stats': client_stats
    }

def run_non_dp_quantile_clustering(client_loaders, num_quantiles, dataset_name, cluster_on):
    clustering = NonDPQuantileClustering(
        max_clusters=10, num_quantiles=num_quantiles,
        dataset_name=dataset_name, cluster_on=cluster_on
    )
    client_stats = clustering.compute_client_statistics(client_loaders)
    assignments, quality = clustering.cluster_clients(client_loaders)
    return {
        'num_quantiles': num_quantiles,
        'num_clusters': clustering.num_clusters,
        'quality': quality,
        'assignments': assignments,
        'client_stats': client_stats
    }

def save_results_to_csv(results, method_name):
    with open(f'results/{method_name}_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['epsilon', 'num_quantiles', 'num_clusters', 'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'epsilon': result.get('epsilon', 'N/A'),
                'num_quantiles': result['num_quantiles'],
                'num_clusters': result['num_clusters'],
                'silhouette_score': result['quality'].get('silhouette_score'),
                'calinski_harabasz_score': result['quality'].get('calinski_harabasz_score'),
                'davies_bouldin_score': result['quality'].get('davies_bouldin_score')
            })

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        ensure_dir('results')

        dataset_names = ['har', 'mnist', 'cifar10', 'cifar100', 'svhn']
        num_clients = 50
        alpha = 0.5
        unlabeled_ratio = 0.3
        cluster_on_options = ['features', 'labels']

        for dataset_name in dataset_names:
            logging.info(f"\nProcessing dataset: {dataset_name}")

            client_loaders, test_loader, unlabeled_loader = get_dataloader(
                dataset_name, batch_size=64, num_clients=num_clients, 
                alpha=alpha, unlabeled_ratio=unlabeled_ratio
            )

            if client_loaders is None:
                logging.error(f"Failed to load {dataset_name} dataset. Skipping...")
                continue

            for cluster_on in cluster_on_options:
                logging.info(f"\nClustering on: {cluster_on}")

                # Grid search for best parameters
                best_epsilon, best_num_quantiles = grid_search_parameters(client_loaders, dataset_name, cluster_on)
                logging.info(f"Best parameters: epsilon={best_epsilon}, num_quantiles={best_num_quantiles}")

                # Run DP quantile clustering with best parameters
                dp_result = run_dp_quantile_clustering(client_loaders, best_epsilon, best_num_quantiles, dataset_name, cluster_on)

                # Run non-DP quantile clustering with same num_quantiles
                non_dp_result = run_non_dp_quantile_clustering(client_loaders, best_num_quantiles, dataset_name, cluster_on)

                # Save results
                save_results_to_csv([dp_result], f'{dataset_name}_{cluster_on}_DP_Quantile')
                save_results_to_csv([non_dp_result], f'{dataset_name}_{cluster_on}_Non_DP_Quantile')

                # Visualize and print metrics
                for method_name, result in [("DP Quantile", dp_result), ("Non-DP Quantile", non_dp_result)]:
                    epsilon_str = f"(epsilon={result['epsilon']})" if 'epsilon' in result else ""
                    title = f"{dataset_name} {cluster_on} {method_name} Clustering {epsilon_str}"
                    
                    visualize_clusters(result['client_stats'], result['assignments'], title)
                    logging.info(f"\nResults for {title}:")
                    logging.info(f"Number of clusters: {result['num_clusters']}")
                    logging.info(f"Number of quantiles: {result['num_quantiles']}")
                    print_quality_metrics(result['quality'], title)

            # Print dataset statistics
            logging.info(f"\nDataset: {dataset_name}")
            logging.info(f"Number of clients: {num_clients}")
            logging.info(f"Alpha (for non-IID distribution): {alpha}")
            logging.info(f"Unlabeled ratio: {unlabeled_ratio}")
            logging.info(f"Samples in first client: {len(client_loaders[0].dataset)}")
            logging.info(f"Samples in test set: {len(test_loader.dataset)}")
            logging.info(f"Samples in unlabeled set: {len(unlabeled_loader.dataset)}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()