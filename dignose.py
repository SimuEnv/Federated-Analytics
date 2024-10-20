import logging
import numpy as np
from data.data_loader import get_dataloader
from dp_summarization import DPMean, DPSum, DPVariance, DPQuantile

def summarize_data(data, client_id):
    """Summarize the data for mean, sum, variance, and quantiles."""
    mean = np.mean(data)
    data_sum = np.sum(data)
    variance = np.var(data)
    quantiles = np.quantile(data, [0.25, 0.5, 0.75])
    
    logging.info(f"\n--- {client_id} Summary ---")
    logging.info(f"Shape: {data.shape}")
    logging.info(f"Mean: {mean}")
    logging.info(f"Sum: {data_sum}")
    logging.info(f"Variance: {variance}")
    logging.info(f"Quantiles: {quantiles}")

def log_raw_data(client_loader, client_id):
    """Log raw pixel data for inspection."""
    for idx, (data, _) in enumerate(client_loader):
        if idx == 0:  # Log only the first batch for simplicity
            logging.info(f"Client {client_id} Raw Data Sample: {data.numpy()[0]}")
            break

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define parameters
    dataset_name = 'mnist'
    num_clients = 5

    logging.info(f"\nLoading dataset: {dataset_name}")

    # Load data for clients
    client_loaders, _, _, _ = get_dataloader(dataset_name, batch_size=32, num_clients=num_clients)

    if client_loaders is None:
        logging.error(f"Failed to load {dataset_name} dataset. Exiting...")
        return

    # Perform basic summarization (without DP)
    for idx, client_loader in enumerate(client_loaders):
        client_data = np.concatenate([x.numpy() for x, _ in client_loader])
        
        # Log raw data for diagnostics
        log_raw_data(client_loader, idx + 1)
        
        # Log data summary for each client
        summarize_data(client_data, f"Client {idx + 1}")

if __name__ == "__main__":
    main()
