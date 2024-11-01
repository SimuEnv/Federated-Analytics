# Privacy-Preserving Clustering in Federated Analytics for Healthcare

This project explores privacy-preserving clustering for healthcare applications using federated analytics, differential privacy (DP), and clustering methods like agglomerative clustering. The approach offers effective clustering without compromising patient privacy, ideal for settings such as patient activity monitoring and chronic disease management.



## Project Overview
This project evaluates federated statistical functions with various DP noise mechanisms to achieve privacy-preserving clustering. The key focus is on the variance and quantile functions under different DP settings. Agglomerative clustering is utilized for its strength in capturing data structure, ensuring high clustering quality even under privacy constraints. 

The approach avoids sending actual patient data, instead relying on federated analytics, fostering trust and adoption in privacy-sensitive healthcare environments.

## Features
- **Privacy-Preserving Clustering**: Employs Local and Centralized DP with Laplace, Exponential, and Gaussian noise to protect sensitive data.
- **Federated Analytics**: Enables effective clustering without transferring raw patient data.
- **Clustering Methods**: Uses Agglomerative Clustering and KMeans to achieve high-quality clustering.
- **Healthcare Applications**: Suitable for use cases like chronic disease monitoring and patient activity tracking.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SimuEnv/Federated-Analytics.git
   cd Federated-Analytics

   pip install -r requirements.txt
python run_summary.py

