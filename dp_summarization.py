import numpy as np

class DPMean:
    def __init__(self, epsilon, noise_type='laplace', delta=1e-5):
        self.epsilon = epsilon
        self.noise_type = noise_type
        self.delta = delta

    def add_gaussian_noise(self, data, sensitivity, epsilon, delta):
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        return np.random.normal(0, sigma, 1)

    def add_exponential_noise(self, data, sensitivity, epsilon):
        scale = sensitivity / epsilon
        return np.random.exponential(scale)


    
    def add_ldp_noise(self, data, epsilon):
        p = np.exp(epsilon) / (np.exp(epsilon) + 1)
        random_values = np.random.rand(*data.shape)  # Match the shape of the data
        return np.where(random_values < p, data, 1 - data)  # Apply LDP noise element-wise



    def summarize(self, data):
        mean = np.mean(data)
        sensitivity = 1.0  # Sensitivity for mean
        if self.noise_type == 'laplace':
            noise = np.random.laplace(0, sensitivity / self.epsilon, 1)
        elif self.noise_type == 'gaussian':
            noise = self.add_gaussian_noise(mean, sensitivity, self.epsilon, self.delta)
        elif self.noise_type == 'exponential':
            noise = self.add_exponential_noise(mean, sensitivity, self.epsilon)
        elif self.noise_type == 'ldp':
            data = self.add_ldp_noise(data, self.epsilon)
            return np.mean(data)
        return mean + noise


class DPSum:
    def __init__(self, epsilon, noise_type='laplace', delta=1e-5):
        self.epsilon = epsilon
        self.noise_type = noise_type
        self.delta = delta

    def add_gaussian_noise(self, data, sensitivity, epsilon, delta):
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        return np.random.normal(0, sigma, 1)

    def add_exponential_noise(self, data, sensitivity, epsilon):
        scale = sensitivity / epsilon
        return np.random.exponential(scale)

    def add_ldp_noise(self, data, epsilon):
        p = np.exp(epsilon) / (np.exp(epsilon) + 1)
        random_values = np.random.rand(*data.shape)  # Match the shape of the data
        return np.where(random_values < p, data, 1 - data)  # Apply LDP noise element-wise

    def summarize(self, data):
        total = np.sum(data)
        sensitivity = 1.0  # Sensitivity for sum
        if self.noise_type == 'laplace':
            noise = np.random.laplace(0, sensitivity / self.epsilon, 1)
        elif self.noise_type == 'gaussian':
            noise = self.add_gaussian_noise(total, sensitivity, self.epsilon, self.delta)
        elif self.noise_type == 'exponential':
            noise = self.add_exponential_noise(total, sensitivity, self.epsilon)
        elif self.noise_type == 'ldp':
            data = self.add_ldp_noise(data, self.epsilon)
            return np.sum(data)
        return total + noise


class DPVariance:
    def __init__(self, epsilon, noise_type='laplace', delta=1e-5):
        self.epsilon = epsilon
        self.noise_type = noise_type
        self.delta = delta

    def add_gaussian_noise(self, data, sensitivity, epsilon, delta):
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        return np.random.normal(0, sigma, 1)

    def add_exponential_noise(self, data, sensitivity, epsilon):
        scale = sensitivity / epsilon
        return np.random.exponential(scale)

    def add_ldp_noise(self, data, epsilon):
        p = np.exp(epsilon) / (np.exp(epsilon) + 1)
        random_values = np.random.rand(*data.shape)  # Match the shape of the data
        return np.where(random_values < p, data, 1 - data)  # Apply LDP noise element-wise

    def summarize(self, data):
        variance = np.var(data)
        sensitivity = 1.0  # Sensitivity for variance
        if self.noise_type == 'laplace':
            noise = np.random.laplace(0, sensitivity / self.epsilon, 1)
        elif self.noise_type == 'gaussian':
            noise = self.add_gaussian_noise(variance, sensitivity, self.epsilon, self.delta)
        elif self.noise_type == 'exponential':
            noise = self.add_exponential_noise(variance, sensitivity, self.epsilon)
        elif self.noise_type == 'ldp':
            data = self.add_ldp_noise(data, self.epsilon)
            return np.var(data)
        return variance + noise


class DPQuantile:
    def __init__(self, epsilon, noise_type='laplace', delta=1e-5):
        self.epsilon = epsilon
        self.noise_type = noise_type
        self.delta = delta

    def add_gaussian_noise(self, data, sensitivity, epsilon, delta):
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        return np.random.normal(0, sigma, 1)

    def add_exponential_noise(self, data, sensitivity, epsilon):
        scale = sensitivity / epsilon
        return np.random.exponential(scale)

    def add_ldp_noise(self, data, epsilon):
        p = np.exp(epsilon) / (np.exp(epsilon) + 1)
        random_values = np.random.rand(*data.shape)  # Match the shape of the data
        return np.where(random_values < p, data, 1 - data)  # Apply LDP noise element-wise

    def summarize(self, data, quantile=0.5):
        q_value = np.quantile(data, quantile)
        sensitivity = 1.0  # Sensitivity for quantile
        if self.noise_type == 'laplace':
            noise = np.random.laplace(0, sensitivity / self.epsilon, 1)
        elif self.noise_type == 'gaussian':
            noise = self.add_gaussian_noise(q_value, sensitivity, self.epsilon, self.delta)
        elif self.noise_type == 'exponential':
            noise = self.add_exponential_noise(q_value, sensitivity, self.epsilon)
        elif self.noise_type == 'ldp':
            data = self.add_ldp_noise(data, self.epsilon)
            return np.quantile(data, quantile)
        return q_value + noise
