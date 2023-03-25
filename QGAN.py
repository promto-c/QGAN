import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding, AngleEmbedding
from pennylane.templates import BasicEntanglerLayers
from pennylane.templates import StronglyEntanglingLayers
from pennylane.ops import PauliZ

from typing import Tuple

class QGAN:
    def __init__(self, dev: qml.device, num_layers: int, step_size: float):
        ''' Initialize the QGAN object.

        Args:
            dev (qml.device): Quantum device to run the circuits.
            num_layers (int): Number of layers for the generator and discriminator circuits.
            step_size (float): Step size for the gradient descent optimizer.
        '''
        self.dev = dev
        self.num_layers = num_layers
        self.step_size = step_size
        self.gen_weights = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_layers, 2, 3))
        self.disc_weights = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_layers, 2, 3))

        # Define the QNodes
        @qml.qnode(dev)
        def generator_qnode(inputs: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
            self.qgan_generator(inputs, weights)
            return qml.expval(PauliZ(0)), qml.expval(PauliZ(1))

        @qml.qnode(dev)
        def discriminator_qnode(inputs: np.ndarray, weights: np.ndarray) -> float:
            self.qgan_discriminator(inputs, weights)
            return qml.expval(PauliZ(0))

        self.generator_qnode = generator_qnode
        self.discriminator_qnode = discriminator_qnode

    def qgan_generator(self, inputs: np.ndarray, weights_g: np.ndarray) -> np.ndarray:
        ''' Define the quantum circuit for the generator.

        Args:
            inputs (np.ndarray): Input noise for the generator.
            weights_g (np.ndarray): Weights for the generator circuit.

        Returns:
            np.ndarray: A sample from the output of the generator circuit.
        '''
        # Apply AngleEmbedding to encode the inputs as rotation angles for qubits 0 and 1
        AngleEmbedding(inputs[:2], wires=[0, 1])

        # Apply BasicEntanglerLayers and StronglyEntanglingLayers using the weights
        BasicEntanglerLayers(weights_g[:2].reshape(1, -1, 2), wires=[0, 1])
        StronglyEntanglingLayers(weights_g[2:].reshape(self.num_layers-2, 2, 3), wires=[0, 1])

        # Measure the first qubit in the Pauli-Z basis
        return qml.expval(PauliZ(0))

    def qgan_discriminator(self, inputs: np.ndarray, weights: np.ndarray) -> float:
        ''' Define the quantum circuit for the discriminator.

        Args:
            inputs (np.ndarray): Input data for the discriminator.
            weights (np.ndarray): Weights for the discriminator circuit.

        Returns:
            float: The expectation value of the PauliZ operator on the first qubit.
        '''
        # Ensure the input data is a one-dimensional array
        if inputs.ndim == 0:
            inputs = np.array([inputs])

        # Pad the input data with zeros to match the required length for AmplitudeEmbedding
        padded_inputs = np.pad(inputs, (0, 4 - len(inputs)), mode='constant')

        AmplitudeEmbedding(padded_inputs, wires=[0, 1], normalize=True)
        BasicEntanglerLayers(weights[:, :, :2], wires=[0, 1])
        StronglyEntanglingLayers(weights, wires=[0, 1])
        return qml.expval(PauliZ(0))

    def qgan_cost(self, weights_g: np.ndarray, weights_d: np.ndarray, inputs: np.ndarray, real_samples: np.ndarray) -> Tuple[float, float]:
        ''' Define the cost function for the Quantum Generative Adversarial Network (QGAN).
        
        Args:
            weights_g (np.ndarray): Weights for the generator circuit.
            weights_d (np.ndarray): Weights for the discriminator circuit.
            inputs (np.ndarray): Input noise for the generator.
            real_samples (np.ndarray): Real samples from the target distribution.

        Returns:
            Tuple[float, float]: The generator loss and the discriminator loss.
        '''

        # Generate fake samples
        fake_samples = np.array([self.generator_qnode(sample, weights_g) for sample in inputs], dtype=float)

        # Calculate discriminator outputs for real and fake samples
        real_discriminator_outputs = np.array([self.discriminator_qnode(sample, weights_d) for sample in real_samples], dtype=float)
        fake_discriminator_outputs = np.array([self.discriminator_qnode(sample, weights_d) for sample in fake_samples], dtype=float)

        # Calculate the generator and discriminator losses
        g_loss = -np.mean(np.log(fake_discriminator_outputs))
        d_loss = -np.mean(np.log(real_discriminator_outputs) + np.log(1 - fake_discriminator_outputs))

        return g_loss, d_loss


    def train_qgan(self, real_samples: np.ndarray, num_epochs: int, batch_size: int):
        ''' Train the QGAN using the provided real data samples, number of epochs, and batch size.

        Args:
            real_samples (np.ndarray): Real data samples.
            num_epochs (int): Number of epochs for training.
            batch_size (int): Size of the batch for each training iteration.
        '''
        optimizer = qml.GradientDescentOptimizer(self.step_size)
        for i in range(num_epochs):
            noise = np.random.normal(size=(batch_size, 2))
            g_loss, d_loss = self.qgan_cost(self.gen_weights, self.disc_weights, noise, real_samples)
            self.gen_weights, self.disc_weights = optimizer.step(lambda v: self.qgan_cost(v[0], v[1], noise, real_samples), [self.gen_weights, self.disc_weights])
            print("Epoch:", i + 1, "Generator Loss:", g_loss, "Discriminator Loss:", d_loss)

# Example usage
dev = qml.device("default.qubit", wires=2)
real_samples = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
num_epochs = 100
batch_size = 4
qgan = QGAN(dev, num_layers=4, step_size=0.1)
qgan.train_qgan(real_samples, num_epochs, batch_size)
