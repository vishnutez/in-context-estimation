import math
from abc import ABC, abstractmethod
import torch
import scipy as sp

# ------------------------------------------------------------
# Modulation samplers
# ------------------------------------------------------------

class ModulationSampler:
    def __init__(self, signal_set, n_tx_antennas=1):
        self.signal_set = signal_set  # (signal_set_size,) complex64
        self.n_tx_antennas = n_tx_antennas
        self.signal_set_size = len(signal_set)

    def sample(self, n_points, b_size, seeds=None):
        if seeds is None:
            signal_ids = torch.randint(0, self.signal_set_size, (b_size, n_points, self.n_tx_antennas))
            signals = self.signal_set[signal_ids]
        else:
            signal_ids = torch.zeros(b_size, n_points, self.n_tx_antennas, dtype=torch.long)
            signals = torch.zeros(b_size, n_points, self.n_tx_antennas, dtype=torch.complex64)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                signal_ids[i] = torch.randint(0, self.signal_set_size, (n_points, self.n_tx_antennas), generator=generator)
                signals[i] = self.signal_set[signal_ids[i]]
        return signals, signal_ids


class QAMSampler(ModulationSampler):

    def __init__(self, n_tx_antennas=1, signal_set_size=16):
        # Create a QAM signal set as the Cartesian product of I and Q levels
        n_real = int(math.sqrt(signal_set_size))
        assert n_real * n_real == signal_set_size, "signal_set_size must be a perfect square"

        levels = torch.arange(n_real, dtype=torch.float32)
        levels = levels - levels.mean()  # center around zero

        # Cartesian product: all (I, Q) pairs -> (signal_set_size,) complex64
        grid_i, grid_q = torch.meshgrid(levels, levels, indexing="ij")
        signal_set = (grid_i.reshape(-1) + 1j * grid_q.reshape(-1)).to(torch.complex64)

        super().__init__(signal_set=signal_set, n_tx_antennas=n_tx_antennas)


class PSKSampler(ModulationSampler):
    def __init__(self, n_tx_antennas=1, signal_set_size=4):
        # Create a PSK signal set as unit-magnitude points on the complex plane -> (signal_set_size,) complex64
        angles = torch.arange(signal_set_size, dtype=torch.float32) * (2 * math.pi / signal_set_size)
        signal_set = (torch.cos(angles) + 1j * torch.sin(angles)).to(torch.complex64)
        super().__init__(signal_set=signal_set, n_tx_antennas=n_tx_antennas)



# ------------------------------------------------------------
# Channel samplers
# ------------------------------------------------------------

class ChannelSampler(ABC):
    def __init__(self, n_tx_antennas=1, n_rx_antennas=1):
        self.n_tx_antennas = n_tx_antennas
        self.n_rx_antennas = n_rx_antennas

    @abstractmethod
    def sample(self, b_size, n_points, seeds=None):
        """Sample channels of shape (b_size, n_rx_antennas, n_tx_antennas) as complex64."""
        pass


class RayleighBlockFadingChannelSampler(ChannelSampler):
    def __init__(self, n_tx_antennas=1, n_rx_antennas=1):
        super().__init__(n_tx_antennas=n_tx_antennas, n_rx_antennas=n_rx_antennas)

    def sample(self, b_size, n_points, seeds=None):
        if seeds is not None:
            channels = torch.zeros(b_size, self.n_rx_antennas, self.n_tx_antennas, dtype=torch.complex64)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                channels[i] = (torch.randn(self.n_rx_antennas, self.n_tx_antennas, generator=generator) + 1j * torch.randn(self.n_rx_antennas, self.n_tx_antennas, generator=generator)) / math.sqrt(2)
        else:
            channels = (torch.randn(b_size, self.n_rx_antennas, self.n_tx_antennas) + 1j * torch.randn(b_size, self.n_rx_antennas, self.n_tx_antennas)) / math.sqrt(2)
        
        # Repeat the channel for each point in the sequence (b_size, n_points, n_rx_antennas, n_tx_antennas)
        channels = channels.unsqueeze(1).expand(-1, n_points, -1, -1)
        return channels



class DopplerSpreadChannelSampler(ChannelSampler):
    def __init__(self, n_tx_antennas=1, n_rx_antennas=1, fc=1e9,
                 velocity_min=None, velocity_max=None, velocities=None, symbol_duration=1e-3):
        """
        Velocity is sampled per call to sample(). Specify exactly one of:
          - velocities:                 list of velocities to choose from uniformly at random
          - velocity_min / velocity_max: continuous uniform range [velocity_min, velocity_max]
        If velocity_min == velocity_max, that fixed velocity is always used.
        """
        super().__init__(n_tx_antennas=n_tx_antennas, n_rx_antennas=n_rx_antennas)
        self.fc = float(fc)
        self.SPEED_OF_LIGHT = 299_792_458  # meters per second
        self.symbol_duration = float(symbol_duration)

        if velocities is not None:
            self.velocities = torch.tensor(velocities, dtype=torch.float32)
            self.velocity_min = None
            self.velocity_max = None
        elif velocity_min is not None and velocity_max is not None:
            self.velocities = None
            self.velocity_min = velocity_min
            self.velocity_max = velocity_max
        else:
            raise ValueError("Specify either velocities or both velocity_min and velocity_max")

    def _sample_velocity(self, generator=None):
        """Sample a single velocity scalar."""
        if self.velocities is not None:
            idx = torch.randint(0, len(self.velocities), (1,), generator=generator).item()
            return self.velocities[idx].item()
        elif self.velocity_min == self.velocity_max:
            return self.velocity_min
        else:
            return (torch.rand(1, generator=generator).item() * (self.velocity_max - self.velocity_min) + self.velocity_min)

    def _get_channel_time_correlation_matrix(self, n_points, v):
        # Time correlation matrix: corresponding to Clarke-Jakes Doppler spectrum
        max_doppler_shift = self.fc * v / self.SPEED_OF_LIGHT # in Hz
        lags = 2 * math.pi * max_doppler_shift * torch.arange(n_points) * self.symbol_duration # lags in radians
        return torch.from_numpy(sp.special.j0(sp.linalg.toeplitz(lags)))

    def _sample_single(self, n_points, v, generator=None):
        """Generate one channel realization (n_points, n_rx, n_tx) for a given velocity."""
        autocorrelation_matrix = self._get_channel_time_correlation_matrix(n_points, v)
        eigenvalues, eigenvectors = torch.linalg.eigh(autocorrelation_matrix)
        L = eigenvectors * torch.sqrt(eigenvalues.clamp(min=0)).unsqueeze(0)

        z = (torch.randn(n_points, self.n_rx_antennas, self.n_tx_antennas, generator=generator)
             + 1j * torch.randn(n_points, self.n_rx_antennas, self.n_tx_antennas, generator=generator)) / math.sqrt(2)
        return torch.einsum('mn,nrt->mrt', L.to(torch.complex64), z)

    def sample(self, b_size, n_points, seeds=None):
        channels = torch.zeros(b_size, n_points, self.n_rx_antennas, self.n_tx_antennas, dtype=torch.complex64)
        generator = torch.Generator()

        if seeds is not None:
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                v = self._sample_velocity(generator)
                channels[i] = self._sample_single(n_points, v, generator)
        else:
            for i in range(b_size):
                v = self._sample_velocity(generator)
                channels[i] = self._sample_single(n_points, v, generator)

        return channels



class CustomBlockFadingChannelSampler(ChannelSampler):
    def __init__(self, channel_dataset):
        """
        Args:
            channel_dataset: tensor of shape (dataset_size, n_rx_antennas, n_tx_antennas), complex64
        """
        dataset_size, n_rx_antennas, n_tx_antennas = channel_dataset.shape
        super().__init__(n_tx_antennas=n_tx_antennas, n_rx_antennas=n_rx_antennas)
        self.channel_dataset = channel_dataset
        self.dataset_size = dataset_size

    def sample(self, b_size, n_points, seeds=None):
        if seeds is not None:
            channels = torch.zeros(b_size, self.n_rx_antennas, self.n_tx_antennas, dtype=torch.complex64)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                idx = torch.randint(0, self.dataset_size, (1,), generator=generator).item()
                channels[i] = self.channel_dataset[idx]
        else:
            indices = torch.randint(0, self.dataset_size, (b_size,))
            channels = self.channel_dataset[indices]

        # Repeat the channel for each point in the sequence (b_size, n_points, n_rx_antennas, n_tx_antennas)
        channels = channels.unsqueeze(1).expand(-1, n_points, -1, -1)
        return channels
