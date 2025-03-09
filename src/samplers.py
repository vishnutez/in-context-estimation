import math
import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {"gaussian": GaussianSampler, 
                        "uniform": UniformSampler, 
                        "qpsk": QPSKSampler, 
                        "signal": SignalSampler, 
                        "signal_qam": SignalSamplerQAM}
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False, seed=None):
    n_dims = len(eigenvalues)
    if seed is None:
        random_matrix = torch.randn(n_dims, n_dims)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        random_matrix = torch.randn(n_dims, n_dims, generator=generator)
    U, _, _ = torch.linalg.svd(random_matrix)
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


def sample_scale(method, n_dims, normalize=False, seed=None):
    if method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True, seed=seed)
        return scale
    return None


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, positive_orthant=False):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.positive_orthant = positive_orthant

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)

        if self.positive_orthant:
            # make all the coordinates positive
            xs_b = torch.abs(xs_b)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class UniformSampler(DataSampler):
    def __init__(self, n_dims, a=0, b=1, bias=None, scale=None, positive_orthant=False):
        super().__init__(n_dims)
        self.a = a
        self.b = b
        self.bias = bias
        self.scale = scale
        self.positive_orthant = positive_orthant

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.rand(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.rand(n_points, self.n_dims, generator=generator)

        if self.positive_orthant:
            # make all the coordinates positive
            xs_b = torch.abs(xs_b)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0

        xs_b = (self.b - self.a) * xs_b + self.a
        return xs_b


class QPSKSampler(DataSampler):
    """
    Complex numbers where the real and imaginary parts can be either -1 or 1.
    """
    def __init__(self, n_dims):
        # Ignore n_dims. It's not used here; the dimension is fixed at 2.
        super().__init__(n_dims=2)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):

        if seeds is None:
            xs_b = torch.randint(low=0, high=2, size=(b_size, n_points, self.n_dims))  # 0s and 1s
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims, dtype=torch.int64)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randint(low=0, high=2, size=(n_points, self.n_dims))

        xs_b = (xs_b * 2) - 1
        xs_b = xs_b.to(torch.float32)
        return xs_b
    

class SignalSampler(DataSampler):
    """
    Complex numbers sampled from a fixed signal set
    """
    def __init__(self, n_dims=2):
        # Ignore n_dims. It's not used here; the dimension is fixed at 2.
        super().__init__(n_dims=2)
        complex_sig_set = torch.tensor([1+1j,1-1j,-1+1j,-1-1j])  # Using QPSK for now
        sig_set_real = torch.real(complex_sig_set).unsqueeze(1)
        sig_set_imag = torch.imag(complex_sig_set).unsqueeze(1)
        self.sig_set = torch.concat((sig_set_real, sig_set_imag), dim=1)
        self.sig_set_size = len(complex_sig_set)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):

        if seeds is None:
            sig_ids = torch.randint(low=0, high=self.sig_set_size, size=(b_size, n_points,))  # 0s and 1s
            sig = self.sig_set[sig_ids]
        else:
            sig_ids = torch.zeros(b_size, n_points, dtype=torch.int64)
            sig = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                sig_ids[i] = torch.randint(low=0, high=self.sig_set_size, size=(n_points,))
                sig[i] = self.sig_set[sig_ids[i]]

        return sig, sig_ids



class SignalSamplerQAM(DataSampler):
    """
    Complex numbers sampled from a fixed signal set
    """
    def __init__(self, n_dims=2, M=16):
        # Ignore n_dims. It's not used here; the dimension is fixed at 2.
        super().__init__(n_dims=2)

        # complex_sig_set = torch.tensor([1+1j,1-1j,-1+1j,-1-1j])  # Using QPSK for now
        # sig_set_real = torch.real(complex_sig_set).unsqueeze(1)
        # sig_set_imag = torch.imag(complex_sig_set).unsqueeze(1)
        # self.sig_set = torch.concat((sig_set_real, sig_set_imag), dim=1)

        # create a sig set of 16-QAM
        M_real = math.sqrt(M)

        sig_real_pos = torch.arange(M_real)
        sig_real_mean = torch.mean(sig_real_pos)
        sig_real = sig_real_pos - sig_real_mean

        sig_real_2D = sig_real.unsqueeze(0)
        sig_real_2D =  sig_real_2D.repeat((int(M_real),1))

        sig_imag_2D = sig_real_2D.transpose(0,1)
        sig_imag_2D = torch.flip(sig_imag_2D, dims=[0]).flatten().unsqueeze(-1)

        sig_real_2D = sig_real_2D.flatten().unsqueeze(-1)
        sig_set = torch.cat((sig_real_2D, sig_imag_2D), dim=-1)
        power = torch.mean(sig_set[:, 0]**2 + sig_set[:, 1]**2)
        self.unit_sig_set = sig_set / math.sqrt(power)
        # power_after_norm = torch.mean(self.unit_sig_set[:, 0]**2 + self.unit_sig_set[:, 1]**2)  # sanity check for power
        self.sig_set_size = len(self.unit_sig_set)


    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):

        if seeds is None:
            sig_ids = torch.randint(low=0, high=self.sig_set_size, size=(b_size, n_points,))  # 0s and 1s
            sig = self.unit_sig_set[sig_ids]
        else:
            sig_ids = torch.zeros(b_size, n_points, dtype=torch.int64)
            sig = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                sig_ids[i] = torch.randint(low=0, high=self.sig_set_size, size=(n_points,))
                sig[i] = self.unit_sig_set[sig_ids[i]]

        return sig, sig_ids

