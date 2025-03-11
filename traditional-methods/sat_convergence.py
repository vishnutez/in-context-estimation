import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

# Get a QAM constellation of size M
def get_qam_constellation(set_size=4):
    """
    Get a QAM constellation of size M.

    :param set_size: Size of the QAM constellation
    :return: QAM constellation of the specified size
    """
    set_size_real = int(np.sqrt(set_size))
    set_size_imag = set_size_real

    x_real_vec = np.linspace(-2*set_size_real+1, 2*set_size_real-1, set_size_real)
    x_imag_vec = x_real_vec
    x_real, x_imag = np.meshgrid(x_real_vec, x_imag_vec)

    x = x_real + 1j * x_imag  # Complex set of symbols

    # Normalize the energy to 1
    x = x / np.sqrt(np.mean(np.abs(x)**2))

    # Stack to get real and imaginary parts of the symbols (set_size, 2)
    x_vec = np.stack((np.real(x).ravel(), np.imag(x).ravel()), axis=-1)
    return x_vec


# Get QPSK constellation
def get_qpsk_constellation():
    """
    Get a QPSK constellation of size M.

    :param set_size: Size of the QPSK constellation
    :return: QPSK constellation of the specified size
    """
    x_vec = np.array([[1, 1],[1, -1],[-1, 1],[-1, -1]])
    return x_vec

# Get PSK constellation
def get_psk_constellation(set_size=4):
    """
    Get a PSK constellation of size M.

    :param set_size: Size of the PSK constellation
    :return: PSK constellation of the specified size
    """
    x = np.exp(1j * np.linspace(0, 2*np.pi, set_size, endpoint=False))
    x_vec = np.stack((np.real(x), np.imag(x)), axis=-1)
    return x_vec


def gen_time_invariant_process(set_x,
                            max_context_len=10,
                            batch_size=100,
                            d=4,
                            snr_db=-5,
                            sigma_h2=1/2):
    """
    Generate a time-invariant process with a given channel and noise variance.

    :param set_x: Set of symbols
    :param max_context_len: Maximum context length
    :param batch_size: Batch size
    :param d: Dimension of the complex symbols
    :param snr_db: SNR in dB
    :param sigma_h2: Channel variance
    :return: y_past, y_q, sig, s_q, H
    """

    sigma2 = 10**(-snr_db/10)
    sigma = np.sqrt(sigma2)
    sigma_h = np.sqrt(sigma_h2)

    # Generate h
    h_real, h_imag = sigma_h * np.random.randn(batch_size, 1, d), sigma_h * np.random.randn(batch_size, 1, d)
    h = h_real + 1j*h_imag

    # Generate z
    z_real, z_imag = sigma * np.random.randn(batch_size, max_context_len, d), sigma*np.random.randn(batch_size, max_context_len, d)
    z = z_real + 1j * z_imag  

    # Generate s and x
    s =  np.random.choice(len(set_x), size=(batch_size, max_context_len), replace=True)
    x_vec = set_x[s]
    x_real, x_imag = x_vec[:,:,0], x_vec[:,:,1]
    x = (x_real + 1j * x_imag).reshape(batch_size, max_context_len, 1)

    # Generate y
    y = x * h + z
    y_real, y_imag = np.real(y), np.imag(y)
    y_vec = np.concatenate((y_real, y_imag), axis=-1)

    y_past = torch.from_numpy(y_vec[:,:-1,:]).float()

    yq = torch.from_numpy(y_vec[:,-1,:]).unsqueeze(1).float()
    s_past = torch.from_numpy(s[:,:-1])
    sq = torch.from_numpy(s[:,-1])


    h_row_1 = np.concatenate((h_real, h_imag), axis=-1) # 1, 2d
    h_row_2 = np.concatenate((-h_imag, h_real), axis=-1)  # 1, 2d
    h_mat = np.concatenate((h_row_1, h_row_2), axis=-2).transpose((0,2,1)) # b, 2d, 2 after T

    return y_past, yq, s_past, sq, h_mat


class SAT(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.w_mat = torch.nn.Parameter(torch.randn(size=(2*d, 2*d)))

    def forward(self, s_past, yq, y_past, size_s):
        """
        :param s_past: b, n
        :param yq: b, 1, 2d
        :param y_past: b, n, 2d
        :param size_s: S
        :return: post_est: b, S        
        
        """
        b, n, d = y_past.shape

        e_s = torch.eye(size_s)[s_past]  # b, n, S

        y_past_t = torch.transpose(y_past, 2, 1)  # b, d, n

        attn_logits = (yq @ self.w_mat @ y_past_t).squeeze(1)  # b, n

        attn_scores = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # b, n, 1

        post_est =  torch.sum(attn_scores * e_s, dim=1)  # b, S

        return post_est
    

def true_post_est(yq, h_mat, snr_db, set_x):

    """
    :param yq: (b, 1, 2d)
    :param h_mat: (b, 2d, 2)
    :param set_x: (S, 2)
    """
    sigma2 = 10**(-snr_db/10)
    # print('sigma2 = ', sigma2)
    yq = yq.numpy()

    yq_t = yq.transpose((0, 2, 1))  # b, 2d, 1
    hx = h_mat @ set_x.T  # b, 2d, S

    scores = np.sum(-np.abs(yq_t - hx)**2, axis=1) / sigma2 / 2  # (b, :, S) -> b, S

    scores_true = torch.from_numpy(scores)
    post_true = torch.softmax(scores_true, dim=-1)

    scores_inner_np = (yq @ h_mat @ set_x.T).squeeze(1) / sigma2  # b, S
    scores_inner = torch.from_numpy(scores_inner_np)
    post_inner = torch.softmax(scores_inner, dim=-1)

    return post_true, post_inner

def cross_entropy(input_dist, tar, tol=1e-30):
    """
    tar: b,
    input_dist: b, S
    """
    input_dist_reg = (input_dist + tol)
    return torch.mean(-torch.log(input_dist_reg[torch.arange(len(tar)), tar]))


def train(model, set_x, batch_size=128, snr_db = -5, d=4, max_context_len=50, n_epochs=10000):

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    size_s = len(set_x)

    pbar = tqdm(range(n_epochs))
    print('Training')

    for t in pbar:

        y_past, yq, s_past, sq, _ = gen_time_invariant_process(set_x=set_x, 
                                                                batch_size=batch_size, 
                                                                snr_db=snr_db, 
                                                                d=d, 
                                                                max_context_len=max_context_len+1)
        post_est = model(s_past=s_past, y_past=y_past, yq=yq, size_s=size_s)
        loss = cross_entropy(input_dist=post_est, tar=sq)  # cross entropy
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_description(f'loss = {loss}')

    torch.save(model.state_dict(), 'model.pt')
    return model

def test(model, set_x, batch_size=128, snr_db = -5, d=4, max_context_len=50):

    ce_model = np.zeros(max_context_len)
    size_s = len(set_x)

    print('Testing')

    
    pbar = tqdm(range(max_context_len))
    for i in pbar:
        y_past, yq, s_past, sq, _ = gen_time_invariant_process(set_x=set_x, 
                                                        batch_size=batch_size, 
                                                        snr_db=snr_db, 
                                                        d=d, 
                                                        max_context_len=i+2)
        
        post_est = model(s_past=s_past, y_past=y_past, yq=yq, size_s=size_s)
        ce = cross_entropy(input_dist=post_est, tar=sq)
        ce_model[i] = ce.detach().numpy()
        pbar.set_description(f'CE = {ce_model[i]}')

    return ce_model

import argparse

parser = argparse.ArgumentParser(description='SAT Convergence')
parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('--snr_db', type=int, default=-5, help='SNR in dB')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--max_context_len', type=int, default=700, help='Maximum context length')
parser.add_argument('--d', type=int, default=4, help='Dimension of the complex symbols')
parser.add_argument('--set_size', type=int, default=4, help='Size of the constellation set')
parser.add_argument('--signal_type', type=str, default='qam', help='Type of signal (qpsk, psk, qam)')
args = parser.parse_args()

n_epochs = args.n_epochs
snr_db = args.snr_db
batch_size = args.batch_size
max_context_len = args.max_context_len
d = args.d
set_size = args.set_size
signal_type = args.signal_type

if signal_type == 'qpsk':
    set_x = get_qpsk_constellation()
elif signal_type == 'psk':
    set_x = get_psk_constellation(set_size=set_size)
elif signal_type == 'qam':
    set_x = get_qam_constellation(set_size=set_size)
else:
    raise ValueError("Invalid signal type")


set_size = len(set_x)

my_sat = SAT(d=d) 
train(model=my_sat, set_x=set_x, batch_size=batch_size, snr_db=snr_db, d=d, max_context_len=max_context_len, n_epochs=n_epochs)

n_samples = 5000
test_max_context_len = 500

ce_sat = test(model=my_sat, set_x=set_x, batch_size=n_samples, snr_db=snr_db, d=d, max_context_len=test_max_context_len)


n_samples_true = 50000
y_past, yq, s_past, sq, h_mat = gen_time_invariant_process(set_x=set_x, 
                                                    batch_size=n_samples_true, 
                                                    d=d, 
                                                    max_context_len=3, 
                                                    snr_db=snr_db)


true_post, inner_post = true_post_est(yq=yq, h_mat=h_mat, snr_db=snr_db, set_x=set_x)  # does not depend on y_past
ce_true = cross_entropy(input_dist=true_post, tar=sq, tol=0).detach().numpy()
ce_inner = cross_entropy(input_dist=inner_post, tar=sq, tol=0).detach().numpy()

print('CE true = ', ce_true)
print('CE Inner = ', ce_inner)

n_points = np.arange(1, test_max_context_len+1)

plt.plot(n_points, ce_sat, label='Transformer', marker='s',  markersize=2, color='darkred', lw=1.5)
plt.axhline(y = ce_true, color = 'k', linestyle = '-', label='True Post', lw=1.5) 
# plt.axhline(y = CE_inner, color = 'steelblue', linestyle = '-', label='Inner Post', lw=1.5) 
plt.xlabel("# In-context examples")
plt.ylabel("Cross entropy")
plt.ylim([0,3])
plt.legend()
plt.grid()
plt.title(f"SAT for {set_size}-{signal_type}")
plt.savefig(f"ce_sat_{set_size}{signal_type}_snr_db{snr_db}.png", bbox_inches='tight', dpi=400)
