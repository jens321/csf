
import torch

zdim = 4

linear = torch.nn.Linear(zdim, zdim)

x = torch.randn(1, zdim)
xp = torch.randn(1, zdim)
phis = linear(x)
phisp = linear(xp)
z = torch.randn(100000000, zdim)
z /= torch.norm(z, dim=-1, keepdim=True)

# compute log expected exp
log_expected_exp = torch.exp(((phisp - phis) * z).sum(dim=1)).mean().log()

# compute bessel function
gamma_fn = 1
if zdim == 2:
    bessel_fn = torch.special.i0
    denominator = 1
    numerator = 1
elif zdim == 4:
    numerator = 2
    bessel_fn = torch.special.i1
    denominator = torch.norm(phisp - phis)

bessel_result = torch.log(gamma_fn * bessel_fn(torch.norm(phisp - phis)) * numerator / denominator)

print(log_expected_exp, bessel_result)

