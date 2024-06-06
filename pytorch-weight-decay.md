# A Bayesian Perspective of Weight Decay in PyTorch

*Ethan Harvey, Mikhail Petrov, Michael C. Hughes*

June 11, 2024

Regularization is a common approach to prevent overfitting in deep learning models. Many regularization techniques limit model capacity by adding a parameter norm penalty $\Omega(\theta)$ to the (unpenalized) loss function $f(\theta)$. For example, given a training dataset $\mathcal{D} = \\{x_i, y_i\\}_{i=1}^n$ and probabilistic model $p(y | x, \theta)$, we minimize the regularized negative log-likelihood

```math
    L(\theta) := \underbrace{- \frac{1}{n} \sum_{i=1}^n \log p(y_i | x_i, \theta)}_{f(\theta)}  + \Omega(\theta).
```

### What is weight decay?

Weight decay is a common term for L2 regularization. This regularization technique biases the parameters towards smaller values by adding a regularization term $\Omega(\theta) = \frac{1}{2} ||\theta||_2^2$ to the (unpenalized) loss function $f(\theta)$. In practice, a decay parameter $\lambda > 0$ is tuned on validation data to determine model capacity

```math
    L(\theta) := \underbrace{- \frac{1}{n} \sum_{i=1}^n \log p(y_i | x_i, \theta)}_{f(\theta)}  + \frac{\lambda}{2} ||\theta||_2^2.
```

```python
def mse_loss(y_hat, y):
    return torch.mean((y - y_hat) ** 2)

def l2_penalty(params):
    return torch.sum(params ** 2) / 2
```

### How is weight decay implemented in PyTorch?

Stocastic gradient descent (SGD) in PyTorch implements weight decay via an efficient direct modification of the gradient vector. Given decay parameter $\lambda > 0$, at iteration $t$ the optimizer internally performs an additive incremental update of the gradient with respect to parameters: $\nabla_\theta f(\theta_{t-1}) = \nabla_\theta f(\theta_{t-1}) + \lambda \theta_{t-1}$ where $f(\theta)$ is the (unpenalized) loss function (e.g. cross entropy). This direct modification is equivalent to, but more efficient than, automatic differentiation of a penalized loss that includes an L2 penalty term for $\theta$.

### References

[1] Ethan Harvey, Mikhail Petrov, and Michael C. Hughes. Transfer Learning with Informative Priors: Simple Baselines Better than Previously Reported. *Transactions on Machine Learning Research (TMLR)*, 2024. ISSN 2835-8856.
