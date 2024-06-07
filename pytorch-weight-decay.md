# A Bayesian Perspective of Weight Decay in PyTorch

*Ethan Harvey, Mikhail Petrov, Michael C. Hughes*

June 11, 2024

### What is weight decay?

Regularization is a common approach to reduce overfitting in deep learning models. Weight decay, also known as L2 regularization, reduces overfitting by adding the L2 norm of model parameters to the loss function. This penalty term is known as weight decay, because it biases parameter values toward the origin. Given a training dataset $\mathcal{D} = \\{x_i, y_i\\}_{i=1}^n$ and probabilistic model $p(y | x, \theta)$, we minimize the regularized negative log-likelihood

```math
    L(\theta) := \underbrace{- \frac{1}{n} \sum_{i=1}^n \log p(y_i | x_i, \theta)}_{f(\theta)}  + \frac{\lambda}{2} ||\theta||_2^2
```

where $f(\theta)$ is the (unpenalized) loss function (e.g., mean squared error, cross entropy) and $\lambda \in [0, \infty)$ is a hyperparameter that determines the relative weight of the penalty term compared to the loss function.

### How is weight decay implemented in PyTorch?

Stocastic gradient descent (SGD) in PyTorch implements weight decay via an efficient direct modification of the gradient vector. Given decay parameter $\lambda > 0$, at iteration $t$ the optimizer internally performs an additive incremental update of the gradient with respect to parameters: $\nabla_\theta f(\theta_{t-1}) = \nabla_\theta f(\theta_{t-1}) + \lambda \theta_{t-1}$ where $f(\theta)$ is the (unpenalized) loss function (e.g., mean squared error, cross entropy). This direct modification is equivalent to, but more efficient than, automatic differentiation of a penalized loss that includes an L2 penalty term for $\theta$.

### A Bayesian interpretation

### How to implement weight decay from scratch?


### References

[1] Ethan Harvey, Mikhail Petrov, and Michael C. Hughes. Transfer Learning with Informative Priors: Simple Baselines Better than Previously Reported. *Transactions on Machine Learning Research (TMLR)*, 2024. ISSN 2835-8856.
