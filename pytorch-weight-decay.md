# A Bayesian Perspective of Weight Decay in PyTorch

*Ethan Harvey, Mikhail Petrov, Michael C. Hughes*

June 11, 2024

### What is weight decay?

Regularization is a common approach to reduce overfitting in deep learning models. Weight decay, also known as L2 regularization, reduces overfitting by adding the L2 norm of model parameters to the loss function. This penalty term is known as weight decay, because it biases parameter values toward the origin. Given a training dataset $\mathcal{D} = \\{x_i, y_i\\}_{i=1}^n$ and probabilistic model $p(y|x, \theta)$, we minimize the regularized negative log-likelihood

```math
    L(\theta) := \underbrace{- \frac{1}{n} \sum_{i=1}^n \log p(y_i|x_i, \theta)}_{f(\theta)}  + \frac{\lambda}{2} ||\theta||_2^2
```

where $f(\theta)$ is the (unpenalized) loss function (e.g., mean squared error, cross entropy) and $\lambda \in [0, \infty)$ is a hyperparameter that determines the relative weight of the penalty term compared to the loss function.

### How is weight decay implemented in PyTorch?

Stocastic gradient descent (SGD) in PyTorch implements weight decay via an efficient direct modification of the gradient vector. Given decay parameter $\lambda > 0$, at iteration $t$ the optimizer internally performs an additive incremental update of the gradient with respect to parameters: $\nabla_\theta f(\theta_{t-1}) \leftarrow \nabla_\theta f(\theta_{t-1}) + \lambda \theta_{t-1}$ where $f(\theta)$ is the (unpenalized) loss function (e.g., mean squared error, cross entropy). This direct modification is equivalent to, but more efficient than, automatic differentiation of a penalized loss that includes an L2 penalty term for $\theta$.

### A Bayesian interpretation

Weight decay in PyTorch has a Bayesian interpretation as maximum a-posteriori (MAP) estimation of parameters $\theta$, under the assumption that the parameters $\theta$ follow a Gaussian prior $\mathcal{N}(0, \tau I)$ with scalar precision parameter $\tau = \frac{1}{n\lambda}$. Rescaling by $n$ ensures that minimizing the above loss is equivalent to maximizing the per-example MAP objective: $\frac{1}{n} [ \log p(y|x, \theta) + \log p(\theta) ]$.

```math
\begin{align*} 
    p(\theta | \mathcal{D}) &\propto p(\mathcal{D} | \theta) p(\theta) &\qquad \text{by Bayes rule} \\
    &\propto - \log p(y|x, \theta) - \log p(\theta) &\qquad \text{negative log-likelihood} \\
    &\propto - \frac{1}{n} \log p(y|x, \theta) - \frac{1}{n} \log p(\theta) &\qquad \text{rescale by $1/n$} \\
    &\propto - \frac{1}{n} \log p(y|x, \theta) - \frac{1}{n} \log \mathcal{N}(\theta | 0, \tau I) &\qquad \text{TODO} \\
    &\propto - \frac{1}{n} \log p(y|x, \theta) - \frac{1}{n} \log \frac{1}{(2 \pi)^{D/2}} \frac{1}{|\tau I|^{1/2}} \exp \{ -\frac{1}{2} (\theta - 0)^T (\tau I)^{-1} (\theta - 0) \} &\qquad \text{TODO} \\
    &\propto - \frac{1}{n} \log p(y|x, \theta) + \frac{1}{n} \frac{1}{2 \tau} \theta^T \theta - \text{const} &\qquad \text{TODO} \\
    &\propto - \frac{1}{n} \log p(y|x, \theta) + \frac{1}{n} \frac{n \lambda}{2} ||\theta||_2^2 - \text{const} &\qquad \text{TODO} \\
    &\propto - \frac{1}{n} \log p(y|x, \theta) + \frac{\lambda}{2} ||\theta||_2^2 - \text{const} &\qquad \text{TODO}
\end{align*} 
```

### How to implement weight decay from scratch?

Import packages and initialized model.

Use randomly initialized model and Gaussian noise to generate labels.

Reinitialized model and save parameters so we can train each model from the same initialization.

### References

[1] Ethan Harvey, Mikhail Petrov, and Michael C. Hughes. Transfer Learning with Informative Priors: Simple Baselines Better than Previously Reported. *Transactions on Machine Learning Research (TMLR)*, 2024. ISSN 2835-8856.
