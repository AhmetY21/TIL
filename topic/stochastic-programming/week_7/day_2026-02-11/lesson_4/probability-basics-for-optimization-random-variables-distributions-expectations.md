---
title: "Probability Basics for Optimization: random variables, distributions, expectations"
date: "2026-02-11"
week: 7
lesson: 4
slug: "probability-basics-for-optimization-random-variables-distributions-expectations"
---

# Topic: Probability Basics for Optimization: random variables, distributions, expectations

## 1) Formal definition (what is it, and how can we use it?)

In the context of stochastic programming, understanding random variables, distributions, and expectations is crucial for modeling and solving optimization problems where some parameters are uncertain. These concepts provide the foundation for quantifying and managing this uncertainty.

*   **Random Variable:** A random variable is a variable whose value is a numerical outcome of a random phenomenon. It can be either *discrete* (taking on a finite or countably infinite number of values, e.g., the number of heads in 10 coin flips) or *continuous* (taking on any value within a given range, e.g., temperature). Mathematically, a random variable is a measurable function from a sample space to the real numbers. In optimization, random variables often represent uncertain input parameters like demand, prices, resource availability, or equipment failure rates.

*   **Distribution:** A probability distribution describes the likelihood of different values or outcomes of a random variable. For discrete random variables, this is usually represented by a *probability mass function (PMF)*, which gives the probability of each specific value. For continuous random variables, it's represented by a *probability density function (PDF)*, where the area under the curve between two points represents the probability of the random variable falling within that range. The cumulative distribution function (CDF) gives the probability that a random variable takes on a value less than or equal to a given value.  Common distributions used in stochastic programming include normal, uniform, exponential, Poisson, and binomial distributions. The choice of distribution is crucial and depends on the nature of the uncertainty being modeled.

*   **Expectation:** The expectation (or expected value) of a random variable is the average value we would expect to observe if we were to repeat the random experiment many times.  It's a weighted average of all possible values of the random variable, where the weights are the probabilities of those values.  For a discrete random variable *X* with PMF *p(x)*, the expectation is *E[X] = Σ x * p(x)*.  For a continuous random variable *X* with PDF *f(x)*, the expectation is *E[X] = ∫ x * f(x) dx*.  In stochastic programming, expectations are often used to define objective functions. For example, we might want to minimize the *expected cost* or maximize the *expected profit* under uncertainty.  Furthermore, the concept of expectation extends to functions of random variables. For example, *E[g(X)]* represents the expected value of the function *g* applied to the random variable *X*.

**How we use it in Stochastic Programming:**
These concepts form the basis of stochastic optimization models. We use them to:

1.  **Model Uncertainty:** Represent uncertain parameters in the optimization problem as random variables with specified distributions.
2.  **Formulate Objectives:** Define objective functions based on expected values of cost, profit, or other performance metrics that depend on the random variables.
3.  **Construct Constraints:** Incorporate probabilistic constraints to manage the risk of violating feasibility conditions due to the uncertainty. These constraints might limit the probability of exceeding a certain resource limit or ensure a certain service level with a given probability.
4.  **Evaluate Solutions:** Assess the performance of different solutions under various scenarios generated from the distributions of the random variables. This allows for robust decision-making that considers the impact of uncertainty.

## 2) Application scenario

Consider a supply chain management problem where a company needs to decide how much inventory to order for a product. The demand for the product is uncertain and can be modeled as a random variable.

*   **Random Variable:** The demand for the product in a given period, represented by the random variable *D*.
*   **Distribution:** Assume *D* follows a normal distribution with a mean of 1000 units and a standard deviation of 200 units. This means *D ~ N(1000, 200^2)*.
*   **Expectation:** The expected demand is *E[D] = 1000* units.

The company wants to minimize its total cost, which includes ordering costs, holding costs for excess inventory, and shortage costs for unmet demand. Let:

*   *Q* be the order quantity (the decision variable).
*   *c* be the cost per unit ordered.
*   *h* be the holding cost per unit of excess inventory.
*   *s* be the shortage cost per unit of unmet demand.

The total cost can be expressed as a function of *Q* and *D*:

```
Cost(Q, D) = cQ + h * max(Q - D, 0) + s * max(D - Q, 0)
```

The company's objective is to minimize the *expected* total cost:

```
Minimize E[Cost(Q, D)] = cQ + E[h * max(Q - D, 0) + s * max(D - Q, 0)]
```

The expectation term involves integrating over the probability density function of the demand *D*. Solving this optimization problem requires knowledge of the distribution of *D* and techniques for evaluating the expected value. This is a classic example of a stochastic programming problem where the uncertainty in demand is explicitly considered.

## 3) Python method (if possible)

We can use Python libraries like `NumPy` and `SciPy` to work with random variables, distributions, and expectations. Specifically, `SciPy.stats` provides tools for defining and working with various probability distributions. `NumPy` can be used for numerical integration to calculate expected values.

```python
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

# Parameters
c = 10  # Cost per unit ordered
h = 2   # Holding cost per unit
s = 5   # Shortage cost per unit
mu = 1000  # Mean demand
sigma = 200 # Standard deviation of demand

# Define the cost function
def cost(Q, D):
    return c * Q + h * np.maximum(Q - D, 0) + s * np.maximum(D - Q, 0)

# Define the expected cost function
def expected_cost(Q):
    # Define the function to integrate (cost * PDF)
    def integrand(D):
        return cost(Q, D) * norm.pdf(D, loc=mu, scale=sigma)

    # Integrate from -infinity to infinity (approximated by a range)
    #Note: Consider integration bounds carefully.  They can influence accuracy
    result, _ = quad(integrand, mu - 5*sigma, mu + 5*sigma)  # Integrate from mean-5sigma to mean+5sigma

    return result

# Example: Calculate expected cost for an order quantity of 1000
Q = 1000
expected_cost_value = expected_cost(Q)
print(f"Expected cost for Q = {Q}: {expected_cost_value}")


# Example: Find optimal Q (simple search - more sophisticated optimization needed for real problems)
Q_values = np.arange(500, 1501, 10)
expected_costs = [expected_cost(q) for q in Q_values]
optimal_Q = Q_values[np.argmin(expected_costs)]
print(f"Approximate Optimal Q: {optimal_Q}")
```

This code calculates the expected cost for a given order quantity *Q* and provides an example of a simple search to find an approximate optimal *Q*. Note that for more complex problems, specialized optimization algorithms are required to efficiently find the optimal solution.

## 4) Follow-up question

How does the choice of the probability distribution for a random variable impact the optimal solution in a stochastic programming problem? What are some strategies for choosing an appropriate distribution when historical data is limited or unavailable?