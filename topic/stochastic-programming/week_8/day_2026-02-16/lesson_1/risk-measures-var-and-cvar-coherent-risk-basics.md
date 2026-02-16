---
title: "Risk Measures: VaR and CVaR (coherent risk basics)"
date: "2026-02-16"
week: 8
lesson: 1
slug: "risk-measures-var-and-cvar-coherent-risk-basics"
---

# Topic: Risk Measures: VaR and CVaR (coherent risk basics)

## 1) Formal definition (what is it, and how can we use it?)

*   **Value at Risk (VaR):** VaR at a confidence level α (typically 95% or 99%) represents the maximum loss that is *not exceeded* with a probability of α within a given time horizon. More formally, for a random variable *X* representing the loss (so larger values are worse), VaR<sub>α</sub>(X) is the α-quantile of the distribution of *X*. That is,

    VaR<sub>α</sub>(X) = inf{x ∈ ℝ : P(X ≤ x) ≥ α}

    It answers the question: "What is the worst loss I can expect to experience α% of the time?".

    We use VaR to quantify the potential loss in a portfolio or investment over a specific period. It is a widely used risk measure in finance, but it has limitations because it doesn't say anything about the losses *beyond* the VaR level.

*   **Conditional Value at Risk (CVaR):** CVaR, also known as Expected Shortfall (ES), is the expected loss given that the loss exceeds the VaR level. It's the average of the losses that occur in the (1-α)% tail of the distribution. Formally:

    CVaR<sub>α</sub>(X) = E[X | X ≥ VaR<sub>α</sub>(X)]

    CVaR addresses the limitation of VaR by considering the magnitude of losses beyond the VaR. It answers the question: "If the worst α% scenarios occur, what is the *average* loss I should expect?".

    CVaR is considered a *coherent* risk measure, satisfying the properties of:

    *   **Translation invariance:** Adding a constant to the loss shifts the CVaR by the same constant.
    *   **Subadditivity:** The CVaR of a portfolio is less than or equal to the sum of the CVaRs of the individual assets.  This is crucial for diversification benefits; VaR is NOT always subadditive.
    *   **Positive homogeneity:** Scaling the loss by a positive constant scales the CVaR by the same constant.
    *   **Monotonicity:** If one loss is always less than another, its CVaR is also less than the other's CVaR.

    We use CVaR for more robust risk management because it considers the tail risk (losses beyond the VaR) and its coherence properties encourage diversification.

## 2) Application scenario

Imagine an investment firm managing a portfolio of stocks. They want to assess the risk of their portfolio using VaR and CVaR.

*   **VaR:** They calculate the 95% VaR for the portfolio over a one-day period to be $1 million. This means there's a 5% chance they could lose more than $1 million in a single day. This gives a baseline understanding of the potential downside.

*   **CVaR:** They then calculate the 95% CVaR to be $1.5 million. This means that if the portfolio experiences a loss exceeding the VaR (the worst 5% of scenarios), the average loss will be $1.5 million. This provides a more complete picture of the potential losses in extreme situations.

The CVaR provides a clearer view of the magnitude of loss that could be incurred when the VaR level is breached. Knowing both helps the firm set appropriate risk controls and capital reserves. Because CVaR is subadditive, the firm can use it to optimize portfolio diversification strategies.  VaR can actually *discourage* diversification in some scenarios because it is not coherent.

## 3) Python method (if possible)

```python
import numpy as np
import scipy.stats as st

def calculate_var(data, alpha):
  """
  Calculates Value at Risk (VaR) for a given dataset and confidence level.

  Args:
    data: A numpy array of loss/return values.
    alpha: The confidence level (e.g., 0.95 for 95% VaR).

  Returns:
    The VaR value.
  """
  return np.quantile(data, 1 - alpha)


def calculate_cvar(data, alpha):
  """
  Calculates Conditional Value at Risk (CVaR) for a given dataset and confidence level.

  Args:
    data: A numpy array of loss/return values.
    alpha: The confidence level (e.g., 0.95 for 95% CVaR).

  Returns:
    The CVaR value.
  """
  var = calculate_var(data, alpha)
  cvar = np.mean(data[data >= var])  # Assuming 'data' represents losses. If returns, change >= to <=.
  return cvar


# Example usage:
returns = np.random.normal(0, 0.01, 1000)  # Simulate daily returns with 1% std dev.
losses = -returns # convert to losses

alpha = 0.95

var_95 = calculate_var(losses, alpha)
cvar_95 = calculate_cvar(losses, alpha)

print(f"95% VaR: {var_95:.4f}")
print(f"95% CVaR: {cvar_95:.4f}")


#Alternative approach using scipy.stats
# Assuming 'losses' is the numpy array representing losses
alpha = 0.95
var_95_alternative = st.scoreatpercentile(losses, alpha*100) #scipy uses percentile, so multiply alpha by 100

print(f"95% VaR (scipy): {var_95_alternative:.4f}")
```

## 4) Follow-up question

How can we estimate VaR and CVaR when we *don't* have a large historical dataset? For example, how could we use Monte Carlo simulation or Extreme Value Theory to estimate these risk measures?