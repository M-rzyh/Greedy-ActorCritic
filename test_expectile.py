import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def expectile_loss(y_true, y_pred, tau=0.7):
    """Compute expectile loss"""
    residual = y_true - y_pred
    return np.mean(np.where(residual > 0, tau * residual**2, (1 - tau) * residual**2))


def fit_expectile_regression(X, y, tau=0.7, max_iter=1000, tol=1e-6):
    """Fit linear regression using expectile loss"""
    n_samples, n_features = X.shape

    # Initialize weights
    weights = np.zeros(n_features)
    bias = 0.0

    # Add intercept column to X
    X_with_intercept = np.column_stack([np.ones(n_samples), X])

    # Iterative reweighted least squares
    for iteration in range(max_iter):
        # Compute predictions
        y_pred = X_with_intercept @ np.concatenate([[bias], weights])
        residuals = y - y_pred

        # Compute weights for each sample
        sample_weights = np.where(residuals > 0, tau, 1 - tau)

        # Weighted least squares update
        W = np.diag(sample_weights)
        XtWX = X_with_intercept.T @ W @ X_with_intercept
        XtWy = X_with_intercept.T @ W @ y

        # Solve weighted normal equations
        try:
            params_new = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            params_new = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

        # Check convergence
        if iteration > 0:
            param_change = np.max(np.abs(params_new - params_old))
            if param_change < tol:
                break

        params_old = params_new.copy()
        bias = params_new[0]
        weights = params_new[1:]

    return weights, bias


def true_q_function(actions):
    """Define a true Q(s,a) function - using a non-linear function"""
    # Example: quadratic function with some complexity
    return 10 + 2 * actions - 0.3 * actions**2 + 0.01 * actions**3


def main():
    np.random.seed(42)

    # Generate true Q-values
    n_samples = 200
    actions_train = np.random.uniform(-10, 10, n_samples)

    # True Q-values with asymmetric noise (more negative outliers)
    true_q_train = true_q_function(actions_train)
    noise = np.random.normal(0, 2, n_samples)
    # Add some asymmetric outliers (simulating suboptimal behavior)
    outlier_mask = np.random.rand(n_samples) < 0.15
    noise[outlier_mask] -= np.random.exponential(5, outlier_mask.sum())

    q_observed = true_q_train + noise

    # Prepare data for regression (using polynomial features for better approximation)
    X_train = np.column_stack([actions_train, actions_train**2, actions_train**3])

    # Fit MSE-based linear regression
    mse_model = LinearRegression()
    mse_model.fit(X_train, q_observed)

    # Fit expectile regression with different tau values
    tau_values = [0.5, 0.7, 0.9]
    expectile_models = {}

    for tau in tau_values:
        weights, bias = fit_expectile_regression(X_train, q_observed, tau=tau)
        expectile_models[tau] = (weights, bias)

    # Generate test points for smooth plotting
    actions_test = np.linspace(-10, 10, 300)
    X_test = np.column_stack([actions_test, actions_test**2, actions_test**3])

    # Compute predictions
    true_q_test = true_q_function(actions_test)
    mse_pred = mse_model.predict(X_test)

    expectile_preds = {}
    for tau in tau_values:
        weights, bias = expectile_models[tau]
        expectile_preds[tau] = X_test @ weights + bias

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: MSE vs Expectile (tau=0.7)
    ax1 = axes[0, 0]
    ax1.scatter(actions_train, q_observed, alpha=0.4, s=20, label='Observed Q(s,a)', color='gray')
    ax1.plot(actions_test, true_q_test, 'k-', linewidth=2, label='True Q(s,a)', alpha=0.7)
    ax1.plot(actions_test, mse_pred, 'b--', linewidth=2, label='MSE Regression')
    ax1.plot(actions_test, expectile_preds[0.7], 'r-', linewidth=2, label='Expectile (Ä=0.7)')
    ax1.set_xlabel('Action', fontsize=11)
    ax1.set_ylabel('Q(s, a)', fontsize=11)
    ax1.set_title('MSE vs Expectile Regression', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Different tau values for expectile
    ax2 = axes[0, 1]
    ax2.scatter(actions_train, q_observed, alpha=0.4, s=20, label='Observed Q(s,a)', color='gray')
    ax2.plot(actions_test, true_q_test, 'k-', linewidth=2, label='True Q(s,a)', alpha=0.7)
    colors = ['green', 'orange', 'red']
    for tau, color in zip(tau_values, colors):
        ax2.plot(actions_test, expectile_preds[tau], linewidth=2,
                label=f'Expectile (Ä={tau})', color=color, linestyle='--')
    ax2.set_xlabel('Action', fontsize=11)
    ax2.set_ylabel('Q(s, a)', fontsize=11)
    ax2.set_title('Expectile Regression with Different Ä', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals for MSE
    ax3 = axes[1, 0]
    mse_residuals_train = q_observed - mse_model.predict(X_train)
    ax3.scatter(actions_train, mse_residuals_train, alpha=0.5, s=30, color='blue')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax3.set_xlabel('Action', fontsize=11)
    ax3.set_ylabel('Residuals', fontsize=11)
    ax3.set_title('MSE Regression Residuals', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Residuals for Expectile (tau=0.7)
    ax4 = axes[1, 1]
    weights_07, bias_07 = expectile_models[0.7]
    expectile_pred_train = X_train @ weights_07 + bias_07
    expectile_residuals_train = q_observed - expectile_pred_train
    ax4.scatter(actions_train, expectile_residuals_train, alpha=0.5, s=30, color='red')
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax4.set_xlabel('Action', fontsize=11)
    ax4.set_ylabel('Residuals', fontsize=11)
    ax4.set_title('Expectile (Ä=0.7) Regression Residuals', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('expectile_vs_mse_comparison.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'expectile_vs_mse_comparison.png'")
    plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("REGRESSION COMPARISON STATISTICS")
    print("="*60)

    mse_train_error = np.mean((q_observed - mse_model.predict(X_train))**2)
    print(f"\nMSE Regression:")
    print(f"  Training MSE: {mse_train_error:.4f}")

    for tau in tau_values:
        weights, bias = expectile_models[tau]
        pred_train = X_train @ weights + bias
        train_error = np.mean((q_observed - pred_train)**2)
        exp_loss = expectile_loss(q_observed, pred_train, tau)
        print(f"\nExpectile Regression (Ä={tau}):")
        print(f"  Training MSE: {train_error:.4f}")
        print(f"  Expectile Loss: {exp_loss:.4f}")

    print("\n" + "="*60)
    print("\nKEY INSIGHTS:")
    print("- MSE regression minimizes squared errors equally for over/under-predictions")
    print("- Expectile regression with Ä > 0.5 penalizes under-predictions more")
    print("- Higher Ä values push the fitted curve toward upper quantiles")
    print("- For RL, Ä=0.7-0.9 helps avoid underestimation of Q-values")
    print("="*60)


if __name__ == "__main__":
    main()
