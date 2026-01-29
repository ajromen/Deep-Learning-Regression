import numpy as np
from matplotlib import pyplot as plt


def get_subplot_layout(choice: str):
    if choice == "activation":
        return 2, 4
    elif choice == "layers":
        return 2, 2
    elif choice == "optimizer":
        return 2, 2
    elif choice == 'single':
        return 1, 1
    else:
        raise ValueError("Unknown comparison type")

def visualise(models, flags, data):
    if not flags.visualise: return
    if flags.activation == "all":
        visualize_models(models, data, mode="activation")
    elif flags.layers == "all":
        visualize_models(models, data, mode="layers")
    elif flags.optimizer == "all":
        visualize_models(models, data, mode="optimizer")
    else:
        visualize_models(models, data)


def visualize_models(models, data, mode: str = 'single'):
    rows, cols = get_subplot_layout(mode)

    fig = plt.figure(figsize=(cols * 4, rows * 4))

    dim = "3d" if models[0].input_size == 2 else None

    for i, model in enumerate(models):
        ax = fig.add_subplot(rows, cols, i + 1, projection=dim)

        plot_model(model, ax, data)

    plt.tight_layout()
    plt.show()


def plot_model(model, ax, data: np.ndarray):
    n = len(data)
    data = data[:500]
    X = model.x_test
    Y = model.y_test
    y_hat = model.forward_pass(X)

    if model.input_size == 2 and model.output_size == 1:
        x1_min, x1_max = data[:, 0].min(), data[:, 0].max()
        x2_min, x2_max = data[:, 1].min(), data[:, 1].max()

        x1_grid, x2_grid = np.meshgrid(
            np.linspace(x1_min, x1_max, 50),
            np.linspace(x2_min, x2_max, 50)
        )

        grid_points = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
        y_grid_pred = model.forward_pass(grid_points).reshape(x1_grid.shape)

        ax.plot_surface(
            x1_grid, x2_grid, y_grid_pred,
            alpha=0.6, cmap="viridis"
        )
        ax.scatter(data[:, 0], data[:, 1], data[:, -1], color="red", s=30)

        ax.set_xlabel(model.c_names[0])
        ax.set_ylabel(model.c_names[1])
        ax.set_zlabel(model.c_names[2])

    elif model.input_size == 1 and model.output_size == 1:
        idx = np.argsort(X[:, 0])
        X_sorted = X[idx]
        Y_sorted = Y[idx]
        y_hat_sorted = y_hat[idx]

        ax.plot(X_sorted[:, 0], Y_sorted[:, 0], color="blue", label="True")
        ax.plot(X_sorted[:, 0], y_hat_sorted[:, 0], color="red", label="Pred", alpha=0.7)

        ax.scatter(X_sorted[:, 0], Y_sorted[:, 0], color="blue", s=15, alpha=0.3)
        ax.scatter(X_sorted[:, 0], y_hat_sorted[:, 0], color="red", s=15, alpha=0.3)

        ax.set_xlabel(model.c_names[0])
        ax.set_ylabel(model.c_names[1])
        ax.legend()

    else:
        ax.text(0.5, 0.5, "Unsupported input/output size", ha="center", va="center")

    ax.set_title(model.name)

    return model.loss.calculate_loss(y_hat, Y)
