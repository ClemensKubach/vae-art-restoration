import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_dim_reduced_latent_space(z: torch.Tensor, labels: list[str], show: bool = True):
    """
    Perform t-SNE dimensional reduction on a PyTorch tensor and plot the 2D space.

    Args:
        z (torch.Tensor): Input tensor of shape (batch_size, latent_dims).
        labels (list[str]): List of labels for each sample in the batch.
        show (bool): Whether to show the plot or not.

    Returns:
        None
    """
    assert z.ndim == 2, f'Expected z to have 2 dimensions, got {z.ndim} instead.'
    assert z.shape[0] == len(labels), f'Expected z to have the same number of samples as labels, got {z.shape[0]} and {len(labels)} instead.'

    # Convert the PyTorch tensor to a numpy array
    z_np = z.detach().cpu().numpy()

    # Apply t-SNE to reduce the dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z_np)

    label_set = set(labels)

    # Plot the 2D space using PyPlot
    plt.figure(figsize=(8, 6))
    for label in label_set:
        x = [z_2d[i][0] for i in range(len(z_2d)) if labels[i] == label]
        y = [z_2d[i][1] for i in range(len(z_2d)) if labels[i] == label]
        plt.scatter(x, y, s=10, label=label)
    plt.title(f'{z.shape[-1]}-dimensional Latent Space in 2D')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    if show:
        plt.show()
    else:
        return plt


if __name__ == "__main__":
    # Generate some random data.
    z = torch.randn(32, 100)

    # Perform dimensional reduction and plot the results.
    plot_dim_reduced_latent_space(z)
