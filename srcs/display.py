import matplotlib.pyplot as plt
import numpy as np
from srcs.Scaler import Scaler


def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    ax1.plot(train_losses, label='training Loss')
    ax1.plot(val_losses, label='validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curves - Loss')
    ax1.legend()

    ax2.plot(train_accuracies, label='train acc')
    ax2.plot(val_accuracies, label='validation acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Learning Curves - Accuracy')
    ax2.legend()

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    contour = ax3.contourf(X, Y, Z, levels=100, cmap='jet')
    fig.colorbar(contour, ax=ax3, label="Cost function")
    
    scaler = Scaler(method="minmax") # Norm
    norm_losses = scaler.fit_transform(train_losses)

    # Create spiral path
    t = np.linspace(0, 4*np.pi, len(train_losses))
    radius = 3 * (1 - np.exp(-norm_losses*2))
    path_x = radius * np.cos(t)
    path_y = radius * np.sin(t)
    
    ax3.plot(path_x, path_y, color='white', alpha=0.7)
    ax3.plot(path_x, path_y, 'x', color='yellow', linewidth=2)
    ax3.plot(path_x[0], path_y[0], 'wo', markersize=8, label='Start')
    ax3.plot(path_x[-1], path_y[-1], 'ro', markersize=8, label='End')
    
    ax3.set_xlabel("X1")
    ax3.set_ylabel("X2")
    ax3.set_title("Loss Landscape & Gradient Path")
    ax3.legend()

    plt.tight_layout()
    plt.show()
