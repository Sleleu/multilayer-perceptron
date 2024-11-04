import matplotlib.pyplot as plt


def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
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

    plt.tight_layout()
    plt.show()
