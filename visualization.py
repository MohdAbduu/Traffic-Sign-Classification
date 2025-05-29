import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import torch

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, id_to_label, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    labels = [id_to_label[i] for i in range(len(id_to_label))]

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"plots/{title.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

def plot_sample_predictions(model, dataloader, id_to_label, model_name):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(12, 10))

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(min(8, len(images))):
                img = images[i].permute(1, 2, 0).numpy()
                true_label = id_to_label[labels[i].item()]
                pred_label = id_to_label[predicted[i].item()]

                plt.subplot(2, 4, i+1)
                plt.imshow(img)
                plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
                plt.axis('off')

                images_shown += 1
                if images_shown == 8:
                    break
            break  # Show only 1 batch

    plt.tight_layout()
    plt.savefig(f"plots/{model_name.lower()}_sample_predictions.png")
    plt.close()

def plot_classification_report(y_true, y_pred, label_map):
    report = classification_report(y_true, y_pred, target_names=list(label_map.values()), output_dict=True)
    labels = list(label_map.values())
    precision = [report[label]['precision'] for label in labels]
    recall = [report[label]['recall'] for label in labels]
    f1 = [report[label]['f1-score'] for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(18, 6))
    plt.bar(x - width, precision, width=width, label='Precision')
    plt.bar(x, recall, width=width, label='Recall')
    plt.bar(x + width, f1, width=width, label='F1 Score')
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('Score')
    plt.title('Precision / Recall / F1 Score per Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/classification_metrics.png')
    plt.close()

def plot_training_history(train_losses, test_losses, train_acc, test_acc, model_name):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'plots/{model_name.lower()}_training_history.png')
    plt.close()

def compare_models(histories):
    epochs = range(1, len(histories[0]['train_losses']) + 1)

    plt.figure(figsize=(14, 5))
    for history in histories:
        plt.plot(epochs, history['test_acc'], label=f"{history['name']} Test Acc")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()

def show_example_predictions(model, images, labels, label_map, model_name):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    images_np = images.cpu().permute(0, 2, 3, 1).numpy()
    images_np = np.clip(images_np * 0.5 + 0.5, 0, 1)  # Unnormalize if needed

    plt.figure(figsize=(20, 8))
    for i in range(min(10, len(images))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images_np[i])
        actual = label_map[labels[i].item()]
        predicted = label_map[preds[i].item()]
        plt.title(f"A: {actual}\nP: {predicted}", fontsize=9)
        plt.axis('off')
    plt.suptitle(f"{model_name} Predictions", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/{model_name.lower()}_predictions.png')
    plt.close()

def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1
