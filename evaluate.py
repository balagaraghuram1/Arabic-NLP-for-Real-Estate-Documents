from sklearn.metrics import classification_report

def evaluate_classification(true_labels, predicted_labels):
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))
