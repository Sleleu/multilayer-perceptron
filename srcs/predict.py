import numpy as np
import pandas as pd
from srcs.Mlp import MLP
from srcs.Scaler import Scaler
from srcs.utils import GREEN, YELLOW, CYAN, MAGENTA, END, load

def predict(model_path: str, data_path: str, scaler_params=None):
    m_data = np.load(model_path, allow_pickle=True).item()

    model = MLP(
        hidden_layer_sizes=m_data['hidden_layer_sizes'],
        output_layer_size=m_data['output_layer_size'],
        activation=m_data['activation'],
        output_activation=m_data['output_activation'],
        loss=m_data['loss']
    )
    try:
        X = pd.read_csv(data_path, header=None)
    except Exception as e:
        print(f"{YELLOW}Error loading data: {e}{END}")
        exit(1)

    if scaler_params:
        scaler = Scaler(method=scaler_params['method'])
        if scaler_params['method'] == 'z_score':
            scaler.mean = scaler_params['mean']
            scaler.scale = scaler_params['scale']
        else:
            scaler.min = scaler_params['min']
            scaler.max = scaler_params['max']
        X = scaler.transform(X)
    
    # PRED
    probabilities, _ = model.feed_forward(X, m_data['W'], m_data['b'])
    predictions = np.argmax(probabilities, axis=1).round(decimals=2)
    confidence = np.max(probabilities, axis=1)
    
    # Print stats
    print(f"\n{GREEN}Prediction Statistics:{END}")
    print(f"{MAGENTA}Average confidence: {GREEN}{np.mean(confidence):.4f}{END}")
    print(f"{MAGENTA}Min confidence: {GREEN}{np.min(confidence):.4f}{END}")
    print(f"{MAGENTA}Max confidence: {GREEN}{np.max(confidence):.4f}{END}")
    
    print(f"\n{GREEN}Class distribution:{END}")
    _, counts = np.unique(predictions, return_counts=True)
    classes = ('B', 'M')
    for i, c in enumerate(counts):
        print(f"{MAGENTA}Class {classes[i]}: {CYAN}{c} samples {GREEN}({c/len(predictions)*100:.2f}%){END}")
    
    # Map to labels B or M
    label_mapping = {0: 'B', 1: 'M'}
    predicted_labels = [label_mapping[pred] for pred in predictions]
    
    # Output DF
    results = pd.DataFrame({
        'Row': range(1, len(predictions) + 1),
        'Prediction': predicted_labels,
        'Confidence': np.round(confidence, decimals=4),
        'Probability_B': np.round(probabilities[:, 0], decimals=4),
        'Probability_M': np.round(probabilities[:, 1], decimals=4)
    })
    
    output_path = 'predictions.csv'
    results.to_csv(output_path, index=False)
    print(f"\n{GREEN}Predictions saved to {CYAN}'{output_path}'{END}")

    print(f"\n{GREEN}Sample predictions (first 5):{CYAN}")
    print(results.head())
    print(f"{END}", end='')
    
    # print(predictions)
    # y_test = load("data/processed/test/y_test.csv", header=None).to_numpy().ravel().T
    # print(y_test)
    return predicted_labels, probabilities