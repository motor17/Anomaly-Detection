# -*- coding: utf-8 -*-
"""Flexible Anomaly Detection Code"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
import openpyxl

def load_data_from_text(data_text, start_marker="DataValue", vgate_col=1, idrain_col=4):
    """Load Vgate and Idrain values from the provided text data."""
    lines = data_text.splitlines()
    vgate_values = []
    idrain_values = []
    
    data_started = False
    print("Lines being processed:")
    for i, line in enumerate(lines):
        print(f"Line {i}: {line}")
        if not data_started and start_marker in line:  # Only start once
            data_started = True
            print(f"Data started at line {i}")
            continue
        if data_started and line.strip():
            values = line.split()
            print(f"Split values: {values}")
            if (len(values) > max(vgate_col, idrain_col) and 
                values[vgate_col].replace('.', '', 1).replace('-', '', 1).isdigit() and 
                values[idrain_col].replace('.', '', 1).replace('-', '', 1).isdigit()):
                vgate_values.append(float(values[vgate_col]))
                idrain_values.append(float(values[idrain_col]))
                print(f"Added: Vgate={values[vgate_col]}, Idrain={values[idrain_col]}")
            elif not line.strip().startswith('DataValue'):  # Stop if non-numeric and not DataValue
                print(f"Stopping at non-numeric line {i}: {line}")
                break
    
    return vgate_values, idrain_values

def calculate_gm(vgate_values, idrain_values):
    """Calculate transconductance (Gm) using the point-to-point method."""
    gm_values = []
    for i in range(1, len(vgate_values)):
        vgate_diff = vgate_values[i] - vgate_values[i - 1]
        idrain_diff = idrain_values[i] - idrain_values[i - 1]
        gm = idrain_diff / vgate_diff if vgate_diff != 0 else 0
        gm_values.append(gm)
    return gm_values

def save_anomalies_to_excel(anomalies, filename="anomalous_data.xlsx"):
    """Save anomalies to an Excel file."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Vgate", "Gm"])
    for anomaly in anomalies:
        ws.append([anomaly[0], anomaly[1]])
    wb.save(filename)

def anomaly_detection_plot(vgate_values, gm_values, anomaly_scores, threshold, plot_type="scatter"):
    """Generate different types of anomaly detection plots."""
    X = np.array(list(zip(vgate_values[1:], gm_values)))
    anomalies = X[anomaly_scores < threshold]

    if plot_type == "scatter":
        plt.figure(figsize=(8, 6))
        plt.scatter(vgate_values[1:], gm_values, label='Original Data')
        plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red', label='Anomalies')
        plt.xlabel('Vgate (V)')
        plt.ylabel('Gm (S)')
        plt.title('Anomaly Detection using Isolation Forest')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif plot_type == "dual_axis":
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(vgate_values[1:], gm_values, label='Gm', color='blue')
        ax1.set_xlabel('Vgate (V)')
        ax1.set_ylabel('Gm (S)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(vgate_values[1:], anomaly_scores, label='Anomaly Score', color='red', linestyle='--')
        ax2.set_ylabel('Anomaly Score', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        for i in range(len(anomaly_scores)):
            if anomaly_scores[i] < threshold:
                ax1.scatter(vgate_values[i+1], gm_values[i], color='red', s=50, 
                           label='Anomaly' if i == 0 else "")
        
        plt.title('Anomaly Detection using Isolation Forest')
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        plt.grid(True)
        plt.show()

    save_anomalies_to_excel(anomalies)

def anomaly_score_map(vgate_values, gm_values, anomaly_scores):
    """Generate an anomaly score map."""
    X = np.array(list(zip(vgate_values[1:], gm_values)))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100),
                         np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    plt.colorbar(label='Anomaly Score')
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='black', marker='o', label='Data Points')
    plt.xlabel('Scaled Vgate')
    plt.ylabel('Scaled Gm')
    plt.title('Anomaly Score Map')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data from file
    with open("c:/Users/User/Desktop/Habit Tracker/document.txt", "r") as file:
        data_text = file.read()

    # Load data
    vgate_values, idrain_values = load_data_from_text(data_text)
    
    # Debug: Check if data is loaded
    print("Vgate values:", vgate_values)
    print("Idrain values:", idrain_values)
    
    if len(vgate_values) < 2 or len(idrain_values) < 2:
        print("Error: Insufficient data points to perform anomaly detection. Need at least 2 points.")
    else:
        # Calculate Gm
        gm_values = calculate_gm(vgate_values, idrain_values)
        
        # Prepare data for Isolation Forest
        X = np.array(list(zip(vgate_values[1:], gm_values)))
        
        # Check if X is valid
        if X.size == 0:
            print("Error: No valid data for Isolation Forest after processing.")
        else:
            model = IsolationForest(contamination='auto', random_state=42)
            model.fit(X)
            anomaly_scores = model.decision_function(X)

            # Interactive scatter plot
            slider = widgets.FloatSlider(value=-0.08, min=-5, max=2, step=0.01, 
                                         description='Anomaly Threshold:', continuous_update=True)
            interact(lambda threshold: anomaly_detection_plot(vgate_values, gm_values, anomaly_scores, threshold, "scatter"), 
                     threshold=slider)

            # Interactive dual-axis plot
            slider_2 = widgets.FloatSlider(value=-0.1116, min=-5, max=2, step=0.01, 
                                           description='Anomaly Threshold:', continuous_update=True)
            interact(lambda threshold: anomaly_detection_plot(vgate_values, gm_values, anomaly_scores, threshold, "dual_axis"), 
                     threshold=slider_2)

            # Anomaly score map
            anomaly_score_map(vgate_values, gm_values, anomaly_scores)
