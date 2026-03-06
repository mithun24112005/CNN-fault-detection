import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Fault Detector",
    page_icon="🖥️",
    layout="centered"  # Simple centered layout
)

# --- 2. Model Loading ---
@st.cache_resource
def load_ai_model():
    # List of possible filenames
    possible_files = ['fault_cnn_model.keras']
    
    for file in possible_files:
        if os.path.exists(file):
            try:
                return load_model(file)
            except:
                continue
    return None

model = load_ai_model()

# --- 3. Core Logic & Feature Engineering ---

FAULT_LABELS = {0: "✅ No Fault", 1: "⚠️ Bitflip", 2: "❌ Opcode Fault"}

def calculate_truth(a, b, opcode):
    """Calculate the mathematically correct result (32-bit unsigned)"""
    mask = 0xFFFFFFFF
    if opcode == 0: return (a + b) & mask   # ADD
    elif opcode == 1: return (a - b) & mask # SUB
    elif opcode == 2: return a & b          # AND
    elif opcode == 3: return a | b          # OR
    elif opcode == 4: return a ^ b          # XOR
    return 0

def get_binary_list(n):
    """Convert integer to list of 32 bits"""
    return [int(b) for b in format(n, '032b')]

def run_prediction(a, b, opcode, faulty_val):
    """
    Prepares data with Enhanced Features (Bit + Bit Count) and predicts.
    """
    # 1. Ground Truth
    true_val = calculate_truth(a, b, opcode)
    
    # 2. Find Errors (XOR)
    error_val = faulty_val ^ true_val
    error_bits = get_binary_list(error_val)
    
    # 3. Feature Engineering (CRITICAL: Must match training data)
    # We need shape (1, 32, 2) -> [[bit, total_errors], ...]
    total_errors = sum(error_bits)
    features = [[bit, total_errors] for bit in error_bits]
    
    # Reshape for CNN
    input_tensor = np.array(features).reshape(1, 32, 2)
    
    # 4. Predict
    probs = model.predict(input_tensor, verbose=0)
    pred_idx = np.argmax(probs)
    confidence = probs[0][pred_idx] * 100
    
    return FAULT_LABELS[pred_idx], confidence, error_val, true_val

# --- 4. The Interface ---

st.title(" Fault Detector")
st.write("Detect hardware errors using cnn model.")

# Check model status
if not model:
    st.error("⚠️ Model file not found! Please put 'fault_cnn_model.h5' in this folder.")
    st.stop()

# Tabbed Interface for Simplicity
tab1, tab2 = st.tabs([" Manual Test", " Batch File Upload"])

with tab1:
    st.subheader("Input Data")
    
    c1, c2 = st.columns(2)
    with c1:
        a_in = st.number_input("Operand A", value=100, step=1)
        op_in = st.selectbox("Operation", [0, 1, 2, 3, 4], 
                             format_func=lambda x: ["ADD", "SUB", "AND", "OR", "XOR"][x])
    with c2:
        b_in = st.number_input("Operand B", value=200, step=1)
    
    st.markdown("---")
    
    # Interactive Fault Simulation
    correct_res = calculate_truth(a_in, b_in, op_in)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        # Allow user to manually type a faulty result
        res_in = st.number_input("ALU Output (Edit this to simulate fault)", value=correct_res)

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        pred, conf, err_int, truth = run_prediction(a_in, b_in, op_in, res_in)
        
        # -- Results --
        st.divider()
        
        # Main Result
        if "No Fault" in pred:
            st.success(f"**Result:** {pred} ({conf:.1f}%)")
        else:
            st.error(f"**Result:** {pred} ({conf:.1f}%)")
        
        # Bit Visualization
        st.write("### Error Map (32-bit)")
        err_bin_str = format(err_int, '032b')
        
        # Simple HTML for bits - works in Dark/Light mode
        # Red = Error, Gray = Correct
        html_bits = "<div style='font-family: monospace; font-size: 1.5rem; letter-spacing: 3px; word-wrap: break-word;'>"
        for b in err_bin_str:
            color = "#FF4B4B" if b == '1' else "#888888" # Red vs Gray
            opacity = "1.0" if b == '1' else "0.3"
            html_bits += f"<span style='color:{color}; opacity:{opacity}; font-weight:bold;'>{b}</span>"
        html_bits += "</div>"
        
        st.markdown(html_bits, unsafe_allow_html=True)
        st.caption("Red '1' = Flipped Bit")

with tab2:
    st.write("Upload a CSV with columns: `a`, `b`, `opcode`, `faulty_result`")
    
    file = st.file_uploader("Choose CSV", type="csv")
    
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head(3))
        
        if st.button("Run Batch Analysis"):
            results = []
            bar = st.progress(0)
            
            for i, row in df.iterrows():
                p_type, p_conf, _, _ = run_prediction(
                    int(row['a']), int(row['b']), int(row['opcode']), int(row['faulty_result'])
                )
                results.append(p_type)
                bar.progress((i+1)/len(df))
                
            df['AI_Prediction'] = results
            
            # Summary
            st.write("### Summary")
            counts = df['AI_Prediction'].value_counts()
            st.bar_chart(counts)
            
            st.dataframe(df)
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "analysis.csv", "text/csv")