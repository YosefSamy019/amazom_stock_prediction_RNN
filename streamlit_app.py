import math
import pickle

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model

WINDOW_SIZE = 35
TABLE_WIDTH = 5
MODEL_PATH = r'models_cache/deep_gru.keras'
MIN_MAX_SCALER_PATH = r'encoders_scalers/min-max.pickle'


def pkl_load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_resources():
    min_max_scaler = pkl_load_obj(MIN_MAX_SCALER_PATH)
    model = load_model(MODEL_PATH)


    initial_stock = np.array([[0.76390083],
       [0.7550527 ],
       [0.75555558],
       [0.75370715],
       [0.75445467],
       [0.75321779],
       [0.77008498],
       [0.79559631],
       [0.78893648],
       [0.7710228 ],
       [0.71815155],
       [0.71380223],
       [0.71225285],
       [0.67866804],
       [0.67745837],
       [0.69198777],
       [0.69219168],
       [0.64315325],
       [0.65836223],
       [0.62734625],
       [0.63319061],
       [0.62950734],
       [0.63392461],
       [0.66295619],
       [0.66316009],
       [0.66957528],
       [0.64524633],
       [0.71592251],
       [0.65020728],
       [0.63363915],
       [0.60277266],
       [0.57418962],
       [0.58124365],
       [0.53485561],
       [0.51578663]])
    initial_stock = min_max_scaler.inverse_transform(initial_stock).flatten()

    return min_max_scaler, model, initial_stock


def predict(prediction_length=10):
    min_max_scaler, model, initial_stock = load_resources()

    X = np.array(initial_stock).reshape((WINDOW_SIZE, 1))
    X_scaled = min_max_scaler.transform(X)

    output_predict = [initial_stock[-1]]

    for _ in range(prediction_length):
        y_scaled = model.predict(X_scaled.reshape(1, WINDOW_SIZE, 1), verbose=0)
        y = min_max_scaler.inverse_transform([[y_scaled.flatten()[0]]])[0][0]
        output_predict.append(y)

        y_scaled_clipped = y_scaled.flatten()[0]
        X_scaled = np.append(X_scaled.flatten(), y_scaled_clipped)[1:].reshape(WINDOW_SIZE, 1)

    return output_predict


def main():
    _, _, initial_stock = load_resources()

    st.set_page_config(
        page_title='Amazon Stock Prediction',
        page_icon='📈',
        layout='wide'
    )

    with open('README.md', 'r', encoding='UTF-8') as f:
        st.write(f.read())

    st.divider()

    t = 1
    for row_i in range(math.ceil(WINDOW_SIZE / TABLE_WIDTH)):
        cols = st.columns(TABLE_WIDTH)

        for col_i, col in enumerate(cols):
            if t > WINDOW_SIZE:
                continue
            initial_stock[t - 1] = col.number_input(label=f'Time {t}', value=float(initial_stock[t - 1]))
            t += 1

    prediction_length = st.number_input(label='Prediction Iterations', min_value=1, max_value=30, step=1, value=5)

    output_predict = predict(prediction_length)

    # Prepare x and y for plotting
    x_input = np.arange(len(initial_stock))
    x_pred = np.arange(len(initial_stock) - 1, len(initial_stock) + prediction_length - 1)

    fig = go.Figure()

    fig.add_scatter(
        x=x_input,
        y=initial_stock,
        mode='lines+markers',
        name='Input Stock',
        line=dict(color='#573200')
    )

    fig.add_scatter(
        x=x_pred,
        y=output_predict,
        mode='lines+markers',
        name='Predicted Stock',
        line=dict(color='#006400')  # Dark green
    )

    fig.update_layout(
        title='Close Price vs Time',
        xaxis_title='Time',
        yaxis_title='Close Price',
        plot_bgcolor='#F3F0D9',
        paper_bgcolor='#F3F0D9',
        legend_title_font_color="#573200",
        title_font=dict(size=16, color='#573200'),
        font=dict(size=14, color='#FFFFFF'),
    )

    fig.update_xaxes(
        title_font=dict(size=16, color='#573200', family='Georgia, serif'),
        tickfont=dict(size=14, color='#784600', family='Georgia, serif'),
        showgrid=True,
        gridcolor='lightgray'
    )

    fig.update_yaxes(
        title_font=dict(size=16, color='#573200', family='Georgia, serif'),
        tickfont=dict(size=14, color='#784600', family='Georgia, serif'),
        showgrid=True,
        gridcolor='lightgray'
    )

    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
