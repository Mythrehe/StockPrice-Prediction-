import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import plotly.graph_objects as go
from datetime import timedelta
import os

STOCK_CSV = r"C:\Users\Rodger Daniel\Downloads\Stockprice prediction using cnn\data\Microsoft_stock_data.csv"
LABELS_CSV = r"C:\Users\Rodger Daniel\Downloads\Stockprice prediction using cnn\labels.csv"
MODEL_PATH = r"C:\Users\Rodger Daniel\Downloads\Stockprice prediction using cnn\stock_cnn_model.h5"
IMAGES_DIR = r"C:\Users\Rodger Daniel\Downloads\Stockprice prediction using cnn\images-20250930T181822Z-1-001\images\Microsoft_stock_data"

IMG_SIZE = (64, 64)
WINDOW_DAYS = 20


stock_df = pd.read_csv(STOCK_CSV)
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df = stock_df.sort_values('Date').reset_index(drop=True)

labels_df = pd.read_csv(LABELS_CSV)
labels_df['date'] = pd.to_datetime(labels_df['date'])
labels_df = labels_df.sort_values('date').reset_index(drop=True)

model = load_model(MODEL_PATH)


st.title("Stock Trend Predictor")
st.markdown("Predict stock trend (rise/fall/neutral) and visualize stock chart.")


user_date = st.date_input(
    "Select stock date (1999-01-01 to 2026-09-19 for predictions):",
    value=stock_df['Date'].max(),
    min_value=pd.to_datetime("1999-01-01"),
    max_value=pd.to_datetime("2026-09-19")
)

user_date = pd.to_datetime(user_date)
last_stock_date = stock_df['Date'].max()
max_future_date = pd.to_datetime("2026-09-19")  


def get_previous_valid_date(ref_date, df_dates):
    """Get nearest previous available date in df_dates <= ref_date"""
    dates = df_dates[df_dates <= ref_date]
    if len(dates) == 0:
        return None
    return dates.max()

def get_future_reference_date(future_date):
    """Get previous year's same day stock for future prediction"""
    prev_year_date = future_date - pd.DateOffset(years=1)
    available_dates = stock_df['Date']
    
    closest_date = get_previous_valid_date(prev_year_date, available_dates)
    return closest_date

def is_weekend(date):
    return date.weekday() >= 5



# Check if selected date is weekend
if is_weekend(user_date):
    st.warning(f"Stock Market is closed on weekends ({user_date.date()}).")
else:
    # Check if date exists in stock_df
    if user_date <= last_stock_date:
        if user_date not in stock_df['Date'].values:
            st.warning(f"Stock Market closed (holiday) on {user_date.date()}. Showing previous available date.")
            user_date = get_previous_valid_date(user_date, stock_df['Date'])
            st.info(f"Showing data for {user_date.date()}.")
        
        idx = stock_df[stock_df['Date'] == user_date].index[0]
        prev_close = stock_df.loc[idx-1, 'Close'] if idx > 0 else stock_df.loc[idx, 'Close']
        today_close = stock_df.loc[idx, 'Close']
        trend_label = "rise" if today_close > prev_close else "fall" if today_close < prev_close else "neutral"
        st.markdown(f"**Actual Trend on {user_date.date()}: {trend_label}**")
        
        # show chart image
        chart_row = labels_df[labels_df['date'] == user_date]
        if not chart_row.empty:
            chart_path = chart_row['filepath'].values[0]
            if os.path.exists(chart_path):
                st.image(chart_path)
            else:
            
                chart_window = stock_df.loc[max(0, idx-WINDOW_DAYS):idx, 'Close']
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=chart_window.values, mode='lines+markers',
                                         line=dict(color='grey', width=2), marker=dict(size=4)))
                fig.update_layout(title=f"Stock Trend for {user_date.date()}",
                                  xaxis_title="Day", yaxis_title="Close Price", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            chart_window = stock_df.loc[max(0, idx-WINDOW_DAYS):idx, 'Close']
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=chart_window.values, mode='lines+markers',
                                     line=dict(color='grey', width=2), marker=dict(size=4)))
            fig.update_layout(title=f"Stock Trend for {user_date.date()}",
                              xaxis_title="Day", yaxis_title="Close Price", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # Future prediction
    else:
        st.info(f"Predicting future trend for {user_date.date()}")

        # If future date is weekend
        if is_weekend(user_date):
            st.warning(f"Stock Market is closed on weekends ({user_date.date()}).")
        else:
            ref_date = get_future_reference_date(user_date)
            if ref_date is None:
                st.error("Cannot find reference stock data for prediction.")
            else:
                idx = stock_df[stock_df['Date'] == ref_date].index[0]
                prev_close = stock_df.loc[idx-1, 'Close'] if idx > 0 else stock_df.loc[idx, 'Close']
                today_close = stock_df.loc[idx, 'Close']
                trend_label = "rise" if today_close > prev_close else "fall" if today_close < prev_close else "neutral"
                st.markdown(f"**Predicted Trend on {user_date.date()}: {trend_label}**")

            
                chart_row = labels_df[labels_df['date'] == ref_date]
                if not chart_row.empty:
                    chart_path = chart_row['filepath'].values[0]
                    if os.path.exists(chart_path):
                        st.image(chart_path)
                    else:
                        chart_window = stock_df.loc[max(0, idx-WINDOW_DAYS):idx, 'Close']
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=chart_window.values, mode='lines+markers',
                                                 line=dict(color='grey', width=2), marker=dict(size=4)))
                        fig.update_layout(title=f"Stock Trend for {user_date.date()} (based on {ref_date.date()})",
                                          xaxis_title="Day", yaxis_title="Close Price", template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    chart_window = stock_df.loc[max(0, idx-WINDOW_DAYS):idx, 'Close']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=chart_window.values, mode='lines+markers',
                                             line=dict(color='grey', width=2), marker=dict(size=4)))
                    fig.update_layout(title=f"Stock Trend for {user_date.date()} (based on {ref_date.date()})",
                                      xaxis_title="Day", yaxis_title="Close Price", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)














