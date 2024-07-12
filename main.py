import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Constants
Start = "2010-01-01"
Today = date.today().strftime("%Y-%m-%d")

st.markdown(
    """
    <style>
    .scrolling-text {
        white-space: nowrap;
        overflow: hidden;
        box-sizing: border-box;
    }

    .scrolling-text div {
        display: inline-block;
        padding-left: 100%;
        animation: scroll-text 20s linear infinite;
    }

    @keyframes scroll-text {
        from {
            transform: translate(0, 0);
        }
        to {
            transform: translate(-100%, 0);
        }
    }
    </style>
    <div class="scrolling-text">
        <div>Disclaimer: The information and analysis provided in this application are for educational and informational purposes only. They should not be construed as financial or investment advice. Always conduct your own research or consult with a financial advisor before making investment decisions.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Streamlit app title
st.title("Stock Prediction Application")

st.text("You can get the symbol of stocks from finance.yahoo.com.")
select_stock = st.text_input("Enter the stock to be predicted:", "AAPL")

n_years = st.slider("Years of prediction:", 1, 5)
period = n_years * 365

# Function to load data with caching
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, Start, Today)
    data.reset_index(inplace=True)
    return data

# Loading data
data_load_state = st.text("Loading data...")
data = load_data(select_stock)
data_load_state.text("Loading data done!")

# Display raw data
st.subheader("Raw data: (From 2010 till now)")
st.write(data)

# Calculate moving averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA100'] = data['Close'].rolling(window=100).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Function to plot raw data with moving averages
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.update_layout(title_text="Visual chart for the data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

def plot_MA_data():
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], name="MA50"))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['MA100'], name="MA100"))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['MA200'], name="MA200"))
    fig1.update_layout(title_text="Market averages chart for the data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)
plot_MA_data()

st.text(
    "Crossover Signals: (For your reference) \n"
    "- Golden Cross: MA50 crosses above MA200, typically a bullish signal.\n"
    "- Death Cross: MA50 crosses below MA200, usually a bearish signal.\n"
    "- MA100 Crosses:\n"
    "  - MA100 above MA200: Indicates strengthening medium-term momentum.\n"
    "  - MA100 below MA200: Indicates weakening medium-term momentum.\n"
    "- MA50 above MA100: Short-term momentum strengthening.\n"
    "- MA50 below MA100: Short-term momentum weakening."
)

# Preparing data for Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Forecasting with Prophet
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Display forecasted data
st.subheader("Forecasted data:")
st.write(forecast.tail())

# Plot forecasted data
st.write("Forecasted data visualization:")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

# Plot forecast components
st.write("Forecast components:")
fig2 = model.plot_components(forecast)
st.write(fig2)
