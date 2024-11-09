#Libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import datetime, timedelta
import requests
import matplotlib.dates as mdates


# Fetch the S&P 500 stock symbols from Wikipedia
@st.cache_data  # Cache to avoid re-downloading data
def load_sp500_symbols():
    # Retrieve S&P 500 stock list from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    sp500 = pd.read_html(html)[0]  # Wikipedia table is the first in the list
    return sp500['Symbol'].tolist(), sp500[['Symbol', 'Security']]

# Load S&P 500 symbols and names
symbols, company_data = load_sp500_symbols()

# Define dashboard layout
st.sidebar.title("📈 Arajem Aboudi - Financial Dashboard 📉")
st.sidebar.subheader("Make your selection")

# Stock selection dropdown in the sidebar
stock_symbol = st.sidebar.selectbox("Select a stock", symbols)

# Date range options for selecting stock history period
date_ranges = {
    "1M": timedelta(days=30),
    "3M": timedelta(days=90),
    "6M": timedelta(days=180),
    "YTD": timedelta(days=(datetime.now() - datetime(datetime.now().year, 1, 1)).days),
    "1Y": timedelta(days=365),
    "3Y": timedelta(days=3 * 365),
    "5Y": timedelta(days=5 * 365) 
}
# Sidebar input for date range selection
date_range = st.sidebar.selectbox("Select Date Range", list(date_ranges.keys()))
start_date = datetime.now() - date_ranges[date_range] if date_ranges[date_range] else None
end_date = datetime.now()

# Display selected company name based on symbol
company_name = company_data[company_data['Symbol'] == stock_symbol]['Security'].values[0]
st.sidebar.write(f"**Selected Company:** {company_name}")

# Display selected date range in the sidebar
st.sidebar.write(f"**Selected Date Range:** {date_range}")

# Button to fetch and update stock data for selected symbol and date range
if st.sidebar.button("Update Data"):
    # Load stock data from Yahoo Finance
    stock = yf.Ticker(stock_symbol)
    data = stock.history(start=start_date, end=end_date)
    st.write("Data has been updated.")
    
    # Store the CSV data for download
    csv_data = data.to_csv().encode('utf-8')
    
    # Display download button for CSV after data is fetched
    st.sidebar.download_button(
        label="Download Updated Data as CSV",
        data=csv_data,
        file_name=f"{stock_symbol}_data_{date_range}.csv",
        mime="text/csv"
    )

# Initialize stock data for selected symbol
stock = yf.Ticker(stock_symbol)

# Create tabs for dashboard sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Summary", "Chart", "Financials", "Monte Carlo Simulation", "Brief Analysis", "Portfolio Management"])

# Summary Tab: Display company and shareholder information
with tab1:
    st.subheader("Stock Summary")
    info = stock.info
    shareholders = stock.major_holders
    col1, col2 = st.columns(2)
    
    with col1:
        # Display key company details
        st.write(f"**Company:** {info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}")
    
    with col2:
        # Display major shareholders
        st.write("**Major Shareholders**")
        st.write(shareholders)

    # Display a short company summary with option to expand for full description
    summary = info.get('longBusinessSummary', 'N/A')
    summary_length = 300
    if len(summary) > summary_length:
        short_summary = summary[:summary_length] + "..."
        st.write(f"**Summary:** {short_summary}")
        if st.button("Read more about the company"):
            st.write(f"**Full Summary:** {summary}")
    else:
        st.write(f"**Summary:** {summary}")

# Chart Tab: Display stock price chart
with tab2:
    st.subheader("Stock Price Chart")
    # Options for chart interval and type (Line or Candlestick)
    interval = st.selectbox("Select Time Interval", ["1d", "1mo"], index=0)
    chart_type = st.selectbox("Select Chart Type", ["Line", "Candlestick"], index=0)
    
    # Fetch historical data for selected date range and interval
    data = stock.history(start=start_date, end=end_date, interval=interval)
    if data.empty:
        st.error("No data available for the selected date range and interval.")
    else:
        # Calculate 50-Day SMA for daily interval
        if interval == "1d":
            data["SMA_50"] = data["Close"].rolling(window=50).mean()
        
        fig = go.Figure()
        
        # Plot line or candlestick chart based on user selection
        if chart_type == "Line":
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='lightblue', width=2)))
        else:
            fig.add_trace(go.Candlestick(
                x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Candlestick",
                increasing_line_color='green', decreasing_line_color='red'
            ))
        
        # Plot 50-day SMA and volume bars
        if interval == "1d":
            fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], mode="lines", name="50-Day SMA", line=dict(color='purple', width=1.5)))
        
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker=dict(color='rgba(0, 139, 139)'), opacity=0.3, yaxis="y2"))
        
        # Chart layout configuration
        fig.update_layout(
            height=600, yaxis=dict(title="Price", showgrid=True),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, range=[0, data['Volume'].max()*4]),
            xaxis=dict(title="Date", showgrid=True), title=f"{stock_symbol} Price Chart ({date_range} - Interval: {interval})"
        )
        st.plotly_chart(fig)

# Financials Tab: Display company financial statements
with tab3:
    st.subheader("Financial Statements")
    # Choose statement type (Income, Balance, Cash Flow) and period (Annual or Quarterly)
    statement_type = st.selectbox("Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"])
    period = st.selectbox("Period", ["Annual", "Quarterly"])
    if statement_type == "Income Statement":
        st.write(stock.financials if period == "Annual" else stock.quarterly_financials)
    elif statement_type == "Balance Sheet":
        st.write(stock.balance_sheet if period == "Annual" else stock.quarterly_balance_sheet)
    else:
        st.write(stock.cashflow if period == "Annual" else stock.quarterly_cashflow)

# Monte Carlo Simulation Tab: Forecast future stock prices
with tab4:
    st.subheader("Monte Carlo Simulation for Future Stock Prices") 
    if data.empty:
        st.error("No data available for Monte Carlo simulation.")
    else:
        # User-defined parameters for the number of simulations and time horizon
        n_simulations = st.selectbox("Number of Simulations", [200, 500, 1000])
        time_horizon = st.selectbox("Time Horizon (days)", [30, 60, 90])
        
        # Calculate daily returns, mean, and standard deviation
        daily_returns = data['Close'].pct_change().dropna()
        mean_return = daily_returns.mean()
        std_dev = daily_returns.std()
        
        # Initialize and run simulations
        simulations = np.zeros((time_horizon, n_simulations))
        last_price = data['Close'][-1]
        for i in range(n_simulations):
            price = last_price
            for t in range(time_horizon):
                price *= (1 + np.random.normal(mean_return, std_dev))
                simulations[t, i] = price

        # Calculate Value at Risk (VaR) at 95% confidence
        VaR_95 = np.percentile(simulations[-1], 5)
        st.write(f" Value at Risk (VaR) at 95% confidence interval: ${VaR_95:.2f}")
        
        # Plot simulation paths
        plt.figure(figsize=(10, 6))
        plt.plot(simulations)
        current_price_line = plt.axhline(y=last_price, color='blue', linewidth=2)
        plt.title(f"{n_simulations} Monte Carlo Simulations for {stock_symbol} over {time_horizon} Days")
        plt.legend([current_price_line], [f'Current stock price: ${last_price:.2f}'])
        plt.xlabel("Day")

# Brief Analysis Tab: Display basic financial metrics
with tab5:
    st.subheader("Brief Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Beta", stock.info.get("beta", "N/A"))
    col2.metric("P/E Ratio", stock.info.get("trailingPE", "N/A"))
    col3.metric("Earnings per Share (EPS)", stock.info.get("trailingEps", "N/A"))

# Portfolio Management Tab: Allows users to build a custom portfolio
with tab6:
    st.subheader("Portfolio Builder")
    st.write("Select up to 3 stocks and specify their weights to build a portfolio.")
    
    # Allow users to select up to 3 stocks and specify weights
    selected_stocks = [st.selectbox("Stock Symbol", symbols) for _ in range(3)]
    weights = [st.number_input("Weight (%)", min_value=0, max_value=100, value=0) for _ in range(3)]
    
    if sum(weights) != 100:
        st.warning("Total weight should be 100% for a balanced portfolio.")
    else:
        st.success("Portfolio is balanced!")
        
# Users can view general stock information, examine detailed financials, conduct simulations, perform brief analysis, and build portfolios.
