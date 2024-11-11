import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import datetime, timedelta
import requests
import matplotlib.dates as mdates
from matplotlib import colormaps



# Fetch the S&P 500 stock symbols from Wikipedia
@st.cache_data  # Cache to avoid re-downloading data
def load_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    sp500 = pd.read_html(html)[0]  # Wikipedia table is the first in the list
    return sp500['Symbol'].tolist(), sp500[['Symbol', 'Security']]

# Load S&P 500 symbols and names
symbols, company_data = load_sp500_symbols()


# Define dashboard layout
st.sidebar.title("📈 Arajem Aboudi - Financial Dashboard 📉")

st.sidebar.subheader("Make your selection")
stock_symbol = st.sidebar.selectbox("Select a stock", symbols)

# Options for date range
date_ranges = {
    "1M": timedelta(days=30),
    "3M": timedelta(days=90),
    "6M": timedelta(days=180),
    "YTD": timedelta(days=(datetime.now() - datetime(datetime.now().year, 1, 1)).days),
    "1Y": timedelta(days=365),
    "3Y": timedelta(days=3 * 365),
    "5Y": timedelta(days=5 * 365) 
}
date_range = st.sidebar.selectbox("Select Date Range", list(date_ranges.keys()))
start_date = datetime.now() - date_ranges[date_range] if date_ranges[date_range] else None
end_date = datetime.now()

# Display selected stock name
company_name = company_data[company_data['Symbol'] == stock_symbol]['Security'].values[0]
st.sidebar.write(f"**Selected Company:** {company_name}")

# Display the selected date range in the sidebar
st.sidebar.write(f"**Selected Date Range:** {date_range}")

# Button to fetch and update data
if st.sidebar.button("Update Data"):
    # Load stock data
    stock = yf.Ticker(stock_symbol)
    data = stock.history(start=start_date, end=end_date)
    st.write("Data has been updated.")

    # Store the CSV data in a variable
    csv_data = data.to_csv().encode('utf-8')

    # Display the download button only after data is fetched
    st.sidebar.download_button(
        label="Download Updated Data as CSV",
        data=csv_data,
        file_name=f"{stock_symbol}_data_{date_range}.csv",
        mime="text/csv"
    )


# Load stock data for the selected symbol
stock = yf.Ticker(stock_symbol)

# Create separate tabs for each section
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Chart", "Financials", "Monte Carlo Simulation", "Portfolio Management"])

# Summary tab
with tab1:
    st.subheader("Stock Summary")
    
    # Attempt to retrieve stock info
    info = stock.info if stock else {}
    shareholders = stock.major_holders if stock else None
    col1, col2 = st.columns(2)

    with col1:
        # Safely retrieve and display each field with fallback for missing data
        st.write(f"**Company:** {info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")

        # Handle market cap safely
        market_cap = info.get('marketCap', None)
        if market_cap:
            st.write(f"**Market Cap:** {market_cap:,}")
        else:
            st.write("**Market Cap:** N/A")

    with col2:
        st.write("**Major Shareholders**")
        st.write(shareholders)

    summary = info.get('longBusinessSummary', 'N/A')
    summary_length = 300
    if len(summary) > summary_length:
        short_summary = summary[:summary_length] + "..."
        st.write(f"**Summary:** {short_summary}")
        if st.button("Read more about the company"):
            st.write(f"**Full Summary:** {summary}")
    else:
        st.write(f"**Summary:** {summary}")

# Define the date range and fetch data accordingly
if date_range == "MAX":
    start_date = None  # "MAX" range - use entire available data history
else:
    start_date = datetime.now() - date_ranges[date_range]

end_date = datetime.now()

# Chart tab
with tab2:
    st.subheader("Stock Price Chart")
    interval = st.selectbox("Select Time Interval", ["1d", "1mo"], index=0)
    chart_type = st.selectbox("Select Chart Type", ["Line", "Candlestick"], index=0)
    data = stock.history(start=start_date, end=end_date, interval=interval)

    if data.empty:
        st.error("No data available for the selected date range and interval.")
    else:
        if interval == "1d":
            data["SMA_50"] = data["Close"].rolling(window=50).mean()

        fig = go.Figure()
    
    if chart_type == "Line":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='lightblue', width=2)))  # Line color changed to blue
    else:
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Candlestick",
            increasing_line_color='green', decreasing_line_color='red'  # Changed colors for candlestick
        ))

    if interval == "1d":
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], mode="lines", name="50-Day SMA", line=dict(color='purple', width=1.5)))  # SMA color changed to orange

    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker=dict(color='rgba(0, 139, 139)'), opacity=0.3, yaxis="y2"))  # Volume bar color changed

    fig.update_layout(
        height=600, yaxis=dict(title="Price", showgrid=True),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, range=[0, data['Volume'].max()*4]),
        xaxis=dict(title="Date", showgrid=True), title=f"{stock_symbol} Price Chart ({date_range} - Interval: {interval})"
    )
    st.plotly_chart(fig)
    
# Financials tab
with tab3:
    st.subheader("Financial Statements")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Statement type selection in the first column
    with col1:
        statement_type = st.selectbox("Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"])
    
    # Period selection in the second column
    with col2:
        period = st.selectbox("Period", ["Annual", "Quarterly"])
    
    # Display the selected financial statement based on the type and period
    if statement_type == "Income Statement":
        st.write(stock.financials if period == "Annual" else stock.quarterly_financials)
    elif statement_type == "Balance Sheet":
        st.write(stock.balance_sheet if period == "Annual" else stock.quarterly_balance_sheet)
    else:
        st.write(stock.cashflow if period == "Annual" else stock.quarterly_cashflow)



# Monte Carlo Simulation tab
with tab4:
    st.subheader("Monte Carlo Simulation for Future Stock Prices")

    if data.empty:
        st.error("No data available for Monte Carlo simulation.")
    else:
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Number of simulations in the first column
        with col1:
            n_simulations = st.selectbox("Number of Simulations", [200, 500, 1000])
        
        # Time horizon in the second column
        with col2:
            time_horizon = st.selectbox("Time Horizon (days)", [30, 60, 90])
        
        daily_returns = data['Close'].pct_change().dropna()
        mean_return = daily_returns.mean()
        std_dev = daily_returns.std()
        simulations = np.zeros((time_horizon, n_simulations))
        last_price = data['Close'][-1]

        for i in range(n_simulations):
            price = last_price
            for t in range(time_horizon):
                price *= (1 + np.random.normal(mean_return, std_dev))
                simulations[t, i] = price

        VaR_95 = np.percentile(simulations[-1], 5)
        

        # Plot the simulations using the "Magma" colormap
        plt.figure(figsize=(10, 6))
        
        # Use the "Magma" colormap for the simulations
        cmap = plt.get_cmap("magma")
        for i in range(n_simulations):
            plt.plot(simulations[:, i], color=cmap(i / n_simulations))  # Apply colormap

        # Use a light grey horizontal line instead of purple
        current_price_line = plt.axhline(y=last_price, color='darkgrey', linewidth=2.5)

        plt.title(f"{n_simulations} Monte Carlo Simulations for {stock_symbol} over {time_horizon} Days")
        plt.legend([current_price_line], [f'Current stock price: ${last_price:.2f}'])
        plt.xlabel("Day")
        plt.ylabel("Price")
        st.pyplot(plt)

        st.write(f"Value at Risk (VaR) at 95% confidence interval: **${VaR_95:.2f}**")

# Portfolio Management Tab
with tab5:
    st.subheader("Portfolio Management")

    # Fetch S&P 500 companies and their sectors from Wikipedia
    @st.cache_data
    def get_sp500_data():
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)  # Read all tables from the Wikipedia page
        sp500_df = tables[0]  # The first table contains the necessary data
        sp500_df = sp500_df[['Symbol', 'GICS Sector']]  # Keep only symbol and sector columns
        sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-', regex=False)  # Clean symbols if needed
        return sp500_df

    # Get the S&P 500 dataset
    sp500_data = get_sp500_data()

    # List of sectors for the dropdown menu
    sectors = sp500_data['GICS Sector'].unique()

    # Add a multi-select menu to select multiple sectors
    selected_sectors = st.multiselect("Select Sectors", sectors, default=sectors[:1])  # Default to first sector

    if not selected_sectors:
        st.warning("Please select at least one sector.")
    else:
        # Filter the symbols based on the selected sectors
        filtered_sp500 = sp500_data[sp500_data['GICS Sector'].isin(selected_sectors)]
        sector_symbols = filtered_sp500['Symbol'].tolist()

        # Select multiple stocks for portfolio
        selected_symbols = st.multiselect("Select Stocks for Portfolio", sector_symbols, default=[sector_symbols[0]])

        # Check if at least one stock is selected
        if not selected_symbols:
            st.warning("Please select at least one stock for your portfolio.")
        else:
            # Initialize equal weights
            weights = np.array([1 / len(selected_symbols)] * len(selected_symbols))

            # Allow user to adjust weights with sum-to-1 constraint
            adjusted_weights = []
            for i, symbol in enumerate(selected_symbols):
                adjusted_weight = st.slider(f"Weight for {symbol} (%)", 0.0, 100.0, weights[i] * 100.0) / 100
                adjusted_weights.append(adjusted_weight)

            # Normalize weights to sum to 1
            weights = np.array(adjusted_weights)
            weights /= weights.sum()

            # Function to fetch portfolio data with caching
            @st.cache_data
            def load_portfolio_data(tickers):
                return yf.download(tickers, start="2023-01-01")["Close"]

            # Fetch historical data and calculate portfolio return
            portfolio_data = load_portfolio_data(selected_symbols)
            daily_returns = portfolio_data.pct_change().dropna()

            if daily_returns.empty:
                st.error("No data available for the selected stocks.")
            else:
                # Calculate annualized return and volatility
                portfolio_return = np.dot(daily_returns.mean() * 252, weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility else 0

                # Display the two charts side by side with different sizes
                col1, col2 = st.columns([1, 2])  # 1:2 width ratio

                # Create a color map based on sectors
                sector_colors = {
                    sector: f"rgb({np.random.randint(100, 255)}, {np.random.randint(100, 255)}, {np.random.randint(100, 255)})"
                    for sector in selected_sectors
                }

                # Assign a color to each stock based on its sector
                stock_colors = [sector_colors[sp500_data.loc[sp500_data['Symbol'] == symbol, 'GICS Sector'].values[0]]
                                for symbol in selected_symbols]

                with col1:
                    st.write("### Portfolio Allocation")
                    # Create a pie chart with colors for sectors
                    allocation_chart = go.Figure(go.Pie(
                        labels=selected_symbols, 
                        values=weights * 100, 
                        hole=0.3,
                        marker=dict(colors=stock_colors)  # Apply sector-based colors
                    ))
                    st.plotly_chart(allocation_chart)

                with col2:
                    st.write("### Portfolio Performance Over Time")
                    cumulative_return = (1 + daily_returns.dot(weights)).cumprod() - 1
                    fig_performance = go.Figure()

                    # Plot each stock's performance with a different color
                    for i, symbol in enumerate(selected_symbols):
                        fig_performance.add_trace(go.Scatter(
                            x=cumulative_return.index,
                            y=(1 + daily_returns[symbol] * weights[i]).cumprod() - 1,
                            mode='lines',
                            name=symbol,
                            line=dict(color=stock_colors[i])  # Use the sector color for each stock
                        ))

                    fig_performance.update_layout(
                        title="Portfolio Cumulative Return Over Time", 
                        xaxis_title="Date", 
                        yaxis_title="Cumulative Return"
                    )
                    st.plotly_chart(fig_performance)
            
            # Display portfolio metrics
            st.write(f"Expected Annual Return: {portfolio_return * 100:.2f}%")
            st.write(f"Portfolio Volatility: {portfolio_volatility * 100:.2f}%")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
