###############################################################################
# Course of Financial Programming - Professor Minh Trung Hoai PHAN

# FINANCIAL DASHBOARD  - Arajem ABOUDI
# Your All-in-One Toolkit for Smarter Investing

###############################################################################


###############################################################################
# Source:   fin_dashboard_revised.py, https://youtu.be/p2pXpcXPoGk?si=jNwo_5XMAbXaXbaW, https://youtu.be/Yk-unX4KnV4?si=nFPFCVqSNtH9dBI8, 
#           https://matplotlib.org/stable/users/explain/colors/colormaps.html, https://stock-dashboard-sp500.streamlit.app/,
#           https://youtu.be/QdPCaHTnpnc?si=DC5Ps3rtztS6lyRh, Yahoo Finance 
###############################################################################
            
# Libraries
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
@st.cache_data  # Cache the result to improve performance
def load_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text  # Get the HTML content of the page
    sp500 = pd.read_html(html)[0]  # Parse the first table containing S&P 500 data
    return sp500['Symbol'].tolist(), sp500[['Symbol', 'Security']]

# Load S&P 500 symbols and associated company names
symbols, company_data = load_sp500_symbols()

# Define dashboard Sidebar and layout
st.sidebar.title("📈 Your All-in-One Toolkit for Smarter Investing 📉")
st.sidebar.subheader("Make your selection")

# Dropdown to select a stock symbol
stock_symbol = st.sidebar.selectbox("Select a stock", symbols)

# Define commonly used date ranges for stock data
date_ranges = {
    "1M": timedelta(days=30),
    "3M": timedelta(days=90),
    "6M": timedelta(days=180),
    "YTD": timedelta(days=(datetime.now() - datetime(datetime.now().year, 1, 1)).days),
    "1Y": timedelta(days=365),
    "3Y": timedelta(days=3 * 365),
    "5Y": timedelta(days=5 * 365)
}

# Get start and end dates based on the user's selection
date_range = st.sidebar.selectbox("Select Date Range", list(date_ranges.keys()))
start_date = datetime.now() - date_ranges[date_range] if date_ranges[date_range] else None
end_date = datetime.now()

# Display the selected stock's name and date range in the sidebar
company_name = company_data[company_data['Symbol'] == stock_symbol]['Security'].values[0]
st.sidebar.write(f"**Selected Company:** {company_name}")
st.sidebar.write(f"**Selected Date Range:** {date_range}")

# Button to fetch and update stock data
if st.sidebar.button("Update Data"):
    stock = yf.Ticker(stock_symbol)  # Fetch the stock data
    data = stock.history(start=start_date, end=end_date)  # Retrieve historical data
    st.write("Data has been updated.")  # Inform the user

    # Enable the user to download the updated data
    csv_data = data.to_csv().encode('utf-8')
    st.sidebar.download_button(
        label="Download Updated Data as CSV",
        data=csv_data,
        file_name=f"{stock_symbol}_data_{date_range}.csv",
        mime="text/csv"
    )

# Add a sidebar image for visual appeal
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/8/8f/Yahoo%21_Finance_logo_2021.png",
    caption="Source: Yahoo Finance",
    width=175
)
st.sidebar.write("**App made by Arajem ABOUDI - MBD 2024/25**")

# Load stock data for the selected stock symbol
stock = yf.Ticker(stock_symbol)

# Create tabs for different sections of the dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Chart", "Financials", "Monte Carlo Simulation", "Portfolio Management"])

# Summary Tab
with tab1:
    st.subheader("Stock Summary")
    info = stock.info if stock else {}  # Fetch stock information
    shareholders = stock.major_holders if stock else None
    col1, col2 = st.columns(2)  # Two-column layout

    with col1:
        # Display company details with fallback for missing information
        st.write(f"**Company:** {info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        market_cap = info.get('marketCap', None)
        if market_cap:
            st.write(f"**Market Cap:** {market_cap:,}")
        else:
            st.write("**Market Cap:** N/A")

    with col2:
        # Display major shareholders information
        st.write("**Major Shareholders**")
        st.write(shareholders)

    # Display a truncated business summary
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

    # User input: Time interval selection for the stock data
    interval = st.selectbox("Select Time Interval", ["1d", "1mo"], index=0)

    # User input: Chart type selection - Line or Candlestick
    chart_type = st.selectbox("Select Chart Type", ["Line", "Candlestick"], index=0)

    # Fetching stock data for the selected time range and interval
    data = stock.history(start=start_date, end=end_date, interval=interval)

    if data.empty:
        # Error message if no data is available
        st.error("No data available for the selected date range and interval.")
    else:
        # Calculate the 50-Day Simple Moving Average (SMA) for daily or monthly intervals
        if interval in ["1d", "1mo"]:
            data["SMA_50"] = data["Close"].rolling(window=50).mean()

        # Initialize the Plotly figure
        fig = go.Figure()
    
    # Plot the selected chart type
    if chart_type == "Line":
        # Add line trace for closing price with a light blue color
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'], mode='lines', name='Close Price', 
            line=dict(color='lightblue', width=2)
        ))
    else:
        # Add candlestick trace with green for increasing and red for decreasing prices
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Candlestick",
            increasing_line_color='green', decreasing_line_color='red'
        ))

    # Add 50-Day SMA line for daily or monthly intervals
    if interval in ["1d", "1mo"]:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["SMA_50"], mode="lines", name="50-Day SMA", 
            line=dict(color='purple', width=1.5)
        ))

    # Add volume bar chart with a dark cyan color and transparency
    fig.add_trace(go.Bar(
        x=data.index, y=data['Volume'], name='Volume', 
        marker=dict(color='rgba(0, 139, 139)'), opacity=0.3, yaxis="y2"
    ))

    # Customize the layout of the chart
    fig.update_layout(
        height=600,  # Set the height of the chart
        yaxis=dict(title="Price", showgrid=True),  # Primary y-axis for price
        yaxis2=dict(  # Secondary y-axis for volume
            title="Volume", overlaying="y", side="right", showgrid=False, 
            range=[0, data['Volume'].max() * 4]
        ),
        xaxis=dict(title="Date", showgrid=True),  # x-axis for dates
        title=f"{stock_symbol} Price Chart ({date_range} - Interval: {interval})"  # Chart title
    )
    
    st.plotly_chart(fig)

# Financials tab
with tab3:
    st.subheader("Financial Statements")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # User input: Selection of financial statement type
    with col1:
        statement_type = st.selectbox("Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"])
    
    # User input: Selection of reporting period (Annual or Quarterly)
    with col2:
        period = st.selectbox("Period", ["Annual", "Quarterly"])
    
    # Display the selected financial statement based on the type and period
    if statement_type == "Income Statement":
        # Show income statement (annual or quarterly based on user selection)
        st.write(stock.financials if period == "Annual" else stock.quarterly_financials)
        # Show balance sheet (annual or quarterly based on user selection)
    elif statement_type == "Balance Sheet":
        st.write(stock.balance_sheet if period == "Annual" else stock.quarterly_balance_sheet)
    else:
        # Show cash flow statement (annual or quarterly based on user selection)
        st.write(stock.cashflow if period == "Annual" else stock.quarterly_cashflow)



# Monte Carlo Simulation tab
with tab4:
    st.subheader("Monte Carlo Simulation")
    
    # Check if there is sufficient data for simulation
    if data.empty:
        st.error("No data available for Monte Carlo simulation.")
    else:
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Number of simulations in the first column
        with col1:
            n_simulations = st.selectbox("Number of Simulations", [200, 500, 1000] # Options for simulations
                                        )
        
        # Time horizon for simulation in the second column
        with col2:
            time_horizon = st.selectbox("Time Horizon (days)", [30, 60, 90] # Options for time horizon
                                       )
        
        # Calculate daily returns and their statistics
        daily_returns = data['Close'].pct_change().dropna()  # Compute daily percentage change
        mean_return = daily_returns.mean()  # Mean of daily returns
        std_dev = daily_returns.std()  # Standard deviation of daily returns
        
        #Initialize an array to store simulated prices
        simulations = np.zeros((time_horizon, n_simulations))
        last_price = data['Close'][-1]  # Get the last closing price as the starting price

        # Perform Monte Carlo simulation
        for i in range(n_simulations):  # Loop through each simulation
            price = last_price  # Start with the last known price
            for t in range(time_horizon):  # Simulate prices for the given time horizon
                price *= (1 + np.random.normal(mean_return, std_dev))  # Apply random returns
                simulations[t, i] = price  # Store simulated price

        # Calculate Value at Risk (VaR) at 95% confidence level
        VaR_95 = np.percentile(simulations[-1], 5)  # 5th percentile of the final simulated prices

        # Plot the Monte Carlo simulation results
        plt.figure(figsize=(10, 6))  # Set figure size
        
        # Use "Magma" colormap for better visualization
        cmap = plt.get_cmap("magma")
        for i in range(n_simulations):  # Loop through each simulation to plot
            plt.plot(simulations[:, i], color=cmap(i / n_simulations))  # Apply colormap

        # Highlight the current stock price with a horizontal line
        current_price_line = plt.axhline(
            y=last_price, 
            color='darkgrey', 
            linewidth=2.5, 
            linestyle='--', 
            label=f'Current stock price: ${last_price:.2f}'
        )


        plt.title(f"{n_simulations} Monte Carlo Simulations for {stock_symbol} over {time_horizon} Days")
        plt.legend([current_price_line], [f'Current stock price: ${last_price:.2f}'])
        plt.xlabel("Day")
        plt.ylabel("Price")
        st.pyplot(plt)

        st.write(f"Value at Risk (VaR) at 95% confidence interval: **${VaR_95:.2f}**")

# Portfolio Management Tab
with tab5:
    st.subheader("Portfolio Management")

# Paragraph description
    st.write("""
The tab in this dashboard allows the user to create and monitor a custom stock portfolio. 
First, you can choose specific sectors and stocks, tailoring your investments to industries you believe in. 


With adjustable sliders, you set the percentage weight for each stock, giving you control over the balance of your portfolio. 
Once configured, you will see a visual breakdown of your portfolio allocation with a pie chart, as well as a performance graph 
that tracks the portfolio’s cumulative return over time. 

This tool provides a clear and interactive way to explore how your 
selected investments perform together, helping the user to make informed investment decisions.
""")  

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

    
