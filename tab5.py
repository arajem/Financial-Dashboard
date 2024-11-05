# Define dashboard layout
st.sidebar.title("📈 Arajem Aboudi - Financial Dashboard 📉")

st.sidebar.subheader("Make your selection")
stock_symbol = st.sidebar.selectbox("Select a stock", symbols)
update_button = st.sidebar.button("Update Data")

# Display selected stock name
company_name = company_data[company_data['Symbol'] == stock_symbol]['Security'].values[0]
st.sidebar.write(f"**Selected Company:** {company_name}")

# Options for date range
date_ranges = {
    "1M": timedelta(days=30),
    "3M": timedelta(days=90),
    "6M": timedelta(days=180),
    "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
    "1Y": timedelta(days=365),
    "3Y": timedelta(days=3 * 365),
    "5Y": timedelta(days=5 * 365)
}
date_range = st.sidebar.selectbox("Select Date Range", list(date_ranges.keys()))
start_date = datetime.now() - date_ranges[date_range] if date_range != "MAX" else None
end_date = datetime.now()

# Display the selected date range in the sidebar
st.sidebar.write(f"**Selected Date Range:** {date_range}")

# Load stock data for the selected symbol
stock = yf.Ticker(stock_symbol)

# Create separate tabs for each section
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Summary", "Chart", "Financials", "Monte Carlo Simulation", "My Own Analysis", "Peer Comparison"])

# Summary tab
with tab1:
    st.subheader("Stock Summary")
    info = stock.info
    shareholders = stock.major_holders
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Company:** {info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}")

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

# Chart tab
with tab2:
    st.subheader("Stock Price Chart")
    interval = st.selectbox("Select Time Interval", ["1d", "1mo", "1y"], index=0)
    chart_type = st.selectbox("Select Chart Type", ["Line", "Candlestick"], index=0)
    data = stock.history(start=start_date, end=end_date, interval=interval)

    if interval == "1d":
        data["SMA_50"] = data["Close"].rolling(window=50).mean()

    fig = go.Figure()
    if chart_type == "Line":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    else:
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Candlestick"
        ))

    if interval == "1d":
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], mode="lines", name="50-Day SMA", line=dict(color='orange', width=1.5)))

    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker=dict(color='blue'), opacity=0.3, yaxis="y2"))
    fig.update_layout(
        height=600, yaxis=dict(title="Price", showgrid=True),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, range=[0, data['Volume'].max()*4]),
        xaxis=dict(title="Date", showgrid=True), title=f"{stock_symbol} Price Chart ({date_range} - Interval: {interval})"
    )
    st.plotly_chart(fig)

# Financials tab
with tab3:
    st.subheader("Financial Statements")
    statement_type = st.selectbox("Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"])
    period = st.selectbox("Period", ["Annual", "Quarterly"])
    if statement_type == "Income Statement":
        st.write(stock.financials if period == "Annual" else stock.quarterly_financials)
    elif statement_type == "Balance Sheet":
        st.write(stock.balance_sheet if period == "Annual" else stock.quarterly_balance_sheet)
    else:
        st.write(stock.cashflow if period == "Annual" else stock.quarterly_cashflow)

# Monte Carlo Simulation tab
with tab4:
    st.subheader("Monte Carlo Simulation for Future Stock Prices")
    n_simulations = st.selectbox("Number of Simulations", [200, 500, 1000])
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
    st.write(f" Value at Risk (VaR) at 95% confidence interval: ${VaR_95:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(simulations)
    plt.title(f"{n_simulations} Monte Carlo Simulations for {stock_symbol} over {time_horizon} Days")
    plt.xlabel("Days")
    plt.ylabel("Price")
    st.pyplot(plt)

# Your Own Analysis tab
with tab5:
    st.subheader("Your Own Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Key Financial Metrics")
        metrics = {
            "Revenue": stock.financials.loc['Total Revenue'].sum(),
            "Net Income": stock.financials.loc['Net Income'].sum(),
            "EPS": stock.info.get('trailingEps', 'N/A'),
            "P/E Ratio": stock.info.get('trailingPE', 'N/A'),
            "Debt-to-Equity Ratio": stock.info.get('debtToEquity', 'N/A'),
            "Return on Equity (ROE)": f"{stock.info.get('returnOnEquity', 'N/A') * 100:.2f}%",
            "Dividend Yield": f"{stock.info.get('dividendYield', 'N/A') * 100:.2f}%"
        }
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        st.table(metrics_df)

    with col2:
        st.write("### Stock Performance")
        performance_data = {
            "1-Year Price Change (%)": ((data['Close'][-1] - data['Close'][0]) / data['Close'][0]) * 100,
            "52-Week High": data['Close'].max(),
            "52-Week Low": data['Close'].min(),
            "Average Trading Volume": data['Volume'].mean(),
            "Beta": stock.info.get('beta', 'N/A')
        }
        performance_df = pd.DataFrame(list(performance_data.items()), columns=['Metric', 'Value'])
        st.table(performance_df)

    st.write("### Stock Performance Over Time")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price', fill='tozeroy', fillcolor='#ADD8E6'))
    fig_line.update_layout(
        title=f"{stock_symbol} - 1 Year Stock Performance",
        xaxis_title="Date",
        yaxis_title="Closing Price (USD)",
        template="plotly_white"
    )
    st.plotly_chart(fig_line)

# Peer Comparison Tab
with tab6:
    st.subheader("Peer Comparison")
    peer_symbols = company_data["Symbol"].tolist()  # Get a list of symbols without sector filtering
    peer_symbols.remove(stock_symbol)  # Exclude selected stock

    # Display peer metrics
    peer_metrics = []
    for symbol in peer_symbols[:5]:  # Limit to top 5 peers
        peer_stock = yf.Ticker(symbol)
        peer_info = peer_stock.info
        peer_metrics.append({
            "Symbol": symbol,
            "P/E Ratio": peer_info.get("trailingPE", "N/A"),
            "Market Cap": peer_info.get("marketCap", "N/A"),
            "Revenue": peer_info.get("totalRevenue", "N/A")
        })
    st.write(pd.DataFrame(peer_metrics))
