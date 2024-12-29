import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from phi.agent.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()
import os

# Get API key from Streamlit secrets
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Enhanced stock symbol mappings
COMMON_STOCKS = {
    # US Stocks
    'NVIDIA': 'NVDA',
    'APPLE': 'AAPL',
    'GOOGLE': 'GOOGL',
    'MICROSOFT': 'MSFT',
    'TESLA': 'TSLA',
    'AMAZON': 'AMZN',
    'META': 'META',
    'NETFLIX': 'NFLX',
    # Indian Stocks - NSE
    'TCS': 'TCS.NS',
    'RELIANCE': 'RELIANCE.NS',
    'INFOSYS': 'INFY.NS',
    'WIPRO': 'WIPRO.NS',
    'HDFC': 'HDFCBANK.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'SBIN': 'SBIN.NS',
    'MARUTI': 'MARUTI.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'HCLTECH': 'HCLTECH.NS',
    'ITC': 'ITC.NS',
    'AXISBANK': 'AXISBANK.NS'
}

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stock-header {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        color: white; /* Change text color to white for better contrast */
        text-align: center;
        padding: 1rem;
        background: linear-gradient(to right, #1f77b4, #004080); /* Stronger gradient for contrast */
        border-radius: 10px;
    }
    .news-card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        transition: transform 0.2s;
    }
    .news-card:hover {
        transform: translateX(5px);
    }
    .stButton>button {
        width: 100%;
    }
    .market-indicator {
        font-size: 16px;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
    st.session_state.watchlist = set()
    st.session_state.analysis_history = []
    st.session_state.last_refresh = None

def initialize_agents():
    """Initialize all agent instances with improved error handling"""
    if not st.session_state.agents_initialized:
        try:
            st.session_state.web_agent = Agent(
                name="Web Search Agent",
                role="Search the web for the information",
                model=Groq(api_key=GROQ_API_KEY),
                tools=[
                    GoogleSearch(fixed_language='english', fixed_max_results=5),
                    DuckDuckGo(fixed_max_results=5)
                ],
                instructions=['Always include sources and verification'],
                show_tool_calls=True,
                markdown=True
            )

            st.session_state.finance_agent = Agent(
                name="Financial AI Agent",
                role="Providing financial insights",
                model=Groq(api_key=GROQ_API_KEY),
                tools=[
                    YFinanceTools(
                        stock_price=True,
                        company_news=True,
                        analyst_recommendations=True,
                        historical_prices=True
                    )
                ],
                instructions=["Provide detailed analysis with data visualization"],
                show_tool_calls=True,
                markdown=True
            )

            st.session_state.multi_ai_agent = Agent(
                name='A Stock Market Agent',
                role='A comprehensive assistant specializing in stock market analysis',
                model=Groq(api_key=GROQ_API_KEY),
                team=[st.session_state.web_agent, st.session_state.finance_agent],
                instructions=["Provide comprehensive analysis with multiple data sources"],
                show_tool_calls=True,
                markdown=True
            )

            st.session_state.agents_initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing agents: {str(e)}")
            return False

def get_symbol_from_name(stock_name):
    """Enhanced function to fetch stock symbol from full stock name"""
    try:
        # Clean up input
        stock_name = stock_name.strip().upper()
        
        # First check if it's in our common stocks dictionary
        if stock_name in COMMON_STOCKS:
            return COMMON_STOCKS[stock_name]
        
        # Check if it's already a valid symbol
        ticker = yf.Ticker(stock_name)
        try:
            info = ticker.info
            if info and 'symbol' in info:
                return stock_name
        except:
            pass
        
        # Try Indian stock market (NSE)
        try:
            indian_symbol = f"{stock_name}.NS"
            ticker = yf.Ticker(indian_symbol)
            info = ticker.info
            if info and 'symbol' in info:
                return indian_symbol
        except:
            # Try BSE
            try:
                bse_symbol = f"{stock_name}.BO"
                ticker = yf.Ticker(bse_symbol)
                info = ticker.info
                if info and 'symbol' in info:
                    return bse_symbol
            except:
                pass
        
        st.error(f"Could not find valid symbol for {stock_name}")
        return None
    except Exception as e:
        st.error(f"Error processing {stock_name}: {str(e)}")
        return None

def get_stock_data(symbol, period="1y"):
    """Enhanced function to fetch stock data with proper cache handling"""
    try:
        # Create a new ticker instance
        stock = yf.Ticker(symbol)
        
        # Fetch data with error handling
        try:
            info = stock.info
            if not info:
                raise ValueError("No data retrieved for symbol")
        except Exception as info_error:
            # If .NS suffix is missing for Indian stocks, try adding it
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                try:
                    indian_symbol = f"{symbol}.NS"
                    stock = yf.Ticker(indian_symbol)
                    info = stock.info
                    symbol = indian_symbol
                except:
                    # Try Bombay Stock Exchange
                    try:
                        bse_symbol = f"{symbol}.BO"
                        stock = yf.Ticker(bse_symbol)
                        info = stock.info
                        symbol = bse_symbol
                    except:
                        raise info_error
            else:
                raise info_error

        # Fetch historical data
        hist = stock.history(period=period, interval="1d", auto_adjust=True)
        
        if hist.empty:
            raise ValueError("No historical data available")
            
        return info, hist
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def create_price_chart(hist_data, symbol):
    """Create an interactive price chart using plotly"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name='Price'
    ))
    
    # Add moving averages
    ma20 = hist_data['Close'].rolling(window=20).mean()
    ma50 = hist_data['Close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(x=hist_data.index, y=ma20, name='20 Day MA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=hist_data.index, y=ma50, name='50 Day MA', line=dict(color='blue')))
    
    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Price',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig

def create_volume_chart(hist_data):
    """Create enhanced volume chart using plotly"""
    # Calculate volume moving average
    volume_ma = hist_data['Volume'].rolling(window=20).mean()
    
    fig = go.Figure()
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=hist_data.index,
        y=hist_data['Volume'],
        name='Volume',
        marker_color='rgba(31, 119, 180, 0.3)'
    ))
    
    # Add volume moving average
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=volume_ma,
        name='20 Day Volume MA',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Trading Volume Analysis',
        yaxis_title='Volume',
        template='plotly_white',
        height=400
    )
    
    return fig

def format_large_number(number):
    """Format large numbers into readable format"""
    if number >= 1e12:
        return f"${number/1e12:.2f}T"
    elif number >= 1e9:
        return f"${number/1e9:.2f}B"
    elif number >= 1e6:
        return f"${number/1e6:.2f}M"
    else:
        return f"${number:,.2f}"

def display_metrics(info):
    """Display enhanced key metrics in a grid"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        pe_ratio = info.get('trailingPE', 'N/A')
        if pe_ratio != 'N/A':
            pe_ratio = f"{pe_ratio:.2f}"
        st.metric("P/E Ratio", pe_ratio)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        eps = info.get('trailingEps', 'N/A')
        if eps != 'N/A':
            eps = f"{eps:.2f}"
        st.metric("EPS", eps)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        debt_to_equity = info.get('debtToEquity', 'N/A')
        if debt_to_equity != 'N/A':
            debt_to_equity = f"{debt_to_equity:.2f}"
        st.metric("Debt to Equity", debt_to_equity)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        roe = info.get('returnOnEquity', 'N/A')
        if roe != 'N/A':
            roe = f"{roe * 100:.2f}"  # Display as percentage
        st.metric("ROE", roe)
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        peg_ratio = info.get('pegRatio', 'N/A')
        if peg_ratio != 'N/A':
            peg_ratio = f"{peg_ratio:.2f}"
        st.metric("PEG Ratio", peg_ratio)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        dividend_yield = info.get('dividendYield', 'N/A')
        if dividend_yield != 'N/A':
            dividend_yield = f"{dividend_yield * 100:.2f}"  # Display as percentage
        st.metric("Dividend Yield", dividend_yield)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_ratio = info.get('currentRatio', 'N/A')
        if current_ratio != 'N/A':
            current_ratio = f"{current_ratio:.2f}"
        st.metric("Current Ratio", current_ratio)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        ps_ratio = info.get('priceToSalesTrailing12Months', 'N/A')
        if ps_ratio != 'N/A':
            ps_ratio = f"{ps_ratio:.2f}"
        st.metric("P/S Ratio", ps_ratio)
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        profit_margin = info.get('profitMargins', 'N/A')
        if profit_margin != 'N/A':
            profit_margin = f"{profit_margin * 100:.2f}"  # Display as percentage
        st.metric("Profit Margin", profit_margin)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        operating_margin = info.get('operatingMargins', 'N/A')
        if operating_margin != 'N/A':
            operating_margin = f"{operating_margin * 100:.2f}"  # Display as percentage
        st.metric("Operating Margin", operating_margin)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        quick_ratio = info.get('quickRatio', 'N/A')
        if quick_ratio != 'N/A':
            quick_ratio = f"{quick_ratio:.2f}"
        st.metric("Quick Ratio", quick_ratio)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        revenue_growth = info.get('revenueGrowth', 'N/A')
        if revenue_growth != 'N/A':
            revenue_growth = f"{revenue_growth * 100:.2f}"  # Display as percentage
        st.metric("Revenue Growth", revenue_growth)
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        gross_margins = info.get('grossMargins', 'N/A')
        if gross_margins != 'N/A':
            gross_margins = f"{gross_margins * 100:.2f}"  # Display as percentage
        st.metric("Gross Margins", gross_margins)
        st.markdown('</div>', unsafe_allow_html=True)


def main():

    # Main content
    st.markdown('<h1 class="stock-header">Advanced Stock Market Analysis By Tejash Mishra</h1>', 
                unsafe_allow_html=True)
    
    # Search and Analysis Section
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_input = st.text_input(
            "Enter Stock Name or Symbol",
            help="Enter company name (e.g., NVIDIA) or symbol (e.g., NVDA)"
        )
    with col2:
        date_range = st.selectbox(
            "Select Time Range",
            ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"], key="time_range"
        )
        # Convert selected range to yfinance period format
        period_map = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "5 Years": "5y"
        }
        period = period_map[date_range]

    if st.button("Analyze", type="primary"):
        if not stock_input:
            st.error("Please enter a stock name or symbol.")
            return

        # Convert input to symbol
        stock_symbol = get_symbol_from_name(stock_input)
        if stock_symbol:
            try:
                # Initialize agents
                if initialize_agents():
                    # Show loading spinner
                    with st.spinner(f"Analyzing {stock_symbol}..."):
                        # Fetch fresh stock data
                        info, hist = get_stock_data(stock_symbol, period=period)
                        
                        if info and hist is not None:
                            # Display market status
                            market_status = "🟢 Market Open" if info.get('regularMarketOpen') else "🔴 Market Closed"
                            st.markdown(f"<div class='market-indicator'>{market_status}</div>", unsafe_allow_html=True)
                            
                            # Create tabs for different sections
                            overview_tab, charts_tab = st.tabs([
                                "Overview", "Charts"
                            ])
                            
                            def safe_float(value):
                                    """Try to convert a value to float. If not possible, return the value as is."""
                                    try:
                                        return float(value)
                                    except (ValueError, TypeError):
                                        return value

                        with overview_tab:
                            st.markdown("### Company Overview")
                            st.write(info.get('longBusinessSummary', 'No description available.'))
                            
                            # Display key metrics
                            st.markdown("### Key Metrics")
                            display_metrics(info)
                            
                            # Additional company information
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### Company Details")
                                st.write(f"Sector: {info.get('sector', 'N/A')}")
                                st.write(f"Industry: {info.get('industry', 'N/A')}")
                                st.write(f"Country: {info.get('country', 'N/A')}")
                                st.write(f"Employees: {info.get('fullTimeEmployees', 'N/A'):,}")
                            
                            with col2:
                                st.markdown("### Trading Information")
                                st.write(f"Exchange: {info.get('exchange', 'N/A')}")
                                st.write(f"Currency: {info.get('currency', 'N/A')}")
                                st.write(f"Volume: {info.get('volume', 'N/A'):,}")
                            
                            # Valuation Metrics
                            st.markdown("### Valuation Metrics")
                            col1, col2 = st.columns(2)
                            with col1:
                                pe_ratio = safe_float(info.get('trailingPE', 'N/A'))
                                st.write(f"P/E Ratio: {pe_ratio if pe_ratio == 'N/A' else f'{pe_ratio:.2f}'}")
                                
                                pb_ratio = safe_float(info.get('priceToBook', 'N/A'))
                                st.write(f"P/B Ratio: {pb_ratio if pb_ratio == 'N/A' else f'{pb_ratio:.2f}'}")
                            
                            with col2:
                                ev_ebitda = safe_float(info.get('enterpriseToEbitda', 'N/A'))
                                st.write(f"EV/EBITDA: {ev_ebitda if ev_ebitda == 'N/A' else f'{ev_ebitda:.2f}'}")
                            
                            # Profitability Metrics
                            st.markdown("### Profitability Metrics")
                            col1, col2 = st.columns(2)
                            with col1:
                                gross_margin = safe_float(info.get('grossMargins', 'N/A')) * 100 if info.get('grossMargins') else 'N/A'
                                st.write(f"Gross Profit Margin: {gross_margin if gross_margin == 'N/A' else f'{gross_margin:.2f}%'}")
                                
                                operating_margin = safe_float(info.get('operatingMargins', 'N/A')) * 100 if info.get('operatingMargins') else 'N/A'
                                st.write(f"Operating Profit Margin: {operating_margin if operating_margin == 'N/A' else f'{operating_margin:.2f}%'}")
                            
                            with col2:
                                net_margin = safe_float(info.get('profitMargins', 'N/A')) * 100 if info.get('profitMargins') else 'N/A'
                                st.write(f"Net Profit Margin: {net_margin if net_margin == 'N/A' else f'{net_margin:.2f}%'}")
                            
                            # Growth Metrics
                            st.markdown("### Growth Metrics")
                            col1, col2 = st.columns(2)
                            with col1:
                                earnings_growth = safe_float(info.get('earningsGrowth', 'N/A')) * 100 if info.get('earningsGrowth') else 'N/A'
                                st.write(f"Earnings Growth (5Y): {earnings_growth if earnings_growth == 'N/A' else f'{earnings_growth:.2f}%'}")
                                
                                revenue_growth = safe_float(info.get('revenueGrowth', 'N/A')) * 100 if info.get('revenueGrowth') else 'N/A'
                                st.write(f"Revenue Growth (5Y): {revenue_growth if revenue_growth == 'N/A' else f'{revenue_growth:.2f}%'}")
                            
                            with col2:
                                future_earnings_growth = safe_float(info.get('futureEarningGrowth', 'N/A')) * 100 if info.get('futureEarningGrowth') else 'N/A'
                                st.write(f"Future Earnings Growth (Projected): {future_earnings_growth if future_earnings_growth == 'N/A' else f'{future_earnings_growth:.2f}%'}")
                            
                            # Risk Metrics
                            st.markdown("### Risk Metrics")
                            col1, col2 = st.columns(2)
                            with col1:
                                beta = safe_float(info.get('beta', 'N/A'))
                                st.write(f"Beta: {beta if beta == 'N/A' else f'{beta:.2f}'}")
                                
                                debt_to_equity = safe_float(info.get('debtToEquity', 'N/A'))
                                st.write(f"Debt-to-Equity Ratio: {debt_to_equity if debt_to_equity == 'N/A' else f'{debt_to_equity:.2f}'}")
                            
                            with col2:
                                interest_coverage = safe_float(info.get('interestCoverageRatio', 'N/A'))
                                st.write(f"Interest Coverage Ratio: {interest_coverage if interest_coverage == 'N/A' else f'{interest_coverage:.2f}'}")
                            
                            # Dividend Metrics
                            st.markdown("### Dividend Metrics")
                            col1, col2 = st.columns(2)
                            with col1:
                                dividend_yield = safe_float(info.get('dividendYield', 'N/A')) * 100 if info.get('dividendYield') else 'N/A'
                                st.write(f"Dividend Yield: {dividend_yield if dividend_yield == 'N/A' else f'{dividend_yield:.2f}%'}")
                                
                                dividend_payout = safe_float(info.get('dividendPayoutRatio', 'N/A'))
                                st.write(f"Dividend Payout Ratio: {dividend_payout if dividend_payout == 'N/A' else f'{dividend_payout:.2f}'}")
                            
                            # Technical Analysis
                            st.markdown("### Technical Analysis")
                            col1, col2 = st.columns(2)
                            with col1:
                                rsi = safe_float(info.get('rsi', 'N/A'))
                                st.write(f"RSI: {rsi if rsi == 'N/A' else f'{rsi:.2f}'}")
                                
                                moving_average_50 = safe_float(info.get('fiftyDayMovingAverage', 'N/A'))
                                st.write(f"50-Day MA: {moving_average_50 if moving_average_50 == 'N/A' else f'{moving_average_50:.2f}'}")
                            
                            with col2:
                                moving_average_200 = safe_float(info.get('twoHundredDayMovingAverage', 'N/A'))
                                st.write(f"200-Day MA: {moving_average_200 if moving_average_200 == 'N/A' else f'{moving_average_200:.2f}'}")

                                        


                            with charts_tab:
                                # Price chart with candlesticks for better price movement visualization
                                st.markdown("### Price Analysis")
                                
                                # Create candlestick chart using Plotly
                                fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                                                    open=hist['Open'],
                                                                    high=hist['High'],
                                                                    low=hist['Low'],
                                                                    close=hist['Close'],
                                                                    name="Candlesticks")])
                                
                                fig.update_layout(title=f'{stock_symbol} Price Candlestick Chart',
                                                xaxis_title='Date',
                                                yaxis_title='Price (INR)',
                                                xaxis_rangeslider_visible=False)
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Volume chart for analyzing trading volume over time
                                st.markdown("### Volume Analysis")
                                volume_chart = create_volume_chart(hist)
                                st.plotly_chart(volume_chart, use_container_width=True)

                                # Technical indicators with additional insights and charts
                                st.markdown("### Technical Indicators")
                                col1, col2, col3 = st.columns(3)

                                # RSI (Relative Strength Index) with overbought and oversold zones
                                with col1:
                                    rsi = hist['Close'].diff()
                                    rsi_pos = rsi.copy()
                                    rsi_neg = rsi.copy()
                                    rsi_pos[rsi_pos < 0] = 0
                                    rsi_neg[rsi_neg > 0] = 0
                                    rsi_14_pos = rsi_pos.rolling(window=14).mean()
                                    rsi_14_neg = abs(rsi_neg.rolling(window=14).mean())
                                    rsi_14 = 100 - (100 / (1 + rsi_14_pos / rsi_14_neg))
                                    
                                    rsi_last = rsi_14.iloc[-1]
                                    rsi_status = "Neutral"
                                    if rsi_last > 70:
                                        rsi_status = "Overbought"
                                    elif rsi_last < 30:
                                        rsi_status = "Oversold"
                                    
                                    st.metric("RSI (14)", f"{rsi_last:.2f}", delta=rsi_status)
                                
                                # Moving Averages and Crossover Signal
                                with col2:
                                    ma20 = hist['Close'].rolling(window=20).mean()
                                    ma50 = hist['Close'].rolling(window=50).mean()
                                    cross_signal = "Bullish" if ma20.iloc[-1] > ma50.iloc[-1] else "Bearish"
                                    
                                    st.metric("MA20 (20-Day)", f"{ma20.iloc[-1]:.2f}")
                                    st.metric("MA50 (50-Day)", f"{ma50.iloc[-1]:.2f}")
                                    st.metric("MA Cross Signal", cross_signal)
                                
                                # Volatility calculation and risk analysis
                                with col3:
                                    volatility = hist['Close'].pct_change().std() * (252 ** 0.5) * 100
                                    st.metric("Annualized Volatility", f"{volatility:.2f}%")
                                    
                                    # Volatility chart (using rolling standard deviation for a better visualization)
                                    volatility_chart = hist['Close'].rolling(window=30).std() * (252 ** 0.5) * 100
                                    st.line_chart(volatility_chart, use_container_width=True)
                                    
                                # Additional chart for moving average convergence divergence (MACD)
                                st.markdown("### MACD - Moving Average Convergence Divergence")
                                short_window = 12
                                long_window = 26
                                signal_window = 9

                                # MACD calculation
                                short_ema = hist['Close'].ewm(span=short_window, adjust=False).mean()
                                long_ema = hist['Close'].ewm(span=long_window, adjust=False).mean()
                                macd = short_ema - long_ema
                                signal_line = macd.ewm(span=signal_window, adjust=False).mean()

                                macd_chart = {
                                    'MACD': macd,
                                    'Signal Line': signal_line
                                }

                                st.line_chart(macd_chart, use_container_width=True)
                                
                                # Price chart with Bollinger Bands
                                st.markdown("### Bollinger Bands")
                                window = 20
                                rolling_mean = hist['Close'].rolling(window=window).mean()
                                rolling_std = hist['Close'].rolling(window=window).std()

                                upper_band = rolling_mean + (rolling_std * 2)
                                lower_band = rolling_mean - (rolling_std * 2)

                                st.line_chart({
                                    'Close': hist['Close'],
                                    'Upper Band': upper_band,
                                    'Lower Band': lower_band
                                }, use_container_width=True)



            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display analysis history
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("### Recent Analysis History")
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(history_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
        This advanced stock market analysis tool combines:
        - Real-time market data analysis
        - AI-powered insights and predictions
        - Technical and fundamental analysis
        - News and sentiment analysis
        - Interactive charts and visualizations
        
        Features:
        - Support for both US and Indian markets (NSE/BSE)
        - Company name and symbol resolution
        - Watchlist management
        - Multiple timeframe analysis
        - Technical indicators
        
        AS IT USE LLM MODEL IT CAN MAKE MISTAKE,
        IF THE MODEL DOES NOT REPONDS THEN RERUN!!
    """)
    
    # Display last refresh time if available
    if st.session_state.last_refresh:
        st.markdown(f"<div class='market-indicator'>Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</div>", 
                   unsafe_allow_html=True)

if __name__ == "__main__":
    main() 