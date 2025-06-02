import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="OHLC Statistics Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data(symbol, start_date, end_date):
    """Load stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        return None

def resample_data(data, frequency, start_day):
    """Resample data to weekly or monthly based on starting day"""
    if frequency == "Weekly":
        # Map day names to numbers (Monday=0, Sunday=6)
        day_map = {'Monday': 'W-MON', 'Tuesday': 'W-TUE', 'Wednesday': 'W-WED',
                  'Thursday': 'W-THU', 'Friday': 'W-FRI', 'Saturday': 'W-SAT', 'Sunday': 'W-SUN'}
        freq = day_map[start_day]
    else:  # Monthly
        freq = 'MS' if start_day == 1 else f'MS'

    # Resample OHLC data
    resampled = data.resample(freq).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    return resampled

def calculate_statistics(data, selected_day="All Days"):
    """Calculate comprehensive statistics for the data"""
    if data.empty:
        return {}

    # Filter by day of week if specified
    if selected_day != "All Days":
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                  'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        filtered_data = data[data.index.dayofweek == day_map[selected_day]]
    else:
        filtered_data = data

    if filtered_data.empty:
        return {}

    # Basic calculations
    df = filtered_data.copy()

    # Range calculations (High - Low)
    df['Range_Points'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Range_Points'] / df['Open']) * 100

    # Body calculations (abs(Close - Open))
    df['Body_Points'] = abs(df['Close'] - df['Open'])
    df['Body_Pct'] = (df['Body_Points'] / df['Open']) * 100

    # Gap calculations (current open vs previous close)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Gap_Points'] = df['Open'] - df['Prev_Close']
    df['Gap_Pct'] = (df['Gap_Points'] / df['Prev_Close']) * 100

    # Separate gap up and gap down
    df['Gap_Up_Points'] = df['Gap_Points'].where(df['Gap_Points'] > 0, 0)
    df['Gap_Down_Points'] = abs(df['Gap_Points'].where(df['Gap_Points'] < 0, 0))
    df['Gap_Up_Pct'] = df['Gap_Pct'].where(df['Gap_Pct'] > 0, 0)
    df['Gap_Down_Pct'] = abs(df['Gap_Pct'].where(df['Gap_Pct'] < 0, 0))

    # Green/Red days from previous close
    df['Change_From_Prev_Close_Points'] = df['Close'] - df['Prev_Close']
    df['Change_From_Prev_Close_Pct'] = (df['Change_From_Prev_Close_Points'] / df['Prev_Close']) * 100
    df['Green_From_Prev_Points'] = df['Change_From_Prev_Close_Points'].where(df['Change_From_Prev_Close_Points'] > 0, 0)
    df['Red_From_Prev_Points'] = abs(df['Change_From_Prev_Close_Points'].where(df['Change_From_Prev_Close_Points'] < 0, 0))
    df['Green_From_Prev_Pct'] = df['Change_From_Prev_Close_Pct'].where(df['Change_From_Prev_Close_Pct'] > 0, 0)
    df['Red_From_Prev_Pct'] = abs(df['Change_From_Prev_Close_Pct'].where(df['Change_From_Prev_Close_Pct'] < 0, 0))

    # Green/Red days from current day open
    df['Change_From_Open_Points'] = df['Close'] - df['Open']
    df['Change_From_Open_Pct'] = (df['Change_From_Open_Points'] / df['Open']) * 100
    df['Green_From_Open_Points'] = df['Change_From_Open_Points'].where(df['Change_From_Open_Points'] > 0, 0)
    df['Red_From_Open_Points'] = abs(df['Change_From_Open_Points'].where(df['Change_From_Open_Points'] < 0, 0))
    df['Green_From_Open_Pct'] = df['Change_From_Open_Pct'].where(df['Change_From_Open_Pct'] > 0, 0)
    df['Red_From_Open_Pct'] = abs(df['Change_From_Open_Pct'].where(df['Change_From_Open_Pct'] < 0, 0))

    # Calculate statistics
    stats = {}

    # Helper function to calculate min, max, avg, std
    def calc_stats(series, name):
        series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(series_clean) > 0:
            return {
                f'{name}_min': series_clean.min(),
                f'{name}_max': series_clean.max(),
                f'{name}_avg': series_clean.mean(),
                f'{name}_std': series_clean.std()
            }
        return {f'{name}_min': 0, f'{name}_max': 0, f'{name}_avg': 0, f'{name}_std': 0}

    # Range statistics
    stats.update(calc_stats(df['Range_Points'], 'range_points'))
    stats.update(calc_stats(df['Range_Pct'], 'range_pct'))

    # Body statistics
    stats.update(calc_stats(df['Body_Points'], 'body_points'))
    stats.update(calc_stats(df['Body_Pct'], 'body_pct'))

    # Gap up statistics
    gap_up_data = df[df['Gap_Points'] > 0]
    if not gap_up_data.empty:
        stats.update(calc_stats(gap_up_data['Gap_Up_Points'], 'gap_up_points'))
        stats.update(calc_stats(gap_up_data['Gap_Up_Pct'], 'gap_up_pct'))
    else:
        stats.update({'gap_up_points_min': 0, 'gap_up_points_max': 0, 'gap_up_points_avg': 0, 'gap_up_points_std': 0})
        stats.update({'gap_up_pct_min': 0, 'gap_up_pct_max': 0, 'gap_up_pct_avg': 0, 'gap_up_pct_std': 0})

    # Gap down statistics
    gap_down_data = df[df['Gap_Points'] < 0]
    if not gap_down_data.empty:
        stats.update(calc_stats(gap_down_data['Gap_Down_Points'], 'gap_down_points'))
        stats.update(calc_stats(gap_down_data['Gap_Down_Pct'], 'gap_down_pct'))
    else:
        stats.update({'gap_down_points_min': 0, 'gap_down_points_max': 0, 'gap_down_points_avg': 0, 'gap_down_points_std': 0})
        stats.update({'gap_down_pct_min': 0, 'gap_down_pct_max': 0, 'gap_down_pct_avg': 0, 'gap_down_pct_std': 0})

    # Green days from previous close
    green_prev_data = df[df['Change_From_Prev_Close_Points'] > 0]
    if not green_prev_data.empty:
        stats.update(calc_stats(green_prev_data['Green_From_Prev_Points'], 'green_prev_points'))
        stats.update(calc_stats(green_prev_data['Green_From_Prev_Pct'], 'green_prev_pct'))
    else:
        stats.update({'green_prev_points_min': 0, 'green_prev_points_max': 0, 'green_prev_points_avg': 0, 'green_prev_points_std': 0})
        stats.update({'green_prev_pct_min': 0, 'green_prev_pct_max': 0, 'green_prev_pct_avg': 0, 'green_prev_pct_std': 0})

    # Red days from previous close
    red_prev_data = df[df['Change_From_Prev_Close_Points'] < 0]
    if not red_prev_data.empty:
        stats.update(calc_stats(red_prev_data['Red_From_Prev_Points'], 'red_prev_points'))
        stats.update(calc_stats(red_prev_data['Red_From_Prev_Pct'], 'red_prev_pct'))
    else:
        stats.update({'red_prev_points_min': 0, 'red_prev_points_max': 0, 'red_prev_points_avg': 0, 'red_prev_points_std': 0})
        stats.update({'red_prev_pct_min': 0, 'red_prev_pct_max': 0, 'red_prev_pct_avg': 0, 'red_prev_pct_std': 0})

    # Green days from current open
    green_open_data = df[df['Change_From_Open_Points'] > 0]
    if not green_open_data.empty:
        stats.update(calc_stats(green_open_data['Green_From_Open_Points'], 'green_open_points'))
        stats.update(calc_stats(green_open_data['Green_From_Open_Pct'], 'green_open_pct'))
    else:
        stats.update({'green_open_points_min': 0, 'green_open_points_max': 0, 'green_open_points_avg': 0, 'green_open_points_std': 0})
        stats.update({'green_open_pct_min': 0, 'green_open_pct_max': 0, 'green_open_pct_avg': 0, 'green_open_pct_std': 0})

    # Red days from current open
    red_open_data = df[df['Change_From_Open_Points'] < 0]
    if not red_open_data.empty:
        stats.update(calc_stats(red_open_data['Red_From_Open_Points'], 'red_open_points'))
        stats.update(calc_stats(red_open_data['Red_From_Open_Pct'], 'red_open_pct'))
    else:
        stats.update({'red_open_points_min': 0, 'red_open_points_max': 0, 'red_open_points_avg': 0, 'red_open_points_std': 0})
        stats.update({'red_open_pct_min': 0, 'red_open_pct_max': 0, 'red_open_pct_avg': 0, 'red_open_pct_std': 0})

    # Additional interesting statistics
    stats['total_trading_days'] = len(filtered_data)
    stats['green_days_count'] = len(green_prev_data)
    stats['red_days_count'] = len(red_prev_data)
    stats['green_days_percentage'] = (stats['green_days_count'] / stats['total_trading_days']) * 100 if stats['total_trading_days'] > 0 else 0
    stats['gap_up_days'] = len(gap_up_data)
    stats['gap_down_days'] = len(gap_down_data)
    stats['no_gap_days'] = stats['total_trading_days'] - stats['gap_up_days'] - stats['gap_down_days']

    return stats

def display_statistics_table(stats, title):
    """Display statistics in a formatted table"""
    if not stats:
        st.warning("No data available for the selected criteria.")
        return

    st.subheader(title)

    # Create organized sections
    sections = {
        "Range Movement (High - Low)": {
            "Points": ['range_points_min', 'range_points_max', 'range_points_avg', 'range_points_std'],
            "Percentage": ['range_pct_min', 'range_pct_max', 'range_pct_avg', 'range_pct_std']
        },
        "Body Range (|Close - Open|)": {
            "Points": ['body_points_min', 'body_points_max', 'body_points_avg', 'body_points_std'],
            "Percentage": ['body_pct_min', 'body_pct_max', 'body_pct_avg', 'body_pct_std']
        },
        "Gap Up": {
            "Points": ['gap_up_points_min', 'gap_up_points_max', 'gap_up_points_avg', 'gap_up_points_std'],
            "Percentage": ['gap_up_pct_min', 'gap_up_pct_max', 'gap_up_pct_avg', 'gap_up_pct_std']
        },
        "Gap Down": {
            "Points": ['gap_down_points_min', 'gap_down_points_max', 'gap_down_points_avg', 'gap_down_points_std'],
            "Percentage": ['gap_down_pct_min', 'gap_down_pct_max', 'gap_down_pct_avg', 'gap_down_pct_std']
        },
        "Green Days (from Previous Close)": {
            "Points": ['green_prev_points_min', 'green_prev_points_max', 'green_prev_points_avg', 'green_prev_points_std'],
            "Percentage": ['green_prev_pct_min', 'green_prev_pct_max', 'green_prev_pct_avg', 'green_prev_pct_std']
        },
        "Red Days (from Previous Close)": {
            "Points": ['red_prev_points_min', 'red_prev_points_max', 'red_prev_points_avg', 'red_prev_points_std'],
            "Percentage": ['red_prev_pct_min', 'red_prev_pct_max', 'red_prev_pct_avg', 'red_prev_pct_std']
        },
        "Green Days (from Current Open)": {
            "Points": ['green_open_points_min', 'green_open_points_max', 'green_open_points_avg', 'green_open_points_std'],
            "Percentage": ['green_open_pct_min', 'green_open_pct_max', 'green_open_pct_avg', 'green_open_pct_std']
        },
        "Red Days (from Current Open)": {
            "Points": ['red_open_points_min', 'red_open_points_max', 'red_open_points_avg', 'red_open_points_std'],
            "Percentage": ['red_open_pct_min', 'red_open_pct_max', 'red_open_pct_avg', 'red_open_pct_std']
        }
    }

    for section_name, section_data in sections.items():
        st.write(f"**{section_name}**")

        # Create DataFrame for this section
        rows = []
        for unit, keys in section_data.items():
            row = {'Metric': unit}
            row['Min'] = f"{stats.get(keys[0], 0):.2f}"
            row['Max'] = f"{stats.get(keys[1], 0):.2f}"
            row['Average'] = f"{stats.get(keys[2], 0):.2f}"
            row['Std Dev'] = f"{stats.get(keys[3], 0):.2f}"
            rows.append(row)

        df_section = pd.DataFrame(rows)
        st.dataframe(df_section, hide_index=True)
        st.write("")

def main():
    st.title("📊 OHLC Statistics Dashboard")

    # Input controls
    col1, col2, col3 = st.columns(3)

    with col1:
        # Popular Indian instruments with their Yahoo Finance symbols
        instruments = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN",
            "BANK NIFTY": "^NSEBANK",
            "NIFTY IT": "^CNXIT",
            "RELIANCE": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "HDFC BANK": "HDFCBANK.NS",
            "INFOSYS": "INFY.NS",
            "ICICI BANK": "ICICIBANK.NS",
            "Custom Symbol": "CUSTOM"
        }

        selected_instrument = st.selectbox("Select Instrument", list(instruments.keys()))

        if selected_instrument == "Custom Symbol":
            symbol = st.text_input("Enter Yahoo Finance Symbol (e.g., AAPL, TSLA)")
        else:
            symbol = instruments[selected_instrument]

    with col2:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", value=datetime.now())

    with col3:
        frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])

        if frequency == "Weekly":
            start_day = st.selectbox("Week Starting Day",
                                   ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        elif frequency == "Monthly":
            start_day = st.number_input("Month Starting Day", min_value=1, max_value=28, value=1)
        else:
            start_day = None

    # Day filter
    day_filter = st.selectbox("Filter by Day of Week",
                             ["All Days", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    if st.button("Analyze", type="primary"):
        if symbol and start_date < end_date:
            with st.spinner("Loading and analyzing data..."):
                # Load data
                data = load_data(symbol, start_date, end_date)

                if data is not None and not data.empty:
                    # Resample if needed
                    if frequency != "Daily":
                        data = resample_data(data, frequency, start_day)

                    # Calculate statistics
                    stats = calculate_statistics(data, day_filter)

                    if stats:
                        # Display total trading days
                        st.metric("Total Trading Days", stats.get('total_trading_days', 0))

                        # Additional summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Green Days", stats.get('green_days_count', 0))
                        with col2:
                            st.metric("Red Days", stats.get('red_days_count', 0))
                        with col3:
                            st.metric("Gap Up Days", stats.get('gap_up_days', 0))
                        with col4:
                            st.metric("Gap Down Days", stats.get('gap_down_days', 0))

                        st.metric("Green Days %", f"{stats.get('green_days_percentage', 0):.1f}%")

                        # Display detailed statistics
                        display_statistics_table(stats, f"Statistics for {selected_instrument} ({day_filter})")
                    else:
                        st.error("Unable to calculate statistics. Please check your data selection.")
                else:
                    st.error("No data found for the selected instrument and date range.")
        else:
            st.error("Please provide valid inputs.")

if __name__ == "__main__":
    main()
