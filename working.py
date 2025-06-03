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
    """Load stock data from Yahoo Finance with extra buffer for previous day calculations"""
    try:
        # Add buffer days to get previous data for gap calculations
        buffer_start = start_date - timedelta(days=10)  # Buffer for weekends/holidays
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=buffer_start, end=end_date)
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
        freq = 'MS' if start_day == 1 else 'MS'

    # Resample OHLC data
    resampled = data.resample(freq).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    return resampled

def calculate_statistics(data, start_date=None, end_date=None):
    """Calculate comprehensive statistics for the data"""
    if data.empty:
        return {}

    # Filter data to the actual date range requested (after getting buffer data)
    if start_date and end_date:
        # Keep one day before start_date for previous close calculations
        analysis_start = start_date - timedelta(days=1)
        mask = (data.index.date >= analysis_start) & (data.index.date <= end_date)
        data = data[mask]

    # Filter by day of week if specified
    # if selected_day != "All Days":
    #     day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    #               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    #     filtered_data = data[data.index.dayofweek == day_map[selected_day]]
    # else:
    filtered_data = data

    if filtered_data.empty:
        return {}

    # Basic calculations
    df = filtered_data.copy()

    # Range calculations (High - Low) - doesn't need previous data
    df['Range_Points'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Range_Points'] / df['Open']) * 100

    # Body calculations (abs(Close - Open)) - doesn't need previous data
    df['Body_Points'] = abs(df['Close'] - df['Open'])
    df['Body_Pct'] = (df['Body_Points'] / df['Open']) * 100

    # Gap calculations (current open vs previous close) - needs previous data
    df['Prev_Close'] = df['Close'].shift(1)
    df['Gap_Points'] = df['Open'] - df['Prev_Close']
    df['Gap_Pct'] = (df['Gap_Points'] / df['Prev_Close']) * 100



    # Green/Red candles from previous close - needs previous data
    df['Change_From_Prev_Close_Points'] = df['Close'] - df['Prev_Close']
    df['Change_From_Prev_Close_Pct'] = (df['Change_From_Prev_Close_Points'] / df['Prev_Close']) * 100

    # Now filter to actual requested date range (removing the buffer day)
    if start_date:
        df = df[df.index.date >= start_date]

    # Continue with calculations on filtered data
    # Separate gap up and gap down
    df['Gap_Up_Points'] = df['Gap_Points'].where(df['Gap_Points'] > 0, 0)
    df['Gap_Down_Points'] = abs(df['Gap_Points'].where(df['Gap_Points'] < 0, 0))
    df['Gap_Up_Pct'] = df['Gap_Pct'].where(df['Gap_Pct'] > 0, 0)
    df['Gap_Down_Pct'] = abs(df['Gap_Pct'].where(df['Gap_Pct'] < 0, 0))

    df['Green_From_Prev_Points'] = df['Change_From_Prev_Close_Points'].where(df['Change_From_Prev_Close_Points'] > 0, 0)
    df['Red_From_Prev_Points'] = abs(df['Change_From_Prev_Close_Points'].where(df['Change_From_Prev_Close_Points'] < 0, 0))
    df['Green_From_Prev_Pct'] = df['Change_From_Prev_Close_Pct'].where(df['Change_From_Prev_Close_Pct'] > 0, 0)
    df['Red_From_Prev_Pct'] = abs(df['Change_From_Prev_Close_Pct'].where(df['Change_From_Prev_Close_Pct'] < 0, 0))

    # Green/Red candles from current day open - doesn't need previous data
    df['Change_From_Open_Points'] = df['Close'] - df['Open']
    df['Change_From_Open_Pct'] = (df['Change_From_Open_Points'] / df['Open']) * 100
    df['Green_From_Open_Points'] = df['Change_From_Open_Points'].where(df['Change_From_Open_Points'] > 0, 0)
    df['Red_From_Open_Points'] = abs(df['Change_From_Open_Points'].where(df['Change_From_Open_Points'] < 0, 0))
    df['Green_From_Open_Pct'] = df['Change_From_Open_Pct'].where(df['Change_From_Open_Pct'] > 0, 0)
    df['Red_From_Open_Pct'] = abs(df['Change_From_Open_Pct'].where(df['Change_From_Open_Pct'] < 0, 0))

    # Calculate statistics
    stats = {}

    # Helper function to calculate min, max, avg, std and std ranges
    def calc_stats(series, name, is_always_positive=False):
        series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(series_clean) > 0:
            mean_val = series_clean.mean()
            std_val = series_clean.std()
            min_val = series_clean.min()
            max_val = series_clean.max()

            # Calculate std ranges, but don't let them go below 0 if the metric is always positive
            std_ranges = {}
            for i in [1, 2, 3]:
                lower = mean_val - (i * std_val)
                upper = mean_val + (i * std_val)

                # For metrics that are always positive (like gaps, body ranges),
                # don't let the lower bound go negative
                if is_always_positive and lower < 0:
                    lower = 0

                std_ranges[f'{name}_{i}std_lower'] = lower
                std_ranges[f'{name}_{i}std_upper'] = upper

            return {
                f'{name}_min': min_val,
                f'{name}_max': max_val,
                f'{name}_avg': mean_val,
                f'{name}_std': std_val,
                **std_ranges
            }

        return {
            f'{name}_min': 0, f'{name}_max': 0, f'{name}_avg': 0, f'{name}_std': 0,
            f'{name}_1std_lower': 0, f'{name}_1std_upper': 0,
            f'{name}_2std_lower': 0, f'{name}_2std_upper': 0,
            f'{name}_3std_lower': 0, f'{name}_3std_upper': 0
        }

    # Range statistics (always positive)
    stats.update(calc_stats(df['Range_Points'], 'range_points', is_always_positive=True))
    stats.update(calc_stats(df['Range_Pct'], 'range_pct', is_always_positive=True))

    # Body statistics (always positive)
    stats.update(calc_stats(df['Body_Points'], 'body_points', is_always_positive=True))
    stats.update(calc_stats(df['Body_Pct'], 'body_pct', is_always_positive=True))

    # Gap up statistics (always positive when they exist)
    gap_up_data = df[df['Gap_Points'] > 0]
    if not gap_up_data.empty:
        stats.update(calc_stats(gap_up_data['Gap_Up_Points'], 'gap_up_points', is_always_positive=True))
        stats.update(calc_stats(gap_up_data['Gap_Up_Pct'], 'gap_up_pct', is_always_positive=True))
    else:
        for suffix in ['min', 'max', 'avg', 'std', '1std_lower', '1std_upper', '2std_lower', '2std_upper', '3std_lower', '3std_upper']:
            stats[f'gap_up_points_{suffix}'] = 0
            stats[f'gap_up_pct_{suffix}'] = 0

    # Gap down statistics (always positive when they exist)
    gap_down_data = df[df['Gap_Points'] < 0]
    if not gap_down_data.empty:
        stats.update(calc_stats(gap_down_data['Gap_Down_Points'], 'gap_down_points', is_always_positive=True))
        stats.update(calc_stats(gap_down_data['Gap_Down_Pct'], 'gap_down_pct', is_always_positive=True))
    else:
        for suffix in ['min', 'max', 'avg', 'std', '1std_lower', '1std_upper', '2std_lower', '2std_upper', '3std_lower', '3std_upper']:
            stats[f'gap_down_points_{suffix}'] = 0
            stats[f'gap_down_pct_{suffix}'] = 0

    # Green candles from previous close (always positive when they exist)
    green_prev_data = df[df['Change_From_Prev_Close_Points'] > 0]
    if not green_prev_data.empty:
        stats.update(calc_stats(green_prev_data['Green_From_Prev_Points'], 'green_prev_points', is_always_positive=True))
        stats.update(calc_stats(green_prev_data['Green_From_Prev_Pct'], 'green_prev_pct', is_always_positive=True))
    else:
        for suffix in ['min', 'max', 'avg', 'std', '1std_lower', '1std_upper', '2std_lower', '2std_upper', '3std_lower', '3std_upper']:
            stats[f'green_prev_points_{suffix}'] = 0
            stats[f'green_prev_pct_{suffix}'] = 0

    # Red candles from previous close (always positive when they exist)
    red_prev_data = df[df['Change_From_Prev_Close_Points'] < 0]
    if not red_prev_data.empty:
        stats.update(calc_stats(red_prev_data['Red_From_Prev_Points'], 'red_prev_points', is_always_positive=True))
        stats.update(calc_stats(red_prev_data['Red_From_Prev_Pct'], 'red_prev_pct', is_always_positive=True))
    else:
        for suffix in ['min', 'max', 'avg', 'std', '1std_lower', '1std_upper', '2std_lower', '2std_upper', '3std_lower', '3std_upper']:
            stats[f'red_prev_points_{suffix}'] = 0
            stats[f'red_prev_pct_{suffix}'] = 0

    # Green candles from current open (always positive when they exist)
    green_open_data = df[df['Change_From_Open_Points'] > 0]
    if not green_open_data.empty:
        stats.update(calc_stats(green_open_data['Green_From_Open_Points'], 'green_open_points', is_always_positive=True))
        stats.update(calc_stats(green_open_data['Green_From_Open_Pct'], 'green_open_pct', is_always_positive=True))
    else:
        for suffix in ['min', 'max', 'avg', 'std', '1std_lower', '1std_upper', '2std_lower', '2std_upper', '3std_lower', '3std_upper']:
            stats[f'green_open_points_{suffix}'] = 0
            stats[f'green_open_pct_{suffix}'] = 0

    # Red candles from current open (always positive when they exist)
    red_open_data = df[df['Change_From_Open_Points'] < 0]
    if not red_open_data.empty:
        stats.update(calc_stats(red_open_data['Red_From_Open_Points'], 'red_open_points', is_always_positive=True))
        stats.update(calc_stats(red_open_data['Red_From_Open_Pct'], 'red_open_pct', is_always_positive=True))
    else:
        for suffix in ['min', 'max', 'avg', 'std', '1std_lower', '1std_upper', '2std_lower', '2std_upper', '3std_lower', '3std_upper']:
            stats[f'red_open_points_{suffix}'] = 0
            stats[f'red_open_pct_{suffix}'] = 0

    # Additional interesting statistics
    stats['total_candles'] = len(df)
    stats['green_candles_count'] = len(green_open_data)
    stats['red_candles_count'] = len(red_open_data)
    stats['flat_candles_count'] = stats['total_candles'] - stats['green_candles_count'] - stats['red_candles_count']
    stats['green_candles_percentage'] = (stats['green_candles_count'] / stats['total_candles']) * 100 if stats['total_candles'] > 0 else 0
    stats['gap_up_candles'] = len(gap_up_data)
    stats['gap_down_candles'] = len(gap_down_data)
    stats['no_gap_candles'] = stats['total_candles'] - stats['gap_up_candles'] - stats['gap_down_candles']

    return stats

def display_statistics_table(stats, title):
    """Display statistics in a formatted table with std ranges"""
    if not stats:
        st.warning("No data available for the selected criteria.")
        return

    st.subheader(title)

    # Create organized sections
    sections = {
        "Total Range (High - Low)": {
            "Points": 'range_points',
            "Percentage": 'range_pct'
        },
        "Body Range (|Close - Open|)": {
            "Points": 'body_points',
            "Percentage": 'body_pct'
        },
        "Gap Up": {
            "Points": 'gap_up_points',
            "Percentage": 'gap_up_pct'
        },
        "Gap Down": {
            "Points": 'gap_down_points',
            "Percentage": 'gap_down_pct'
        },
        "Green Candles (from Previous Close)": {
            "Points": 'green_prev_points',
            "Percentage": 'green_prev_pct'
        },
        "Red Candles (from Previous Close)": {
            "Points": 'red_prev_points',
            "Percentage": 'red_prev_pct'
        },
        "Green Candles (from Current Open)": {
            "Points": 'green_open_points',
            "Percentage": 'green_open_pct'
        },
        "Red Candles (from Current Open)": {
            "Points": 'red_open_points',
            "Percentage": 'red_open_pct'
        }
    }

    for section_name, section_data in sections.items():
        st.write(f"**{section_name}**")

        # Create DataFrame for this section
        rows = []
        for unit, key_prefix in section_data.items():
            row = {
                'Metric': unit,
                'Min': f"{stats.get(f'{key_prefix}_min', 0):.1f}",
                'Max': f"{stats.get(f'{key_prefix}_max', 0):.1f}",
                'Average': f"{stats.get(f'{key_prefix}_avg', 0):.1f}",
                # 'Std Dev': f"{stats.get(f'{key_prefix}_std', 0):.1f}",
                '1σ Range': f"{stats.get(f'{key_prefix}_1std_lower', 0):.1f} to {stats.get(f'{key_prefix}_1std_upper', 0):.1f}",
                '2σ Range': f"{stats.get(f'{key_prefix}_2std_lower', 0):.1f} to {stats.get(f'{key_prefix}_2std_upper', 0):.1f}",
                '3σ Range': f"{stats.get(f'{key_prefix}_3std_lower', 0):.1f} to {stats.get(f'{key_prefix}_3std_upper', 0):.1f}"
            }
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
    # day_filter = st.selectbox("Filter by Day of Week",
    #                          ["All Days", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

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
                    stats = calculate_statistics(data, start_date, end_date)

                    if stats:
                        # Display summary metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Candles", stats.get('total_candles', 0))
                        with col2:
                            st.metric("Green Candles", stats.get('green_candles_count', 0))
                        with col3:
                            st.metric("Red Candles", stats.get('red_candles_count', 0))
                        # with col4:
                        #     st.metric("Flat Candles", stats.get('flat_candles_count', 0))
                        # with col5:
                        #     st.metric("Green Candles %", f"{stats.get('green_candles_percentage', 0):.1f}%")

                        # col1, col2, col3 = st.columns(3)
                        # with col1:
                        #     st.metric("Gap Up Candles", stats.get('gap_up_candles', 0))
                        # with col2:
                        #     st.metric("Gap Down Candles", stats.get('gap_down_candles', 0))
                        # with col3:
                        #     st.metric("No Gap Candles", stats.get('no_gap_candles', 0))

                        # Verification
                        # total_check = stats.get('green_candles_count', 0) + stats.get('red_candles_count', 0) + stats.get('flat_candles_count', 0)
                        # st.info(f"Verification: {stats.get('green_candles_count', 0)} + {stats.get('red_candles_count', 0)} + {stats.get('flat_candles_count', 0)} = {total_check} (equals Total Trading Days: {stats.get('total_candles', 0)})")

                        # Display detailed statistics
                        display_statistics_table(stats, f"Statistics for {selected_instrument}")
                    else:
                        st.error("Unable to calculate statistics. Please check your data selection.")
                else:
                    st.error("No data found for the selected instrument and date range.")
        else:
            st.error("Please provide valid inputs.")

if __name__ == "__main__":
    main()
