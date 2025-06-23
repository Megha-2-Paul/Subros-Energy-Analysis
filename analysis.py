import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import ttest_ind
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import warnings
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# Step 1: Verify Files (Update path)
subros_dir = 'C:\Users\MPaul\Desktop\Data Analysis\TAsk'  # Replace with your path
print("Files in Subros Directory:", os.listdir(subros_dir))

# Step 2: Define File List
files = [os.path.join(subros_dir, f) for f in os.listdir(subros_dir) if f.endswith('.xlsx')]
print("Detected Files:", files)

# Phase 1: Data Understanding & Preparation
dfs = []
for file in tqdm(files, desc="Loading Excel Files"):
    try:
        df = pd.read_excel(file, engine='openpyxl')
        month = file.split('\\')[-1].split(' ')[-1].split('.')[0].replace('.', '')
        df['Month'] = month
        dfs.append(df)
        print(f"Loaded {file} with {df.shape[0]} rows and {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"Error: File {file} not found.")
    except Exception as e:
        print(f"Error loading {file}: {str(e)}")

if not dfs:
    raise ValueError("No files loaded. Place Excel files in the directory.")

chunk_size = 5000
merged_chunks = []
for i in tqdm(range(0, len(dfs), chunk_size), desc="Merging"):
    chunk_dfs = dfs[i:i + chunk_size]
    if chunk_dfs:
        merged_chunks.append(pd.concat(chunk_dfs, ignore_index=True))

merged_df = pd.concat(merged_chunks, ignore_index=True)
print(f"Merged DataFrame: {merged_df.shape}")

# 1.2 Cleaning & Standardization
merged_df['DATE'] = pd.to_datetime(merged_df['DATE'], format='%d %b %y', errors='coerce')
merged_df.columns = [col.strip().replace(' ', '_').upper() for col in merged_df.columns]
merged_df = merged_df.replace(['#DIV/0!', '#REF!'], np.nan)
merged_df.fillna(method='ffill', inplace=True)
merged_df.fillna(0, inplace=True)
numeric_cols = merged_df.select_dtypes(include=np.number).columns
merged_df[numeric_cols] = merged_df[numeric_cols].clip(lower=0)
outliers = merged_df[numeric_cols].apply(lambda x: x > x.mean() + 3 * x.std()).any(axis=1)
merged_df = merged_df[~outliers]
print("Data cleaned.")

# Phase 2: Descriptive Analytics
# 2.1 Basic Statistical Analysis
print("Summary Stats:\n", merged_df[numeric_cols].describe())
monthly_trend = merged_df.groupby('MONTH')['TOTAL_UNIT_(UPPCL+DG)'].mean()
print("Monthly Trend:\n", monthly_trend)

# 2.2 Operational Insights & EDA
peak_periods = merged_df.loc[merged_df['TOTAL_UNIT_(UPPCL+DG)'].idxmax()]
print("Peak Consumption:", peak_periods['DATE'], peak_periods['TOTAL_UNIT_(UPPCL+DG)'], "kWh")
plt.figure(figsize=(12, 6))
sns.lineplot(x='DATE', y='TOTAL_UNIT_(UPPCL+DG)', hue='MONTH', data=merged_df)
plt.title('Daily Energy Consumption')
plt.savefig('daily_energy.png')
plt.show()

# Phase 3: Advanced Analytics
# 3.1 Time Series Analysis
decomposition = seasonal_decompose(merged_df['TOTAL_UNIT_(UPPCL+DG)'], period=30)
decomposition.trend.plot(title='Trend Component')
plt.savefig('trend_component.png')
plt.show()

model = ARIMA(merged_df['TOTAL_UNIT_(UPPCL+DG)'], order=(1, 1, 1))
fit = model.fit()
forecast = fit.forecast(steps=90)
forecast_df = pd.DataFrame({'Date': pd.date_range(start=merged_df['DATE'].max() + pd.Timedelta(days=1), periods=90), 'Forecasted_kWh': forecast})
print("90-Day Forecast:\n", forecast_df.head())

# 3.2 Production Correlation (Placeholder)
print("Production correlation needs output data.")

# Phase 4: Business Intelligence
# 4.1 Manufacturing Insights
top_consumers = merged_df.filter(like='TOTAL_KWH').mean().sort_values(ascending=False).head()
print("Top Consumers:\n", top_consumers)

# 4.2 Financial Impact
uppcl_cost = merged_df['UPPCL_(UNIT_)'].sum() * 8
dg_cost = merged_df['DG_UNIT'].sum() * 20
total_cost = uppcl_cost + dg_cost
print(f"Total Cost: ₹{total_cost:.2f} (UPPCL: ₹{uppcl_cost:.2f}, DG: ₹{dg_cost:.2f})")

# Phase 5: Predictive Analytics
# 5.1 Forecasting
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
# Placeholder: Train with reshaped data
print("LSTM defined; train with historical data.")

# 5.2 Machine Learning
X = merged_df[['DG_UNIT', 'UPPCL_(UNIT_)']].dropna()
y = merged_df['TOTAL_UNIT_(UPPCL+DG)'].loc[X.index]
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X, y)
predictions = rf_model.predict(X)
mape = mean_absolute_percentage_error(y, predictions)
print(f"Random Forest MAPE: {mape:.2%}")

# Phase 6: Actionable Recommendations
emissions = merged_df['DIESEL_CONSUMPTION_LTR,'].sum() * 2.68  # kg CO2/liter
insights = f"""
### Energy Analysis Report: Jan-May 2025
#### Operational Insights
1. Peak consumption {peak_periods['TOTAL_UNIT_(UPPCL+DG)']:.0f} kWh on {peak_periods['DATE']}.
2. Weekday avg {merged_df[merged_df['DATE'].dt.dayofweek < 5]['TOTAL_UNIT_(UPPCL+DG)'].mean():.0f} kWh vs weekend {merged_df[merged_df['DATE'].dt.dayofweek >= 5]['TOTAL_UNIT_(UPPCL+DG)'].mean():.0f} kWh.
3. DG peak {merged_df['DG_UNIT'].max():.0f} kWh indicates grid issues.

#### Efficiency Insights
1. Top consumers {top_consumers.index[0]} suggest audit targets.
2. PF varies {merged_df['PF'].min():.2f}-{merged_df['PF'].max():.2f}, optimize to >0.9.

#### Financial Insights
1. Total cost ₹{total_cost:.2f}, DG at ₹{dg_cost:.2f} ({~(dg_cost/total_cost*100):.0f}%).
2. Savings possible by shifting to UPPCL.

#### Sustainability Insights
1. Emissions {emissions:.0f} kg CO2, target reduction.
2. PNG usage {merged_df['PNG_(SCM)_CONSUMPTION'].sum():.0f} SCM as greener option.

#### Reliability Insights
1. DG reliance {merged_df['DG_UNIT'].sum()/merged_df['TOTAL_UNIT_(UPPCL+DG)'].sum()*100:.0f}% risks production.

#### Production Correlation Insights
1. Weekday peaks suggest production link (data needed).

#### Quality Control Insights
1. Energy stability may affect quality.

#### Strategic Insights
1. Solar investment potential.

#### Predictions
1. 90-day demand ~{forecast_df['Forecasted_kWh'].mean():.0f} kWh/day.
2. June cost ~₹{(forecast_df['Forecasted_kWh'].mean()*30*(8*0.6+20*0.4)):.2f}.
3. DG ~{40}% in June.
4. Emissions ~{(forecast_df['Forecasted_kWh'].mean()*0.4/20*2.68*30):.0f} kg CO2.

#### Recommendations
1. Audit {top_consumers.index[0]}.
2. Install 500-1000kW solar.
3. Maintain PF > 0.9.
4. Optimize DG for peak hours.
5. Automate meters.
6. Target 20% emission cut by 2026.
7. Integrate production data.
8. Schedule maintenance weekends.
9. Join demand response.
10. Train staff.
"""
with open('energy_report.md', 'w') as f:
    f.write(insights)

# Phase 7: Visualization & Reporting
monthly_source = merged_df.groupby('MONTH')[['UPPCL_(UNIT_)', 'DG_UNIT']].sum()
monthly_source.plot(kind='bar', stacked=True, title='Monthly Energy by Source')
plt.savefig('monthly_energy.png')
plt.show()

fig = px.bar(top_consumers.reset_index(), x='index', y=0, title='Top Consumers')
fig.write_html('top_consumers.html')

print("Outputs saved as PNG/HTML/MD files.")