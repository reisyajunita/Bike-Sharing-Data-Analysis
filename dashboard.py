import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
sns.set_theme(style="dark")

# Menyiapkan day_df dan hour_df
def create_daily_user_df(day_df):
    daily_user_df = day_df.resample(rule="D", on="dteday").agg({
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum"
    })
    daily_user_df = daily_user_df.reset_index()
    daily_user_df.rename(columns={"casual": "Total Casual", "registered": "Total Registered", "cnt": "Total User"}, inplace=True)
    return daily_user_df
def create_workingday_vs_holiday_df(day_df):
    workingday_total = day_df[day_df["workingday"] == 1]["cnt"].sum()
    holiday_total = day_df[day_df['holiday'] == 1]['cnt'].sum()

    # Membuat DataFrame baru dengan hasil perhitungan
    workingday_vs_holiday_df = pd.DataFrame({
        "Type": ["Working Day", "Holiday"],
        "Total": [workingday_total, holiday_total]
    })
    return workingday_vs_holiday_df
def create_hours_rental_df(hour_df):
    mode_hour = hour_df['hr'].mode()[0]
    median_hour = hour_df['hr'].median()

    hours_rental_df = pd.DataFrame({
        "Statistic": ["Mode", "Median"],
        "Hour": [mode_hour, median_hour]
    })
    return hours_rental_df
def create_year_2011_2012_df(day_df):
    day_df["yr"] = day_df["dteday"].dt.year
    year_counts = day_df.groupby("yr")["cnt"].sum()

    year_2011_2012_df = pd.DataFrame({
        "Year": year_counts.index,
        "Count": year_counts.values
    })
    return year_2011_2012_df
def create_season_df(day_df):
    season_df = day_df.groupby('season')['cnt'].sum().sort_values(ascending=False)
    return season_df
def create_linear_regression_analysis_df(X_train, X_test, y_train, y_test):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_lr = linear_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    return linear_model, y_pred_lr, mse_lr, r2_lr
def create_random_forest_analysis(X_train, X_test, y_train, y_test):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    return rf_model, y_pred_rf, mse_rf, r2_rf

# Meload Data
day_df = pd.read_csv("day.csv")
hour_df = pd.read_csv("hour.csv")

# Mengatur kolom tanggal
datetime_column = ["dteday"]
day_df.sort_values("dteday", inplace=True)
day_df.reset_index(inplace=True)

for column in datetime_column:
    day_df[column] = pd.to_datetime(day_df[column])

# Membuat komponen filter
min_date = day_df["dteday"].min()
max_date = day_df["dteday"].max()

with st.sidebar:
    st.image("logo.png")
        # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
main_df = day_df[(day_df["dteday"] >= str(start_date)) & 
            (day_df["dteday"] <= str(end_date))]
sec_df = hour_df[(hour_df["dteday"] >= str(start_date)) & 
            (day_df["dteday"] <= str(end_date))]

# Menggunakan fungsi-fungsi yang sudah dibuat
daily_user_df = create_daily_user_df(main_df)
workingday_vs_holiday_df = create_workingday_vs_holiday_df(main_df)
hours_rental_df = create_hours_rental_df(sec_df)
year_2011_2012_df = create_year_2011_2012_df(main_df)
season_df = create_season_df(main_df)

X = main_df[["temp", "hum", "windspeed"]] 
y = main_df["cnt"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
linear_model, y_pred_lr, mse_lr, r2_lr = create_linear_regression_analysis_df(X_train, X_test, y_train, y_test) 
rf_model, y_pred_rf, mse_rf, r2_rf = create_random_forest_analysis(X_train, X_test, y_train, y_test)

# Melengkapi Dashboard dengan berbagai Visualisasi Data
st.header("Bike Sharing Dashboard :bike:")
st.subheader("Daily Users")
# Mengatur kolom untuk layout
col1, col2, col3 = st.columns(3)

# kolom 1: Total Casual
with col1:
    total_casual = daily_user_df["Total Casual"].sum()
    st.metric("Total Casual", value=f"{total_casual:,}")

# kolom 2: Total Registered 
with col2:
    total_registered = daily_user_df["Total Registered"].sum() 
    st.metric("Total Registered", value=f"{total_registered:,}")

# kolom 3: Total Users
with col3:
    total_users = daily_user_df["Total User"].sum()
    st.metric("Total All User", value=f"{total_users:,}")

# Membuat figure dan Axis
fig, ax = plt.subplots(figsize=(16, 8))

# Plot Data
ax.plot(
    daily_user_df["dteday"],  # sumbu x: tanggal
    daily_user_df["Total User"],  # sumbu y: jumlah
    marker='o', 
    linewidth=2,
    color="#90CAF9"
)
# Mengatur Parameter Tick
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)

# Menampilkan Perbandingan Workingday dengan Holiday dalam bentuk Pie Chart
st.subheader("Comparison of Workingday and Holiday Renters")
fig, ax = plt.subplots(figsize=(16, 8))
colors = ["#85A947", "#d3d3d3"]
values = workingday_vs_holiday_df["Total"]
labels = workingday_vs_holiday_df["Type"]
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, textprops={'fontsize': 14})
plt.axis('equal')
plt.title("Perbandingan Jumlah Hari Kerja dan Hari Libur")
st.pyplot(fig)

# Menampilkan Waktu Paling Banyak terjadinya Penyewaan Sepeda
st.subheader("The Most Bike Rental Hours")
fig, ax = plt.subplots(figsize=(16, 8))
colors = ["#85A947"]

mode_hour = hour_df['hr'].mode()[0]
median_hour = hour_df['hr'].median()

sns.histplot(data=hour_df, x="hr", bins=24, color=colors[0])
plt.title("Distribusi Jam Penyewaan Bike Sharing")
plt.xlabel("Hour")
plt.ylabel(None)

plt.axvline(mode_hour, color='red', linestyle='--', label='Mode')
plt.axvline(median_hour, color='green', linestyle='-', label='Median')

plt.legend()
st.pyplot(fig)

# Melihat Rentang Total Penyewaan Bike Sharing dari Tahun 2011 sampai 2012
st.subheader("Total Bike Rental from 2011 to 2012")
fig, ax = plt.subplots(figsize=(16, 8))
day_df["yr"] = day_df["dteday"].dt.year
year_counts = day_df.groupby("yr")["cnt"].sum()
colors = ["#85A947", "#85A947"]

sns.barplot(data=day_df, x="yr", y="cnt",estimator="sum", errorbar=None, hue="yr", palette=colors) 
plt.title("Distribusi Penyewaan Bike Sharing Berdasarkan Tahun", fontsize=15, fontweight='bold')
plt.xlabel("VS")
plt.ylabel("Total Count")
plt.legend(title="Year")
plt.xticks(ticks=[0, 1], labels=["2011", "2012"])
st.pyplot(fig)

# Musim apa yang paling banyak penyewaan sepeda?
st.subheader("The Most Bike Rental Seasons")
fig, ax = plt.subplots(figsize=(16, 8))
fav_season = day_df.groupby('season')['cnt'].sum().sort_values(ascending=False)
colors = ["#D3D3D3", "#D3D3D3","#85A947", "#D3D3D3"]
sns.barplot(data=day_df, x="season", y="cnt", estimator=sum, errorbar=None, hue="season", palette=colors)
plt.xticks(ticks=[0, 1, 2, 3], labels=["Winter", "Spring", "Summer", "Fall"])
plt.xlabel("Season")
plt.ylabel(None)
plt.title("Jumlah Penyewaan Bike Sharing pe Musim")

st.pyplot(fig)

# Analisis Regresi Linear
st.subheader("Analisis Regresi Linear")
st.write(f"Mean Squared Error: {mse_lr}")
st.write(f"R-squared: {r2_lr}")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_lr, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted Values (Linear Regression)')
st.pyplot(fig)

# Melakukan Grid Search untuk Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

best_params = grid_search_rf.best_params_ 
best_score = -grid_search_rf.best_score_


# Menggunakan parameter terbaik dari Grid Search untuk melatih model
best_rf_model = RandomForestRegressor(**best_params, random_state=42)
best_rf_model.fit(X_train, y_train)
y_pred_best_rf = best_rf_model.predict(X_test)
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)

# Analisis Random Forest
st.subheader("Analisis Random Forest")
st.write(f"Mean Squared Error: {mse_rf}")
st.write(f"R-squared: {r2_rf}")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_rf, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted Values (Random Forest)')
st.pyplot(fig)

# Analisis Random Forest yang Dioptimalkan
st.subheader("Analisis Random Forest yang Dioptimalkan")
st.write(f"Mean Squared Error: {mse_best_rf}")
st.write(f"R-squared: {r2_best_rf}")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_best_rf, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted Values (Optimized Random Forest)') 
st.pyplot(fig)

st.caption("Dibuat oleh: Reisya Junita (2025)")
