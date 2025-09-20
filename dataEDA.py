import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#  Is there a relationship between the daily minimum and maximum temperature?
#  Can you predict the maximum temperature given the minimum temperature?

summary = pd.read_csv('SummaryofWeather.csv')
weatherStation = pd.read_csv('WeatherStationLocations.csv')

pd.set_option('display.width', None)

summary = summary.iloc[:,[0,1,2,4,5,6,9,10,11]]

summary["Date"] = pd.to_datetime(summary["Date"], format='%Y-%m-%d')

summary["YR"] = summary["Date"].dt.year
summary["MO"] = summary["Date"].dt.month
summary["DA"] = summary["Date"].dt.day

# print(summary["Precip"].unique())

invalid_rows = summary[pd.to_numeric(summary["Precip"], errors="coerce").isna() & summary["Precip"].notna()]
mask = pd.to_numeric(summary["Precip"], errors="coerce").isna() & summary["Precip"].notna()

"""print(invalid_rows)

df_dropped = summary.drop(summary[mask].index)

plt.figure(figsize=(10,10))
sns.heatmap(df_dropped.corr()*10, annot=True)
plt.savefig("withPrecip.png")
plt.close()"""

summary["Precip"] = pd.to_numeric(summary["Precip"], errors="coerce")
summary["Precip"] = summary["Precip"].fillna(summary["Precip"].median())

summary["Precip"] = summary["Precip"].astype(float)

# print(summary)
# (summary.info())
# print(summary.isnull().sum())

# print(summary.describe())

plt.figure(figsize=(10,10))
sns.heatmap(summary.corr(), annot=True)
plt.savefig("heatMap.png")
plt.close()

columns = summary.columns
corr_columns = []

for column in columns:
    if abs(summary["MaxTemp"].corr(summary[column])) > 0.85:
        corr_columns.append(column)
# print(corr_columns)

# print(summary["STA"].unique())
# print(weatherStation["WBAN"].unique())

weatherStation = weatherStation.rename(columns={"WBAN": "STA"})

df = summary.merge(weatherStation, how="inner", on="STA")

df = df.drop(columns=["NAME","STATE/COUNTRY ID","LAT","LON"],axis=1)

df["ELEV"] = df["ELEV"].replace(9999, np.nan)
df["ELEV"] = df["ELEV"].fillna(df["ELEV"].mean())

print(df)
print(df.info())
print(df.isnull().sum())

features = ["MinTemp","Precip","ELEV","Latitude","Longitude","DA","MO","YR","MaxTemp"]

"""
sns.pairplot(df[features])
plt.savefig("pairplot.png")
plt.close()
"""

plt.figure(figsize=(10,10))
sns.heatmap(df[features].corr(), annot=True)
plt.savefig("featuresCorrelation.png")
plt.close()

# ["MinTemp","Precip","ELEV","Latitude","Longitude","DA","MO","YR","MaxTemp"]

plt.subplots(figsize=(50,10))
plt.subplot(1,5,1)
sns.scatterplot(x=df["MinTemp"], y=df["MaxTemp"])
plt.title("Min-Max")
plt.subplot(1,5,2)
sns.scatterplot(x=df["Precip"], y=df["MaxTemp"])
plt.title("Precip-Max")
plt.subplot(1,5,3)
sns.scatterplot(x=df["ELEV"], y=df["MaxTemp"])
plt.title("ELEV-Max")
plt.subplot(1,5,4)
sns.scatterplot(x=df["Latitude"], y=df["MaxTemp"])
plt.title("Latitude-Max")
plt.subplot(1,5,5)
sns.scatterplot(x=df["Longitude"], y=df["MaxTemp"])
plt.title("Longitude-Max")
plt.tight_layout()
plt.savefig("scatterplot.png")
plt.close()

plt.figure(figsize=(10,10))
sns.scatterplot(x=df["Precip"], y=df["ELEV"])
plt.title("ELEV-Precip")
plt.savefig("ELEVprecip.png")
plt.close()

df.to_csv("maxTemperatureData.csv", index=False)