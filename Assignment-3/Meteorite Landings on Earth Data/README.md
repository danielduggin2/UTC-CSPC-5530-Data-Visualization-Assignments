# Overview
Credits: https://www.kaggle.com/datasets/nafayunnoor/meteorite-landings-on-earth-data?resource=download

This project analyzes meteorite landings worldwide using NASA's Open Data Portal. The dataset, provided by The Meteoritical Society, includes 34,513 recorded meteorites with details such as:

Name & Type: Meteorite classification

Mass (grams): Weight of the meteorite

Fall Status: Whether the meteorite was observed falling ("Fell") or later discovered ("Found")

Year: The year of discovery or fall

Geographical Data: Latitude, longitude, and geo-coordinates

This analysis helps us understand meteorite impact locations, trends over time, and mass distribution.

Installation & Dependencies

To run this project, install the necessary Python libraries:

```
pip install pandas numpy geopandas folium matplotlib seaborn
```

Dataset Preprocessing

1️⃣ Loading the Dataset

We load the dataset into a Pandas DataFrame and inspect its structure:

```python
import pandas as pd

# Load dataset
df = pd.read_csv("Meteorite_Landings.csv")

# Display info and first few rows
print(df.info())
print(df.head())
```

2️⃣ Handling Missing Values

We check for and remove missing values in essential fields (latitude, longitude, mass).

```python
# Check for missing values
print(df.isnull().sum())

# Drop rows with missing essential values
df = df.dropna(subset=['reclat', 'reclong', 'mass (g)'])

# Fill missing years with 0
df['year'] = df['year'].fillna(0).astype(int)
```

3️⃣ Data Cleaning & Standardization

We rename columns for consistency and convert data types.

```python
# Rename columns
df = df.rename(columns={
    'name': 'Name',
    'id': 'ID',
    'recclass': 'Type',
    'mass (g)': 'Mass_g',
    'fall': 'Fall_Status',
    'year': 'Year',
    'reclat': 'Latitude',
    'reclong': 'Longitude',
})
```

Exploratory Data Analysis (EDA)

1️⃣ Meteorite Discoveries Over Time

We visualize how meteorite discoveries have changed over time:

```python
import matplotlib.pyplot as plt

# Plot discoveries over time
df['year'].value_counts().sort_index().plot(kind='line', figsize=(10,5))
plt.xlabel("Year")
plt.ylabel("Number of Meteorites")
plt.title("Meteorite Discoveries Over Time")
plt.show()
```

📌 Figure 1: Meteorite Discoveries Over Time



2️⃣ Distribution of Meteorite Mass

We analyze the distribution of meteorite mass, using a logarithmic scale for better visualization.

```python
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df['mass_g'], bins=50, kde=True)
plt.xscale("log")  # Log scale because mass varies greatly
plt.xlabel("Mass (g)")
plt.ylabel("Count")
plt.title("Distribution of Meteorite Mass")
plt.show()
```

📌 Figure 2: Meteorite Mass Distribution



Geospatial Analysis: Meteorite Impact Locations

We plot meteorite landings on an interactive world map using Folium.

```python
import folium
import numpy as np
from IPython.display import display

# Create a world map
m = folium.Map(location=[0, 0], zoom_start=2)

# Sample 1000 meteorites for performance
sample_df = df.sample(n=min(1000, len(df)), random_state=42)

# Add meteorite landings as points
for _, row in sample_df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=max(2, np.log(row['mass_g'] + 1)),  
        color='red' if row['fall_status'] == 'Fell' else 'blue',
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['name']} ({row['year']}) - {row['mass_g']}g"
    ).add_to(m)

# Display map
display(m)
```

📌 Figure 3: Meteorite Impact Locations



Key Insights

1️⃣ Meteorite Discoveries Increased in Recent Years: Most discoveries happened after 1800, likely due to advances in science and exploration.2️⃣ Mass Distribution is Highly Skewed: A few massive meteorites exist, but most are small.3️⃣ Impact Locations Show Clusters: Many meteorites are found in deserts (Sahara, Antarctica) and North America, where they are easier to detect.

Future Work

✅ Cluster Analysis: Identify regions with the most meteorite falls.✅ Machine Learning: Predict potential impact zones based on past data.✅ Correlating with Population Density: Are meteorites found more where people live?

Conclusion

This project provides valuable insights into meteorite landings worldwide. By leveraging geospatial analysis, we can better understand where and when meteorites fall, aiding researchers in astronomy, planetary science, and geology. 🚀

Acknowledgments

Dataset provided by https://www.kaggle.com/datasets/nafayunnoor/meteorite-landings-on-earth-data?resource=download.

Code developed in Python using Pandas, NumPy, Matplotlib, Seaborn, and Folium.
