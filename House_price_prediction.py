# ============================================================
#   House Price Prediction using Linear Regression
#   Task 2 - Machine Learning Internship
#   Made by: Sachin Avasthi
# ============================================================

# Step 1 - Import karo saari zaruri libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("Libraries import ho gayi!")
print("=" * 50)


# ============================================================
# Step 2 - Dataset banana (Kaggle jaise realistic data)
# ============================================================

print("\nDataset bana raha hoon...")

# Random seed set karo taaki result har baar same aaye
np.random.seed(42)

# Kitne ghar chahiye dataset mein
total_houses = 1000

# Location types
locations = ['Prime', 'Urban', 'Suburban', 'Rural']

# Random values generate karo har feature ke liye
location     = np.random.choice(locations, total_houses, p=[0.15, 0.35, 0.35, 0.15])
rooms        = np.random.randint(2, 9, total_houses)
size_sqft    = np.random.randint(600, 4000, total_houses)
age_years    = np.random.randint(1, 50, total_houses)
bathrooms    = np.clip(rooms // 2 + np.random.randint(-1, 2, total_houses), 1, 5)
garage_cars  = np.random.choice([0, 1, 2], total_houses, p=[0.2, 0.5, 0.3])
floor        = np.random.randint(1, 4, total_houses)
school_rating = np.random.uniform(3.0, 10.0, total_houses).round(1)

# Distance city se kitni door hai
distance_city = np.where(location == 'Prime',    np.random.uniform(1, 5, total_houses),
                np.where(location == 'Urban',    np.random.uniform(5, 20, total_houses),
                np.where(location == 'Suburban', np.random.uniform(20, 50, total_houses),
                                                 np.random.uniform(50, 150, total_houses))))

# Price calculate karo (formula se)
location_price = {'Prime': 3.5, 'Urban': 2.5, 'Suburban': 1.8, 'Rural': 1.0}
base_price = np.array([location_price[loc] for loc in location])

price = (
    base_price    * 20000
    + rooms       * 8000
    + size_sqft   * 45
    + bathrooms   * 5000
    + garage_cars * 10000
    - age_years   * 1200
    + school_rating * 3000
    - distance_city * 500
    + floor       * 2000
    + np.random.normal(0, 15000, total_houses)  # thoda random noise add karo
)

# Price ko reasonable range mein rakhna
price = np.clip(price, 50000, 1500000)

# Ab saara data ek DataFrame mein daalo
dataset = pd.DataFrame({
    'Location'        : location,
    'Rooms'           : rooms,
    'Size_sqft'       : size_sqft,
    'Age_years'       : age_years,
    'Bathrooms'       : bathrooms,
    'Garage_cars'     : garage_cars,
    'Floor'           : floor,
    'School_Rating'   : school_rating,
    'Distance_City_km': distance_city.round(1),
    'Price'           : price.round(-3).astype(int)
})

print(f"Dataset ready! Total rows: {dataset.shape[0]}, Total columns: {dataset.shape[1]}")
print("\nPehle 5 rows:")
print(dataset.head())


# ============================================================
# Step 3 - Dataset ko samjho (EDA - Exploratory Data Analysis)
# ============================================================

print("\n" + "=" * 50)
print("Dataset ki basic info:")
print("=" * 50)

print("\nDataset ka size:", dataset.shape)
print("\nColumn names:", list(dataset.columns))
print("\nKoi missing values hain?")
print(dataset.isnull().sum())

print("\nBasic statistics:")
print(dataset.describe().round(2))

print(f"\nSabse sasta ghar : Rs. {dataset['Price'].min():,}")
print(f"Sabse mehanga ghar: Rs. {dataset['Price'].max():,}")
print(f"Average price     : Rs. {dataset['Price'].mean():,.0f}")


# ============================================================
# Step 4 - Data Preprocessing (data ko model ke liye ready karo)
# ============================================================

print("\n" + "=" * 50)
print("Data Preprocessing...")
print("=" * 50)

# Label Encoding - Location (text) ko number mein convert karo
# Kyunki model numbers samajhta hai, text nahi
le = LabelEncoder()
dataset['Location_Encoded'] = le.fit_transform(dataset['Location'])

print("\nLocation encoding (text -> number):")
for name, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {name} -> {code}")

# Features (X) aur Target (y) alag karo
# Features = jo cheezein hum model ko batate hain
# Target   = jo cheez hum predict karna chahte hain (Price)
features = ['Location_Encoded', 'Rooms', 'Size_sqft', 'Age_years',
            'Bathrooms', 'Garage_cars', 'Floor', 'School_Rating', 'Distance_City_km']

X = dataset[features]   # Input features
y = dataset['Price']    # Output (price)

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target  (y) shape: {y.shape}")

# Train Test Split - data ko 2 parts mein baanto
# 80% se model seekhega, 20% se hum test karenge
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining data  : {X_train.shape[0]} rows")
print(f"Testing data   : {X_test.shape[0]} rows")

# Feature Scaling - sab features ko same scale par laao
# Kyunki Size_sqft ka range bahut bada hai vs Rooms ka range
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nFeature scaling done! (StandardScaler use kiya)")


# ============================================================
# Step 5 - Model banana aur train karna
# ============================================================

print("\n" + "=" * 50)
print("Linear Regression Model Train ho raha hai...")
print("=" * 50)

# Model banao
model = LinearRegression()

# Model ko training data se seekhne do
model.fit(X_train_scaled, y_train)

print("\nModel train ho gaya!")
print(f"Model Intercept (b0): Rs. {model.intercept_:,.0f}")

# Har feature ka coefficient dekho
print("\nHar feature ka coefficient:")
for feature_name, coef in zip(features, model.coef_):
    direction = "upar" if coef > 0 else "neeche"
    print(f"  {feature_name:<22}: {coef:>10,.0f}  (price {direction} jata hai)")


# ============================================================
# Step 6 - Model ka test karo (Evaluation)
# ============================================================

print("\n" + "=" * 50)
print("Model Evaluation...")
print("=" * 50)

# Training data par predict karo
y_pred_train = model.predict(X_train_scaled)

# Testing data par predict karo
y_pred_test = model.predict(X_test_scaled)

# Metrics calculate karo
train_r2   = r2_score(y_train, y_pred_train)
test_r2    = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae   = mean_absolute_error(y_test, y_pred_test)

print(f"\n  R2 Score  (Training) : {train_r2:.4f}")
print(f"  R2 Score  (Testing)  : {test_r2:.4f}   <-- yeh important hai!")
print(f"  RMSE      (Training) : Rs. {train_rmse:,.0f}")
print(f"  RMSE      (Testing)  : Rs. {test_rmse:,.0f}")
print(f"  MAE       (Testing)  : Rs. {test_mae:,.0f}")
print(f"\n  Matlab: Model {test_r2*100:.1f}% accuracy se price predict kar sakta hai")


# ============================================================
# Step 7 - Graphs banana (Visualization)
# ============================================================

print("\n" + "=" * 50)
print("Graphs bana raha hoon...")
print("=" * 50)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('House Price Prediction - Linear Regression Analysis',
             fontsize=16, fontweight='bold', y=1.02)

# --- Graph 1: Actual vs Predicted ---
axes[0, 0].scatter(y_test, y_pred_test, color='steelblue', alpha=0.5, s=20)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Line')
axes[0, 0].set_title(f'Actual vs Predicted Price\n(R2 = {test_r2:.3f})')
axes[0, 0].set_xlabel('Actual Price (Rs.)')
axes[0, 0].set_ylabel('Predicted Price (Rs.)')
axes[0, 0].legend()

# --- Graph 2: Residuals (errors) ---
residuals = y_test - y_pred_test
axes[0, 1].scatter(y_pred_test, residuals, color='coral', alpha=0.5, s=20)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[0, 1].set_title('Residual Plot\n(Error Distribution)')
axes[0, 1].set_xlabel('Predicted Price')
axes[0, 1].set_ylabel('Error (Actual - Predicted)')

# --- Graph 3: Feature Coefficients ---
coef_values = model.coef_
bar_colors = ['green' if c > 0 else 'red' for c in coef_values]
axes[0, 2].barh(features, coef_values, color=bar_colors)
axes[0, 2].axvline(x=0, color='black', linewidth=1)
axes[0, 2].set_title('Feature Coefficients\n(Har feature ka impact)')
axes[0, 2].set_xlabel('Coefficient Value')

# --- Graph 4: Price Distribution ---
axes[1, 0].hist(dataset['Price'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
axes[1, 0].axvline(dataset['Price'].mean(), color='red', linestyle='--',
                    label=f"Average: Rs.{dataset['Price'].mean():,.0f}")
axes[1, 0].set_title('Price Distribution\n(Kitne ghar kis price range mein)')
axes[1, 0].set_xlabel('Price (Rs.)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend()

# --- Graph 5: Average Price by Location ---
avg_by_location = dataset.groupby('Location')['Price'].mean().sort_values(ascending=False)
bar_colors_loc = ['#FF6B35', '#004E89', '#1A936F', '#888888']
axes[1, 1].bar(avg_by_location.index, avg_by_location.values,
               color=bar_colors_loc, edgecolor='white')
axes[1, 1].set_title('Average Price by Location\n(Location ka price par asar)')
axes[1, 1].set_xlabel('Location')
axes[1, 1].set_ylabel('Average Price (Rs.)')

# --- Graph 6: Correlation Heatmap ---
numeric_cols = ['Rooms', 'Size_sqft', 'Age_years', 'Bathrooms',
                'Garage_cars', 'School_Rating', 'Distance_City_km', 'Price']
corr_matrix = dataset[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=axes[1, 2], linewidths=0.5, annot_kws={'size': 7})
axes[1, 2].set_title('Correlation Heatmap\n(Features ka aapas mein relation)')

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("Graphs save ho gaye -> 'model_analysis.png'")


# ============================================================
# Step 8 - Naye ghar ki price predict karo
# ============================================================

print("\n" + "=" * 50)
print("Naye Gharon ki Price Predict karna...")
print("=" * 50)

# Kuch sample houses banao jinka price predict karna hai
new_houses = pd.DataFrame({
    'Location'        : ['Prime', 'Urban', 'Suburban', 'Rural'],
    'Rooms'           : [5,        3,        4,          2],
    'Size_sqft'       : [2000,    1000,     1500,       700],
    'Age_years'       : [5,        15,       10,         30],
    'Bathrooms'       : [3,        2,        2,          1],
    'Garage_cars'     : [2,        1,        1,          0],
    'Floor'           : [2,        1,        2,          1],
    'School_Rating'   : [9.0,      7.0,      6.5,        5.0],
    'Distance_City_km': [3.0,      12.0,     25.0,       80.0]
})

# Location ko encode karo
new_houses['Location_Encoded'] = le.transform(new_houses['Location'])

# Sirf features select karo
new_X = new_houses[features]

# Scale karo
new_X_scaled = scaler.transform(new_X)

# Price predict karo
predicted_prices = model.predict(new_X_scaled)

print("\nPredicted Prices:")
print("-" * 55)
for i in range(len(new_houses)):
    print(f"  Ghar {i+1}: {new_houses['Location'].iloc[i]:<10} | "
          f"{new_houses['Rooms'].iloc[i]} rooms | "
          f"{new_houses['Size_sqft'].iloc[i]} sqft | "
          f"Rs. {predicted_prices[i]:>10,.0f}")

print("\n" + "=" * 50)
print("TASK 2 COMPLETE! Model successfully ban gaya!")
print(f"Final R2 Score: {test_r2:.4f} ({test_r2*100:.1f}% accurate)")
print("=" * 50)
