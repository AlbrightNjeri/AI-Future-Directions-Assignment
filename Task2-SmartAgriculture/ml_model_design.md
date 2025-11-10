# LSTM Model Design for Crop Yield Prediction

## Model Architecture

**Input Shape:** (sequence_length=30, features=25)
- 30 days of historical data
- 25 features per timestep

## Model Layers

1. **LSTM Layer 1:** 64 units, return_sequences=True
   - Processes temporal patterns
   - Dropout: 0.2

2. **LSTM Layer 2:** 32 units, return_sequences=False
   - Extracts high-level temporal features
   - Dropout: 0.2

3. **Dense Layer 1:** 16 units, ReLU activation
   - Feature abstraction

4. **Dense Layer 2:** 8 units, ReLU activation
   - Further dimensionality reduction

5. **Output Layer:** 1 unit, Linear activation
   - Yield prediction (kg/ha)

## Total Parameters: 12,737

## Training Specification

- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (learning_rate=0.001)
- **Batch Size:** 32
- **Epochs:** 50 (with early stopping)
- **Validation Split:** 20%

## Expected Performance

- **RMSE:** <5% of average yield
- **R² Score:** >0.92
- **MAE:** <3% of average yield

## Features Used (25 Total)

**Current State (10 features):**
- Soil Moisture (%)
- Soil Temperature (°C)
- Air Temperature (°C)
- Humidity (%)
- Soil pH
- Nitrogen (mg/kg)
- Phosphorus (mg/kg)
- Potassium (mg/kg)
- Rainfall (mm)
- Wind Speed (m/s)

**Historical Trends (7 features):**
- Moisture trend
- Temperature trend
- Humidity trend
- NPK depletion rate
- Disease pressure index
- Water stress indicator
- Pest activity index

**Derived Features (8 features):**
- Days since planting
- Growth stage
- Field history
- Weather forecast
- And 4 more domain-specific features