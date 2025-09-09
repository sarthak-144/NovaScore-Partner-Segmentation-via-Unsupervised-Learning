# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import MetricFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Suppress a specific FutureWarning from sklearn
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.cluster._kmeans')

# --- Data Generation Logic (from dummy_data.ipynb) ---
# This function is cached to avoid regenerating data on every run
@st.cache_data
def generate_data(num_records=50000):
    """
    Generates the biased synthetic dataset. Based on dummy_data.ipynb.
    """
    np.random.seed(42)
    
    def calculate_loyalty_score(partner_since_date, trip_frequency):
        days_as_partner = (datetime.now() - partner_since_date).days
        tenure_score = 1 - np.exp(-days_as_partner / 1000)  
        trips_score = 1 - np.exp(-trip_frequency / 200)      
        return round(0.6 * tenure_score + 0.4 * trips_score, 3)

    data = []
    # ... (Paste the entire data generation loop from your dummy_data.ipynb here) ...
    # It's the large 'for i in range(NUM_RECORDS):' loop.
    # For brevity, it is omitted here, but you must include it.
    partner_types = ['Driver', 'Merchant']
    driver_vehicle_types = ['Cab Driver', 'Food Delivery Rider']
    merchant_categories = ['Small', 'Medium', 'Large']

    for i in range(num_records):
        gender = np.random.choice(['Male', 'Female'], p=[0.85, 0.15])
        partner_type = np.random.choice(partner_types, p=[0.7, 0.3])
        
        record = {'gender': gender, 'partner_type': partner_type}
        
        if partner_type == 'Driver':
            record['vehicle_type'] = np.random.choice(driver_vehicle_types, p=[0.6, 0.4])
            days_ago = np.random.randint(30, 365 * 4)
            record['partner_since_date'] = datetime.now() - timedelta(days=days_ago)
            record['geographic_location'] = np.random.choice(['Urban', 'Rural'], p=[0.75, 0.25])
            
            if record['geographic_location'] == 'Urban':
                record['shift'] = np.random.choice(['Day', 'Night'], p=[0.4, 0.6])
            else:
                record['shift'] = np.random.choice(['Day', 'Night'], p=[0.8, 0.2])
            
            if record['vehicle_type'] == 'Cab Driver':
                base_rate_range = (18, 25)
            else:
                base_rate_range = (12, 18)

            if record['geographic_location'] == 'Rural':
                base_rate_range = tuple(x * 0.8 for x in base_rate_range)

            if gender == 'Female':
                base_rate_range = tuple(x * 0.9 for x in base_rate_range)

            if record['shift'] == 'Night':
                base_rate_range = tuple(x * 1.2 for x in base_rate_range)

            hours_online = np.random.uniform(100, 250)
            earnings_per_hour = round(np.random.uniform(*base_rate_range), 2)
            total_earnings = hours_online * earnings_per_hour

            record['earnings_per_hour'] = earnings_per_hour
            record['total_earnings'] = round(total_earnings, 2)
            record['hours_online'] = int(round(hours_online, 0))

            trip_freq = np.random.uniform(80, 200)
            if record['geographic_location'] == 'Rural':
                trip_freq *= 0.9
            record['trip_frequency'] = int(round(trip_freq, 0))

            record['num_insurance_claims'] = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.7, 0.15, 0.05, 0.05, 0.025, 0.025])
            
            if record['num_insurance_claims'] <= 1:
                rating = np.random.uniform(4.5, 5.0) 
            elif record['num_insurance_claims'] <= 3:
                rating = np.random.uniform(4.0, 4.7)
            else:
                rating = np.random.uniform(3.5, 4.2) 
            record['customer_rating'] = round(rating, 1)

            weekly_earnings = np.random.normal(
                loc=total_earnings / 4, 
                scale=total_earnings / 20, 
                size=12
            )
            record['consistency_index'] = round(np.std(weekly_earnings), 2)

            record['loyalty_score'] = calculate_loyalty_score(record['partner_since_date'], trip_freq)
        
        else: #Merchants
            record['merchant_category'] = np.random.choice(merchant_categories, p=[0.6, 0.3, 0.1])
            record['geographic_location'] = np.random.choice(['Urban', 'Rural'], p=[0.8, 0.2])
            
            if record['merchant_category'] == 'Small':
                base_gmv = np.random.uniform(50_000, 150_000)
            elif record['merchant_category'] == 'Medium':
                base_gmv = np.random.uniform(200_000, 500_000)
            else:
                base_gmv = np.random.uniform(600_000, 2_000_000)
            
            gmv = base_gmv
            if gender == 'Female':
                gmv *= 0.9
            if record['geographic_location'] == 'Rural':
                gmv *= 0.7
                
            record['gross_merchandise_volume'] = round(gmv, 2)
            record['average_order_value'] = round(gmv / np.random.randint(500, 5000), 2)
            record['sales_growth'] = round(np.random.uniform(-0.05, 0.20), 2)

            monthly_revenues = []
            for _ in range(12):
                month_rev = gmv / 12 * np.random.uniform(0.8, 1.2)
                monthly_revenues.append(month_rev)
            coef_var = np.std(monthly_revenues) / np.mean(monthly_revenues)
            record['revenue_stability_score'] = round(1 - min(coef_var / 0.6, 1), 2)

            record['store_availability'] = round(np.random.uniform(0.85, 1.0), 2)
            record['preparation_speed'] = round(np.random.uniform(10, 25), 1)
            record['order_accuracy'] = round(np.random.uniform(0.95, 1.0), 3)
            record['customer_retention'] = round(np.random.uniform(0.4, 0.8), 2)
        
        data.append(record)

    df = pd.DataFrame(data)
    df.to_csv('nova_custom_biased_dataset.csv', index=False)
    return df

# --- Model Training and Analysis Logic (from main.ipynb) ---

@st.cache_data
def run_segmentation_and_scoring(df):
    """
    Performs PCA, K-Means clustering, and assigns the NovaScore.
    Based on Blocks 1, 4, and 5 from main.ipynb.
    """
    # Block 5 Logic: Recompute clusters on original feature space for NovaScore
    feature_columns = [
        'earnings_per_hour', 'customer_rating', 'loyalty_score', 
        'consistency_index', 'num_insurance_claims',
        'revenue_stability_score', 'customer_retention', 
        'sales_growth', 'order_accuracy'
    ]
    X_scaled = StandardScaler().fit_transform(df[feature_columns].fillna(0))
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_scaled)
    df['kmeans_cluster'] = kmeans.labels_
    
    # Calculate distances for scoring
    centroids = kmeans.cluster_centers_
    distances = euclidean_distances(X_scaled, centroids)
    df['cluster_distance'] = [distances[i, c] for i, c in enumerate(df['kmeans_cluster'])]
    df['norm_dist'] = df.groupby('kmeans_cluster')['cluster_distance'].transform(lambda x: (x.max() - x) / (x.max() - x.min() + 1e-9))
    
    # Define scoring ranges (these are based on the analysis in your notebook)
    driver_score_map = {1: (8.5, 10.0), 2: (6.5, 8.4), 4: (1.0, 6.4)}
    merchant_score_map = {3: (8.0, 10.0), 0: (5.0, 7.9)}
    
    def assign_novascore_math(row):
        cluster = row['kmeans_cluster']
        score_map = driver_score_map if row['partner_type'] == 'Driver' else merchant_score_map
        low, high = score_map.get(cluster, (0, 0))
        return round(low + row['norm_dist'] * (high - low), 2)

    df['NovaScore'] = df.apply(assign_novascore_math, axis=1)
    return df

def run_bias_mitigation(df, threshold):
    """
    Performs bias audit and mitigation.
    Based on Blocks 6, 7, and 8 from main.ipynb.
    """
    df['is_creditworthy'] = (df['NovaScore'] >= threshold).astype(int)
    
    # Combine gender and location for intersectional analysis
    df['gender_location'] = df['gender'] + "_" + df['geographic_location']
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['is_creditworthy'])
    
    # Prepare data for ThresholdOptimizer
    X_train = train_df[['NovaScore']]
    y_train_true = train_df['is_creditworthy']
    sensitive_features_train = train_df['gender_location']
    
    X_test = test_df[['NovaScore']]
    y_test_true = test_df['is_creditworthy']
    sensitive_features_test = test_df['gender_location']

    # Train a simple base model
    base_estimator = CalibratedClassifierCV(LogisticRegression(max_iter=500, solver='liblinear'), method='sigmoid', cv=5)
    base_estimator.fit(X_train, y_train_true)
    
    # Get original (biased) predictions
    original_pred = base_estimator.predict(X_test)
    
    # Train and apply the ThresholdOptimizer
    optimizer = ThresholdOptimizer(estimator=base_estimator, constraints="demographic_parity", prefit=True)
    optimizer.fit(X_train, y_train_true, sensitive_features=sensitive_features_train)
    mitigated_pred = optimizer.predict(X_test, sensitive_features=sensitive_features_test)
    
    return y_test_true, sensitive_features_test, original_pred, mitigated_pred

def plot_mitigation_impact(original_pred, mitigated_pred):
    """
    Visualizes the partner movement matrix.
    """
    cm = confusion_matrix(original_pred, mitigated_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['After: Not Creditworthy', 'After: Creditworthy'],
                yticklabels=['Before: Not Creditworthy', 'Before: Creditworthy'])
    ax.set_title('Partner Movement After Bias Mitigation', fontsize=16)
    ax.set_ylabel('Original Biased Prediction', fontsize=12)
    ax.set_xlabel('New Mitigated Prediction', fontsize=12)
    return fig

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("⚖️ Project-Nova: A Fair Credit Scoring Model")
st.write("This app demonstrates building a fairer, more inclusive credit scoring model for gig economy workers using alternative data while actively mitigating biases.")

# Load or generate data
try:
    df = pd.read_csv('nova_custom_biased_dataset.csv')
except FileNotFoundError:
    with st.spinner('Biased dataset not found. Generating a new one... (this may take a moment)'):
        df = generate_data()
    st.success("New biased dataset generated successfully!")

# Run segmentation and scoring
scored_df = run_segmentation_and_scoring(df.copy())

# --- Main App Layout ---
st.sidebar.header("Controls")
credit_threshold = st.sidebar.slider("Creditworthiness Threshold", min_value=1.0, max_value=10.0, value=5.5, step=0.1)

st.header("Phase 1: Unsupervised Partner Segmentation")
with st.expander("See Segmentation Details", expanded=False):
    st.write("We use K-Means clustering to segment partners into different groups based on their performance metrics after reducing dimensionality with PCA.")
    st.image('ref/SystemDesign.png', caption='System Design Diagram', use_column_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Optimal k for Drivers")
        st.image('ref/elbow_plot_drivers.png', use_column_width=True)
        st.image('ref/silhouette_plot_drivers.png', use_column_width=True)
    with col2:
        st.subheader("Optimal k for Merchants")
        st.image('ref/elbow_plot_merchants.png', use_column_width=True)
        st.image('ref/silhouette_plot_merchants.png', use_column_width=True)

st.subheader("Generated NovaScore")
st.write(f"Based on the clustering, a `NovaScore` is assigned. Partners with a score above the threshold of **{credit_threshold}** are considered 'Creditworthy'.")
st.dataframe(scored_df[['partner_type', 'gender', 'geographic_location', 'kmeans_cluster', 'NovaScore']].head())


st.header("Phase 2: Scoring Audit and Bias Mitigation")
# Run mitigation logic
y_true, sensitive_features, original_pred, mitigated_pred = run_bias_mitigation(scored_df, credit_threshold)

# --- Fairness Audit Section ---
st.subheader("Fairness Audit")
st.write("We audit the initial predictions for fairness based on demographic parity (equal selection rates across groups). A large disparity indicates bias.")

metrics = {"selection_rate": lambda y_true, y_pred: y_pred.mean()}
# Before Mitigation
before_audit = MetricFrame(metrics=metrics, y_true=y_true, y_pred=original_pred, sensitive_features=sensitive_features)
# After Mitigation
after_audit = MetricFrame(metrics=metrics, y_true=y_true, y_pred=mitigated_pred, sensitive_features=sensitive_features)

col1, col2 = st.columns(2)
with col1:
    st.markdown("##### Before Mitigation")
    st.dataframe(before_audit.by_group)
    st.metric(label="Disparity (Difference in Selection Rate)", value=f"{before_audit.difference()['selection_rate']:.4f}")

with col2:
    st.markdown("##### After Mitigation")
    st.dataframe(after_audit.by_group)
    st.metric(label="Disparity (Difference in Selection Rate)", value=f"{after_audit.difference()['selection_rate']:.4f}", delta=f"{after_audit.difference()['selection_rate'] - before_audit.difference()['selection_rate']:.4f}")

st.success(f"Bias mitigation reduced the disparity in selection rates from {before_audit.difference()['selection_rate']:.4f} to **{after_audit.difference()['selection_rate']:.4f}**.")


# --- Impact Visualization ---
st.subheader("Impact of Mitigation")
st.write("This matrix shows how many partners had their creditworthiness status changed by the mitigation algorithm to ensure a fairer outcome.")
impact_fig = plot_mitigation_impact(original_pred, mitigated_pred)
st.pyplot(impact_fig)

upgraded_count = confusion_matrix(original_pred, mitigated_pred)[0, 1]
downgraded_count = confusion_matrix(original_pred, mitigated_pred)[1, 0]
st.info(f"**Analysis:** {upgraded_count} partners were upgraded from 'Not Creditworthy' to 'Creditworthy', while {downgraded_count} were downgraded to achieve fairness.")