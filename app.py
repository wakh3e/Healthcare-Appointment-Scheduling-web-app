"""
Hospital Appointment No-Show Prediction System
Main Flask application with ML model integration and time series analysis
"""

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'hospital_appointment_system_2023'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Global variables for model and scaler
model = None
deep_scaler = None

def train_ml_model():
    """
    Train ML model using the provided dataset
    This replaces the sample model creation with actual training
    """
    try:
        print("Training ML model from dataset...")
        
        # Try to load the dataset
        if os.path.exists('Dataset.csv'):
            df = pd.read_csv('Dataset.csv')
            print(f"Dataset loaded successfully with {len(df)} records")
        else:
            print("Dataset.csv not found. Using sample data for training...")
            return create_sample_model()
        
        # Data preprocessing similar to train_model.py
        # Convert dates to datetime (fixing typo from original train_model.py)
        df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
        df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
        
        # Create DaysUntilAppointment feature (fixing typo)
        df['DaysUntilAppointment'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
        
        # Ensure DaysUntilAppointment is non-negative
        df['DaysUntilAppointment'] = df['DaysUntilAppointment'].clip(lower=0)
        
        # Encode categorical variables
        df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
        
        # Encoding target (handle both string and numeric)
        if df['No-show'].dtype == 'object':
            df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
        
        # Select features for training (using same features as prediction form)
        features = ['Gender', 'Age', 'SMS_received', 'DaysUntilAppointment']
        
        # Check if all required features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}. Using available features.")
            features = [f for f in features if f in df.columns]
        
        X = df[features]
        y = df['No-show']
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully with accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))
        
        # Save model and scaler
        joblib.dump(rf_model, 'no_show_model.pkl')
        joblib.dump(scaler, 'deep_scaler.pkl')
        
        print("Model and scaler saved successfully")
        return rf_model, scaler
        
    except Exception as e:
        print(f"Error training model: {e}")
        print("Falling back to sample model...")
        return create_sample_model()

def create_sample_model():
    """
    Create a sample ML model if dataset is not available
    """
    print("Creating sample ML model with synthetic data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Gender': np.random.choice([0, 1], n_samples, p=[0.55, 0.45]),
        'Age': np.concatenate([
            np.random.randint(0, 18, 200),
            np.random.randint(18, 40, 300),
            np.random.randint(40, 60, 300),
            np.random.randint(60, 100, 200)
        ]),
        'SMS_received': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'DaysUntilAppointment': np.random.randint(0, 30, n_samples),
        'No-show': np.zeros(n_samples)
    }
    
    # Realistic no-show probabilities
    for i in range(n_samples):
        base_prob = 0.2
        if data['Age'][i] < 18: base_prob += 0.1
        if data['Age'][i] > 65: base_prob += 0.05
        if data['SMS_received'][i] == 0: base_prob += 0.15
        if data['DaysUntilAppointment'][i] > 7: base_prob += 0.1
        data['No-show'][i] = np.random.binomial(1, min(base_prob, 0.6))
    
    df = pd.DataFrame(data)
    X = df[['Gender', 'Age', 'SMS_received', 'DaysUntilAppointment']]
    y = df['No-show']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    joblib.dump(rf_model, 'no_show_model.pkl')
    joblib.dump(scaler, 'deep_scaler.pkl')
    
    print("Sample model created successfully")
    return rf_model, scaler

def perform_time_series_analysis():
    """
    Perform time series analysis similar to time_series_analysis.py
    """
    try:
        if os.path.exists('Dataset.csv'):
            df = pd.read_csv('Dataset.csv')
            print("Performing time series analysis on dataset...")
        else:
            df = load_or_create_data()
            if 'No-show' in df.columns and df['No-show'].dtype == 'object':
                df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
            print("Performing time series analysis on sample data...")
        
        # Convert appointment day to datetime
        df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
        
        # Extract weekday and month
        df['Weekday'] = df['AppointmentDay'].dt.day_name()
        df['Month'] = df['AppointmentDay'].dt.month_name()
        
        # Create plots
        plots = {}
        
        # No-show counts by weekday
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Weekday', hue='No-show', palette=['#27ae60', '#e74c3c'])
        plt.title('No-shows by Weekday', fontsize=14, fontweight='bold')
        plt.xlabel('Weekday')
        plt.ylabel('Number of Appointments')
        plt.legend(title='Attendance', labels=['Attended', 'No-show'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        img_buf.seek(0)
        plots['weekday'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close()

        # No-show counts by month
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='Month', hue='No-show', palette=['#27ae60', '#e74c3c'], 
                     order=month_order)
        plt.title('No-shows by Month', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Number of Appointments')
        plt.legend(title='Attendance', labels=['Attended', 'No-show'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        img_buf.seek(0)
        plots['month'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Save summary
        summary = df.groupby(['Month', 'Weekday'])['No-show'].mean().reset_index()
        summary.to_csv('no_show_summary.csv', index=False)
        
        print("Time series analysis completed successfully")
        return plots
        
    except Exception as e:
        print(f"Error in time series analysis: {e}")
        return {}

def load_or_create_data():
    """
    Load existing appointment data or create realistic sample data
    """
    try:
        if os.path.exists('appointments.csv'):
            df = pd.read_csv('appointments.csv')
            print("Loaded existing appointments data")
        else:
            # Create comprehensive sample data
            np.random.seed(42)
            n_records = 500
            
            base_date = datetime.now()
            scheduled_dates = [base_date - timedelta(days=np.random.randint(1, 60)) for _ in range(n_records)]
            appointment_dates = [sd + timedelta(days=np.random.randint(1, 30)) for sd in scheduled_dates]
            
            data = {
                'PatientId': np.random.randint(1000, 9999, n_records),
                'AppointmentID': range(5001, 5001 + n_records),
                'Gender': np.random.choice(['M', 'F'], n_records, p=[0.45, 0.55]),
                'ScheduledDay': [sd.strftime('%Y-%m-%d %H:%M:%S') for sd in scheduled_dates],
                'AppointmentDay': [ad.strftime('%Y-%m-%d %H:%M:%S') for ad in appointment_dates],
                'Age': np.concatenate([
                    np.random.randint(0, 18, 100),
                    np.random.randint(18, 40, 150),
                    np.random.randint(40, 60, 150),
                    np.random.randint(60, 100, 100)
                ]),
                'SMS_received': np.random.choice([0, 1], n_records, p=[0.35, 0.65]),
                'No-show': np.random.choice(['No', 'Yes'], n_records, p=[0.75, 0.25])
            }
            
            df = pd.DataFrame(data)
            df.to_csv('appointments.csv', index=False)
            print("Created sample appointments data")
        
        # Convert date columns and calculate days until appointment
        df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
        df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
        df['DaysUntilAppointment'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
        df['DaysUntilAppointment'] = df['DaysUntilAppointment'].clip(lower=0)
        
        return df
        
    except Exception as e:
        print(f"Error loading/creating data: {e}")
        return pd.DataFrame()

# Initialize model and data on startup
@app.before_first_request
def initialize_app():
    """Initialize ML model and data when app starts"""
    global model, deep_scaler
    
    try:
        model = joblib.load('no_show_model.pkl')
        deep_scaler = joblib.load('deep_scaler.pkl')
        print("ML model and scaler loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}. Training new model...")
        model, deep_scaler = train_ml_model()

@app.route('/')
def index():
    """Home page with system overview"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard with visualizations including time series analysis"""
    df = load_or_create_data()
    
    if df.empty:
        return render_template('dashboard.html', 
                             error="No data available",
                             total_appointments=0,
                             no_show_count=0,
                             no_show_rate=0,
                             plots={})

    # Generate standard visualizations
    plots = generate_visualizations(df)
    
    # Add time series analysis plots
    ts_plots = perform_time_series_analysis()
    plots.update(ts_plots)
    
    # Calculate key metrics
    total_appointments = len(df)
    no_show_count = (df['No-show'] == 'Yes').sum() if df['No-show'].dtype == 'object' else df['No-show'].sum()
    no_show_rate = (no_show_count / total_appointments) * 100 if total_appointments > 0 else 0
    
    return render_template('dashboard.html',
                         plots=plots,
                         total_appointments=total_appointments,
                         no_show_count=no_show_count,
                         no_show_rate=round(no_show_rate, 2))

def generate_visualizations(df):
    """Generate base64 encoded plots for dashboard"""
    plots = {}
    
    try:
        # Plot 1: No-show by Gender
        plt.figure(figsize=(10, 6))
        gender_plot = sns.countplot(data=df, x='Gender', hue='No-show', palette=['#27ae60', '#e74c3c'])
        plt.title('Appointment Attendance by Gender', fontsize=14, fontweight='bold')
        plt.xlabel('Gender')
        plt.ylabel('Number of Appointments')
        plt.legend(title='Attendance', labels=['Attended', 'No-show'])
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        img_buf.seek(0)
        plots['gender'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close()

        # Plot 2: No-show by Age Group
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 70, 100], 
                               labels=['0-18', '19-30', '31-50', '51-70', '71+'])
        plt.figure(figsize=(12, 6))
        age_plot = sns.countplot(data=df, x='AgeGroup', hue='No-show', palette=['#27ae60', '#e74c3c'])
        plt.title('Appointment Attendance by Age Group', fontsize=14, fontweight='bold')
        plt.xlabel('Age Group')
        plt.ylabel('Number of Appointments')
        plt.legend(title='Attendance', labels=['Attended', 'No-show'])
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        img_buf.seek(0)
        plots['age'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close()

        # Plot 3: No-show by SMS Notification
        plt.figure(figsize=(10, 6))
        sms_plot = sns.countplot(data=df, x='SMS_received', hue='No-show', palette=['#27ae60', '#e74c3c'])
        plt.title('Appointment Attendance by SMS Notification', fontsize=14, fontweight='bold')
        plt.xlabel('SMS Received')
        plt.ylabel('Number of Appointments')
        plt.xticks([0, 1], ['No SMS', 'SMS Received'])
        plt.legend(title='Attendance', labels=['Attended', 'No-show'])
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        img_buf.seek(0)
        plots['sms'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close()

        # Plot 4: No-show rate by waiting time
        plt.figure(figsize=(12, 6))
        wait_data = df.groupby('DaysUntilAppointment')['No-show'].apply(
            lambda x: (x == 'Yes').mean() * 100 if x.dtype == 'object' else x.mean() * 100
        ).reset_index()
        wait_plot = sns.lineplot(data=wait_data, x='DaysUntilAppointment', y='No-show')
        plt.title('No-show Rate by Days Until Appointment', fontsize=14, fontweight='bold')
        plt.xlabel('Days Until Appointment')
        plt.ylabel('No-show Rate (%)')
        plt.grid(True, alpha=0.3)
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        img_buf.seek(0)
        plots['wait_time'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close()
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    return plots

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """No-show prediction page with form"""
    if request.method == 'POST':
        if model is None or deep_scaler is None:
            return render_template('predict.html', 
                                 error="Prediction model not available. Please try again later.")
        
        try:
            # Validate and extract form data
            gender = 1 if request.form.get('gender') == 'Male' else 0
            age = int(request.form.get('age', 0))
            sms_received = 1 if request.form.get('sms_received') == 'Yes' else 0
            days_until = int(request.form.get('days_until', 0))
            
            # Validate inputs
            if not (0 <= age <= 120):
                return render_template('predict.html', error="Please enter a valid age (0-120)")
            if days_until < 0:
                return render_template('predict.html', error="Days until appointment cannot be negative")
            
            # Prepare features and make prediction
            features = np.array([[gender, age, sms_received, days_until]])
            
            # Scale features if scaler is available
            if deep_scaler is not None:
                features_scaled = deep_scaler.transform(features)
            else:
                features_scaled = features
            
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            result = {
                'prediction': 'No-show' if prediction == 1 else 'Will Attend',
                'probability': round(probability * 100, 2),
                'confidence': 'high' if probability > 0.7 else 'medium' if probability > 0.4 else 'low',
                'gender': 'Male' if gender == 1 else 'Female',
                'age': age,
                'sms_received': 'Yes' if sms_received == 1 else 'No',
                'days_until': days_until
            }
            
            return render_template('predict.html', result=result)
            
        except Exception as e:
            return render_template('predict.html', error=f"Error processing request: {str(e)}")
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for predictions"""
    if model is None or deep_scaler is None:
        return jsonify({'error': 'Prediction model not available'}), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        gender = 1 if data.get('gender', '').capitalize() == 'Male' else 0
        age = int(data.get('age', 0))
        sms_received = 1 if data.get('sms_received', 'No') == 'Yes' else 0
        days_until = int(data.get('days_until', 0))
        
        # Validate inputs
        if not (0 <= age <= 120):
            return jsonify({'error': 'Age must be between 0 and 120'}), 400
        if days_until < 0:
            return jsonify({'error': 'Days until appointment cannot be negative'}), 400
        
        features = np.array([[gender, age, sms_received, days_until]])
        
        if deep_scaler is not None:
            features_scaled = deep_scaler.transform(features)
        else:
            features_scaled = features
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': round(probability * 100, 2),
            'label': 'No-show' if prediction == 1 else 'Will Attend',
            'risk_level': 'high' if probability > 0.7 else 'medium' if probability > 0.4 else 'low'
        })
        
    except Exception as e:
        return jsonify({'error': f'Invalid request: {str(e)}'}), 400

@app.route('/appointments')
def appointments():
    """Display all appointments"""
    df = load_or_create_data()
    
    if df.empty:
        return render_template('appointments.html', appointments=[])
    
    # Convert to list of dictionaries for template
    appointments_data = []
    for _, row in df.iterrows():
        appt_data = {
            'PatientId': row['PatientId'],
            'AppointmentID': row['AppointmentID'],
            'Gender': row['Gender'],
            'ScheduledDay': row['ScheduledDay'].strftime('%Y-%m-%d %H:%M'),
            'AppointmentDay': row['AppointmentDay'].strftime('%Y-%m-%d %H:%M'),
            'Age': row['Age'],
            'SMS_received': row['SMS_received'],
            'No-show': row['No-show'],
            'DaysUntilAppointment': row['DaysUntilAppointment']
        }
        appointments_data.append(appt_data)
    
    return render_template('appointments.html', appointments=appointments_data)

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        'status': 'healthy' if model and deep_scaler else 'degraded',
        'model_loaded': model is not None,
        'scaler_loaded': deep_scaler is not None,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Endpoint to retrain the model with current data"""
    try:
        global model, deep_scaler
        model, deep_scaler = train_ml_model()
        return jsonify({'status': 'success', 'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Prevent caching of images
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    print("🚀 Starting Hospital Appointment System...")
    print("📊 Loading ML models and data...")
    initialize_app()
    print("✅ System ready! Starting web server...")
    app.run(debug=True, host='0.0.0.0', port=5000)