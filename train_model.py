import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



df = pd.read('Dataset.csv')

#Convert dates to datetime
df['ScheduleDay'] = pd.to_datetime(df['ScheduleDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])


#Create DaysuntilAppointment feature
df['DaysUntilAppointment'] = df['Appointmentday'] - df['Scheduleday']

#Encode categorical variables
df['Gender'] = df['Gender'].map({'F':0 , 'M':1})

#Encoding target
df['No-show'] = df['No-show'].map({'No':0, 'Yes':1})


#define features and target
x= df.drop(columns=['No-show'], axis=1)
y = df['No-show']

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size= 0.2, random_state=42 )

#train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

#Evaluate model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'RandomForest Model Accuracy: {accuracy*100:.2f}%')

#Save model
joblib.dump(model, 'no_show_model.pkl')
print("Model saved as no_show_model.pkl")