import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Dataset.csv')

#Map No-show to binary
df['No-show'] = df['No-show'].map({'No':0, 'Yes':1})

if 'ScheduledDay' in df.columns:
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
  
#convert appointment and schedule day to datetime
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

#Extract weekday and month
df['Weekday'] = df['AppointmentDay'].dt.day_name()
df['Month'] = df['AppointmentDay'].dt.month_name()

#plot no-show counts by weekday
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Weekday', hue='No-show')
plt.title('No-shows by weekday')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('no_shows_by_weekday.png')



#plot no-show counts by month
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Month', hue='No-show', order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
plt.title('No-shows by month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('no_shows_by_month.png')


#save summary csv
summary = df.groupby(['Month', 'Weekday'])['No-show'].mean().reset_index()
summary.to_csv('no_show_summary.csv', index=False)

print("time_series_analysis completed and plots saved.") 

