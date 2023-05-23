import pandas as pd
df = pd.read_csv('data.csv')

df['recoveredPopulation'] = df['recovered'].apply(lambda x: 0 if x == 0 else 1)
df['symptom_date'] = pd.to_datetime(df['symptom_onset'], format='%m/%d/%Y', errors='coerce')
df['recovery_date'] = pd.to_datetime(df['recovered'], format='%m/%d/%Y', errors='coerce')

recovered_wuhan_df = df[(df['recoveredPopulation'] ) & 
                        (df['visiting Wuhan'] == 1) & 
                        (df['recovery_date'].notnull() ) & 
                         (df['symptom_date'].notnull() )]

recovered_wuhan_df['recovery_interval'] = ( recovered_wuhan_df['recovery_date'] - recovered_wuhan_df['symptom_date']).dt.days
num_recovered_wuhan = recovered_wuhan_df.shape[0]
print(f'the number of patients who have recovered and visited Wuhan is {num_recovered_wuhan}')

recovered_wuhan_df = recovered_wuhan_df[recovered_wuhan_df['recovery_interval'] >= 0]
average_recovery_interval = recovered_wuhan_df['recovery_interval'].mean()
print(f'the average recovery interval for patients who visited Wuhan is {average_recovery_interval} days')
