import pandas as pd

df = pd.DataFrame(columns=['hex0', 'luv0', 'hex1', 'luv1', 'hex2', 'luv2', 'hex3', 'luv3', 'hex4', 'luv4'])
df.to_csv('/home/artificialcolour/mysite/files/user_data.csv', index=False)
