import pandas as pd
dataset = pd.read_csv('eestec_hackathon_2025_train.tsv' ,sep = '\t',names=['ID', 'Label', 'Statement', 'Subjects', 'Speaker Name', 'Speaker Title', 'State', 'Party Affiliation', 'Credit History: barely-true', 'Credit History: false', 'Credit History: half-true', 'Credit History: mostly-true', 'Credit History: pants-fire', 'Context/Location'])
