import pandas as pd
import plotter


df = pd.read_csv('learningData/LEARNING_EXP_2_20220427-202222.csv')

plotter.plotArr(df['Rewards'], 'Average Rewards', '', '')
plotter.plotArr(df['WaitTime'], 'Average Wait Time', '', '')
plotter.plotArr(df['Steps'], 'Average Steps Per Episode', '', '')

