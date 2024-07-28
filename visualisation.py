import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_prep import calculate_val,coefficient
def visual_plot(x_values,orignal_data,predicted_values):
    plt.plot(x_values, orignal_data, marker='o',label='Orignal_Revenue')
    plt.plot(x_values, predicted_values, marker='x', label='predicted_values')
    plt.title('Revenue orignal and predicted Over Time')
    # plt.ylim(0,10**10)
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.legend()
    plt.show()
def share_effect(data_new,target,orignal_data):
    effect = coefficient(data_new, target).iloc[:5, :]

    # Calculate the sum of the values in the second column (from data_new)
    total_sum_new = effect.iloc[:, 1].sum()

    # Calculate individual contributions as percentages for data_new
    effect['Contribution (%) (effect)'] = (effect.iloc[:, 1] / total_sum_new) * 100

    # Calculate individual contributions as percentages for original_data (first 5 entries)
    total_sum_original = sum(orignal_data[:5])  # Assuming original_data is a list of numeric values
    effect['Contribution (%) (spend)'] = [(val / total_sum_original) * 100 for val in orignal_data[:5]]

    # Drop the original numerical values columns if not needed anymore
    effect = effect.drop(effect.columns[1], axis=1)

    # Plotting the bar chart
    bar_width = 0.35
    index = range(len(effect))
    plt.bar(index, effect['Contribution (%) (effect)'], width=bar_width, color='skyblue', label='Contribution (%) (effect)')
    plt.bar([i + bar_width for i in index], effect['Contribution (%) (spend)'], width=bar_width, color='orange', label='Contribution (%) (spend)')
    plt.xlabel('Variables')  # Adjust label as per your data
    plt.ylabel('Contribution (%)')
    plt.title('share effect vs revenue')  # Adjust title as needed
    plt.xticks(index, effect.iloc[:, 0], rotation=45)  # Rotate x-axis labels if necessary for better visibility
    plt.legend()
    plt.tight_layout()
    plt.show()
def decay_effect(y):
    x_values=[i for i in range(0,len(y))]
    plt.plot(x_values,y,label='Decay Effect')
    plt.title('Decay-Effect')
    plt.yscale('log')
    plt.xlabel('Week')
    plt.ylabel('% carryover')
    plt.legend()
    plt.show()

def area_chart(df,x):
    plt.figure(figsize=(14, 7))
    plt.stackplot(x , df['GOOGLE_SPEND'], df['META_FACEBOOK_SPEND'],
                  df['META_INSTAGRAM_SPEND'], df['META_OTHER_SPEND'], df['TIKTOK_SPEND'],df['Base'],
                  labels=['Base', 'Google Spend', 'Facebook Spend', 'Instagram Spend',
                          'Other Meta Spend', 'TikTok Spend'],
                  colors=['blue', 'orange', 'green', 'red', 'purple', 'cyan'])

    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Effect on Sales')
    plt.show()
