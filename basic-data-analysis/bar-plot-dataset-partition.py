import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

# Load data
train = pd.read_csv(r"")
test = pd.read_csv(r"")

# Process training data
train['labels'] = train['labels'].apply(lambda string: string.split(' '))
mlb = MultiLabelBinarizer()
trainx = pd.DataFrame(mlb.fit_transform(train['labels']), columns=mlb.classes_, index=train.index)
train_label_counts = trainx.sum()

# Process testing data
test['labels'] = test['labels'].apply(lambda string: string.split(' '))
testx = pd.DataFrame(mlb.transform(test['labels']), columns=mlb.classes_, index=test.index)
test_label_counts = testx.sum()

# Combine label counts into a single DataFrame
combined_counts = pd.DataFrame({'Train': train_label_counts, 'Test': test_label_counts}).reset_index()
combined_counts = combined_counts.rename(columns={'index': 'Label'})

# Plot the combined data
plt.rc('font', size=28)
fig, ax = plt.subplots(1, 1, figsize=(22, 8))
bar_width = 0.35
index = np.arange(len(combined_counts))

bar1 = plt.bar(index, combined_counts['Train'], bar_width, label='Train', color='darkblue')
bar2 = plt.bar(index + bar_width, combined_counts['Test'], bar_width, label='Test', color='lightblue')

ax.set_title('Label Counts for Training and Testing Sets', fontsize=28, font='Arial', weight='bold')
ax.set_xlabel('Label', fontsize=22, font='Arial', weight='bold')
ax.set_ylabel('Count', fontsize=22, font='Arial', weight='bold')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(combined_counts['Label'], rotation=90)
ax.legend()

# Show the plot
plt.show()
