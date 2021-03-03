import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


df = pd.read_csv('C:/Users/Roy/Desktop/data/data_new.csv')

#print(df.head())

plt.plot(df['iou_new'])
plt.plot(df['val_iou_new'])
plt.title('Segmentation Coefficient : IoU')
plt.ylabel('iou')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()