# Code Reference: https://www.kaggle.com/vipoooool/plant-diseases-classification-using-alexnet

import pickle
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


with open(".\\ResNet50\\ResNet50_model_history.pkl", 'rb') as f:
    history = pickle.load(f)
    
print('\n\n')
print(history.history.keys())
print('\n\n')


# dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])
    
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

