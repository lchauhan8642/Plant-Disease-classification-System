#Code Reference: https://www.kaggle.com/srikanthreddyt/plant-disease-classification-with-pretrained-vgg16

import pickle
from matplotlib import pyplot
import seaborn as sns
sns.set()


def summarize_diagnostics(history):
    sns.set()

    pyplot.title('Training and Validation Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='valid')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
   
    pyplot.figure()
    pyplot.title('Training and Validation Accuracy')
    pyplot.plot(history.history['acc'], color='blue', label='train')
    pyplot.plot(history.history['val_acc'], color='orange', label='valid')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    pyplot.show()
    
 


if __name__ == "__main__":
        
    with open(".\\VGG16\\vgg16_train_history.pkl", 'rb') as f:
        history = pickle.load(f)
    
    summarize_diagnostics(history)
    
