from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical, plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2, l1_l2
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


def main():
    dataset = pd.read_csv('./data/data.csv')

    X = dataset.iloc[:,2:32] # [all rows, col from index 2 to the last one excluding 'Unnamed: 32']
    y = dataset.iloc[:,1] # [all rows, col one only which contains the classes of cancer]

    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    # Tranform training labels to one-hot encoding
    y_train = np_utils.to_categorical(y_train, 2)

    # Tranform test labels to one-hot encoding
    y_test = np_utils.to_categorical(y_test, 2)

    model = Sequential()

    classes = 2
    hidden_layers = 2
    neurons = [16] * hidden_layers
    print(neurons)
    epochs = 1000

    # Camada de entrada
    model.add(Dense(units=8, activation='sigmoid', input_dim=30))

    # model.add(Dropout(0.2))
    # Camadas escondidas
    for i in range(hidden_layers):
        model.add(Dense(units=neurons[i], activation='sigmoid'))

    # Camada de saída\
    model.add(Dense(units=classes, activation='softmax'))

    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(X_train, y_train,epochs=epochs, batch_size = 20, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)

    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')

    print(f'Train loss: {history.history["loss"][-1]:.3}')
    print(f'Train accuracy: {history.history["acc"][-1]:.3}')

    print_acc_results(history)
    print_loss_results(history)
    print_confusion_matrix(model, X_test, y_test)


def print_acc_results(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Teste'], loc='best')
    plt.grid()
    plt.show()

def print_loss_results(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Entropia Cruzada')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Teste'], loc='best')
    plt.grid()
    plt.show()

def print_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    df_cm = pd.DataFrame(matrix, index = [i for i in "BM"],
                    columns = [i for i in "BM"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    

if __name__ == "__main__":
    main()