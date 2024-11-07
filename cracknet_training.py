import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Input, Model

from sklearn.metrics import confusion_matrix, classification_report


import warnings
warnings.filterwarnings('ignore')


positive_dir = Path('./dataset/Positive')
negative_dir = Path('./dataset/Negative')
BATCH = 64
IMSIZE = 120


def generate_df(image_dir, label):
    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)
    return df


positive_df = generate_df(positive_dir, label='POSITIVE')
negative_df = generate_df(negative_dir, label='NEGATIVE')

all_df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
print(all_df)

train_df, test_df = train_test_split(
    all_df.sample(6000, random_state=1),
    train_size=0.7,
    shuffle=True,
    random_state=1
)

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(IMSIZE, IMSIZE),
    color_mode='rgb',
    class_mode='binary',
    batch_size=BATCH,
    shuffle=True,
    seed=42,
    subset='training'
)

val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(IMSIZE, IMSIZE),
    color_mode='rgb',
    class_mode='binary',
    batch_size=BATCH,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_data = train_gen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(IMSIZE, IMSIZE),
    color_mode='rgb',
    class_mode='binary',
    batch_size=BATCH,
    shuffle=False,
    seed=42
)

inputs = Input(shape=(IMSIZE, IMSIZE, 3))

x = layers.Dense(units=64, activation='relu')(inputs)
x = layers.Dropout(rate=0.2)(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Dense(units=128, activation='relu')(x)
x = layers.Dropout(rate=0.2)(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Dense(units=64, activation='relu')(x)
x = layers.Dropout(rate=0.2)(x)

x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)

x = layers.GlobalAveragePooling2D()(x)

outputs = layers.Dense(units=1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)

fig = px.line(
    history.history,
    y=['loss', 'val_loss'],
    labels={'index': 'Epoch', 'value': 'Loss'},
    title='Training and Validation Loss Over Time'
)

fig.show()
model.save('cracknet_model.h5')


def evaluate_model(model, test_data):
    results = model.evaluate(test_data, verbose=0)
    loss = results[0]
    acc = results[1]

    print('Test Loss: {:.5f}'.format(loss))
    print('Test Accuracy: {:.2f}%'.format(acc * 100))

    y_pred = np.squeeze((model.predict(test_data) >= 0.5).astype('int'))
    cm = confusion_matrix(test_data.labels, y_pred)
    clr = classification_report(test_data.labels, y_pred, target_names=['NEGATIVE', 'POSITIVE'])

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=['NEGATIVE', 'POSITIVE'])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=['NEGATIVE', 'POSITIVE'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print('Classification Report:\n', clr)


evaluate_model(model, test_data)
