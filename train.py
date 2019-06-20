import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

class VideoClassifierLSTM:
    def __init__(self):
        self.batch_size = 128
        self.epochs = 50


    def load_data(self,
                  path,
                  ext='avi',
                  c='rgb',
                  X_npy_path=None,
                  y_npy_path=None):
        X = []
        y = []
        video_num = 0
        color = cv2.COLOR_BGR2GRAY if c == 'gray' or 'g' else cv2.COLOR_BGR2RGB

        if (X_npy_path is not None and y_npy_path is not None):
            assert(os.path.exists(X_npy_path) and os.path.exists(y_npy_path),
                   "There is no X.npy or y.npy file in given path.")
            if (os.path.exists(X_npy_path) and os.path.exists(y_npy_path)):
                X = np.load(X_npy_path)
                y = np.load(y_npy_path)
                return X, y

        for root, dir, files in os.walk(path):
            label = os.path.split(root)[-1]
            print(f"`{label}` class is in progress..")
            for file in tqdm(files):
                if file.endswith(ext):
                    video_num += 1
                    capture = cv2.VideoCapture(os.path.join(root, file))

                    while True:
                        ret, frame = capture.read()

                        if not ret:
                            break

                        img_gray = cv2.cvtColor(frame, color)
                        resized_frame = cv2.resize(img_gray, 
                                                   (224, 224), 
                                                   interpolation = cv2.INTER_AREA)
                        X.append(resized_frame)
                        y.append(label)
        print(f"Total {len(y)} frame(s) out of {video_num} video(s) were loaded.")
        np.save('X.npy', np.array(X))
        np.save('y.npy', np.array(y))

        return X, y

    def train_model(self, X_train, X_val, y_train, y_val, e = 1):
        for i in range(e):
            model = Sequential()
            model.add(LSTM(256, dropout=0.2, input_shape=(len(X_train[0]),
                                                          len(X_train[0][0]))))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1024, activation='relu')) 
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='softmax'))
            sgd = SGD(lr=0.00005, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(optimizer=sgd, 
                          loss='sparse_categorical_crossentropy', 
                          metrics=['accuracy'])
            
            if (os.path.exists('video_LSTM.h5')):
                model.load_weights('video_LSTM.h5')
                
            callbacks = [
                EarlyStopping(monitor='val_loss', 
                              patience=5, 
                              verbose=0),
                ModelCheckpoint('video_LSTM.h5', 
                                monitor='val_loss', 
                                save_best_only=True, 
                                verbose=0),
            ]
            model.fit(X_train, y_train,
                      validation_data=(X_val, y_val),
                      callbacks=callbacks,
                      shuffle=True,
                      batch_size = self.batch_size,
                      epochs= self.epochs,
                      verbose=1,
            )
        return model


if __name__ == "__main__":
    clf = VideoClassifierLSTM()
    X, y = clf.load_data('videos',
                       ext='avi',
                       c='gray',
                       X_npy_path='X.npy',
                       y_npy_path='y.npy')
    y = pd.Series(y)
    y = np.array(pd.get_dummies(y))[:, 1:] # Get dummies, but drop first column..
    X = np.array(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    model = clf.train_model(X_train, X_val, y_train, y_val, e=3)

