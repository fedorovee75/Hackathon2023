import numpy, sklearn.model_selection, tensorflow, matplotlib, pandas
step = 4
max_len = 10000
data_raw = pandas.read_csv("/content/drive/MyDrive/Colab Notebooks/dsc_fc_summed_spectra_2022_v01.csv", delimiter = ',', parse_dates=[0], infer_datetime_format=True, na_values='0', header = None)
data = data_raw.iloc[0:max_len+step,1]
X, Y =[], []
for i in range(len(data)-step):
    X.append(data[i:i+step]); Y.append(data[i+step])
(x_train, x_test, y_train, y_test) = sklearn.model_selection.train_test_split(numpy.array(X), numpy.array(Y), test_size=0.2)
x_train_len = len(x_train)
x_test_len = len(x_test)
x_train = x_train.reshape(x_train_len, step, 1)
x_test = x_test.reshape(x_test_len, step, 1)

inputs = tensorflow.keras.Input(shape=(step, 1))
hidden = tensorflow.keras.layers.SimpleRNN(units=64)(inputs)
outputs = tensorflow.keras.layers.Dense(units=1)(hidden)
srn = tensorflow.keras.Model(inputs=inputs, outputs=outputs)
srn.compile(optimizer="Adam", loss="mean_squared_error")
srn.summary()

srn.fit(x=x_train, y=y_train, batch_size=64, epochs=10, validation_split=0.2, verbose=1, shuffle=True, initial_epoch=0, steps_per_epoch=None, validation_batch_size=64, validation_freq=1)
test_scores = srn.evaluate(x=x_test, y=y_test, batch_size=64, verbose=1, return_dict=False)
print("Test loss:", test_scores)

y_model = srn.predict(x=x_test)
fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(y_test)
ax.plot(y_model)
x=numpy.arange(0, x_test_len)
matplotlib.pyplot.plot(x, y_test, label="y_test", color="b")
matplotlib.pyplot.plot(x, y_model, label="y_model", color="r")
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
