import json
import random
import tensorflow as tf

FILE_PATH = "dataset.json"

def main():
    input_lst = load_data(FILE_PATH)

    X, Y = list(), list()
    for row in input_lst:
        Y.append([row.pop()])
        X.append(row)

    X_test, Y_test = list(), list()
    # Para tirar 1 trem a oeste e 2 trens a leste, descomente a linha abaixo
    test = [random.randint(0, 4), random.randint(0, 3), random.randint(3, 7)]

    # Para tirar 2 trens a oeste e 1 trem a leste, descomente a linha abaixo
    # test = [random.randint(0, 4), random.randint(4, 8), random.randint(4, 7)]

    for i in test:
        X_test.append(X.pop(i))
        Y_test.append(Y.pop(i))

    model = model_builder(len(X[0]))
    model.fit(X, Y, epochs=10000, batch_size=1)
    _, accuracy = model.evaluate(X_test, Y_test)

    print(f"Accuracy: {accuracy * 100:.2f}%")


def load_data(path):
    data = list()
    with open(path, "r") as file:
        json_data = json.load(file)
        for train in json_data.values():
            train_list = list()
            for name, attr in train.items():
                if name == "cars":
                    for car in attr.values():
                        for car_attr in car.values():
                            train_list.append(car_attr)
                else:
                    train_list.append(attr)
            data.append(train_list)
    return data


def model_builder(input_layer_len):
    # define o modelo
    model = tf.keras.Sequential([

        # Tensor de Input
        tf.keras.Input(shape=(input_layer_len)),

        # Camada oculta
        tf.keras.layers.Dense(input_layer_len, activation="linear"),

        # Camada de Output
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    return model

if __name__ == "__main__":
    main()