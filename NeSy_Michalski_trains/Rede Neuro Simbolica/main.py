import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

FILE_PATH = "trains-data.csv"

str_att = {
    "length": ["short", "long"],
    "shape": [
        "closedrect",
        "dblopnrect",
        "ellipse",
        "engine",
        "hexagon",
        "jaggedtop",
        "openrect",
        "opentrap",
        "slopetop",
        "ushaped",
    ],
    "load_shape": ["circlelod", "hexagonlod", "rectanglod", "trianglod"],
    "Class_attribute": ["west", "east"],
}

def main():
    df = read_data(FILE_PATH)
    cols = [[c for c in df.columns if not (str.isdigit(c[-1])) or (str.isdigit(c[-1]) and int(c[-1]) == n)] for n in range(1, 5)]
    data = [df[col] for col in cols]

    for c, df in enumerate(data):
        df.columns = [name if not str.isdigit(name[-1]) else name[:-1] for name in df.columns]
        df["car"] = c + 1

    data = pd.concat(data)
    data.reset_index(level=0, inplace=True)
    data["train"] = data.pop("index")
    data.T[0]
    
    hists = dict()
    question_2_names = ["Train", "Output of flat network", "Desired output", "Class"]
    question_2 = {name: [] for name in question_2_names}

    for validation_i in tqdm.tqdm(range(0, 10)):
        x_train, x_test, y_train, y_test = split(data, validation_i)
        metanet, east, rules = model_2()

        x_train = [np.asarray(x).astype(np.int32) for x in x_train]
        y_train = np.asarray(y_train).astype(np.int32)
        x_test = [np.asarray(x).astype(np.int32) for x in x_test]
        y_test = np.asarray(y_test).astype(np.int32)

        east.compile(loss="binary_crossentropy", optimizer="adam", metrics=["mse", "binary_accuracy"])
    
        hist = east.fit(x_train, y_train, validation_data=[x_test, y_test], epochs=2000, verbose=0)
        hists[validation_i] = hist

        question_2["Train"].append(validation_i)
        question_2["Output of flat network"].append(east.predict(x_test)[0][0])
        question_2["Desired output"].append(y_test[0])
        question_2["Class"].append(str_att["Class_attribute"][int(y_test[0])])

        t2 = pd.DataFrame.from_dict(question_2).round(2)
    t2["Cars Accuracy"] = [hists[i].history["binary_accuracy"][-1] for i in range(0, 10)]
    t2 = t2.reindex(columns=["Train", "Cars Accuracy", "Output of flat network", "Desired output", "Class"])
    print(t2)


def read_data(path):
    df = pd.read_csv(path)
    for k in df:
        for att in str_att:
            if k.startswith(att):
                for i, val in enumerate(df[k]):
                    if val in str_att[att]:
                        df[k][i] = str_att[att].index(val)

    df.replace("\\0", 0, inplace=True)
    df.replace("None", -1, inplace=True)

    return df


def model_2():
    # features
    t = tf.keras.Input(shape=(1,), name="t")
    c = tf.keras.Input(shape=(1,), name="c")
    w = tf.keras.Input(shape=(1,), name="w")
    l = tf.keras.Input(shape=(1,), name="l")
    s = tf.keras.Input(shape=(1,), name="s")
    nc = tf.keras.Input(shape=(1,), name="nc")
    ls = tf.keras.Input(shape=(1,), name="ls")
    nl = tf.keras.Input(shape=(1,), name="nl")
    ncl = tf.keras.Input(shape=(1,), name="ncl")
    x_crc = tf.keras.Input(shape=(1,), name="x_crc")
    x_hex = tf.keras.Input(shape=(1,), name="x_hex")
    x_rec = tf.keras.Input(shape=(1,), name="x_rec")
    x_tri = tf.keras.Input(shape=(1,), name="x_tri")

    # num_cars(t,nc)
    num_cars_ = tf.keras.layers.concatenate([t, nc])
    num_cars_ = tf.keras.layers.Dense(2, activation="linear")(num_cars_)
    num_cars = tf.keras.layers.Dense(1, activation="sigmoid", name="num_cars")(num_cars_)
    num_cars = tf.keras.Model(inputs=[t, nc], outputs=num_cars)

    # num_loads(t,nl)
    num_loads_ = tf.keras.layers.concatenate([t, nl])
    num_loads_ = tf.keras.layers.Dense(2, activation="linear")(num_loads_)
    num_loads = tf.keras.layers.Dense(1, activation="sigmoid", name="num_loads")(num_loads_)
    num_loads = tf.keras.Model(inputs=[t, nl], outputs=num_loads)

    # num_wheels(t,c,w)
    num_wheels_ = tf.keras.layers.concatenate([t, c, w])
    num_wheels_ = tf.keras.layers.Dense(3, activation="linear")(num_wheels_)
    num_wheels = tf.keras.layers.Dense(1, activation="sigmoid", name="num_wheels")(num_wheels_)
    num_wheels = tf.keras.Model(inputs=[t, c, w], outputs=num_wheels)

    # length(t,c,l)
    length_ = tf.keras.layers.concatenate([t, c, l])
    length_ = tf.keras.layers.Dense(3, activation="linear")(length_)
    length = tf.keras.layers.Dense(1, activation="sigmoid", name="length")(length_)
    length = tf.keras.Model(inputs=[t, c, l], outputs=length)

    # shape(t,c,s)
    shape_ = tf.keras.layers.concatenate([t, c, s])
    shape_ = tf.keras.layers.Dense(3, activation="linear")(shape_)
    shape = tf.keras.layers.Dense(1, activation="sigmoid", name="shape")(shape_)
    shape = tf.keras.Model(inputs=[t, c, s], outputs=shape)

    # num_car_loads(t,c,ncl)
    num_car_loads_ = tf.keras.layers.concatenate([t, c, ncl])
    num_car_loads_ = tf.keras.layers.Dense(3, activation="linear")(num_car_loads_)
    num_car_loads = tf.keras.layers.Dense(1, activation="sigmoid", name="num_car_loads")(num_car_loads_)
    num_car_loads = tf.keras.Model(inputs=[t, c, ncl], outputs=num_car_loads)

    # load_shape(t,c,ls)
    load_shape_ = tf.keras.layers.concatenate([t, c, ls])
    load_shape_ = tf.keras.layers.Dense(3, activation="linear")(load_shape_)
    load_shape = tf.keras.layers.Dense(1, activation="sigmoid", name="load_shape")(load_shape_)
    load_shape = tf.keras.Model(inputs=[t, c, ls], outputs=load_shape)

    # next_crc(t,c,x)
    next_crc_ = tf.keras.layers.concatenate([t, c, x_crc])
    next_crc_ = tf.keras.layers.Dense(3, activation="linear")(next_crc_)
    next_crc = tf.keras.layers.Dense(1, activation="sigmoid", name="next_crc")(next_crc_)
    next_crc = tf.keras.Model(inputs=[t, c, x_crc], outputs=next_crc)

    # next_hex_(t,c,x)
    next_hex_ = tf.keras.layers.concatenate([t, c, x_hex])
    next_hex_ = tf.keras.layers.Dense(3, activation="linear")(next_hex_)
    next_hex = tf.keras.layers.Dense(1, activation="sigmoid", name="next_hex")(next_hex_)
    next_hex = tf.keras.Model(inputs=[t, c, x_hex], outputs=next_hex)

    # next_rec(t,c,x)
    next_rec_ = tf.keras.layers.concatenate([t, c, x_rec])
    next_rec_ = tf.keras.layers.Dense(3, activation="linear")(next_rec_)
    next_rec = tf.keras.layers.Dense(1, activation="sigmoid", name="next_rec")(next_rec_)
    next_rec = tf.keras.Model(inputs=[t, c, x_rec], outputs=next_rec)

    # next_tri(t,c,x)
    next_tri_ = tf.keras.layers.concatenate([t, c, x_tri])
    next_tri_ = tf.keras.layers.Dense(3, activation="linear")(next_tri_)
    next_tri = tf.keras.layers.Dense(1, activation="sigmoid", name="next_tri")(next_tri_)
    next_tri = tf.keras.Model(inputs=[t, c, x_tri], outputs=next_tri)

    # east
    east = tf.keras.layers.concatenate(
        [
            num_cars_,
            num_loads_,
            num_wheels_,
            length_,
            shape_,
            num_car_loads_,
            load_shape_,
            next_crc_,
            next_hex_,
            next_rec_,
            next_tri_,
        ]
    )
    east = tf.keras.layers.Dense(3, activation="linear")(east)
    east = tf.keras.layers.Dense(1, activation="sigmoid", name="east")(east)
    east = tf.keras.Model(
        inputs=[t, c, w, l, s, nc, ls, nl, ncl, x_crc, x_hex, x_rec, x_tri],
        outputs=east,
    )

    # metanet
    metanet = tf.keras.Model(
        inputs=east.inputs,
        outputs=[
            num_cars.output,
            num_loads.output,
            num_wheels.output,
            length.output,
            shape.output,
            num_car_loads.output,
            load_shape.output,
            next_crc.output,
            next_hex.output,
            next_rec.output,
            next_tri.output,
            east.output,
        ],
    )

    # rules
    rules = {
        "num_cars": num_cars,
        "num_loads": num_loads,
        "num_wheels": num_wheels,
        "length": length,
        "shape": shape,
        "num_car_loads": num_car_loads,
        "load_shape": load_shape,
        "next_crc": next_crc,
        "next_hex": next_hex,
        "next_rec": next_rec,
        "next_tri": next_tri,
    }

    return metanet, east, rules


def sort_inputs(X):
    t = X["train"].T
    c = X["car"].T
    w = X["num_wheels"].T
    l = X["length"].T
    s = X["shape"].T
    nc = X["Number_of_cars"].T
    ls = X["load_shape"].T
    nl = X["Number_of_different_loads"].T
    ncl = X["num_loads"].T
    x_crc = np.sum(X[[col for col in X if col.endswith("circle")]], axis=1).T
    x_hex = np.sum(X[[col for col in X if col.endswith("hexagon")]], axis=1).T
    x_rec = np.sum(X[[col for col in X if col.endswith("rectangle")]], axis=1).T
    x_tri = np.sum(X[[col for col in X if col.endswith("triangle")]], axis=1).T

    return [t, c, w, l, s, nc, ls, nl, ncl, x_crc, x_hex, x_rec, x_tri]


def split(data, val_train):
    x_train = data.query(f"train != {val_train}")
    x_test = data.query(f"train == {val_train}")
    y_train = np.array(x_train.pop("Class_attribute"))
    y_test = np.array(x_test.pop("Class_attribute"))
    x_train = sort_inputs(x_train)
    x_test = sort_inputs(x_test)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    main()