from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def split_train_test(csv_path, test_size, random_state, out_train_path, out_test_path):
    train_df = pd.read_csv(csv_path).set_index("id")
    train_names = train_df.index.values
    train_labels = np.asarray(train_df['label'].values)

    x_train, x_test, y_train, y_test = train_test_split(train_names, train_labels, test_size=test_size, random_state=random_state)
    df_train = pd.DataFrame(data={'id':x_train, 'label':y_train})
    df_test = pd.DataFrame(data={'id':x_test, 'label':y_test})

    if out_test_path is not None and out_train_path is not None:
        df_train.to_csv(out_train_path)
        df_test.to_csv(out_test_path)

    else:
        return df_train, df_test
