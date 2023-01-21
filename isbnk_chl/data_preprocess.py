import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


def prepare_monthly_expenditures(df, df_train, df_test):
    df["tarih"] = df["tarih"].astype(str)

    # eksi degerleri duzeltme
    for i in df[(df["islem_adedi"] <= 0)]["islem_adedi"].index:
        df["islem_adedi"][i] = df["islem_adedi"][i] * -1

    # pivot table
    yeni_df = df.pivot_table(index="musteri", columns=["sektor", "tarih"], values=["islem_adedi", "aylik_toplam_tutar"])
    yeni_df = yeni_df.fillna(0)

    train_df = yeni_df.loc[df_train["musteri"]]
    test_df = yeni_df.loc[df_test["musteri"]]

    return train_df, test_df


def preprocess_train(df, df_mon):
    df.drop(["tarih"], axis=1, inplace=True)

    df.loc[df["is_durumu"] == "51be29729b", "meslek_grubu"] = df.meslek_grubu.replace(np.nan, 'label1')
    df.loc[df["is_durumu"] == "eb35a5eb6b", "meslek_grubu"] = df.meslek_grubu.replace(np.nan, 'label2')
    df.loc[df["is_durumu"] == "289777e76d", "meslek_grubu"] = df.meslek_grubu.replace(np.nan, 'label3')
    df.loc[df["is_durumu"] == "d36d84f51a", "meslek_grubu"] = df.meslek_grubu.replace(np.nan, 'label4')
    df.loc[df["is_durumu"] == "915be3a7a4", "meslek_grubu"] = df.meslek_grubu.replace(np.nan, 'label5')

    df["egitim"].fillna(df["egitim"].mode()[0], inplace=True)
    df["is_durumu"].fillna(df["is_durumu"].mode()[0], inplace=True)
    df["meslek_grubu"].fillna(df["meslek_grubu"].mode()[0], inplace=True)

    column_numeric = ["yas", "kidem_suresi"]

    df = handle_outliers(df, column_numeric)

    df = pd.concat([df, df_mon], axis=1)

    df = pd.get_dummies(df, columns=(["egitim", "is_durumu", "meslek_grubu"]))

    # drop label colmns
    df = df.drop(["target"], axis=1)

    return df


def handle_outliers(df, columns):
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        under = q1 - 1.5 * iqr
        over = q3 + 1.5 * iqr
        outliers = df[(df[column] < under) | (df[column] > over)][column]
        values = df[(df[column] >= under) & (df[column] <= over)][column]

        for i in range(len(df[column])):
            if df[column][i] > over:
                df[column][i] = over
            if df[column][i] < under:
                df[column][i] = under

        return df


def main():
    df_train = pd.read_csv("./data/train.csv")
    df_test = pd.read_csv("./data/test.csv")
    df_mon = pd.read_csv("./data/monthly_expenditures.csv")
    target = df_train['target']

    df_mon_train, df_mon_test = prepare_monthly_expenditures(df_mon, df_train, df_test)

    df_train = preprocess_train(df_train, df_mon_train)

    scaler = StandardScaler()
    scaler_array = scaler.fit_transform(df_train)
    scaler_df = pd.DataFrame(scaler_array, columns=df_train.columns, index=df_train.index)

    # Create correlation matrix
    corr_matrix = scaler_df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
    # Drop features
    scaler_df.drop(to_drop, axis=1, inplace=True)

    scaler_df.to_csv("last_train.csv")


if __name__ == '__main__':
    main()
