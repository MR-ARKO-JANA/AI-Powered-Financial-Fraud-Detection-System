import pandas as pd 

# df = pd.read_csv("data/transactions_data.csv")

# print(df.head())
def preprocess_data(df:pd.DataFrame):

    print(df.isnull().sum())

    df['amount'] = df['amount'].astype(str).str.replace('$', '', regex=False).astype(float)
    df['amount'] = df['amount'].fillna(0)
    df['merchant_id'] = df['merchant_id'].fillna("UNKNOWN")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df['merchant_id'] = df['merchant_id'].astype(str)
    df['merchant_id'] = df['merchant_id'].str.strip().str.upper()

    df['merchant_id'] = df['merchant_id'].replace({
        "AMZN*MKTP": "AMAZON",
        "AMZN": "AMAZON",
        "FLIPKART*INDIA": "FLIPKART"
    })

    df = df.drop_duplicates()
    # df.to_csv("data/clean_transactions.csv", index=False)
    return df