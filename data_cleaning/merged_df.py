import pandas as pd


def main():

    load_df: pd.DataFrame = pd.read_pickle('./dataset/load_df_ped.pkl')
    purchase_df: pd.DataFrame = pd.read_pickle('./dataset/purchase_df_ped.pkl')

    def load_to_dict(order: pd.Series):
        df = load_df[(load_df['uuid_ind'] == order['uuid_ind'])
                     & (load_df['session_id'] == order['session_id'])]
        df = df[df['timestamp'] <= order['timestamp']]
        return [row.to_dict() for i, row in df.iterrows()]

    loads = purchase_df[['uuid_ind', 'session_id',
                         'timestamp']].apply(load_to_dict, axis=1)

    valid_order = loads.map(len) > 0
    purchase_df = purchase_df[valid_order].reset_index(drop=True)
    loads = loads[valid_order].reset_index(drop=True)

    merged_df = pd.concat(
        [purchase_df, loads.rename('loads').to_frame()], axis=1
    )
    print(merged_df)
    merged_df.to_pickle('./dataset/merged_df.pkl')


main()
