import json
import pandas as pd
import pickle
import re
import time
import warnings

from helpers import BloomFilter
from sseclient import SSEClient as EventSource

warnings.filterwarnings("ignore")

URL = 'https://stream.wikimedia.org/v2/stream/recentchange'


def feature_extraction(datafile):
    import pandas as pd
    if datafile.endswith('.gz'):
        df = pd.read_csv(datafile, sep='\t', compression='gzip', )
    else:
        df = pd.read_csv(datafile, sep='\t', )
    short_df = df[['timestamp', 'user', 'bot', 'type', 'comment']].copy()
    short_df.dropna(inplace=True)
    short_df['timestamp'] = pd.to_datetime(short_df['timestamp'].loc[:], unit='s')
    short_df = short_df.assign(requests=1)
    user_df = short_df.set_index('timestamp').groupby('user')[['bot', 'requests', ]].resample("1S", label='right').sum()
    user_df['bot'] = [1 if x > 0.5 else 0 for x in user_df['bot']]
    user_df = user_df.reset_index()
    user_df = user_df[user_df['requests'] != 0]
    user_df = user_df.groupby('user', as_index=False)[['bot', 'requests']].mean()
    user_df['bot'] = user_df['bot'].astype(int)

    def count_digits(string):
        return sum(c.isdigit() for c in string)

    user_df['n_digits_name'] = user_df['user'].apply(count_digits)
    find_lead_digits = lambda name: len(re.findall('^\d+', name)[0]) if name[0].isdigit() else 0
    user_df['lead_digits_name'] = user_df['user'].apply(find_lead_digits)

    def unique_ratio(string):
        return (len(set(string)) / len(string))

    user_df['uniq_char_ratio_name'] = user_df['user'].apply(unique_ratio)
    user_df['uniq_char_ratio_name'] = user_df['uniq_char_ratio_name'].round(3)
    user_df['bot_in_name'] = (user_df['user'].str.lower().str.contains('bot')).astype(int)
    dummies_df = short_df.join(short_df['type'].str.get_dummies())
    dummies_df = dummies_df.groupby('user').sum()
    dummies_df = dummies_df.drop(['bot', 'requests'], axis=1).reset_index()
    try:
        dummies_df = dummies_df.drop('142', axis=1)
    except KeyError:
        pass
    comment_df = short_df[['user', 'comment']]
    comment_df['len_comment'] = comment_df['comment'].str.len()

    def find_alnum_num(name):
        for el in name:
            if type(el) != str:
                return 0
            else:
                return sum(el.isalnum() for el in name)

    comment_df['alnum_ratio_comment'] = comment_df['comment'].astype("str").apply(find_alnum_num) / comment_df[
        'comment'].str.len()
    comment_df['bot_in_comment'] = (comment_df['comment'].str.lower().str.contains('bot'))
    comment_df = comment_df.drop('bot_in_comment', axis=1)
    mean_df = comment_df.groupby('user').mean().rename(
        columns={'len_comment': 'len_comment_avg', 'alnum_ratio_comment': 'alnum_ratio_comment_avg'})
    min_df = comment_df.groupby('user').min().rename(
        columns={'len_comment': 'len_comment_min', 'alnum_ratio_comment': 'alnum_ratio_comment_mix'})
    max_df = comment_df.groupby('user').max().rename(
        columns={'len_comment': 'len_comment_max', 'alnum_ratio_comment': 'alnum_ratio_comment_max'})
    user_df = user_df.merge(dummies_df, on='user', how='outer')
    user_df = user_df.merge(min_df, on='user', how='outer')
    user_df = user_df.merge(mean_df, on='user', how='outer')
    user_df = user_df.merge(max_df, on='user', how='outer')
    user_df.drop(['comment_x', 'comment_y'], axis=1, inplace=True)
    return user_df


def get_model(mode_path):
    model = pickle.load(open(mode_path, 'rb'))
    return model


def main():
    model = get_model('model/best_model.sav')

    results = []

    params_grid = [
        {'classifier_time_limit': 20., 'bloom_filter_time_limit': 20., },
        {'classifier_time_limit': 20., 'bloom_filter_time_limit': 15., },
        {'classifier_time_limit': 20., 'bloom_filter_time_limit': 10., },
        {'classifier_time_limit': 20., 'bloom_filter_time_limit': 5., },
        {'classifier_time_limit': 20., 'bloom_filter_time_limit': 3., },
        {'classifier_time_limit': 10., 'bloom_filter_time_limit': 3., },
        {'classifier_time_limit': 5., 'bloom_filter_time_limit': 3., },
        {'classifier_time_limit': 1., 'bloom_filter_time_limit': 1., },
    ]

    for params in params_grid:
        result = params.copy()

        t_0 = time.time()
        dataset = []

        time_threshold = params['classifier_time_limit']
        print(f'Running stream for {time_threshold} mins...\n')

        # stream to generate data for black list
        for event in EventSource(URL):  # start streaming
            if event.event == 'message':
                try:
                    change = json.loads(event.data)
                except ValueError:
                    continue

                dataset.append(change)

                if (time.time() - t_0) // 60 > time_threshold:
                    break

        filename = f'data/stream_test_data_{int(time_threshold)}mins_{t_0}.csv'
        df = pd.DataFrame(dataset)
        df.to_csv(filename, sep='\t')
        print(f'Generated file {filename}\n')

        t1 = time.time()

        user_df = feature_extraction(filename)
        X_val = user_df.drop(['bot', 'user'], axis=1)

        y_pred = model.predict(X_val)

        black_list = [user for pred, user in zip(y_pred, user_df['user']) if pred]
        print(f'Black list length is {len(black_list)}\n')

        # generate bloom filter
        bloom_filter = BloomFilter(len(black_list), 0.1)
        for item in black_list:
            bloom_filter.add(item)

        t2 = time.time()
        print(f'Generated Bloom filter with {bloom_filter.size, bloom_filter.hash_count}\n')
        print(f'Took {t2 - t1} seconds to preprocess, predict and create bloom\n')

        gd_dataset = []
        bloom_filter_users = []
        t_0 = time.time()

        time_threshold = params['bloom_filter_time_limit']
        print(f'Running stream for {time_threshold} mins...\n')

        # stream to generate data for evaluation
        for event in EventSource(URL):  # start streaming
            if event.event == 'message':
                try:
                    change = json.loads(event.data)
                except ValueError:
                    continue

                gd_dataset.append(change)

                user_name = change['user']
                if bloom_filter.check(user_name):
                    bloom_filter_users.append(user_name)

                if (time.time() - t_0) // 60 > time_threshold:
                    break

        filename = f'data/stream_validation_data_{int(time_threshold)}mins_{t_0}.csv'
        df = pd.DataFrame(gd_dataset)
        df.to_csv(filename, sep='\t')
        print(f'Generated ground truth file {filename}\n')

        gd_black_set = set(df[df.bot].user.unique())
        bloom_filter_set = set(bloom_filter_users)

        result.update({
            'real_bots_count': len(gd_black_set),
            'bloom_bots_count': len(bloom_filter_set),
            'acc_rel_real': len(bloom_filter_set.intersection(gd_black_set)) / len(gd_black_set),
            'acc_rel_bloom': len(bloom_filter_set.intersection(gd_black_set)) / len(bloom_filter_set),
            'intersec_count': len(bloom_filter_set.intersection(gd_black_set)),
        })

        print(f'Result - {result}\n')
        results.append(result)

        save_results(results)

    print(results)

    save_results(results)


def save_results(results):
    with open('results.json', 'w') as output_file:
        json.dump(results, output_file)


if __name__ == '__main__':
    main()
