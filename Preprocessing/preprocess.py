import pandas as pd
import json
import os


def split_detailed_topics(df, converter_dct):
    freq2 = df.groupby(['detailed_topic']).size().sort_values(ascending=False)

    detailed2sequential_id = converter_dct['label2id']
    sequential_id2detailed = converter_dct['id2label']

    temp_dct = {'topic_id': [],
                'size': []}

    for i, detailed_topic in enumerate(freq2.keys()):
        new_id = i

        detailed2sequential_id[detailed_topic] = new_id
        sequential_id2detailed[new_id] = detailed_topic

        size = freq2[detailed_topic]
        temp_dct['topic_id'].append(new_id)
        temp_dct['size'].append(size)

    return pd.DataFrame(temp_dct).set_index('topic_id')


def label_reducing(df, converter_dct, frequency_dataset):
    THRESHOLD = 10

    labels_reduce = set()

    for i in converter_dct['id2label'].keys():
        if frequency_dataset.loc[i]['size'] < THRESHOLD:
            labels_reduce.add(i)
    
    reduced_df = df.copy()
    reduced_df["detailed_topic"] = reduced_df["detailed_topic"].map(lambda x: x if converter_dct['label2id'][x] not in labels_reduce else "undefined")

    freq3 = reduced_df.groupby(['detailed_topic']).size().sort_values(ascending=False)

    detailed2sequential_id = converter_dct['label2id_reduced']
    sequential_id2detailed = converter_dct['id2label_reduced']

    for i, detailed_topic in enumerate(freq3.keys()):
        new_id = i

        detailed2sequential_id[detailed_topic] = new_id
        sequential_id2detailed[new_id] = detailed_topic
    
    return reduced_df

def preprocess_main(new_db_name="clean_data.csv"):
    df = pd.read_csv(os.path.join("Database", "db_2.csv"), sep=';', encoding='utf-8')
    df.dropna(inplace=True)
    df.drop(columns=['topic_id', 'president_topic'], inplace=True)

    converter_dct = {
        'label2id': {},
        'id2label': {},

        'label2id_reduced': {},
        'id2label_reduced': {}
    }


    frequency_dataset = split_detailed_topics(df, converter_dct)
    reduced_df = label_reducing(df, converter_dct, frequency_dataset)

    with open(os.path.join("Assets", "converter.json"), "w", encoding='utf-8') as json_file:
        json.dump(converter_dct, json_file, ensure_ascii=False, indent=4)
    
    # df.to_csv(os.path.join("Database", "clean_data.csv"), index=False)
    reduced_df.to_csv(os.path.join("Database", new_db_name), sep=';', encoding='utf-8', index=False)


if __name__ == "__main__":
    preprocess_main()