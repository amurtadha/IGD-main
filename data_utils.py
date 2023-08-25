import random

from torch.utils.data import Dataset
import  json
from tqdm import tqdm
import numpy as np
import re


class custom_dataset(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, labels):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        labels = {label: _ for _,label in enumerate(labels)}
        print(labels)
        data = json.load(open(fname))

        all_data=[]
        for d in tqdm(data):
            text, label = d['text'], d['label']
            if label not in labels: continue


            inputs = tokenizer.encode_plus(text.strip().lower(), None, add_special_tokens=True,
                                           max_length=max_seq_len, truncation='only_first', padding='max_length',
                                           return_token_type_ids=True)

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]

            assert len(input_ids) <= max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')

            tem = {
                'text': text,
                'input_ids': input_ids,
                'segments_ids': segment_ids,
                'input_mask': input_mask,
                'label': labels[label]
            }
            all_data.append(tem)
            # if len(all_data)>1000:break
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)

class custom_dataset_pl(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, dataset, number_candidates=5):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len

        data = json.load(open(fname))

        all_data=[]
        for ind , d in enumerate (tqdm(data)):
            text, label = d['text'], d['label'].strip()
            label = re.findall(r'\d+', label)[0]

            inputs = tokenizer.encode_plus(text.strip().lower(), None, add_special_tokens=True,
                                           max_length=max_seq_len, truncation='only_first', padding='max_length',
                                           return_token_type_ids=True)

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]

            assert len(input_ids) <= max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')


            if 'train' in fname:
                candidates = d['candidates']
                if len(candidates)>=number_candidates:

                    candidates =candidates[:number_candidates]
                if not len(candidates):
                    candidates = [text]
                for c in candidates:
                        inputs_c = tokenizer.encode_plus(c.strip().lower(), None, add_special_tokens=True,
                                                       max_length=max_seq_len, truncation='only_first',
                                                       padding='max_length',
                                                       return_token_type_ids=True)

                        assert len(input_ids) <= max_seq_len
                        input_ids_c = np.asarray(inputs_c['input_ids'], dtype='int64')
                        input_mask_c = np.asarray(inputs_c['attention_mask'], dtype='int64')
                        segment_ids_c = np.asarray(inputs_c["token_type_ids"], dtype='int64')

                        data_temp = {
                                'text': text,
                                'input_ids': input_ids,
                                'segments_ids': segment_ids,
                                'input_mask': input_mask,
                                'input_ids_c': input_ids_c,
                                'segments_ids_c': segment_ids_c,
                                'input_mask_c': input_mask_c,
                                'label': int(label)
                            }
                        all_data.append(data_temp)
            else:
                data_temp = {
                            'text': text,
                            'input_ids': input_ids,
                            'segments_ids': segment_ids,
                            'input_mask': input_mask,

                            'label': 0
                        }
                all_data.append(data)


            if ind>25000 and dataset=='YELP-2' :break
            # if ind > 1000 :break
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)
