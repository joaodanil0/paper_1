
import pandas as pd
import numpy as np
import os

from src.commom import Commom

# def extract_labels(labels, amount_labels):
#   len_array = labels.shape[0]
#   if amount_labels > len_array : amount_labels = len_array
#   idxs = np.random.choice(range(len_array), amount_labels, replace=False)
#   extracted_items = labels[idxs]  
#   return extracted_items

# def sample_df_test(df, labels, random_sample):
#   new_df = pd.DataFrame()
#   for label in labels:
#     df_aux = df[df.LABEL == label].copy()
#     sample = df_aux.sample(random_state=random_sample)
#     new_df = pd.concat([new_df, sample])  
#   return new_df

test_file = pd.read_csv("bases/db_Samplescomaisde3aps_semSala42e49_Janela10Mean.csv")
print(test_file)
# ticks = np.linspace(1,len(Commom.label_points), 20).astype(int)
# random_samples = np.arange(30)

# for random_sample in random_samples:
#   os.mkdir("bases/test_folder/" + str(random_sample))
#   for tick in ticks:
#     test_labels = extract_labels(np.array(Commom.label_points), tick)

#     i = test_labels.shape[0]
#     df_test = sample_df_test(test_file, test_labels, random_sample)

#     df_test.to_csv("bases/test_folder/" + str(random_sample) + "/" + str(i) + ".csv", index=False)





