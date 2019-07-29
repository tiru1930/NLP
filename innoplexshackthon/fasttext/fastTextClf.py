import pandas as pd 
import fasttext as ft
import os 
from sklearn.utils import resample
import numpy as np
import spacy 

nlp = spacy.load('en') 


# train_df = pd.read_csv("../data/train_processed.csv")
# test_Data  = pd.read_csv("../data/test_processed.csv")

# # test_tOlRoBf.csv
# # train_F3WbcTw.csv

# # train_df = pd.read_csv("../data/train_F3WbcTw.csv")
# # test_Data  = pd.read_csv("../data/test_tOlRoBf.csv")

# train_df["strlen"] = train_df["processed_text"].str.split().str.len()
# test_Data["strlen"] = test_Data["processed_text"].str.split().str.len()

# # _a = train_df.loc[train_df["strlen"] >= 1000].strlen.count()
# # _b = train_df.loc[train_df["strlen"] >= 750].strlen.count()
# # _c = train_df.loc[train_df["strlen"] >= 250].strlen.count()
# # _d = train_df.loc[train_df["strlen"] >= 180].strlen.count()
# # _e = train_df.loc[train_df["strlen"] >= 100].strlen.count()
# # _f = train_df.loc[train_df["strlen"] < 100].strlen.count()
# # _g = train_df.loc[train_df["strlen"] >= 1500].strlen.count()
# # print (" # of Reviews by Length \n %s >1000 words \n %s >750 words \
# # 			\n %s >250 words \n %s >180 words \n %s >100 words \n %s <100 words\n %s >=5000\n" % (_a,_b,_c,_d,_e,_f,_g))

# print(train_df.groupby("sentiment").count())
# train_df = train_df.loc[train_df["strlen"] <= 500]
# print(train_df.groupby("sentiment").count())

# df_positive = train_df[train_df.sentiment==0]
# df_negative = train_df[train_df.sentiment==1]
# df_nuteral = train_df[train_df.sentiment==2]

# # df_nuteral_downsampled = resample(df_nuteral, 
# #                                  replace=False,    
# #                                  n_samples=3000,    
# #                                  random_state=123)

# df_nuteral_downsampled = df_nuteral
# train_data = pd.concat([df_nuteral_downsampled, df_positive,df_negative])

# print(train_data.groupby("sentiment").count())

# train = train_data[["sentiment","processed_text"]]
# # train = train_data[["sentiment","text"]]
# train["sentiment_label"] = train["sentiment"].apply(lambda text : "__label__"+str(text))

# train[["sentiment_label","processed_text"]].to_csv("train_to_fastText.csv",index=False,sep="\t")
# # train[["sentiment_label","text"]].to_csv("train_to_fastText.csv",index=False,sep="\t")
# train_data = os.path.join('.', 'train_to_fastText.csv')
# model = ft.train_supervised(input=train_data, epoch=100, lr=1.0, wordNgrams=2, verbose=2, minCount=1,loss="hs") 




# def pred(text):
# 	pr = model.predict(text.replace("\n"," ").strip(),k=1)
# 	return pr[0][0].replace("__label__"," ").strip()


# test_Data["sentiment"] = test_Data["processed_text"].apply(pred)
# # test_Data["sentiment"] = test_Data["text"].apply(pred)
# test_Data[["unique_hash","sentiment"]].to_csv("submession_2.csv",index=False)	
# print(test_Data.groupby("sentiment").count())



train_df = pd.read_csv("../data/train_processed.csv")
test_df  = pd.read_csv("../data/test_processed.csv")

# print(train_df["processed_text"])
# def remove_words_based_on_lenght(text):

# 	return ' '.join(imp_words)

# train_df["_text"] = train_df["processed_text"].apply(remove_words_based_on_lenght)
# test_df["_text"] = test_df["processed_text"].apply(remove_words_based_on_lenght)

# test_df["_label"] = test_df["sentiment"].apply(lambda text : "__label__"+str(text))
# test_df[["_label","_text"]].to_csv("train_to_fastText.csv",index=False,sep="\t",header=False)


# train_data = os.path.join('.', 'train_to_fastText.csv')
# model = ft.train_supervised(input=train_data, epoch=50, lr=1.0, wordNgrams=2, verbose=2, minCount=1,loss="hs") 




# def pred(text):
# 	pr = model.predict(text.replace("\n"," ").strip(),k=1)
# 	return int(pr[0][0].replace("__label__"," ").strip())


# test_df["sentiment"] = test_df["processed_text"].apply(pred)
# test_df[["unique_hash","sentiment"]].to_csv("submession_2.csv",index=False)	
# print(test_df.groupby("sentiment").count())

# train_df["predict"] = train_df["processed_text"].apply(pred)
# train_df["sentiment"] = train_df["sentiment"].apply(lambda x : int(x))
# train_df["match"] =  np.where(train_df["predict"] == train_df["sentiment"], 1 , 0)
# print(sum(train_df["match"].values)/len(train_df))
