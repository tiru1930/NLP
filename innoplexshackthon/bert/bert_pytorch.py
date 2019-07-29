import torch
import pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm_notebook, trange
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from multiprocessing import Pool, cpu_count
from tools import *
import convert_examples_to_features
from sklearn.utils.extmath import softmax
import numpy as np 
import logging
import pandas as pd
from sklearn.utils import resample 

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# train_df = pd.read_csv("./data/train_processed.csv")
train_df = pd.read_csv("../data/train_F3WbcTw.csv")
train_df["strlen"] = train_df["text"].str.split().str.len()
print(train_df.groupby("sentiment").count())
train_df = train_df.loc[train_df["strlen"] <= 1500]
print(train_df.groupby("sentiment").count())

# train_df_bert = pd.DataFrame({
#     'id':range(len(train_df)),
#     'label':train_df['sentiment'],
#     'alpha':['a']*train_df.shape[0],
#     'text': train_df['processed_text'].replace(r'\n', ' ', regex=True)
# })





df_positive = train_df[train_df.sentiment==0]
df_negative = train_df[train_df.sentiment==1]
df_nuteral = train_df[train_df.sentiment==2]
df_nuteral_downsampled = resample(df_nuteral, 
                                 replace=False,    
                                 n_samples=2000,    
                                 random_state=123)

train_data = pd.concat([df_nuteral_downsampled, df_positive,df_negative])
print(train_data.groupby("sentiment").count())

train_df_bert = pd.DataFrame({
    'id':range(len(train_data)),
    'label':train_data['sentiment'],
    'alpha':['a']*train_data.shape[0],
    # 'text': train_data['processed_text'].replace(r'\n', ' ', regex=True)
    'text': train_data['text'].replace(r'\n', ' ', regex=True)
})


train_df_bert.to_csv('data/train.tsv', sep='\t', index=False, header=False)

# Bert pre-trained model selected in the list: bert-base-uncased, 
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
DATA_DIR = "data/"
TASK_NAME = 'drug_sent_clf'
BERT_MODEL = 'bert-base-cased'
# BERT_MODEL = 'bert-large-cased'
OUTPUT_DIR = 'outputs/drug_sent_clf/'
REPORTS_DIR = 'reports/drug_senti_clf_evaluation_report/'
CACHE_DIR = 'cache/'

MAX_SEQ_LENGTH = 256
TRAIN_BATCH_SIZE = 10
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 20
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"


if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
        REPORTS_DIR += '/report_'+str(len(os.listdir(REPORTS_DIR)))
        os.makedirs(REPORTS_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    REPORTS_DIR += '/report_'+str(len(os.listdir(REPORTS_DIR)))
    os.makedirs(REPORTS_DIR)


if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(OUTPUT_DIR))
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


processor = BinaryClassificationProcessor()
train_examples = processor.get_train_examples(DATA_DIR)
train_examples_len = len(train_examples)

label_list = processor.get_labels() 
num_labels = len(label_list)

num_train_optimization_steps = int(train_examples_len / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
label_map = {label: i for i, label in enumerate(label_list)}
train_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in train_examples]



process_count = cpu_count() - 1
with Pool(process_count) as p:
    train_features = list(tqdm_notebook(p.imap(convert_examples_to_features.convert_example_to_feature, \
    											train_examples_for_processing), total=train_examples_len))

model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=num_labels)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=LEARNING_RATE,
                     warmup=WARMUP_PROPORTION,
                     t_total=num_train_optimization_steps)


global_step = 0
nb_tr_steps = 0
tr_loss = 0


logger.info("***** Running training *****")
logger.info("  Num examples = %d", train_examples_len)
logger.info("  Batch size = %d", TRAIN_BATCH_SIZE)
logger.info("  Num steps = %d", num_train_optimization_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

if OUTPUT_MODE == "classification":
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
elif OUTPUT_MODE == "regression":
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)

model.train()
for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        logits = model(input_ids, segment_ids, input_mask, labels=None)

        if OUTPUT_MODE == "classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif OUTPUT_MODE == "regression":
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        loss.backward()
        print("\r%f" % loss, end='')
        
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(OUTPUT_DIR)


def eval_single(text):
    eval_examples = [InputExample(guid=0, text_a=text, text_b=None, label='1')]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    eval_examples_len = len(eval_examples)

    label_map = {label: i for i, label in enumerate(label_list)}
    eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in eval_examples]

    process_count = cpu_count() - 1
    with Pool(process_count) as p:
           eval_features = list(tqdm_notebook(p.imap(convert_examples_to_features.convert_example_to_feature, 
                            eval_examples_for_processing), total=eval_examples_len))

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if OUTPUT_MODE == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif OUTPUT_MODE == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)


    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        if OUTPUT_MODE == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif OUTPUT_MODE == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    # print(preds)
    pred_probs = softmax(preds)
    # print(pred_probs)
    idx = np.array(pred_probs).argmax()
    # print(idx)
    return idx 

# test_df  = pd.read_csv("./data/test_processed.csv")
test_df  = pd.read_csv("../data/test_tOlRoBf.csv")
test_df["sentiment"] = test_df["text"].apply(eval_single)
test_df.head()
test_df[["unique_hash","sentiment"]].to_csv("submession_bert_1.csv",index=False)   
print(test_df.groupby("sentiment").count())
