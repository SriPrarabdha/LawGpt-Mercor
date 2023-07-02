with open('data.txt', 'r' , encoding='windows-1252') as f:
    data = f.read()

data_left = data[int(0.7*len(data)):]
data_left

limit = 384

def chunker(contexts: list):
    chunks = []
    all_contexts = ''.join(contexts).split('.')
    chunk = []
    for context in all_contexts:
        chunk.append(context)
        if len(chunk) >= 3 and len('.'.join(chunk)) > limit:
            # surpassed limit so add to chunks and reset
            chunks.append('.'.join(chunk).strip()+'.')
            # add some overlap between passages
            chunk = chunk[-2:]
    # if we finish and still have a chunk, add it
    if chunk is not None:
        chunks.append('.'.join(chunk))
    return chunks

chunks = chunker(data_left)
chunks

"""We create the full contexts dataset with this logic like so:"""

data = []
chunks = chunker(data_left)
for i, context in enumerate(chunks):
    data.append({
            'id': f"{i+1472}",
            #Adding 1472 to make the proper id
            #Check the actual ids and correct this 1472 accordingly
            'context': context
        })

from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# check device being run on
if device != 'cuda':
    print("==========\n"+
          "WARNING: You are not running on GPU so this may be slow.\n"+
          "If on Google Colab, go to top menu > Runtime > Change "+
          "runtime type > Hardware accelerator > 'GPU' and rerun "+
          "the notebook.\n==========")

dense_model = SentenceTransformer(
    'msmarco-bert-base-dot-v5',
    device=device
)

emb = dense_model.encode(data[0]['context'])
print(emb.shape)

"""The model returns `768` dimensional dense vectors, this is also reflected in the model attributes."""

from splade.models.transformer_rep import Splade

sparse_model_id = 'naver/splade-cocondenser-ensembledistil'

sparse_model = Splade(sparse_model_id, agg='max')
sparse_model.to(device)  # move to GPU if possible
sparse_model.eval()

"""The model takes tokenized inputs that are built using a tokenizer initialized with the same model ID."""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sparse_model_id)

tokens = tokenizer(data[0]['context'], return_tensors='pt')

"""To create sparse vectors we do:"""

with torch.no_grad():
    sparse_emb = sparse_model(
        d_kwargs=tokens.to(device)
    )['d_rep'].squeeze()

print(sparse_emb.shape)

indices = sparse_emb.nonzero().squeeze().cpu().tolist()
print(len(indices))

"""We have `174` non-zero values, we use them to create a dictionary of index positions to scores like so:"""

values = sparse_emb[indices].cpu().tolist()
sparse = {'indices': indices, 'values': values}

idx2token = {idx: token for token, idx in tokenizer.get_vocab().items()}

"""Then create the mappings like we did with the Pinecone-friendly sparse format above."""

sparse_dict_tokens = {
    idx2token[idx]: round(weight, 2) for idx, weight in zip(indices, values)
}
# sort so we can see most relevant tokens first
sparse_dict_tokens = {
    k: v for k, v in sorted(
        sparse_dict_tokens.items(),
        key=lambda item: item[1],
        reverse=True
    )
}
print(sparse_dict_tokens)

import pinecone


def builder(records: list):
    ids = [x['id'] for x in records]
    contexts = [x['context'] for x in records]
    # create dense vecs
    dense_vecs = dense_model.encode(contexts).tolist()
    # create sparse vecs
    input_ids = tokenizer(
        contexts, return_tensors='pt',
        padding=True, truncation=True
    )
    with torch.no_grad():
        sparse_vecs = sparse_model(
            d_kwargs=input_ids.to(device)
        )['d_rep'].squeeze()
    # convert to upsert format
    upserts = []
    for _id, dense_vec, sparse_vec, context in zip(ids, dense_vecs, sparse_vecs, contexts):
        # extract columns where there are non-zero weights
        indices = sparse_vec.nonzero().squeeze().cpu().tolist()  # positions
        values = sparse_vec[indices].cpu().tolist()  # weights/scores
        # build sparse values dictionary
        sparse_values = {
            "indices": indices,
            "values": values
        }
        # build metadata struct
        metadata = {'context': context}
        # append all to upserts list as pinecone.Vector (or GRPCVector)
        upserts.append({
            'id': _id,
            'values': dense_vec,
            'sparse_values': sparse_values,
            'metadata': metadata
        })
    return upserts

builder(data[:3])

"""Now we initialize our connection to Pinecone using a [free API key](https://app.pinecone.io/)."""

import pinecone

pinecone.init(
    api_key="API_KEY",  # app.pinecone.io
    environment="ENV"  # next to api key in console
)


index_name = 'crimimnal-laws'

pinecone.create_index(
    index_name,
    dimension=dim,
    metric="dotproduct",
    pod_type="s1"
)

"""Initialize with `GRPCIndex` or `Index`:"""

index = pinecone.GRPCIndex(index_name)
index.describe_index_stats()

"""Upsert to sparse-dense is simple:"""

index.upsert(builder(data[:3]))

"""We can repeat this and iterate through (and index) the full dataset:"""

from tqdm.auto import tqdm

batch_size = 64

for i in tqdm(range(0, len(data), batch_size)):
    # extract batch of data
    i_end = min(i+batch_size, len(data))
    batch = data[i:i_end]
    # pass data to builder and upsert
    index.upsert(builder(data[i:i+batch_size]))

"""We can check the number of upserted records aligns with the length of `data`."""

print(len(data), index.describe_index_stats())