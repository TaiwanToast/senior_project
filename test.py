from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 加载中文 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

def extract_keywords(text, top_k=5):
    # 分词并转为模型输入格式
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)

    # 提取词嵌入（最后一层的隐藏状态）
    embeddings = outputs.last_hidden_state[0]  # shape: [seq_len, hidden_size]
    cls_embedding = embeddings[0]  # CLS token embedding
    
    # 计算每个词与 CLS 的余弦相似度
    word_embeddings = embeddings[1:-1]  # 排除 [CLS] 和 [SEP]
    similarity_scores = cosine_similarity(cls_embedding.unsqueeze(0).detach().numpy(), 
                                           word_embeddings.detach().numpy())[0]
    
    # 按相似度排序，提取关键词
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][1:-1])  # 排除 [CLS] 和 [SEP]
    keyword_scores = list(zip(tokens, similarity_scores))
    keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [word for word, score in keywords]

# 测试
text = "我想要買一台電競筆電"
keywords = extract_keywords(text)
print("提取的关键词:", keywords)
