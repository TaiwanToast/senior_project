import chromadb
from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
# ========================================
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
# ========================================
import pymysql
# ========================================
import os
import json
import joblib
# import pypdf
# ========================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ========================================
# from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

class MyChromaAndTFIDF(object):
    def __init__(self, chroma_db_path: str = './chroma_db', tfidf_model_path: str = './tfidf_model', transformers_tokenizer_model: str = 'intfloat/e5-base'):
        # self.__ws_driver = CkipWordSegmenter(device=0)
        # self.__pos_driver = CkipPosTagger(device=0)
        # ========================================
        # self.__sentenceModel = SentenceTransformer(sentence_model)
        # sentence_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__transformers_tokenizer = AutoTokenizer.from_pretrained(transformers_tokenizer_model, device=self.device)
        self.__transformers_model = AutoModel.from_pretrained(transformers_tokenizer_model).to(self.device)
        # self.__transformers_model.to(torch.device('cuda:0'))
        # ========================================
        self.__cromaClient = chromadb.PersistentClient(path=chroma_db_path, settings=Settings(allow_reset=True))
        self.__collection = self.__cromaClient.get_or_create_collection(name="pdf_documents")

        self.__mySQLClient = pymysql.connect(host='localhost', user='root', password='Dd1204889', database='senior_project_1', cursorclass=pymysql.cursors.DictCursor)
        self.__mySQLCursor = self.__mySQLClient.cursor()
        # ========================================
        os.makedirs(tfidf_model_path, exist_ok=True)
        self.__modelDir = tfidf_model_path
        self.__tfidfVectorizerPath = os.path.join(tfidf_model_path, "tfidf_vectorizer.joblib")
        self.__tfidfMatrixPath = os.path.join(tfidf_model_path, "tfidf_matrix.joblib")
        self.__fileIndexPath = os.path.join(tfidf_model_path, "file_index.json")
        self.__fileStoreIndexPath = os.path.join(tfidf_model_path, "file_store_index.json")
        # ========================================
        self.__fileList = []
        if(os.path.exists(self.__fileIndexPath)):
            with open(self.__fileIndexPath, 'r') as f:
                self.__fileList = list(json.load(f).keys())
        self.__fileCount = len(self.__fileList)
        print('MyChromaAndTFIDF initialized')
    
    def __storeVectorInChroma(self, chunks: list, file_name: str, store_name: str) -> None:
        for index, chunk in enumerate(chunks):
            self.__collection.add(
                ids=[f"{file_name}_{index}"],
                embeddings=[self.__sentenceModel.encode(chunk)],
                metadatas=[{"text": chunk, "file_name": file_name, "store_name": store_name}]
            )
        return

    
    def __tfidfStoreData(self, vectorizer, matrix, fileIndex, fileStoreIndex) -> None:
        joblib.dump(vectorizer, self.__tfidfVectorizerPath)
        joblib.dump(matrix, self.__tfidfMatrixPath)
        with open(self.__fileIndexPath, 'w') as f:
            json.dump(fileIndex, f)
        with open(self.__fileStoreIndexPath, 'w') as f:
            json.dump(fileStoreIndex, f)
        return

    def __tfidfLoadData(self):
        if os.path.exists(self.__tfidfVectorizerPath):
            vectorizer = joblib.load(self.__tfidfVectorizerPath)
            matrix = joblib.load(self.__tfidfMatrixPath)
            with open(self.__fileIndexPath, 'r') as f:
                fileIndex = json.load(f)
            with open(self.__fileStoreIndexPath, 'r') as f:
                fileStoreIndex = json.load(f)
        else:
            vectorizer = TfidfVectorizer()
            matrix = None
            fileIndex = {}
            fileStoreIndex = {}
        return vectorizer, matrix, fileIndex, fileStoreIndex
    '''__ckip_tokenize
    # def __ckip_tokenize(self, text: str) -> str:
    #     def clean(sentence_ws, sentence_pos):
    #         # 這裡只保留名詞和動詞，且去除單一字元
    #         short_sentence = []
    #         stop_pos = set(['Nep', 'Nh', 'Nb'])
    #         for word_ws, word_pos in zip(sentence_ws, sentence_pos):
    #             is_N_or_V = word_pos.startswith("V") or word_pos.startswith("N")
    #             is_not_stop_pos = word_pos not in stop_pos
    #             is_not_one_charactor = not (len(word_ws) == 1)
    #             if is_N_or_V and is_not_stop_pos and is_not_one_charactor:
    #                 short_sentence.append(f"{word_ws}")
    #         return " ".join(short_sentence)
        
    #     ws = self.__ws_driver([text])
    #     pos = self.__pos_driver(ws)
    #     short = clean(ws[0], pos[0])
    #     return short
    '''
    
    def addFile(self, file_path: str) -> None:
        def textSpliter(text: str, chunkSize: int = 250, chunkOverlap: int = 20) -> list:
            chunks = []
            flag = False
            for i in range(0, len(text), chunkSize):
                if not flag:
                    chunks.append(text[i:i+chunkSize])
                    flag = True
                else:
                    chunks.append(text[i-chunkOverlap:i+chunkSize])
            return chunks
        '''addCSV(csv_path: str)
        def addCSV(csv_path: str) -> None:
            import pandas as pd
            df = pd.read_csv(csv_path)
            products = df.to_dict(orient='records')
            
            for product in products:
                product_name = '商品名稱' + product['product name']
                product_price = '商品價格' + str(product['product price'])
                product_info = '商品描述' + product['product info']
                store_name = '商店名稱' + product['store name']
                store_phone_number = '商店電話' + str(product['store phone number'])
                store_address = '商店地址' + product['store address']
                sql = f"INSERT INTO products_1 (`product name`, `product price`, `product info`, `store name`, `store phone number`, `store address`) VALUES ('{product['product name']}', {product['product price']}, '{product['product info']}', '{product['store name']}', '{product['store phone number']}', '{product['store address']}');"

                # Store data in MySQL
                try:
                    self.__mySQLCursor.execute(sql)
                    self.__mySQLClient.commit()
                except pymysql.MySQLError as e:
                    print(f"Error inserting data into MySQL: {e}")
                    self.__mySQLClient.rollback()
                    # sql_unsuccess.append(product.update({"sql error": e}))
                    continue

                all_text = product_name + ' ' + product_price + ' ' + product_info + ' ' + store_name + ' ' + store_phone_number + ' ' + store_address
                # Split text into chunks
                tokenizedText = self.__ckip_tokenize(all_text)
                chunks = textSpliter(tokenizedText)

                # Store vectors in Chroma
                self.__storeVectorInChroma(chunks, product['product name'], product['store name'])

                # Store TF-IDF data
                vectorizer, matrix, fileIndex, fileStoreIndex = self.__tfidfLoadData()
                document = list(fileIndex.values()) + [tokenizedText]
                vectorizer = TfidfVectorizer()
                matrix = vectorizer.fit_transform(document)
                file_name = product['product name']
                fileStore_name = product['store name']
                fileStoreIndex[file_name] = fileStore_name
                fileIndex[file_name] = tokenizedText
                self.__tfidfStoreData(vectorizer, matrix, fileIndex, fileStoreIndex)
                print(f"Added {file_name}")
                self.__fileCount += 1
                self.__fileList.append(file_name)
            return
            '''
        
        def addJson(json_path: str) -> None:
            def product_to_text(product: dict) -> str:
                # print(product)
                name = product["name"]
                specs = product["specs"]
                comments = product["comments"]

                specs_text = " ".join(f"{k}: {v}" for k, v in specs.items())
                comments_text = " ".join(comments[:5])  # 可只選前幾則代表性評論

                full_text = f"passage: {name}. Specifications: {specs_text}. User comments: {comments_text}"
                return full_text
            
            def encode_text(text):
                inputs = self.__transformers_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.__transformers_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # L2 normalization
                return embeddings[0]  # shape: (768,)
            
            def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
                last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
                return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            
            files = [file for file in os.listdir(json_path) if file.endswith('.json')]
            
            for file in files:
                json_file = os.path.join(json_path, file)
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for idx, product in enumerate(data):
                    name = product["name"]
                    url = product["link"]
                    text = product_to_text(product)

                    encoded_input = self.__transformers_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                    encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                    with torch.no_grad():
                        # self.__transformers_model.to(torch.device('cuda'))
                        model_output = self.__transformers_model(**encoded_input)
                    # model_output = model_output.cpu().numpy()
                    embedding = average_pool(model_output.last_hidden_state, encoded_input['attention_mask'])
                    embedding = F.normalize(embedding.cpu(), p=2, dim=1).squeeze().numpy()

                    self.__collection.add(
                        documents=[text],
                        embeddings=[embedding],
                        ids=[f"{name}-{idx}"],  # 唯一 ID
                        metadatas=[{
                            "name": name,
                            "url": url
                        }]
                    )
                    self.__fileCount += 1
                    self.__fileList.append(name)

        addJson(file_path)
        return

    '''# def search(self, query: str, top_k: int = 10, hybridSearch: bool = False, use_json_file_storage: bool = False, chroma_weight: float = 0.4, tfidf_weight: float = 0.6):
    #     tokenizedQuery = self.__ckip_tokenize(query)
    #     # print('tokenized query: ', tokenizedQuery)
    #     queryVector = self.__sentenceModel.encode(tokenizedQuery)
    #     vectorizer, matrix, fileIndex, fileStoreIndex = self.__tfidfLoadData()
    #     tfidfQueryVector = vectorizer.transform([tokenizedQuery])

    #     # Search in Chroma
    #     retrieved_files = set()
    #     where_condition = {
    #         "file_name": {"$nin": list(retrieved_files)}
    #     }
    #     chroma_result = []
    #     for i in range(top_k):
    #         where_condition = {
    #             "file_name": {"$nin": list(retrieved_files)}
    #         }
    #         searchResult = self.__collection.query(
    #             query_embeddings=[queryVector],
    #             n_results=1,
    #             where=where_condition if bool(retrieved_files) else None
    #         )
    #         try:
    #             retrieved_files.add(searchResult['metadatas'][0][0]['file_name'])
    #             chroma_result.append({'file_name': searchResult['metadatas'][0][0]['file_name'],
    #                         #    'text': searchResult['metadatas'][0][0]['text'],
    #                            'store_name': searchResult['metadatas'][0][0]['store_name'],
    #                            'score': 1 / searchResult['distances'][0][0]})
    #         except:
    #             break
        
    #     # Search in TF-IDF
    #     cosineSimilarities = cosine_similarity(tfidfQueryVector, matrix).flatten()
    #     topIndices = cosineSimilarities.argsort()[-top_k:][::-1]
    #     tfidf_result = []
    #     files = list(fileIndex.keys())
    #     for index in topIndices:
    #         if index < self.__fileCount:
    #             tfidf_result.append({
    #                 "file_name": files[index],
    #                 "cosine_similarity": cosineSimilarities[index],
    #                 'store_name': fileStoreIndex[files[index]],
    #                 # 'text': fileIndex[files[index]]
    #             })

    #     if not hybridSearch and not use_json_file_storage:
    #         return chroma_result, tfidf_result
    #     elif not hybridSearch and use_json_file_storage:
    #         raise ValueError("use_json_file_storage is not supported for non-hybrid search")
    #     elif hybridSearch and use_json_file_storage:
    #         chroma_scores = {
    #             i['file_name']: i['score']
    #             for i in chroma_result
    #         }
    #         tfidf_scores = {
    #             i['file_name']: i['cosine_similarity']
    #             for i in tfidf_result
    #         }
    #         combinedScores = {}
    #         for file in set(chroma_scores).union(tfidf_scores.keys()):
    #             chroma_score = chroma_scores.get(file, 0)
    #             tfidf_score = tfidf_scores.get(file, 0)
    #             combinedScores[file] = chroma_score * chroma_weight + tfidf_score * tfidf_weight

    #         sorted_result = sorted(combinedScores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    #         result = []
    #         for i in sorted_result:
    #             sql = f"SELECT * FROM products_1 WHERE `product name` = '{i[0]}' and `store name` = '{fileStoreIndex[i[0]]}';"
    #             try:
    #                 self.__mySQLCursor.execute(sql)
    #                 SQLresult = self.__mySQLCursor.fetchall()
    #                 if SQLresult:
    #                     result.append({
    #                         'product name': SQLresult[0][0],
    #                         'product price': SQLresult[0][1],
    #                         'product info': SQLresult[0][2],
    #                         'store name': SQLresult[0][3],
    #                         'store phone number': SQLresult[0][4],
    #                         'store address': SQLresult[0][5]
    #                     })
    #             except pymysql.MySQLError as e:
    #                 print(f"Error fetching data from MySQL: {e}")
    #                 self.__mySQLClient.rollback()
    #                 continue
    #         result = json.dumps(result, ensure_ascii=False)
    #         with open('search_result.json', 'w', encoding='utf-8') as f:
    #             f.write(result)
    #         return
    #     else:
    #         chroma_scores = {
    #             i['file_name']: i['score']
    #             for i in chroma_result
    #         }
    #         tfidf_scores = {
    #             i['file_name']: i['cosine_similarity']
    #             for i in tfidf_result
    #         }
    #         combinedScores = {}
    #         for file in set(chroma_scores).union(tfidf_scores.keys()):
    #             chroma_score = chroma_scores.get(file, 0)
    #             tfidf_score = tfidf_scores.get(file, 0)
    #             combinedScores[file] = chroma_score * chroma_weight + tfidf_score * tfidf_weight

    #         sorted_result = sorted(combinedScores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    #         return chroma_result, tfidf_result, sorted_result'''

    def search(self, query: str, top_k: int = 20):
        def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
                last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
                return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        encoded_input = self.__transformers_tokenizer(query, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.__transformers_model(**encoded_input)
        embedding = average_pool(model_output.last_hidden_state, encoded_input['attention_mask'])
        embedding = F.normalize(embedding.cpu(), p=2, dim=1).squeeze().numpy()

        results = self.__collection.query(
            query_embeddings=[embedding],
            n_results=top_k*2,  # 先取多一點，用來過濾重複
            include=["metadatas", "distances", "documents"]
        )

        # 去除同商品名稱重複（可換成唯一 ID 比較更嚴謹）
        unique_results = []
        seen_names = set()

        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            product_name = metadata["name"]
            if product_name not in seen_names:
                seen_names.add(product_name)
                unique_results.append({
                    "name": product_name,
                    "url": metadata["url"],
                })

        top_k_unique = unique_results[:top_k]
        sql_result = []
        for i in top_k_unique:
            sql = f"SELECT * FROM senior_project_2.products WHERE `name` = '{i['name']}' and `link` = '{i['url']}';"
            try:
                self.__mySQLCursor.execute(sql)
                SQLresult = self.__mySQLCursor.fetchall()
                if SQLresult:
                    # sql_result.append({
                    #     'name': SQLresult[0][0],
                    #     'price': SQLresult[0][1],
                    #     'spacs': SQLresult[0][2],
                    #     'rating': SQLresult[0][3],
                    #     'link': SQLresult[0][4],
                    #     'comments': SQLresult[0][5]
                    # })
                    sql_result.append(SQLresult)
            except pymysql.MySQLError as e:
                print(f"Error fetching data from MySQL: {e}")
                self.__mySQLClient.rollback()
                continue
        sql_result = json.dumps(sql_result, ensure_ascii=False, indent=4)
        with open('search_result.json', 'w+', encoding='utf-8') as f:
            f.write(sql_result)
        # return top_k_unique
    
    def delPDF(self, file_name: str) -> None:
        if file_name not in self.__fileList:
            print(f'{file_name} does not exist')
            return
        
        # delete from chroma
        self.__collection.delete(where={"file_name": file_name})
        # self.__cromaClient.persist()
        
        # delete from TF-IDF
        vectorizer, matrix, fileIndex = self.__tfidfLoadData()
        del fileIndex[file_name]
        document = list(fileIndex.values())
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(document)
        self.__tfidfStoreData(vectorizer, matrix, fileIndex)

        print(f'Deleted {file_name}')
        self.__fileCount -= 1
        del self.__fileList[self.__fileList.index(file_name)]

    def getFileCount(self) -> int:
        return self.__fileCount
    
    def getFileList(self) -> list:
        return self.__fileList


