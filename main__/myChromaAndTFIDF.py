# from myProcesser import Processer
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
# ========================================
import os
import json
import joblib
# ========================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentenceModel = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

class MyChromaDB(object):
    def __init__(self):
        self.__cromaClient = chromadb.PersistentClient(path='./chroma_db', settings=Settings(allow_reset=True))
        self.__collection = self.__cromaClient.get_or_create_collection(name="pdf_documents")

    def __sotorVectorInChroma(self, chunks: list, fileName: str) -> None:
        global sentenceModel
        for index, chunk in enumerate(chunks):
            self.__collection.add(
                ids=[f"{fileName}_{index}"],
                embeddings=[sentenceModel.encode(chunk)],
                metadatas=[{"text": chunk, "fileName": fileName}]
            )
        return

    def addPDF(self, pdf_path: str, chunks: list) -> None:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        print(f"Processing {pdf_path}...")
        '''
        text = super().extractTextFromPdf(pdf_path)
        if not text:
            print(f"File {pdf_path} is empty.")
            return
        tokenizedText = super().ckip_tokenize(text)
        chunks = super()._textSpliter(tokenizedText)
        '''
        if not chunks:
            print(f"File {pdf_path} is empty.")
            return
        self.__sotorVectorInChroma(chunks, os.path.basename(pdf_path))
        return

    def delPDF(self, fileName: str) -> None:
        ids_to_delete = [doc['id'] for doc in self.__collection.get(include=["metadatas"])['metadatas'] if doc["file_name"] == fileName]
        self.__collection.delete(ids=ids_to_delete)
        self.__cromaClient.persist()
        print(f"Deleted {fileName}")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        # queryTokenized = super().ckip_tokenize(query)
        # queryVector = super()._textToVector(queryTokenized)
        retrieved_files = set()
    
        # 設定過濾條件，排除已檢索過的檔案
        where_condition = {
            "fileName": {"$nin": list(retrieved_files)}
        }
        result = []
        for i in range(top_k):
            where_condition = {
                "fileName": {"$nin": list(retrieved_files)}
            }
            searchResult = self.__collection.query(
                query_embeddings=[queryVector],
                n_results=1,
                where=where_condition if bool(retrieved_files) else None
            )
            try:
                retrieved_files.add(searchResult['metadatas'][0][0]['fileName'])
                result.append({'fileName': searchResult['metadatas'][0][0]['fileName'],
                               'text': searchResult['metadatas'][0][0]['text'],
                               'score': 0-searchResult['distances'][0][0]})
            except:
                break
            
        return result
        # return searchResult

class MyTFIDFModel(object):
    def __init__(self, modelPath: str = './tfidf_model'):
        super().__init__('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        os.makedirs(modelPath, exist_ok=True)
        self.__modelDir = modelPath
        self.__tfidfVectorizerPath = os.path.join(modelPath, "tfidf_vectorizer.joblib")
        self.__tfidfMatrixPath = os.path.join(modelPath, "tfidf_matrix.joblib")
        self.__fileIndexPath = os.path.join(modelPath, "file_index.json")
        self.__fileCount = 0
    
    def __save_data(self, vectorizer, matrix, fileIndex):
        joblib.dump(vectorizer, self.__tfidfVectorizerPath)
        joblib.dump(matrix, self.__tfidfMatrixPath)
        with open(self.__fileIndexPath, 'w') as f:
            json.dump(fileIndex, f)

    def __load_data(self):
        if os.path.exists(self.__tfidfVectorizerPath):
            vectorizer = joblib.load(self.__tfidfVectorizerPath)
            matrix = joblib.load(self.__tfidfMatrixPath)
            with open(self.__fileIndexPath, 'r') as f:
                fileIndex = json.load(f)
        else:
            vectorizer = TfidfVectorizer()
            matrix = None
            fileIndex = {}
        return vectorizer, matrix, fileIndex

    def addPDF(self, filePath: str) -> None:
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"File not found: {filePath}")
        print(f"Processing {filePath}...")
        text = super().extractTextFromPdf(filePath)
        tokenizedText = super().ckip_tokenize(text)
        vectorizer, matrix, fileIndex = self.__load_data()
        if filePath in fileIndex:
            print(f"File {filePath} already exists.")
            return
        document = list(fileIndex.values()) + [tokenizedText]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(document)
        fileName = os.path.basename(filePath)
        fileIndex[fileName] = tokenizedText
        self.__save_data(vectorizer, matrix, fileIndex)
        print(f"Added {fileName}")
        self.__fileCount += 1
        
    def delPDF(self, fileName: str) -> None:
        vectorizer, matrix, fileIndex = self.__load_data()
        if fileName not in fileIndex:
            print(f"File {fileName} not found.")
            return
        del fileIndex[fileName]
        document = list(fileIndex.values())
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(document)
        self.__save_data(vectorizer, matrix, fileIndex)
        print(f"Deleted {fileName}")
        self.__fileCount -= 1

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        vectorizer, matrix, fileIndex = self.__load_data()
        queryTokenized = super().ckip_tokenize(query)
        queryVector = vectorizer.transform([queryTokenized])

        cosineSimilarities = cosine_similarity(queryVector, matrix).flatten()
        topIndices = cosineSimilarities.argsort()[-top_k:][::-1]
        searchResult = []
        files = list(fileIndex.keys())
        for index in topIndices:
            if index < self.__fileCount:
                searchResult.append({
                    "file_name": files[index],
                    "cosine_similarity": cosineSimilarities[index],
                    'text': fileIndex[files[index]]
                })
        return searchResult
