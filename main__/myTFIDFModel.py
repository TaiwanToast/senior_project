from myProcesser import Processer
import os
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MyTFIDFModel(Processer):
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


