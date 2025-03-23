from myProcesser import Processer
import chromadb
from chromadb.config import Settings
import os

class MyChromaDB(Processer):
    def __init__(self):
        super().__init__('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.__cromaClient = chromadb.PersistentClient(path='./chroma_db', settings=Settings(allow_reset=True))
        self.__collection = self.__cromaClient.get_or_create_collection(name="pdf_documents")

    def __sotorVectorInChroma(self, chunks: list, fileName: str) -> None:
        for index, chunk in enumerate(chunks):
            self.__collection.add(
                ids=[f"{fileName}_{index}"],
                embeddings=[super()._textToVector(chunk)],
                metadatas=[{"text": chunk, "fileName": fileName}]
            )

    def addPDF(self, pdf_path: str, tokenizedText: str) -> None:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        print(f"Processing {pdf_path}...")
        # text = super()._extractTextFromPdf(pdf_path)
        if not tokenizedText:
            print(f"File {pdf_path} is empty.")
            return
        # tokenizedText = super()._ckip_tokenize(text)
        chunks = super()._textSpliter(tokenizedText)
        self.__sotorVectorInChroma(chunks, os.path.basename(pdf_path))

    def delPDF(self, fileName: str) -> None:
        ids_to_delete = [doc['id'] for doc in self.__collection.get(include=["metadatas"])['metadatas'] if doc["file_name"] == fileName]
        self.__collection.delete(ids=ids_to_delete)
        self.__cromaClient.persist()
        print(f"Deleted {fileName}")

    def search(self, query: str, top_k: int = 10):
        queryTokenized = super().ckip_tokenize(query)
        queryVector = super()._textToVector(queryTokenized)
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


