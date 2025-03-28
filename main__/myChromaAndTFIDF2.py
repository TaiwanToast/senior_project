import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
# ========================================
import os
import json
import joblib
import pypdf
# ========================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ========================================
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

class MyChromaAndTFIDF(object):
    def __init__(self, chroma_db_path: str = './chroma_db', tfidf_model_path: str = './tfidf_model', sentence_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.__ws_driver = CkipWordSegmenter(device=0)
        self.__pos_driver = CkipPosTagger(device=0)
        # ========================================
        self.__sentenceModel = SentenceTransformer(sentence_model)
        # ========================================
        self.__cromaClient = chromadb.PersistentClient(path=chroma_db_path, settings=Settings(allow_reset=True))
        self.__collection = self.__cromaClient.get_or_create_collection(name="pdf_documents")
        # ========================================
        os.makedirs(tfidf_model_path, exist_ok=True)
        self.__modelDir = tfidf_model_path
        self.__tfidfVectorizerPath = os.path.join(tfidf_model_path, "tfidf_vectorizer.joblib")
        self.__tfidfMatrixPath = os.path.join(tfidf_model_path, "tfidf_matrix.joblib")
        self.__fileIndexPath = os.path.join(tfidf_model_path, "file_index.json")
        # ========================================
        self.__fileCount = 0
        self.__fileList = []
    
    def __storeVectorInChroma(self, chunks: list, file_name: str) -> None:
        for index, chunk in enumerate(chunks):
            self.__collection.add(
                ids=[f"{file_name}_{index}"],
                embeddings=[self.__sentenceModel.encode(chunk)],
                metadatas=[{"text": chunk, "file_name": file_name}]
            )
        return
    
    def __tfidfStoreData(self, vectorizer, matrix, fileIndex) -> None:
        joblib.dump(vectorizer, self.__tfidfVectorizerPath)
        joblib.dump(matrix, self.__tfidfMatrixPath)
        with open(self.__fileIndexPath, 'w') as f:
            json.dump(fileIndex, f)
        return

    def __tfidfLoadData(self):
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
    
    def __ckip_tokenize(self, text: str) -> str:
        def clean(sentence_ws, sentence_pos):
            # 這裡只保留名詞和動詞，且去除單一字元
            short_sentence = []
            stop_pos = set(['Nep', 'Nh', 'Nb'])
            for word_ws, word_pos in zip(sentence_ws, sentence_pos):
                is_N_or_V = word_pos.startswith("V") or word_pos.startswith("N")
                is_not_stop_pos = word_pos not in stop_pos
                is_not_one_charactor = not (len(word_ws) == 1)
                if is_N_or_V and is_not_stop_pos and is_not_one_charactor:
                    short_sentence.append(f"{word_ws}")
            return " ".join(short_sentence)
        
        ws = self.__ws_driver([text])
        pos = self.__pos_driver(ws)
        short = clean(ws[0], pos[0])
        return short
    
    def addPDF(self, pdf_path: str) -> None:
        def extractTextFromPdf(pdf_path: str) -> str:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
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

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        print(f"Processing {pdf_path}...")

        # Extract text from PDF
        text = extractTextFromPdf(pdf_path)
        if not text:
            print(f"File {pdf_path} is empty.")
            return

        # Split text into chunks
        # chunks = textSpliter(text)

        tokenizedText = self.__ckip_tokenize(text)
        chunks = textSpliter(tokenizedText)

        # Tokenize each chunk
        # tokenizedChunks = [self.__ckip_tokenize(chunk) for chunk in chunks]
        # tokenizedChunks = []
        # tmpdoc = ''
        # for chunk in chunks:
        #     tmp = self.__ckip_tokenize(chunk)
        #     tokenizedChunks.append(tmp)
        #     tmpdoc += tmp

        # Store vectors in Chroma
        self.__storeVectorInChroma(chunks, os.path.basename(pdf_path))

        # Store TF-IDF data
        vectorizer, matrix, fileIndex = self.__tfidfLoadData()
        document = list(fileIndex.values()) + [tokenizedText]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(document)
        file_name = os.path.basename(pdf_path)
        fileIndex[file_name] = tokenizedText
        self.__tfidfStoreData(vectorizer, matrix, fileIndex)
        print(f"Added {file_name}")
        self.__fileCount += 1
        self.__fileList.append(file_name)
        return

    def search(self, query: str, top_k: int = 10, hybridSearch: bool = False, chroma_weight: float = 0.4, tfidf_weight: float = 0.6):
        tokenizedQuery = self.__ckip_tokenize(query)
        # print('tokenized query: ', tokenizedQuery)
        queryVector = self.__sentenceModel.encode(tokenizedQuery)
        vectorizer, matrix, fileIndex = self.__tfidfLoadData()
        tfidfQueryVector = vectorizer.transform([tokenizedQuery])

        # Search in Chroma
        retrieved_files = set()
        where_condition = {
            "file_name": {"$nin": list(retrieved_files)}
        }
        chroma_result = []
        for i in range(top_k):
            where_condition = {
                "file_name": {"$nin": list(retrieved_files)}
            }
            searchResult = self.__collection.query(
                query_embeddings=[queryVector],
                n_results=1,
                where=where_condition if bool(retrieved_files) else None
            )
            try:
                retrieved_files.add(searchResult['metadatas'][0][0]['file_name'])
                chroma_result.append({'file_name': searchResult['metadatas'][0][0]['file_name'],
                               'text': searchResult['metadatas'][0][0]['text'],
                               'score': 1 / searchResult['distances'][0][0]})
            except:
                break
        
        # Search in TF-IDF
        cosineSimilarities = cosine_similarity(tfidfQueryVector, matrix).flatten()
        topIndices = cosineSimilarities.argsort()[-top_k:][::-1]
        tfidf_result = []
        files = list(fileIndex.keys())
        for index in topIndices:
            if index < self.__fileCount:
                tfidf_result.append({
                    "file_name": files[index],
                    "cosine_similarity": cosineSimilarities[index],
                    'text': fileIndex[files[index]]
                })
        if not hybridSearch:
            return chroma_result, tfidf_result
        else:
            chroma_scores = {
                i['file_name']: i['score']
                for i in chroma_result
            }
            tfidf_scores = {
                i['file_name']: i['cosine_similarity']
                for i in tfidf_result
            }
            combinedScores = {}
            for file in set(chroma_scores).union(tfidf_scores.keys()):
                chroma_score = chroma_scores.get(file, 0)
                tfidf_score = tfidf_scores.get(file, 0)
                combinedScores[file] = chroma_score * chroma_weight + tfidf_score * tfidf_weight

            sorted_result = sorted(combinedScores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return chroma_result, tfidf_result, sorted_result
    
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



