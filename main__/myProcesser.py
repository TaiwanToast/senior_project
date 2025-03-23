from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
from sentence_transformers import SentenceTransformer
import pypdf

class Processer:
    def __init__(self, sentence_model: str):
        self._ws_driver = CkipWordSegmenter(device=0)
        self._pos_driver = CkipPosTagger(device=0)
        self._sentence_model = SentenceTransformer(sentence_model)
    
    # 提取PDF文件中的文本
    def extractTextFromPdf(self, pdf_path: str) -> str:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    
    # 將文本進行分割
    def _textSpliter(self, text: str, chunkSize = 250, chunkOverlap = 20) -> list:
        chunks = []
        flag = False
        for i in range(0, len(text), chunkSize):
            if not flag:
                chunks.append(text[i:i+chunkSize])
                flag = True
            else:
                chunks.append(text[i-chunkOverlap:i+chunkSize])
        return chunks

    # 將文本進行向量化
    def _textToVector(self, text: str) -> list:
        return self._sentence_model.encode(text)

    # 將文本進行斷詞
    def ckip_tokenize(self, text: str) -> str:
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
        
        ws = self._ws_driver([text])
        pos = self._pos_driver(ws)
        short = clean(ws[0], pos[0])
        return short