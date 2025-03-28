# def textSpliter(text: str, chunkSize: int, chunkOverlap: int) -> list:
#     chunks = []
#     flag = False
#     for i in range(0, len(text), chunkSize):
#         if not flag:
#             chunks.append(text[i:i+chunkSize])
#             flag = True
#         else:
#             chunks.append(text[i-chunkOverlap:i+chunkSize])
#     return chunks
    
# if __name__ == "__main__":
#     text = "This Is A Test Text."
#     chunkSize = 5
#     chunkOverlap = 2
#     print(textSpliter(text, chunkSize, chunkOverlap))

l = ['a', 'b']
del l[l.index('a')]
print(l)