from sentence_transformers import SentenceTransformer, util
import numpy as np

model_path = './output/training_sts'

embedder = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = [
    '특허권침해소송의 상대방이 제조 등을 하는 제품 또는 사용하는 방법(이하 ‘침해제품 등’이라고 한다)이 특허발명의 특허권을 침해한다고 하기 위해서는 특허발명의 특허청구범위에 기재된 각 구성요소와 그 구성요소 간의 유기적 결합관계가 침해제품 등에 그대로 포함되어 있어야 한다.',
    '침해제품 등에 특허발명의 특허청구범위에 기재된 구성 중 변경된 부분이 있는 경우에도, 특허발명과 과제 해결원리가 동일하고, 특허발명에서와 실질적으로 동일한 작용효과를 나타내며, 그와 같이 변경하는 것이 그 발명이 속하는 기술분야에서 통상의 지식을 가진 사람이라면 누구나 쉽게 생각해 낼 수 있는 정도라면, 특별한 사정이 없는 한 침해제품 등은 특허발명의 특허청구범위에 기재된 구성과 균등한 것으로서 여전히 특허발명의 특허권을 침해한다고 보아야 한다.']

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = [
    '특허권침해소송의 상대방이 제조 등을 하는 제품이나 사용하는 방법(이하 ‘침해제품 등’)이 특허발명의 특허권을 침해한다 하기위해선 특허발명의 특허청구범위에 기재된 각 구성요소, 그리고 그 구성요소 간의 유기적 결합관계가 침해제품 등에 그대로 포함돼 있어야한다.',
    '침해제품 등에 특허발명의 특허청구범위에 기재된 구성 중 변경된 부분이 있는 경우 또한, 특허발명과 과제 해결원리가 동일하며 특허발명에서와 실질적으로 동일한 작용효과를 나타내고, 이처럼 변경하는 것이 해당 발명이 속하는 기술분야에서 통상의 지식을 가진 사람이라면 누구든지 쉽게 생각해낼수 있는 정도라면, 특별한 사정이 없는한 침해제품 등은 특허발명의 특허청구범위에 기재된 구성과 균등한 것으로 여전히 특허발명의 특허권을 침해한다고 봐야 된다.']

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
#top_k = 5
idx = -1
for query in queries:
    idx = idx + 1
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(
        query_embedding, corpus_embeddings)[0][idx]
    cos_scores = cos_scores.cpu()

    # We use np.argpartition, to only partially sort the top_k results
    #top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\n")
    # print("\nTop 5 most similar sentences in corpus:")

    # for idx in top_results[0:top_k]:
    print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores))
    print("\n\n======================\n\n")
