from ollama import embed
import chromadb as chroma
import CONFIGURATION


ChromaClient = chroma.Client()
informationList = CONFIGURATION.INFORMATION_LIST
embeddingModelUsed = CONFIGURATION.EMBEDDING_MODEL

class embeddingFunction(chroma.EmbeddingFunction):
    def __call__(self, input: chroma.Documents) -> chroma.Embeddings:
        return embed(model=embeddingModelUsed, input=input)['embeddings']

collection = ChromaClient.create_collection(
    name="IDEAsiaFacts",
    embedding_function=embeddingFunction()
)

collection.add(
    ids=[f"id{i}" for i in range(len(informationList))],
    documents=informationList
)

def relevantInformationRelatedTo(topic : str, amount : int = 5):
    return collection.query(
        query_texts=[topic], # Chroma will embed this for you
        n_results=amount # how many results to return
    )["documents"][0]

# print(relevantInformationAbout("Nomor telepon"))