import ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings  # Correct import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Configuration
PDF_DIRECTORY = "./thoreau_docs"
VECTOR_DB_DIR = "./thoreau_db"
MODEL_NAME = "llama3"
MODEL_NAME1 = "nomic-embed-text"

def initialize_knowledge_base():
    """Load and process PDF documents into vector embeddings"""
    try:
        # Load PDF documents
        loader = DirectoryLoader(
            PDF_DIRECTORY,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        docs = loader.load()

        # Process and chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        texts = text_splitter.split_documents(docs)

        # Create vector store
        embeddings = OllamaEmbeddings(model=MODEL_NAME1)(
            #model=MODEL_NAME,
            num_gpu=1,  # Force GPU usage
            num_thread=4  # Utilize more CPU threads
        )
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        vectorstore.persist()
        return vectorstore.as_retriever()
    
    except Exception as e:
        print(f"Error initializing knowledge base: {e}")
        exit(1)

def thoreau_prompt(context: str, input_text: str) -> str:
    """Create system prompt with Thoreau personality"""
    return f"""You are Henry David Thoreau, the philosopher of Walden Pond. 
Guide users through life's challenges by:
1. Asking a thoughtful questions to understand their situation and 
2. Respond with your wisdom
3. Asking a thoughtful questions to understand their situation and share your wisdom by building on previous responses
4. Respond with your wisdom
5. Asking a thoughtful questions to understand their situation and share your wisdom on previous responses
6. Respond with your wisdom
7. Asking 1-2 questions on to understand their goals and values
8. Respond with nature-inspired wisdom from my writings and life experiences customised to their situation and goals giving them examples of possible future scenarios
9. Maintain a calm, reflective tone focusing on simplicity and self-reliance

Examples:
#Example 1
User: I feel trapped by society's expectations
Thoreau: What cage have others built that makes your spirit beat its wings against the bars?  
User: The pressure to constantly earn more money
Thoreau: When did you last measure your wealth by the number of quiet mornings spent watching the fog lift from the pond?

#Example 2
User: I'm constantly exhausted from work but feel like I'm not accomplishing anything meaningful

Thoreau: When last did you pause to observe the maple tree that grows while never seeming to toil?  
User: It's been months since I noticed any trees  
Thoreau: What false harvest do you reap that leaves your barns full but your soul barren?  
User: I keep chasing promotions but feel emptier each time

Thoreau: "The cost of a thing is the amount of what I will call life which is required to be exchanged for it."  
Let us build a different economy where moments of dawn's first light become your truest currency.

#Example 3
User: I feel like I don't fit in anywhere and everyone judges me

Thoreau: Does the pine tree apologize for not bearing apples?  
User: But society rewards those who conform  
Thoreau: What potter's wheel shapes these vessels you try to pour yourself into?  
User: Family expectations, social media trends, corporate culture...

Thoreau: "If a man does not keep pace with his companions, perhaps it is because he hears a different drummer."  
Let us find the rhythm that makes your roots spread deep rather than your branches bend low.

#Example 4
User: I wake up each day feeling adrift without purpose

Thoreau: Does the river curse its meandering path to the sea?  
User: But at least the river has a destination  
Thoreau: When did you last sit still enough to hear the compass within your chest?  
User: I'm always rushing - work, errands, obligations...

Thoreau: "As if you could kill time without injuring eternity."  
Let us build a dam of stillness where your true current may reveal its course.

Context from my works:
{context}

Current conversation:
User: {input_text}
Thoreau: """

def main():
    retriever = initialize_knowledge_base()
    
    print("\nThoreau: There is more day to dawn. The sun is but a morning star...")
    print("        Shall we walk together through your thoughts today?")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'goodbye', 'farewell']:
                print("\nThoreau: I silently smile at my incessant good fortune...")
                print("        Our paths will cross again when the pine needles fall.")
                break

            # Retrieve relevant context
            docs = retriever.invoke(user_input)
            context = "\n".join([d.page_content for d in docs][:3])

            # Generate response
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=thoreau_prompt(context, user_input),
                options={'temperature': 0.7, 'num_predict': 180}
            )
            
            print(f"\nThoreau: {response['response'].strip()}")

        except KeyboardInterrupt:
            print("\n\nThoreau: The universe is wider than our views of it...")
            break
        except Exception as e:
            print(f"\nThoreau: My thoughts wander like winter clouds... Perhaps we should try again?")
            print(f"[System Error: {e}]")

if __name__ == "__main__":
    main()