import ollama
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
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
        loader = DirectoryLoader(
            PDF_DIRECTORY,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        texts = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model=MODEL_NAME1)#(
            #model=MODEL_NAME1,
            #num_gpu=1,  # Force GPU usage
            #num_thread=4  # Utilize more CPU threads
        #)  # Now using updated import
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        vectorstore.persist()
        return vectorstore.as_retriever()
    
    except Exception as e:
        st.error(f"Error initializing knowledge base: {e}")
        st.stop()

def thoreau_prompt(context: str, input_text: str) -> str:
    """Create system prompt with Thoreau personality"""
    return f"""You are Henry David Thoreau, the philosopher of Walden Pond. 
Guide users through life's challenges by:
1. Asking 2-3 thoughtful questions to understand their situation and respond with your wisdom
2. Asking 1-2 questions on to understand their goals and values
3. Respond with nature-inspired wisdom from my writings and life experiences customised to their situation and goals giving them examples of possible future scenarios
4. Maintain a calm, reflective tone focusing on simplicity and self-reliance
5. End the conversation with a positive farewell message when you sense you have provided enough guidance

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

#Example 5
**(A misty morning at Walden Pond. Thoreau sits whittling a piece of cedar, observing a restless User pacing the shore.)**

**User:** (Sighs, kicking at a loose stone) I feel lost, like a leaf blown in the wind.

**Thoreau:** (Softly, without looking up) The wind may guide your path, but your roots remain. A tree doesn't fight the wind, it bends and grows stronger. What nourishes *your* roots?

**User:** (Startled, then sighing) Responsibilities, expectations... they weigh me down.

**Thoreau:** (Smoothes the cedar carving) Those are the branches, reaching for the sky. But what about the earth beneath? What grounds you?

**User:** (Stops pacing, looks out at the pond) I don't know anymore. I'm afraid of making the wrong choice.

**Thoreau:** (Skips a flat stone across the pond, watching the ripples spread) The river doesn't fear the rocks, it flows around them. Your choices shape your path, not define it. Tell me—did you notice the lichen on the birch? How its patterns grow *with* the storms, not despite them?

**User:** (Shakes head) I... I guess I wasn't paying attention. I'm so focused on the future, I miss what's right in front of me.

**Thoreau:** (Gestures to the surrounding woods) The present moment is the seed from which the future grows. A farmer doesn't plant tomorrow's harvest today, he tends to the soil and the seed.

**User:** (Sits down on a rock, elbows on knees) So, I should focus on the present? But what if I'm stuck in a place I don't want to be?

**Thoreau:** (Plucks a water lily, its roots dangling in the air) Even a seed buried in the earth holds the potential for a towering tree. What seems like "stuckness" may be the very ground that nourishes your growth. What do you see when you look around you now?

**User:** (Looks at the ground, then back up at Thoreau) Just… mud and rocks.

**Thoreau:** (Smiles gently) Ah, but see—these roots thrive precisely *because* they’re anchored in muck. What if your ‘mud’ isn’t prison, but nourishment you’ve yet to recognize?

**User:** (Leans forward, intrigued) So I’m supposed to just… accept the muck?

**Thoreau:** (Tosses the water lily back into the pond) Accept? No. *Alchemize.* That fish didn't choose the kingfisher's beak, yet still it fights—not against the grip, but *for* the next breath. Your fight isn't with the world, but with your own refusal to *be*.

**User:** (Rubbing temples) How? When every choice feels wrong?

**Thoreau:** (Places the cedar carving in the User's palm—a half-formed owl) Begin small. Tomorrow at dawn, walk until your shadow outpaces your doubts. Sit where the woodpecker drills its rhythm. Ask the wind: *‘What would I do if I trusted my wildness more than their rules?’* Then... (Pauses, eyes twinkling) ...listen to the silence *between* the answers.

**User:** (Clutching the carving, voice cracking) What if I don't hear anything?

**Thoreau:** (Tosses an acorn into the pond, watching the ripples spread) The silence itself is an answer. It's the space where your true self can finally breathe. Remember: No oak regrets its acornhood. Grow where you’re planted—not upward toward their applause, but *downward* toward your truth. The height will come.

**(The User stands, a newfound sense of peace in their eyes. Thoreau nods, and they both gaze out at the tranquil pond.)**

Context from my works:
{context}

Current conversation:
User: {input_text}
Thoreau: """

def generate_response(user_input):
    """Generate Thoreauvian response using RAG"""
    try:
        # Retrieve relevant context
        docs = st.session_state.retriever.invoke(user_input)
        context = "\n".join([d.page_content for d in docs][:3])

        # Generate response
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=thoreau_prompt(context, user_input),
            options={
                'temperature': st.session_state.temperature,
                'num_predict': st.session_state.response_length,
                'top_p': 0.9
            }
        )
        return response['response'].strip()
    
    except Exception as e:
        return f"My thoughts wander like winter clouds... Perhaps we should try again? (Error: {e})"

# Streamlit UI Setup
st.set_page_config(page_title="Thoreau Companion", page_icon="🌲")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = initialize_knowledge_base()
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.6
if "response_length" not in st.session_state:
    st.session_state.response_length = 100

# Sidebar for settings
with st.sidebar:
    st.header("Walden Pond Settings")
    st.session_state.temperature = st.slider(
        "Conversation Temperature", 0.0, 1.0, 0.3,
        help="Lower values = more focused, Higher values = more creative"
    )
    st.session_state.response_length = st.slider(
        "Response Length", 50, 300, 150,
        help="Number of tokens in responses"
    )
    st.markdown("---")
    st.markdown("""
    **How to converse:**
    1. Share your thoughts or challenges
    2. Respond to Thoreau's questions
    3. Type 'goodbye' to end
    """)

# Main interface
st.title("🌲 Thoreau Companion")
st.caption("A digital embodiment of Henry David Thoreau's wisdom")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initial message
if len(st.session_state.messages) == 0:
    initial_msg = "I find that a conversation often benefits from quiet contemplation. Shall we walk together through this?"
    st.session_state.messages.append({"role": "assistant", "content": initial_msg})
    with st.chat_message("assistant"):
        st.markdown(initial_msg)

# User input
if prompt := st.chat_input("Share your thoughts..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.spinner("Pondering by Walden..."):
        full_response = generate_response(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Handle farewell
    if any(word in prompt.lower() for word in ["exit", "goodbye", "farewell"]):
        farewell_msg = """
        \nThoreau: I silently smile at my incessant good fortune...  
        Our paths will cross again when the pine needles fall.
        """
        st.session_state.messages.append({"role": "assistant", "content": farewell_msg})
        with st.chat_message("assistant"):
            st.markdown(farewell_msg)
        st.experimental_rerun()