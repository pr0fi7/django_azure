from django.shortcuts import render, redirect
from django.http import JsonResponse
import openai

from django.contrib import auth
from django.contrib.auth.models import User
import openai.cli
from .models import Chat
from django.utils import timezone

from .forms import DocumentUploadForm, ChatForm

from io import BytesIO
import pdfplumber
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema.document import Document
from hashlib import md5

from django.contrib.auth.decorators import login_required

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key=os.getenv("OPENAI_API_KEY")
openaiapi = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma.db"

def get_embedding_function():
    # Set the API key either from environment variable or directly
    api_key = os.getenv('OPENAI_API_KEY', openaiapi) 
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings

def add_to_chroma(chunks, chat_id):
    chat_db_directory = os.path.join(CHROMA_PATH, f'chroma_{chat_id}')

    os.makedirs(chat_db_directory, exist_ok=True)

    db = Chroma(
        persist_directory=chat_db_directory, embedding_function=get_embedding_function()
    )
    db.add_documents(chunks)
    
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk['metadata']["id"] not in existing_ids:
            new_chunks.append(chunk)
        else:
            print(f"Skipping chunk with id {chunk['metadata']['id']} - already exists.")
    print('new_chunks:', new_chunks)
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk['metadata']["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("Persisting database...")
        db.persist()
    else:
        print("✅ No new documents to add")

def split_documents(documents: list[Document]):
    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        content_hash = md5(chunk.page_content.encode('utf-8')).hexdigest()[:8]
        current_page_id = f"{source}:{content_hash}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    print('chunks:', chunks)
    return chunks


def ask_openai(message, chat_id):

    chat_db_directory = os.path.join(CHROMA_PATH, f'chroma_{chat_id}')

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chat_db_directory, embedding_function=embedding_function)
    context = db.similarity_search_with_score(message, k=5)

    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Provide a response to the following message:{message} based on following context:{context}"},
        ]
    )
    
    answer = response.choices[0].message.content.strip()
    return answer

# Create your views here.
def chatbot(request, chat_id):
    if chat_id:
        chats = Chat.objects.filter(user=request.user, chat_id=chat_id)

    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_openai(message, chat_id)

        chat = Chat(chat_id = chat_id, user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()

        return JsonResponse({'message': message, 'response': response})
            
    return render(request, 'chatbot.html', {'chats': chats})



@login_required
def start_chat(request):
    if request.method == 'POST':
        message = request.POST.get('message')

        response = openai.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {'role': "system", 'content': f"You are usefull assistant, write an answer to the following message: {message} and create a name for a chat based on this first message, without "" or anything like that: '{message}', write it in the following format: Answer: <answer>, Name: <name>"}
            ]
        )
        answer = response.choices[0].message.content.strip()
        response = answer.split("Name: ")[0].split("Answer: ")[1].strip()
        name = answer.split("Name: ")[1].strip()
        chat_id = name.replace(" ", "_").replace("/", "-")
        
        print('name:', answer)
        # Create the initial chat in the database
        new_chat = Chat.objects.create(
            user=request.user,
            chat_id=chat_id,
            message=message,
            response=response,
        )

        return redirect('chatbot', chat_id=chat_id)

    return render(request, 'start_chat.html')


@login_required
def chat_list_view(request):
    # Get all chats for the logged-in user
    user_chats = Chat.objects.filter(user=request.user).order_by('chat_id')

    # Use a set to track unique chat_ids and a list to store unique chats
    seen_chat_ids = set()
    unique_chats = []

    for chat in user_chats:
        if chat.chat_id not in seen_chat_ids:
            unique_chats.append(chat)
            seen_chat_ids.add(chat.chat_id)
    
    context = {
        'chats': unique_chats
    }
    return render(request, 'chats.html', context)


@login_required
def create_chat_view(request):
    if request.method == "POST":
        form = ChatForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('chats')
    else:
        form = ChatForm()
    
    context = {"form": form}
    return render(request, 'create_chat.html', context)

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('start_chat')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save()
                auth.login(request, user)
                return redirect('start_chat')
            except:
                error_message = 'Error creating account'
                return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = 'Password dont match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('login')

def upload(request, chat_id):
    if request.method == "POST":
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            document = request.FILES['document']
            raw_data = document.read()

            pdf_stream = BytesIO(raw_data)
            
            try:
                all_text = ""
                with pdfplumber.open(pdf_stream) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        all_text += text if text else ""

                print('all_text:', all_text)
                
                if not all_text.strip():
                    return render(request, 'upload.html', {'form': form, 'error': 'Could not extract text from the uploaded PDF.'})
                documents = [Document(page_content=text, metadata={"source": document.name})]
                chunks = split_documents(documents)
                print('chunks:', chunks)
                # Add to Chroma
                add_to_chroma(chunks, chat_id=chat_id)

                return render(request, 'upload.html', {'form': form, 'success': 'PDF file uploaded successfully.'})
                

            except Exception as e:
                return render(request, 'upload.html', {'form': form, 'error': f'Error processing PDF file: {str(e)}'})
    else:
        form = DocumentUploadForm()
    context = {
        'chat_id': chat_id, 
        'form': form
    }
    return render(request, 'upload.html', context= context)
