#!/usr/bin/env python
# coding: utf-8

# In[16]:


#!pip install torch #torchvision torchaudio
#!pip install transformers
#!pip install streamlit


# In[17]:

import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import re
import pickle 
import os


def download_model():
	url = 'https://drive.google.com/uc?id=1-4f9SfmaTLFKPliRnf95HKG2apJI6NWe'
	output = 'bert_bigru_model.pth'
	if not os.path.exists(output):
		gdown.download(url, output, quiet=False) 

#Roman Number IV A into IVA

def standardize_roman_numerals(text):
  if not isinstance(text, str):
    return text  # or you can return a default value like '' if you want to replace non-strings
  roman_pattern = re.compile(r"(?<=\b)(IV|IX|XL|XC|CD|CM|I{1,3}|V|X{1,3}|L|C{1,3}|D|M{1,3})([-\s])([A-Za-z])(?=\b)")
  return roman_pattern.sub(r"\1\3", text)

#Letter PS 1 into PS1

def standardize_ps(text):
  if not isinstance(text, str):
    return text  # or you can return a default value like '' if you want to replace non-strings
  regex_pattern = r'(PS)([\s-]*)(\d+)'
  return re.sub(regex_pattern, r'\1\3', text)

#Roman dictionary
roman_numerals = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
    'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20
  }


def roman_to_number(text) :
  return roman_numerals.get(text.upper(), text)

# Standardize "tipe" or "type" to "type"

def standardize_typetest(text):

  def replace_with_standardized(match):
    type_word = match.group(1)
    number = match.group(2).replace('-', '').replace(' ', '')
    if number.upper() in roman_numerals:
      number = roman_to_number(number.upper())
    return f'{type_word}{number}'

  text = re.sub(r'\b(tipe|type)\b[\s-]*(\b(?:I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|XI{0,3}|IX|XX|[0-9]+)\b)', replace_with_standardized, text, flags=re.IGNORECASE)
  return text

def normalize_text(text):
  if not isinstance(text, str):
    return text  # or you can return a default value like '' if you want to replace non-strings

  # Standardize Covid variations to 'Covid19'
  text = re.sub(r'\bcovid\b', 'Covid', text, flags=re.IGNORECASE)
  text = re.sub(r'Covid[\s-]*19', 'Covid19', text, flags=re.IGNORECASE)
  text = re.sub(r'Covid(?!\d+)', 'Covid19', text, flags=re.IGNORECASE)

  # Replace CKR and its variants with 'cidera kepala ringan'
  text = re.sub(r'\bckr\b', 'cidera kepala ringan', text, flags=re.IGNORECASE)

  # Replace CKR and its variants with 'cidera kepala ringan'
  text = re.sub(r'\bcks\b', 'cidera kepala ringan', text, flags=re.IGNORECASE)

  # Replace stg, stage and its variants with 'stage'
  text = re.sub(r'\bstg\b|\bstage\b', 'stage', text, flags=re.IGNORECASE)

  return text


# In[18]:

download_model()

# Define the model class (ensure it matches your training code)
class StackedGRUModel(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, n_layers, num_classes, bidirectional=True, dropout=0.2):
        super(StackedGRUModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedded = bert_outputs.last_hidden_state
        embedded = self.dropout(embedded)
        gru_out, _ = self.gru(embedded)
        gru_out = self.dropout(gru_out)
        logits = self.fc(gru_out[:, -1, :])
        probs = self.softmax(logits)
        return probs


# In[19]:


# Function to load the model
@st.cache_resource
def load_model():
    bert_model_name = 'bert-base-uncased'
    hidden_dim = 256
    n_layers = 2
    num_classes = 10
    dropout = 0.2
    bidirectional = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StackedGRUModel(bert_model_name, hidden_dim, n_layers, num_classes, bidirectional, dropout).to(device)
    save_path = 'bert_bigru_model.pth'
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model, device


# In[20]:


# Function to load the saved LabelEncoder


def load_label_encoder():
    
    # Path to the saved LabelEncoder
    filepath = 'label_encoder.pkl'  # Adjust the path as needed


    with open(filepath, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder


# In[22]:


# Streamlit app
st.title("Klasifikasi Text dengan BERT and BiGRU")
st.write("Masukkan text untuk klasifikasi:")

# Initialize session state
if 'model' not in st.session_state:
    with st.spinner('Loading model...'):
        st.session_state.model, st.session_state.device = load_model()
        st.session_state.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
        st.session_state.label_encoder = load_label_encoder()   

text_input = st.text_area("Diagnosa primer", "Contoh text .")

st.write("Kode ICD 10: [A16.2, D38.1, I25.1, K30, Z03.1, K01.1]")
st.write(" NN = [J47,S06.0, U07.1, C34.9]")

if st.button("Classify"):
    if text_input:
        try:
            # Preprocess the text
            preprocess = text_input
            preprocess = standardize_roman_numerals(preprocess)
            preprocess = standardize_ps(preprocess)
            preprocess = normalize_text(preprocess)
            preprocess = standardize_typetest(preprocess)
            text_input = preprocess        

            inputs = st.session_state.tokenizer(text_input, return_tensors='pt', padding=True, truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(st.session_state.device)
            attention_mask = inputs['attention_mask'].to(st.session_state.device)

            if st.session_state.model is not None:
                with torch.no_grad():
                    outputs = st.session_state.model(input_ids, attention_mask)
                    predicted_class = torch.argmax(outputs, dim=1).cpu().numpy()  
                    predicted_label = st.session_state.label_encoder.inverse_transform(predicted_class)[0] 
                st.write(f"Prediksi ICD 10: {predicted_label}")
            else:
                st.write("Model tidak dapat dipanggil.")
        except Exception as e:
            st.error(f"Error saat mengklasifikasikan: {e}")
    else:
        st.write("Silahkan masukkan text untuk prediksi klasifikasi.")


# In[ ]:




