import streamlit as st
from transformers import pipeline, BertTokenizer, BertModel, BertForMaskedLM
import torch

st.title('Bert Demo App:')
st.text('Select Bert Model:')
model_in = st.selectbox('Models:', ['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased'],
                        index=0)
option = st.selectbox('Mode:', ['Pre-implemented Pipeline', 'Explain Steps'], index=1)

if option == 'Pre-implemented Pipeline':
    with st.echo():
        unmasker = pipeline('fill-mask', model=model_in)

    sentence = st.text_input('Enter Sentence:', value="Hello I'm a [MASK] model.")
    with st.echo():
        res = unmasker(sentence)

    st.dataframe(res)

else:
    st.text('1) Load Tokenizer from given Bert Model')
    with st.echo():
        tokenizer = BertTokenizer.from_pretrained(model_in)

    sentence = st.text_input('2) Enter Sentence:', value="I want to [MASK] the car because it is cheap . Malaria is a [MASK] disease")
    st.text('2.1)Add [CLS]/[SEP] Tags')
    sentence = "[CLS] " + sentence.strip() + " [SEP]"
    st.text('Sentence: \n' + sentence)
    st.text('3) Tokenize Sentence:')

    with st.echo():
        tokenized_text = tokenizer.tokenize(sentence)
    st.text('4) Tokenized Sentence:')
    st.text(tokenized_text)
    st.text('5) Convert the sequence of tokens to ids using the Bert model Vocab')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    st.text(indexed_tokens)

    st.text('6) Create Input/Prediction Tensors')
    with st.echo():
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([[0] * len(tokenized_text)])

    st.text("7) Load pre-trained model (retrieve weights)")

    with st.echo():
        model = BertForMaskedLM.from_pretrained(model_in)
        # activate evaluation mode
        model.eval()
        # calculate index of masked word
        masked_index = [i for i, x in enumerate(tokenized_text) if x == '[MASK]']

    st.text('8) Predict all Tokens')

    with st.echo():
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)

    st.text('9) For each masked word get the prediction with the highest score and convert it to text again')

    for i in masked_index:
        st.text('Mask ID: '+str(i))
        predicted_index = torch.argmax(predictions[0][0, i]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        st.text(predicted_token)
