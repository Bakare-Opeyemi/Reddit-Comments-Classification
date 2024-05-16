import pandas as pd
import numpy as np
import streamlit as st
from keras.models import load_model
from extraObjects import TransformerEncoder, TokenAndPositionEmbedding, preprocessComment, ensemblePrediction,preprocessBatch
#from keras.models import load_model


@st.cache_data
def load_models():
    """
    Load models
    """
    gru_model = load_model('models/gru_modelv2.h5')
    lstm_model = load_model('models/lstm_modelv2.keras')
    cnn_model = load_model('models/cnn_modelv2.keras')
    transformer_model = load_model('models/trf_modelv2.keras',
                                   custom_objects={"TransformerEncoder": TransformerEncoder, "TokenAndPositionEmbedding": TokenAndPositionEmbedding})
    return gru_model,lstm_model,cnn_model,transformer_model

gru_model,lstm_model,cnn_model,transformer_model = load_models()

prediction_decoding = {0:"medical doctor", 1:"veterinarian", 2:"person who is neither a medical doctor not a veterinarian" }

def main():
    st.title('Social Media Comment Classification')
    html_temp = """
    <div style="background:#051733 ;padding:10px">
    <h2 style="color:white;text-align:center;">Reddit Comment Profession Classifier</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html = True)
    st.divider()

    html_temp2 = """
    <div style="background:#14302f ;padding:10px">
    <p style="color:white;text-align:left;">This project aims to infer the profession of reddit users from their comments </p>
    <h5 style="color:white;text-align:left;">Class Codes</h5>
    <p style="color:white;text-align:left;">Medical Practitioner: 0, Veterinarian: 1, Others:2 </p>
    </div>
    """
    st.markdown(html_temp2, unsafe_allow_html = True)
    st.divider()
    news_story = st.text_area('Enter a Reddit Comment', height=200)

    if st.button('Identify text'):
        input_data = preprocessComment(news_story)
        lstmPred = np.argmax(lstm_model.predict(input_data))
        cnnPred = np.argmax(cnn_model.predict(input_data))
        gruPred = np.argmax(gru_model.predict(input_data))
        trfPred = np.argmax(transformer_model.predict(input_data))
        ensemblePred = ensemblePrediction([cnnPred,gruPred,trfPred,lstmPred])

        st.success("A " + prediction_decoding[ensemblePred] + " made this comment")
    
    st.divider()
    dataset = st.file_uploader("Upload a csv dataset of reddit comments. Comments must be in a 'comments' column", type=["csv"], accept_multiple_files = False)

    if st.button('Identify Batch'):
        df = pd.read_csv(dataset)
        comments = df['comments']
        input_batch = preprocessBatch(comments)
        lstmPred = [np.argmax(prediction) for prediction in lstm_model.predict(input_batch)]
        cnnPred = [np.argmax(prediction) for prediction in cnn_model.predict(input_batch)]
        gruPred = [np.argmax(prediction) for prediction in gru_model.predict(input_batch)]
        trfPred = [np.argmax(prediction) for prediction in transformer_model.predict(input_batch)]
        ensemblePred = [ensemblePrediction([lstmPred[i],cnnPred[i],gruPred[i],trfPred[i]]) for i in range(len(input_batch))]

        st.success("Prediction:" + str(ensemblePred))



if __name__=='__main__': 
    main()