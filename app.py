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
    
    
    cnn_model12 = load_model('models/12cnn_modelv6.keras')
    cnn_model02 = load_model('models/02lstm_model_v7.keras')
    transformer_model = load_model('models/01trf_modelv5.keras',
                                   custom_objects={"TransformerEncoder": TransformerEncoder, "TokenAndPositionEmbedding": TokenAndPositionEmbedding})
    return cnn_model02,cnn_model12,transformer_model

cnn_model02,cnn_model12,transformer_model = load_models()

prediction_decoding = {0:"medical doctor", 1:"veterinarian", 2:"person who is neither a medical doctor nor a veterinarian" }

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
    <p style="color:white;text-align:left;"><b>Class Codes</b><p>
    <p style="color:white;text-align:left;">Medical Practitioner: 0, Veterinarian: 1, Others:2 </p>
    </div>
    """
    st.markdown(html_temp2, unsafe_allow_html = True)
    st.divider()
    news_story = st.text_area('Enter a Reddit Comment',height=200)

    if st.button('Identify text'):
        input_data = preprocessComment(news_story)
        cnnPred12 = np.argmax(cnn_model12.predict(input_data))
        cnnPred02 = np.argmax(cnn_model02.predict(input_data))
        trfPred01 = np.argmax(transformer_model.predict(input_data))
        ensemblePred = ensemblePrediction([cnnPred12,cnnPred02,trfPred01])
        print(cnnPred12)
        print(cnnPred02)
        print(trfPred01)

        st.success("A " + prediction_decoding[ensemblePred] + " made this comment")
    
    st.divider()
    dataset = st.file_uploader("Upload a csv dataset of reddit comments. Comments must be in a 'comments' column", type=["csv"], accept_multiple_files = False)

    if st.button('Identify Batch'):
        df = pd.read_csv(dataset)
        comments = df['comments']
        input_batch = preprocessBatch(comments)
        cnnPred12 = [np.argmax(prediction) for prediction in cnn_model12.predict(input_batch)]
        cnnPred02 = [np.argmax(prediction) for prediction in cnn_model02.predict(input_batch)]
        trfPred01 = [np.argmax(prediction) for prediction in transformer_model.predict(input_batch)]
        ensemblePred = [ensemblePrediction([cnnPred12[i],cnnPred02[i],trfPred01[i]]) for i in range(len(input_batch))]

        st.success("Prediction:" + str(ensemblePred))



if __name__=='__main__': 
    main()