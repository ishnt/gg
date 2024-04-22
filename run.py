import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
def main():
    st.title("Stock Price Analysis")

    # Get user input
    user_input = st.text_input("Enter your Stock Name:")
    # Process the input
    #1
    URL="https://www.google.com/search?q="+user_input+"+stock+price+&sca_esv=b1e24946e2e3e41f&biw=1920&bih=945&tbm=nws&sxsrf=ACQVn0-x4w6-df315UJGx9SSIckz-bSiPw%3A1713773153808&ei=YRomZvT_MJaX4-EPsLK60A0&ved=0ahUKEwi0tLvArtWFAxWWyzgGHTCZDtoQ4dUDCA4&uact=5&oq=amd+stock+price+&gs_lp=Egxnd3Mtd2l6LW5ld3MiEGFtZCBzdG9jayBwcmljZSAyCxAAGIAEGJECGIoFMhEQABiABBiRAhixAxiDARiKBTILEAAYgAQYkQIYigUyCxAAGIAEGJECGIoFMgsQABiABBiRAhiKBTIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAESM0HUABYAHAAeACQAQCYAZgBoAGYAaoBAzAuMbgBA8gBAPgBAZgCAaACnQGYAwCSBwMwLjGgB4sG&sclient=gws-wiz-news"
    response=requests.get(URL)
    print("The response code is:",response)
    soup=BeautifulSoup(response.content,'html.parser')
    headlines=soup.find_all('h3')
    array=[]
    for headline in headlines:
        array.append(headline.text)
     
    #2
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    j=0
    streamlit=[]
    while j<len(array):
        streamlit.append(array[j])
        print(array[j])
        text = array[j]
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = config.id2label[ranking[i]]
            s = scores[ranking[i]]
            streamlit.append(f"{l} {np.round(float(s),4)}")
            print(f"{l} {np.round(float(s),4)}")
        j=j+1   
    
    
    st.write("Output:", streamlit)

def process_input(input_text):
    # You can process the input here, for example, let's just convert it to uppercase
    return input_text.upper()

if __name__ == "__main__":
    main()
