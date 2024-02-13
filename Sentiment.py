import streamlit as st
import numpy as np
import pandas as pd
import urllib.request
import json
import os
import ssl
import plotly.express as px

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

st.set_page_config(page_title="Customer Sentiment", page_icon="📈",layout="wide",initial_sidebar_state='collapsed')

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;                   
                }
        </style>
        """, unsafe_allow_html=True) 


dash_1 = st.container()
dash_2 = st.container()
dash_3 = st.container()
dash_4 = st.container()



with dash_1:
    st.markdown("<h2 style='text-align: center;'>Motopay Customer Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.write("")

with dash_2:
    col1, col2 = st.columns(2)

    with col1:

        label = False

        uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False, type = ['csv'])
        flattened_data = None

        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            dataframe.rating = dataframe.rating.astype(float)
            dataframe.review = dataframe.review.astype(str)
            flattened_data = dataframe[['review', 'rating']].to_json(orient='split')

            st.write(dataframe)

            label = st.button("label")

        if label:
            
            with col2:
                m_dict = json.loads(flattened_data)
                data = {"data": m_dict['data']}
  
                body = str.encode(json.dumps(data))

                url = 'https://sentiment-endpoint-c6174cb2.eastus.inference.ml.azure.com/score'
                # Replace this with the primary/secondary key or AMLToken for the endpoint
                api_key = st.secrets['api-key']
                if not api_key:
                    raise Exception("A key should be provided to invoke the endpoint")

                # The azureml-model-deployment header will force the request to go to a specific deployment.
                # Remove this header to have the request observe the endpoint traffic rules
                headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue' }

                req = urllib.request.Request(url, body, headers)

                try:
                    with urllib.request.urlopen(req) as response:
                        result = json.loads(response.read().decode())

                except urllib.error.HTTPError as error:
                    print("The request failed with status code: " + str(error.code))

                    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
                    print(error.info())
                    print(error.read().decode("utf8", 'ignore'))

                if result is not None:
                    dataframe['label'] = result
                    # print(dataframe.columns)

                grouped_df = dataframe.groupby(['product_id', 'label']).size().reset_index(name='counts')

                # Pivot the DataFrame to have product IDs as rows, labels as columns, and counts as values
                pivot_df = grouped_df.pivot(index='product_id', columns='label', values='counts').fillna(0)

                # st.bar_chart(pivot_df)
                fig = px.bar(grouped_df, x="product_id", y="counts", color="label", barmode="group",
                height=400, title="Customer Reviews on Products")

                # Customize the layout for better readability
                fig.update_layout(xaxis_title="Product ID",
                                yaxis_title="Number of Reviews",
                                legend_title="Review Sentiment",
                                xaxis={'categoryorder':'total descending'}) # This sorts the products by the total count of reviews

                # Display the Plotly chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(pivot_df)