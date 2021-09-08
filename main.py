import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

header = st.beta_container()
dataset = st.beta_container()
model_training = st.beta_container()

with header:
    st.title("Welcome to my Data Science Project")

with dataset:
    st.header("Bike Dataset") 
    
    
    taxi_data=pd.read_csv('data/taxi_data.csv')
    st.write(taxi_data.head(5))

    st.subheader('Trip Duration on the Bike dataset')
    pulocation_dist=pd.DataFrame(taxi_data['tripduration'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)


with model_training:

	st.header("Time to train the model")

	sel_col,disp_col=st.beta_columns(2)
	max_depth=sel_col.slider('What should be max_depth of model?' ,min_value=10,max_value=100,value=20,step=10)

	n_estimators=sel_col.selectbox('How many trees should be there?',options=[100,200,300,400],index=0)
	
	sel_col.text('Total Features in the Bike dataset')
	sel_col.write(taxi_data.columns)
	input_feature=sel_col.text_input("Which should be the input feature?",'tripduration')




	if n_estimators=='No Limit':
		regr = RandomForestRegressor(max_depth=max_depth)
	else:
		regr = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)		
  
        
  
	X=taxi_data[[input_feature]]
	y=taxi_data[['start station id']]

	regr.fit(X,y)
	prediction=regr.predict(y)

	disp_col.subheader('Mean absolute error of the model is:')
	disp_col.write(mean_absolute_error(y,prediction))

	disp_col.subheader('Mean squared error of the model is:')
	disp_col.write(mean_squared_error(y,prediction))


	disp_col.subheader('R squared score of the model is:')
	disp_col.write(r2_score(y,prediction))

	