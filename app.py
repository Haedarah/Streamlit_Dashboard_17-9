#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from datetime import date, time
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
from statsmodels.tsa.arima.model import ARIMA



st.set_page_config(page_title="Ambilio Dashboard",page_icon="✅",layout="wide",initial_sidebar_state="collapsed")



#Reading data, fixing it, and sorting it by date
data=pd.read_csv('Data/supermarket_sales.csv',on_bad_lines='skip')

data['Date']=pd.to_datetime(data['Date']).dt.date
for i in data.index:
	data['Time'][i]=datetime.strptime(data['Time'][i], "%H:%M").time()
	data['Date'][i]=pd.Timestamp.combine(date(data['Date'][i].year,data['Date'][i].month,data['Date'][i].day),
		time(data['Time'][i].hour,data['Time'][i].minute,data['Time'][i].second))

data.drop("Time", axis=1, inplace=True)
data=data.sort_values(by='Date',ignore_index=True)
data=data.set_index(data['Date'])
data = data[~data.index.duplicated(keep='first')]


#Dividing the page into 5 sections
header=st.container()
st.markdown("""---""")
dataset=st.container()
st.markdown("""---""")
datavis=st.container()
st.markdown("""---""")
dataARIMA=st.container()
st.markdown("""---""")
footer=st.container()



#Programming the sidebar
st.sidebar.image("Objects/LOGO1.png",use_column_width=True) 

plotting=data.columns.unique()

st.sidebar.write("- Columns to be visualized:")
vis_parameters=st.sidebar.multiselect("Your columns:",plotting,default='gross income')

st.sidebar.write("- Starting date & the ending date of visualized data:")
vis_start=st.sidebar.date_input("Start Date:",data['Date'].min(),data['Date'].min(),data['Date'].max())
vis_end=st.sidebar.date_input("End Date:",data['Date'].max(),data['Date'].min(),data['Date'].max())

st.sidebar.write("-Plotting type:")
vis_type=st.sidebar.selectbox("Plotting type:",['Line Chart','Bar Chart','Pie Chart','Histogram'])



#Programming the different sections:
with header:
	st.title("Supermarket Sales Visualization - Beta Version")
	st.markdown("This dashboard is meant to visualize the data itself from different aspects.")


with dataset:
	st.subheader("Displaying the first 5 entries of the dataset we are dealing with:")
	st.markdown("You can find & download the dataset [here](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales).")
	
	st.table(data.head(5))


with datavis:
	st.subheader("Plotting the chosen parameters, during the chosen duration and in the chosen plot type:")
	if len(vis_parameters)==0:
		st.write("Choose some parameters!")
	else:
		plotting_data=data[(data['Date']>=vis_start) & (data['Date']<=vis_end)]


		if vis_type=='Line Chart':

			try:
				fig1=plt.figure(figsize=(10,4))
				axes1=fig1.add_axes([0.05,0.05,0.95,0.95])
				axes1.set_xlabel("Date")
				axes1.grid(True,color='0.6')
				for i in range(0,len(vis_parameters)):
					axes1.plot(plotting_data['Date'],plotting_data[vis_parameters[i]],label=vis_parameters[i]
						,lw=2,marker='o',markerfacecolor='r')
				plt.legend(loc=0)
				fig1.autofmt_xdate()
				if len(vis_parameters)>0 :
					st.pyplot(fig1)

			except ValueError:
				st.write("Data can't be visualized using the current parameters. Make some changes in the sidebar!")


		elif vis_type=='Bar Chart':
			
			try:
				fig2=plt.figure(figsize=(10,4))
				axes2=fig2.add_axes([0.05,0.05,0.95,0.95])
				axes2.set_ylabel("Frequency")
				axes2.grid(True,color='0.5')
				for i in range(0,len(vis_parameters)):
					axes2.bar(plotting_data[vis_parameters[i]].unique(),plotting_data[vis_parameters[i]].value_counts(),
						label=vis_parameters[i],width=0.25,edgecolor='k')
				plt.legend(loc=0)
				fig2.autofmt_xdate()
				if len(vis_parameters)>0 :
					st.pyplot(fig2)

			except ValueError:
				st.write("Data can't be visualized using the current parameters. Please make some changes in the sidebar.")


		elif vis_type=='Pie Chart':

			st.write("Only the first chosen parameter will be plotted!")
			try:
				fig3,axes3=plt.subplots()
				fig3.set_figheight(3)
				fig3.set_figwidth(10)
				labels=[]
				if len(vis_parameters)>0 :
					labels=plotting_data[vis_parameters[0]].unique()
				dd=[]

				for i in range(0,len(labels)) :
					sum=0
					for j in plotting_data[vis_parameters[0]]:
						if j==labels[i] :
							sum+=1
					dd.append(float(sum/len(plotting_data)))

				axes3.pie(dd, labels=labels, autopct='%1.2f%%')
				axes3.axis('equal')
				if len(vis_parameters)>0 :
					st.pyplot(fig3)

			except ValueError:
				st.write("Data can't be visualized using the current parameters. Please make some changes in the sidebar.")


		else:

			st.write("Only the first chosen parameter will be plotted!")
			try:
				fig4=plt.figure(figsize=(15,4))
				axes4=fig4.add_axes([0.05,0.05,0.95,0.95])
				axes4.set_ylabel("Frequency")
				axes4.grid(True,color='0.5')
				axes4.hist(plotting_data[vis_parameters[0]],bins=len(plotting_data[vis_parameters[0]].unique()),density=False)
				fig4.autofmt_xdate()
				if len(vis_parameters)>0 :
					st.pyplot(fig4)

			except ValueError:
				st.write("Data can't be visualized using the current parameters. Please make some changes in the sidebar.")
			


with dataARIMA:		

	st.subheader("Applying ARIMA model to predict the future values of gross income:")

	first=st.container()
	second=st.container()
	with first:
		f_col1,f_col2=st.columns(2)
		with f_col1:
			bran=st.selectbox("Choose a branch:",['A','B','C'])
		with f_col2:
			per=st.selectbox("Choose a period for the prediction",['1 month','2 months','3 months'])
	

	with second:
		pred_A=data['gross income'][data['Branch']=='A'].rename('gross income A')
		pred_A.index=pred_A.index.date

		pred_B=data['gross income'][data['Branch']=='B'].rename('gross income B')
		pred_B.index=pred_B.index.date
		
		pred_C=data['gross income'][data['Branch']=='C'].rename('gross income C')
		pred_C.index=pred_C.index.date



		pred_A=pred_A.cumsum()
		pred_A = pred_A[~pred_A.index.duplicated(keep='last')]
		pred_A_idx=[]

		for i in pred_A.index:
			pred_A_idx.append(i)
		
		for i,v in enumerate(pred_A_idx):
			if i!=0:
				pred_A[pred_A_idx[i]]=pred_A[pred_A_idx[i]]-pred_A[pred_A_idx[i-1]]

		pred_B=pred_B.cumsum()
		pred_B = pred_B[~pred_B.index.duplicated(keep='last')]
		pred_B_idx=[]

		for i in pred_B.index:
			pred_B_idx.append(i)
		
		for i,v in enumerate(pred_B_idx):
			if i!=0:
				pred_B[pred_B_idx[i]]=pred_B[pred_B_idx[i]]-pred_B[pred_B_idx[i-1]]

		pred_C=pred_C.cumsum()
		pred_C = pred_C[~pred_C.index.duplicated(keep='last')]
		pred_C_idx=[]

		for i in pred_C.index:
			pred_C_idx.append(i)
		
		for i,v in enumerate(pred_C_idx):
			if i!=0:
				pred_C[pred_C_idx[i]]=pred_C[pred_C_idx[i]]-pred_C[pred_C_idx[i-1]]


		def ad_test(dataset):
		    dftest=adfuller(dataset,autolag ="AIC")
		    st.write("1. ADF : ",dftest[0])
		    st.write("2. P-Value : ",dftest[1])
		    st.write("3. Num Of Lags : ",dftest[2])
		    st.write("4. Num Of Observations Used For ADF Regression:",dftest[3])
		    st.write("5. Critical Values :")
		    for key, val in dftest[4].items():
		        st.write("\t",key, ": ", val)



		if bran=='A':
			pred_A_mean=np.log(pred_A)
			pred_A_mean=pred_A_mean.rolling(2).mean().rename("Rolling mean A")			
			pred_A_mean=pred_A_mean.dropna()
			#ad_test(pred_A_mean)
			#train=pred_A_mean.iloc[:-15]
			#test=pred_A_mean.iloc[-15:]
			#stepwise_fit = auto_arima(pred_A_mean, trace=True, suppress_warnings=True)
			#st.write(stepwise_fit)
			model=ARIMA(pred_A_mean,order=(1,2,1))
			model=model.fit()
			model.summary()
			
			#start=len(train)
			#end=len(train)+len(test)-1
			#pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')
			#pred.index=pred_A_mean[-15:].index

			def tenpower(num):
				return np.e**num
			
			if per=='1 month' :
				index_future_dates=pd.date_range(start='2019-03-31',end='2019-04-29')
				pred=model.predict(start=len(pred_A),end=len(pred_A)+29,typ='levels').rename('ARIMA Predictions')
				pred.index=index_future_dates
			elif per=='2 months' :
				index_future_dates=pd.date_range(start='2019-03-31',end='2019-05-29')
				pred=model.predict(start=len(pred_A),end=len(pred_A)+59,typ='levels').rename('ARIMA Predictions')
				pred.index=index_future_dates
			else :
				index_future_dates=pd.date_range(start='2019-03-31',end='2019-06-28')
				pred=model.predict(start=len(pred_A),end=len(pred_A)+89,typ='levels').rename('ARIMA Predictions')
				pred.index=index_future_dates

			pred=pred.apply(tenpower)
			Plotting=pd.concat([pred_A,pred],axis=1)
			st.line_chart(Plotting)



		if bran=='B':
			pred_B_mean=np.log(pred_B)
			pred_B_mean=pred_B_mean.rolling(4).mean().rename("Rolling mean B")			
			pred_B_mean=pred_B_mean.dropna()
			#ad_test(pred_B_mean)
			#train=pred_B_mean.iloc[:-15]
			#test=pred_B_mean.iloc[-15:]
			#stepwise_fit = auto_arima(pred_B_mean, trace=True, suppress_warnings=True)
			#st.write(stepwise_fit)
			model=ARIMA(pred_B_mean,order=(0,2,4))
			model=model.fit()
			model.summary()	
			#start=len(train)
			#end=len(train)+len(test)-1
			#pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')
			#pred.index=pred_B_mean[-15:].index

			def tenpower(num):
				return np.e**num
			
			if per=='1 month' :
				index_future_dates=pd.date_range(start='2019-03-31',end='2019-04-29')
				pred=model.predict(start=len(pred_B),end=len(pred_B)+29,typ='levels').rename('ARIMA Predictions')
				pred.index=index_future_dates
			elif per=='2 months' :
				index_future_dates=pd.date_range(start='2019-03-31',end='2019-05-29')
				pred=model.predict(start=len(pred_B),end=len(pred_B)+59,typ='levels').rename('ARIMA Predictions')
				pred.index=index_future_dates
			else :
				index_future_dates=pd.date_range(start='2019-03-31',end='2019-06-28')
				pred=model.predict(start=len(pred_B),end=len(pred_B)+89,typ='levels').rename('ARIMA Predictions')
				pred.index=index_future_dates

			pred=pred.apply(tenpower)
			Plotting=pd.concat([pred_B,pred],axis=1)
			st.line_chart(Plotting)



		if bran=='C':
			pred_C_mean=np.log(pred_C)
			pred_C_mean=pred_C_mean.rolling(2).mean().rename("Rolling mean C")			
			pred_C_mean=pred_C_mean.dropna()
			#ad_test(pred_C_mean)
			#train=pred_C_mean.iloc[:-15]
			#test=pred_C_mean.iloc[-15:]
			#stepwise_fit = auto_arima(pred_C_mean, trace=True, suppress_warnings=True)
			#st.write(stepwise_fit)
			model=ARIMA(pred_C_mean,order=(3,2,3))
			model=model.fit()
			model.summary()	
			#start=len(train)
			#end=len(train)+len(test)-1
			#pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')
			#pred.index=pred_C_mean[-15:].index

			def tenpower(num):
				return np.e**num
			
			if per=='1 month' :
				index_future_dates=pd.date_range(start='2019-03-31',end='2019-04-29')
				pred=model.predict(start=len(pred_C),end=len(pred_C)+29,typ='levels').rename('ARIMA Predictions')
				pred.index=index_future_dates
			elif per=='2 months' :
				index_future_dates=pd.date_range(start='2019-03-31',end='2019-05-29')
				pred=model.predict(start=len(pred_C),end=len(pred_C)+59,typ='levels').rename('ARIMA Predictions')
				pred.index=index_future_dates
			else :
				index_future_dates=pd.date_range(start='2019-03-31',end='2019-06-28')
				pred=model.predict(start=len(pred_C),end=len(pred_C)+89,typ='levels').rename('ARIMA Predictions')
				pred.index=index_future_dates

			pred=pred.apply(tenpower)
			Plotting=pd.concat([pred_C,pred],axis=1)
			st.line_chart(Plotting)

with footer:
	st.markdown("Ambilio Technologies")
