import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def costfunction(X,y,theta,m):
    hypo = X.dot(theta)
    J = (1/(2*m)) * (np.sum((hypo-y)**2))
    return [J,theta]


def gradientfunction(X, Y, theta,m,alpha,num_iteration):
    costlist = []
    thetalist =[]
    theta1=[]
    theta2=[]
   
    
    for i in range(0,num_iteration):
        hypo = X.dot(theta)
        error = hypo - Y
        sumation = np.dot(X.T,error)
        
        theta = theta - (alpha/m) * sumation

        [j,theta] = costfunction(X,Y,theta,m)
        """print("Iteration ",end=" ")
        print(i+1,end=" ")
        print("Cost",end=" ")
        print(j)"""
        costlist.append(j)
        thetalist.append(theta)
        thetaf=(theta[0,:].tolist())
        thetaS=(theta[1,:].tolist())
        theta1.append(thetaf)
        theta2.append(thetaS)

        
        

    return [costlist , thetalist,theta1,theta2]

def final_plot(final_hypo,dataframe,m):
    x = dataframe["Hour"].values
    z=x.reshape(m,1)
    dataf=pd.DataFrame(z, columns=['Hour']) 
    dataf["Hypothesis"]=final_hypo
    fig, ax = plt.subplots()
    sns.relplot(x='Hour', y='Score', data=dataframe,kind="scatter", ax=ax)
    sns.relplot(x='Hour', y="Hypothesis", data=dataf,kind="line", ax=ax, color='r')
    plt.show()

def costvisualization(cost,theta0,theta1):
    thetaI = np.array(theta0)
    thetaII=np.array(theta1)
    dataframe = pd.DataFrame(cost,columns=["Cost"])
    dataframe["Theta0"] = thetaI
    dataframe["Theta1"] = thetaII
    fig =plt.figure()
    ax=fig.gca(projection = '3d')
    
    ax.plot_trisurf(dataframe["Theta0"],dataframe["Theta1"],dataframe["Cost"],cmap=plt.cm.viridis,linewidth=0.2)
    plt.show()

def run():
        costlist = []
        thetalist =[]
        theta0=[[0]]
        theta1=[[0]]
        
        
        #dataframe = pd.read_csv("data.csv")
        dataframe=pd.read_csv("C:\\Users\\Ashish\\Desktop\\ex.csv")
        sns.scatterplot(x="Hour",y="Score",data=dataframe)
        plt.show()
        m=len(dataframe["Hour"])
        #initial_theta = np.zeros([2,1])
        initial_theta = np.zeros([2,1])#np.array([[-1],[2]])
        biasterm = np.ones([m,1])
        dataframe["Bias"] = biasterm
        x = dataframe["Hour"].values #so this values convert type dataframe to type mumpy_array
        X=dataframe[["Bias","Hour"]].values
        y = dataframe["Score"].values
        Y = y.reshape(m,1)
        [j,theta] = costfunction(X, Y, initial_theta,m)
        costlist.append(j)
        thetalist.append(theta)
        alpha=0.01
        num_iteration = 1500
        [c,t,thetaI,thetaII] = gradientfunction(X,Y,initial_theta, m,alpha, num_iteration)
        costlist.extend(c)
        thetalist.extend(t)
        theta0.extend(thetaI)
        theta1.extend(thetaII)
        final_min_cost=min(costlist)
        index_final_min_cost = costlist.index(min(costlist))
        theta_min_cost = thetalist[index_final_min_cost]
        print(final_min_cost,end=" ")
        print(theta_min_cost)
        final_hypo =  X.dot(theta_min_cost)
        

        final_plot(final_hypo, dataframe,m)
        costvisualization(costlist,theta0,theta1)

        
        
        
        

        

if __name__ == "__main__":
    run()
    

