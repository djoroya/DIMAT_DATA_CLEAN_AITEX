
from matplotlib import pyplot as plt
import numpy as np
def fitplot(results):
    outputs = results["outputs_train"]
    predicted = results["predicted_train"]
    outputs_test = results["outputs_test"]
    predicted_test = results["predicted_test"]

    et_mean = results["et_mean"]
    et_std = results["et_std"]
    etest_mean = results["etest_mean"]
    etest_std = results["etest_std"]
    fig = plt.figure(figsize=(15,10))
    # padding ssbplot 
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    for i in range(outputs.shape[1]):
        plt.subplot(2, outputs.shape[1], i+1)
        experimental = outputs.iloc[:,i].values
        plt.plot(experimental,predicted[:,i],'.')

        xspan = np.linspace(np.min(experimental),np.max(experimental),100)

        plt.plot(xspan,xspan,'-')

        plt.title(outputs.columns[i])
        plt.xlabel("Experimental")
        plt.ylabel("Prediction")
        if i == 0:
            text_plot = f"error: {et_mean:.2f} +/- {et_std:.2f}"
            plt.text(0.1,0.9,text_plot,transform=plt.gca().transAxes)
        plt.subplot(2, outputs.shape[1], i+1+outputs.shape[1])
        
        experimental = outputs_test.iloc[:,i].values

        plt.plot(experimental,predicted_test[:,i],'.')
        xspan = np.linspace(np.min(experimental),np.max(experimental),100)
        plt.plot(xspan,xspan,'-')
        if i == 0:
            text_plot=  f"error: {etest_mean:.2f} +/- {etest_std:.2f}"
            plt.text(0.1,0.9,text_plot,transform=plt.gca().transAxes)

        plt.xlabel("Experimental")
        plt.ylabel("Prediction")
        plt.title(outputs.columns[i])
