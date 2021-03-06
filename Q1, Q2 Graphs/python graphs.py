import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn.datasets as datasets
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
import sklearn.linear_model
from sklearn.metrics import r2_score


pres2016 = pd.read_csv("MergedMassData.csv")

pres2020 = pd.read_csv("MergedMassData.csv")

gov2014 = pd.read_csv("MergedMassDataGov.csv")

gov2018 = pd.read_csv("MergedMassDataGov.csv")

#cleaning for presidential election

pres2020 = pres2020.replace(",", "", regex = True)

pres2016 = pres2016.replace(",", "", regex = True)

pres2020["Democratic 2020"] = pd.to_numeric(pres2020["Democratic 2020"])

pres2020["Republican 2020"] = pd.to_numeric(pres2020["Republican 2020"])

pres2020["Total Votes Cast 2020"] = pd.to_numeric(pres2020["Total Votes Cast 2020"])

pres2020["Mexican 2019"] = pd.to_numeric(pres2020["Mexican 2019"])

pres2020["Puerto Rican 2019"] = pd.to_numeric(pres2020["Puerto Rican 2019"])

pres2020["Cuban 2019"] = pd.to_numeric(pres2020["Cuban 2019"])

pres2020["Other LatinX 2019"] = pd.to_numeric(pres2020["Other LatinX 2019"])

pres2020["Total Population 2019"] = pd.to_numeric(pres2020["Total Population 2019"]) 

pres2016["Democratic 2016"] = pd.to_numeric(pres2016["Democratic 2016"])

pres2016["Republican 2016"] = pd.to_numeric(pres2016["Republican 2016"])

pres2016["Total Votes Cast"] = pd.to_numeric(pres2016["Total Votes Cast"])

pres2016["Mexican 2016"] = pd.to_numeric(pres2016["Mexican 2016"])

pres2016["Puerto Rican 2016"] = pd.to_numeric(pres2016["Puerto Rican 2016"])

pres2016["Cuban 2016"] = pd.to_numeric(pres2016["Cuban 2016"])

pres2016["Other LatinX 2016"] = pd.to_numeric(pres2016["Other LatinX 2016"])

pres2016["Total Population 2016"] = pd.to_numeric(pres2016["Total Population 2016"])

#cleaning for governors election

gov2014 = gov2014.replace(",", "", regex = True)

gov2018 = gov2018.replace(",", "", regex = True)

gov2018["Democratic 2018"] = pd.to_numeric(gov2018["Democratic 2018"])

gov2018["Republican 2018"] = pd.to_numeric(gov2018["Republican 2018"])

gov2018["Total Votes Cast 2018"] = pd.to_numeric(gov2018["Total Votes Cast 2018"])

gov2018["Mexican 2018"] = pd.to_numeric(gov2018["Mexican 2018"])

gov2018["Puerto Rican 2018"] = pd.to_numeric(gov2018["Puerto Rican 2018"])

gov2018["Cuban 2018"] = pd.to_numeric(gov2018["Cuban 2018"])

gov2018["Other LatinX 2018"] = pd.to_numeric(gov2018["Other LatinX 2018"])

gov2018["Total Population 2018"] = pd.to_numeric(gov2018["Total Population 2018"])

gov2014["Democratic 2014"] = pd.to_numeric(gov2014["Democratic 2014"])

gov2014["Republican 2014"] = pd.to_numeric(gov2014["Republican 2014"])

gov2014["Total Votes Cast 2014"] = pd.to_numeric(gov2014["Total Votes Cast 2014"])

gov2014["Mexican 2014"] = pd.to_numeric(gov2014["Mexican 2014"])

gov2014["Puerto Rican 2014"] = pd.to_numeric(gov2014["Puerto Rican 2014"])

gov2014["Cuban 2014"] = pd.to_numeric(gov2014["Cuban 2014"])

gov2014["Other LatinX 2014"] = pd.to_numeric(gov2014["Other LatinX 2014"])

gov2014["Total Population 2014"] = pd.to_numeric(gov2014["Total Population 2014"]) 

#analyzing changes between the most recent elections for president

changeDemPres = (pres2020["Democratic 2020"] / pres2020["Total Votes Cast 2020"]) - (pres2016["Democratic 2016"] / pres2016["Total Votes Cast"])

changeRepPres = (pres2020["Republican 2020"] / pres2020["Total Votes Cast 2020"]) - (pres2016["Republican 2016"] / pres2016["Total Votes Cast"])

changeMexPres = (pres2020["Mexican 2019"] / pres2020["Total Population 2019"]) - (pres2016["Mexican 2016"] / pres2016["Total Population 2016"])

changePRPres = (pres2020["Puerto Rican 2019"] / pres2020["Total Population 2019"]) - (pres2016["Puerto Rican 2016"] / pres2016["Total Population 2016"])

changeCubanPres = (pres2020["Cuban 2019"] / pres2020["Total Population 2019"]) - (pres2016["Cuban 2016"] / pres2016["Total Population 2016"])

changeOtherLatinXPres = (pres2020["Other LatinX 2019"] / pres2020["Total Population 2019"]) - (pres2016["Other LatinX 2016"] / pres2016["Total Population 2016"])

changeTotalLatinXPres = ((pres2020["Mexican 2019"] + pres2020["Puerto Rican 2019"] + pres2020["Cuban 2019"] + pres2020["Other LatinX 2019"])
 / pres2020["Total Population 2019"]) - ((pres2016["Mexican 2016"] + pres2016["Puerto Rican 2016"] + pres2016["Cuban 2016"] +pres2016["Other LatinX 2016"])
 / pres2016["Total Population 2016"])

#analyzing changes between most recent elections for governors

changeDemGov = (gov2018["Democratic 2018"] / gov2018["Total Votes Cast 2018"]) - (gov2014["Democratic 2014"] / gov2014["Total Votes Cast 2014"])

changeRepGov = (gov2018["Republican 2018"] / gov2018["Total Votes Cast 2018"]) - (gov2014["Republican 2014"] / gov2014["Total Votes Cast 2014"])

changeMexGov = (gov2018["Mexican 2018"] / gov2018["Total Population 2018"]) - (gov2014["Mexican 2014"] / gov2014["Total Population 2014"])

changePRGov = (gov2018["Puerto Rican 2018"] / gov2018["Total Population 2018"]) - (gov2014["Puerto Rican 2014"] / gov2014["Total Population 2014"])

changeCubanGov = (gov2018["Cuban 2018"] / gov2018["Total Population 2018"]) - (gov2014["Cuban 2014"] / gov2014["Total Population 2014"])

changeOtherLatinXGov = (gov2018["Other LatinX 2018"] / gov2018["Total Population 2018"]) - (gov2014["Other LatinX 2014"] / gov2014["Total Population 2014"])

changeTotalLatinXGov = ((gov2018["Mexican 2018"] + gov2018["Puerto Rican 2018"] + gov2018["Cuban 2018"] + gov2018["Other LatinX 2018"])
 / gov2018["Total Population 2018"]) - ((gov2014["Mexican 2014"] + gov2014["Puerto Rican 2014"] + gov2014["Cuban 2014"] + gov2014["Other LatinX 2014"]) / gov2014["Total Population 2014"])

#merge presidential election changes with demographic changes during those years

merged_presidential = pd.DataFrame({"change in democratic support" : changeDemPres, "change in republican support" : changeRepPres, "change in mexican population" : changeMexPres,
                                    "change in puerto rican population" : changePRPres, "change in cuban population" : changeCubanPres, "change in other LatinX population" : changeOtherLatinXPres,
                                    "change in total LatinX population" : changeTotalLatinXPres})

merged_presidential = merged_presidential.replace([np.inf, -np.inf], np.nan)    #replace inf values by NaN, occurs when starting value is zero

merged_presidential = merged_presidential.fillna(0)     #repalce any NaN with 0

#plot presidential election data for mexican population

plt.scatter(merged_presidential["change in mexican population"], merged_presidential["change in democratic support"], color = "blue", label = "democratic support", s= 0.7)

plt.scatter(merged_presidential["change in mexican population"], merged_presidential["change in republican support"], color = "red", label = "republican support", s=0.7)

mexPop = merged_presidential["change in mexican population"].values.reshape(-1, 1)

demSupport = merged_presidential["change in democratic support"].values.reshape(-1, 1)

linregDem = LinearRegression()

linregDem.fit(mexPop, demSupport)

Y_predMexDem = linregDem.predict(mexPop)

mexDemR2 = r2_score(demSupport, Y_predMexDem)

plt.figtext(.7, .73, ("democratic R?? =" + "{:.8f}".format(mexDemR2)))

plt.plot(mexPop, Y_predMexDem, color='cyan', label = "democratic support trend")

repSupport = merged_presidential["change in republican support"].values.reshape(-1, 1)

linregRep = LinearRegression()

linregRep.fit(mexPop, repSupport)

Y_predMexRep = linregRep.predict(mexPop)

mexRepR2 = r2_score(repSupport, Y_predMexRep)

plt.figtext(.7, .71, ("republican R??=" + "{:.8f}".format(mexRepR2)))

plt.plot(mexPop, Y_predMexRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in mexican population")

plt.ylabel("% point change in presidential political support")

plt.title("change in mexican population vs change in presidential support (MA 2016 - 2020)")

plt.show()

#plot presidential election data for puetro rican population

plt.scatter(merged_presidential["change in puerto rican population"], merged_presidential["change in democratic support"], color = "blue", label = "democratic support", s= 0.7)

plt.scatter(merged_presidential["change in puerto rican population"], merged_presidential["change in republican support"], color = "red", label = "republican support", s=0.7)

prPop = merged_presidential["change in puerto rican population"].values.reshape(-1, 1)

linregDem.fit(prPop, demSupport)

Y_predprDem = linregDem.predict(prPop)

prDemR2 = r2_score(demSupport, Y_predprDem)

plt.figtext(.7, .73, ("democratic R?? =" + "{:.8f}".format(prDemR2)))

plt.plot(prPop, Y_predprDem, color='cyan', label = "democratic support trend")

linregRep.fit(prPop, repSupport)

Y_predprRep = linregRep.predict(prPop)

prRepR2 = r2_score(repSupport, Y_predprRep)

plt.figtext(.7, .71, ("republican R??=" + "{:.8f}".format(prRepR2)))

plt.plot(prPop, Y_predprRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in puerto rican population")

plt.ylabel("% point change in presidential political support")

plt.title("change in puerto rican population vs change in presidential support (MA 2016 - 2020)")

plt.show()

#plot presidential election data for cuban population

plt.scatter(merged_presidential["change in cuban population"], merged_presidential["change in democratic support"], color = "blue", label = "democratic support", s=0.7)

plt.scatter(merged_presidential["change in cuban population"], merged_presidential["change in republican support"], color = "red", label = "republican support",s=0.7)

cubanPop = merged_presidential["change in cuban population"].values.reshape(-1, 1)

linregDem.fit(cubanPop, demSupport)

Y_predcubanDem = linregDem.predict(cubanPop)

cubanDemR2 = r2_score(demSupport, Y_predcubanDem)

plt.figtext(.7, .73, ("democratic R?? =" + "{:.8f}".format(cubanDemR2)))

plt.plot(cubanPop, Y_predcubanDem, color='cyan', label = "democratic support trend")

linregRep.fit(cubanPop, repSupport)

Y_predcubanRep = linregRep.predict(cubanPop)

cubanRepR2 = r2_score(repSupport, Y_predcubanRep)

plt.figtext(.7, .71, ("republican R??=" + "{:.8f}".format(cubanRepR2)))

plt.plot(cubanPop, Y_predcubanRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in cuban population")

plt.ylabel("% point change in presidential political support")

plt.title("change in cuban population vs change in presidential support (MA 2016 - 2020)")

plt.show()

#plot presidential election data for other LatinX populations

plt.scatter(merged_presidential["change in other LatinX population"], merged_presidential["change in democratic support"], color = "blue", label = "democratic support",s=0.7)

plt.scatter(merged_presidential["change in other LatinX population"], merged_presidential["change in republican support"], color = "red", label = "republican support",s=0.7)

otherPop = merged_presidential["change in other LatinX population"].values.reshape(-1, 1)

linregDem.fit(otherPop, demSupport)

Y_predotherDem = linregDem.predict(otherPop)

otherDemR2 = r2_score(demSupport, Y_predotherDem)

plt.figtext(.7, .73, ("democratic R?? =" + "{:.8f}".format(otherDemR2)))

plt.plot(otherPop, Y_predotherDem, color='cyan', label = "demcoratic support trend")

linregRep.fit(otherPop, repSupport)

Y_predotherRep = linregRep.predict(otherPop)

otherRepR2 = r2_score(repSupport, Y_predotherRep)

plt.figtext(.7, .71, ("republican R??=" + "{:.8f}".format(otherRepR2)))

plt.plot(otherPop, Y_predotherRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in other LatinX population")

plt.ylabel("% point change in presidential political support")

plt.title("change in other LatinX population vs change in presidential support (MA 2016 - 2020)")

plt.show()

#plot presidential election data for Total LatinX populations

plt.scatter(merged_presidential["change in total LatinX population"], merged_presidential["change in democratic support"], color = "blue", label = "democratic support",s=0.7)

plt.scatter(merged_presidential["change in total LatinX population"], merged_presidential["change in republican support"], color = "red", label = "republican support",s=0.7)

totalPop = merged_presidential["change in total LatinX population"].values.reshape(-1, 1)

linregDem.fit(totalPop, demSupport)

Y_predtotalDem = linregDem.predict(totalPop)

totalDemR2 = r2_score(demSupport, Y_predtotalDem)

plt.figtext(.7, .73, ("democratic R?? =" + "{:.8f}".format(totalDemR2)))

plt.plot(totalPop, Y_predtotalDem, color='cyan', label = "democratic support trend")

linregRep.fit(totalPop, repSupport)

Y_predtotalRep = linregRep.predict(totalPop)

totalRepR2 = r2_score(repSupport, Y_predtotalRep)

plt.figtext(.7, .71, ("republican R??=" + "{:.8f}".format(totalRepR2)))

plt.plot(totalPop, Y_predtotalRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in total LatinX population")

plt.ylabel("% point change in presidential political support")

plt.title("change in total LatinX population vs change in presidential support (MA 2016 - 2020)")

plt.show()

#merge governors election changes with demographic changes during those years

merged_governors = pd.DataFrame({"change in democratic support" : changeDemGov, "change in republican support" : changeRepGov, "change in mexican population" : changeMexGov,
                                    "change in puerto rican population" : changePRGov, "change in cuban population" : changeCubanGov, "change in other LatinX population" : changeOtherLatinXGov,
                                    "change in total LatinX population" : changeTotalLatinXGov})

merged_governors = merged_governors.replace([np.inf, -np.inf], np.nan)    #replace inf values by NaN, occurs when starting value is zero

merged_governors = merged_governors.fillna(0)

#plot governors election data for mexican population

plt.scatter(merged_governors["change in mexican population"], merged_governors["change in democratic support"], color = "blue", label = "democratic support",s=0.7)

plt.scatter(merged_governors["change in mexican population"], merged_governors["change in republican support"], color = "red", label = "republican support",s=0.7)

mexPop = merged_governors["change in mexican population"].values.reshape(-1, 1)

demSupport = merged_governors["change in democratic support"].values.reshape(-1, 1)

linregDem = LinearRegression()

linregDem.fit(mexPop, demSupport)

Y_predMexDem = linregDem.predict(mexPop)

mexDemR2 = r2_score(demSupport, Y_predMexDem)

plt.figtext(.7, .73, ("democratic R?? =" + "{:.8f}".format(mexDemR2)))

plt.plot(mexPop, Y_predMexDem, color='cyan', label = "democratic support trend")

repSupport = merged_governors["change in republican support"].values.reshape(-1, 1)

linregRep = LinearRegression()

linregRep.fit(mexPop, repSupport)

Y_predMexRep = linregRep.predict(mexPop)

mexRepR2 = r2_score(repSupport, Y_predMexRep)

plt.figtext(.7, .71, ("republican R??=" + "{:.8f}".format(mexRepR2)))

plt.plot(mexPop, Y_predMexRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in mexican population")

plt.ylabel("% point change in governors political support")

plt.title("change in mexican population vs change in governors support (MA 2014 - 2018)")

plt.show()

#plot governors election data for puetro rican population

plt.scatter(merged_governors["change in puerto rican population"], merged_governors["change in democratic support"], color = "blue", label = "democratic support",s=0.7)

plt.scatter(merged_governors["change in puerto rican population"], merged_governors["change in republican support"], color = "red", label = "republican support",s=0.7)

prPop = merged_governors["change in puerto rican population"].values.reshape(-1, 1)

linregDem.fit(prPop, demSupport)

Y_predprDem = linregDem.predict(prPop)

prDemR2 = r2_score(demSupport, Y_predprDem)

plt.figtext(.7, .73, ("democratic R?? =" + "{:.8f}".format(prDemR2)))

plt.plot(prPop, Y_predprDem, color='cyan', label = "democratic support trend")

linregRep.fit(prPop, repSupport)

Y_predprRep = linregRep.predict(prPop)

prRepR2 = r2_score(repSupport, Y_predprRep)

plt.figtext(.7, .71, ("republican R??=" + "{:.8f}".format(prRepR2)))

plt.plot(prPop, Y_predprRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in puerto rican population")

plt.ylabel("% point change in governors political support")

plt.title("change in puerto rican population vs change in governors support (MA 2014 - 2018)")

plt.show()

#plot governors election data for cuban population

plt.scatter(merged_governors["change in cuban population"], merged_governors["change in democratic support"], color = "blue", label = "democratic support",s=0.7)

plt.scatter(merged_governors["change in cuban population"], merged_governors["change in republican support"], color = "red", label = "republican support",s=0.7)

cubanPop = merged_governors["change in cuban population"].values.reshape(-1, 1)

linregDem.fit(cubanPop, demSupport)

Y_predcubanDem = linregDem.predict(cubanPop)

cubanDemR2 = r2_score(demSupport, Y_predcubanDem)

plt.figtext(.7, .73, ("democratic R?? =" + "{:.8f}".format(cubanDemR2)))

plt.plot(cubanPop, Y_predcubanDem, color='cyan', label = "democratic support trend")

linregRep.fit(cubanPop, repSupport)

Y_predcubanRep = linregRep.predict(cubanPop)

cubanRepR2 = r2_score(repSupport, Y_predcubanRep)

plt.figtext(.7, .71, ("republican R??=" + "{:.8f}".format(cubanRepR2)))

plt.plot(cubanPop, Y_predcubanRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in cuban population")

plt.ylabel("% point change in governors political support")

plt.title("change in cuban population vs change in governors support (MA 2014 - 2018)")

plt.show()

#plot governors election data for other LatinX populations

plt.scatter(merged_governors["change in other LatinX population"], merged_governors["change in democratic support"], color = "blue", label = "democratic support",s=0.7)

plt.scatter(merged_governors["change in other LatinX population"], merged_governors["change in republican support"], color = "red", label = "republican support",s=0.7)

otherPop = merged_governors["change in other LatinX population"].values.reshape(-1, 1)

linregDem.fit(otherPop, demSupport)

Y_predotherDem = linregDem.predict(otherPop)

otherDemR2 = r2_score(demSupport, Y_predotherDem)

plt.figtext(.7, .73, ("democratic R?? =" + "{:.8f}".format(otherDemR2)))

plt.plot(otherPop, Y_predotherDem, color='cyan', label = "democratic support trend")

linregRep.fit(otherPop, repSupport)

Y_predotherRep = linregRep.predict(otherPop)

otherRepR2 = r2_score(repSupport, Y_predotherRep)

plt.figtext(.7, .71, ("republican R??=" + "{:.8f}".format(otherRepR2)))

plt.plot(otherPop, Y_predotherRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in other LatinX population")

plt.ylabel("% point change in governors political support")

plt.title("change in other LatinX population vs change in governors support (MA 2014 - 2018)")

plt.show()

#plot governors election data for Total LatinX populations

plt.scatter(merged_governors["change in total LatinX population"], merged_governors["change in democratic support"], color = "blue", label = "democratic support",s=0.7)

plt.scatter(merged_governors["change in total LatinX population"], merged_governors["change in republican support"], color = "red", label = "republican support",s=0.7)

totalPop = merged_governors["change in total LatinX population"].values.reshape(-1, 1)

linregDem.fit(totalPop, demSupport)

Y_predtotalDem = linregDem.predict(totalPop)

totalDemR2 = r2_score(demSupport, Y_predtotalDem)

plt.figtext(.7, .73, ("democratic R?? =" + "{:.8f}".format(totalDemR2)))

plt.plot(totalPop, Y_predtotalDem, color='cyan', label = "democratic support trend")

linregRep.fit(totalPop, repSupport)

Y_predtotalRep = linregRep.predict(totalPop)

totalRepR2 = r2_score(repSupport, Y_predtotalRep)

plt.figtext(.7, .71, ("republican R??=" + "{:.8f}".format(totalRepR2)))

plt.plot(totalPop, Y_predtotalRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in total LatinX population")

plt.ylabel("% point change in governors political support")

plt.title("change in total LatinX population vs change in governors support (MA 2014 - 2018)")

plt.show()


