import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn.datasets as datasets
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
import sklearn.linear_model


pres2016 = pd.read_csv("Presidential Demographic (Tracts) + Voting Data (Precincts) - 2016.csv")

pres2020 = pd.read_csv("Presidential Demographic (Tracts) + Voting Data (Precincts) - 2020.csv")

gov2014 = pd.read_csv("Governor Demographic (Tracts) + Voting Data (Precincts) - 2014.csv")

gov2018 = pd.read_csv("Governor Demographic (Tracts) + Voting Data (Precincts) - 2018.csv")

#cleaning for presidential election

pres2020 = pres2020.replace(",", "", regex = True)

pres2016 = pres2016.replace(",", "", regex = True)

pres2020["Joseph R. Biden, Jr."] = pd.to_numeric(pres2020["Joseph R. Biden, Jr."])

pres2020["Donald J. Trump"] = pd.to_numeric(pres2020["Donald J. Trump"])

pres2020["Total Votes Cast"] = pd.to_numeric(pres2020["Total Votes Cast"])

pres2020["Estimate Mexican Population"] = pd.to_numeric(pres2020["Estimate Mexican Population"])

pres2020["Estimate Puerto Rican Population"] = pd.to_numeric(pres2020["Estimate Puerto Rican Population"])

pres2020["Estimate Cuban Population"] = pd.to_numeric(pres2020["Estimate Cuban Population"])

pres2020["Estimate Other Hispanic or Latino"] = pd.to_numeric(pres2020["Estimate Other Hispanic or Latino"])

pres2020["Estimate Total Population"] = pd.to_numeric(pres2020["Estimate Total Population"])

pres2016["Hillary Clinton"] = pd.to_numeric(pres2016["Hillary Clinton"])

pres2016["Donald J. Trump"] = pd.to_numeric(pres2016["Donald J. Trump"])

pres2016["Total Votes Cast"] = pd.to_numeric(pres2016["Total Votes Cast"])

pres2016["Estimate Mexican Population"] = pd.to_numeric(pres2016["Estimate Mexican Population"])

pres2016["Estimate Puerto Rican Population"] = pd.to_numeric(pres2016["Estimate Puerto Rican Population"])

pres2016["Estimate Cuban Population"] = pd.to_numeric(pres2016["Estimate Cuban Population"])

pres2016["Estimate Other Hispanic or Latino"] = pd.to_numeric(pres2016["Estimate Other Hispanic or Latino"])

pres2016["Estimate Total Population"] = pd.to_numeric(pres2016["Estimate Total Population"])

#cleaning for governors election

gov2014 = gov2014.replace(",", "", regex = True)

gov2018 = gov2018.replace(",", "", regex = True)

gov2018["Gonzalez and Palfrey (D)"] = pd.to_numeric(gov2018["Gonzalez and Palfrey (D)"])

gov2018["Baker and Polito (R)"] = pd.to_numeric(gov2018["Baker and Polito (R)"])

gov2018["Total Votes Cast"] = pd.to_numeric(gov2018["Total Votes Cast"])

gov2018["Estimate Mexican Population"] = pd.to_numeric(gov2018["Estimate Mexican Population"])

gov2018["Estimate Puerto Rican Population"] = pd.to_numeric(gov2018["Estimate Puerto Rican Population"])

gov2018["Estimate Cuban Population"] = pd.to_numeric(gov2018["Estimate Cuban Population"])

gov2018["Estimate Other Hispanic or Latino"] = pd.to_numeric(gov2018["Estimate Other Hispanic or Latino"])

gov2018["Estimate Total Population"] = pd.to_numeric(gov2018["Estimate Total Population"])

gov2014["Coakley and Kerrigan (D)"] = pd.to_numeric(gov2014["Coakley and Kerrigan (D)"])

gov2014["Baker and Polito (R)"] = pd.to_numeric(gov2014["Baker and Polito (R)"])

gov2014["Total Votes Cast"] = pd.to_numeric(gov2014["Total Votes Cast"])

gov2014["Estimate Mexican Population"] = pd.to_numeric(gov2014["Estimate Mexican Population"])

gov2014["Estimate Puerto Rican Population"] = pd.to_numeric(gov2014["Estimate Puerto Rican Population"])

gov2014["Estimate Cuban Population"] = pd.to_numeric(gov2014["Estimate Cuban Population"])

gov2014["Estimate Other Hispanic or Latino"] = pd.to_numeric(gov2014["Estimate Other Hispanic or Latino"])

gov2014["Estimate Total Population"] = pd.to_numeric(gov2014["Estimate Total Population"])

#analyzing changes between the most recent elections for president

changeDemPres = (pres2020["Joseph R. Biden, Jr."] / pres2020["Total Votes Cast"]) - (pres2016["Hillary Clinton"] / pres2016["Total Votes Cast"])

changeRepPres = (pres2020["Donald J. Trump"] / pres2020["Total Votes Cast"]) - (pres2016["Donald J. Trump"] / pres2016["Total Votes Cast"])

changeMexPres = (pres2020["Estimate Mexican Population"] / pres2020["Estimate Total Population"]) - (pres2016["Estimate Mexican Population"] / pres2016["Estimate Total Population"])

changePRPres = (pres2020["Estimate Puerto Rican Population"] / pres2020["Estimate Total Population"]) - (pres2016["Estimate Puerto Rican Population"] / pres2016["Estimate Total Population"])

changeCubanPres = (pres2020["Estimate Cuban Population"] / pres2020["Estimate Total Population"]) - (pres2016["Estimate Cuban Population"] / pres2016["Estimate Total Population"])

changeOtherLatinXPres = (pres2020["Estimate Other Hispanic or Latino"] / pres2020["Estimate Total Population"]) - (pres2016["Estimate Other Hispanic or Latino"] / pres2016["Estimate Total Population"])

changeTotalLatinXPres = ((pres2020["Estimate Mexican Population"] + pres2020["Estimate Puerto Rican Population"] + pres2020["Estimate Cuban Population"] + pres2020["Estimate Other Hispanic or Latino"])
 / pres2020["Estimate Total Population"]) - ((pres2016["Estimate Mexican Population"] + pres2016["Estimate Puerto Rican Population"] + pres2016["Estimate Cuban Population"] +pres2016["Estimate Other Hispanic or Latino"])
 / pres2016["Estimate Total Population"])

#analyzing changes between most recent elections for governors

changeDemGov = (gov2018["Gonzalez and Palfrey (D)"] / gov2018["Total Votes Cast"]) - (gov2014["Coakley and Kerrigan (D)"] / gov2014["Total Votes Cast"])

changeRepGov = (gov2018["Baker and Polito (R)"] / gov2018["Total Votes Cast"]) - (gov2014["Baker and Polito (R)"] / gov2014["Total Votes Cast"])

changeMexGov = (gov2018["Estimate Mexican Population"] / gov2018["Estimate Total Population"]) / (gov2014["Estimate Mexican Population"] / gov2014["Estimate Total Population"])

changePRGov = (gov2018["Estimate Puerto Rican Population"] / gov2018["Estimate Total Population"]) / (gov2014["Estimate Puerto Rican Population"] / gov2014["Estimate Total Population"])

changeCubanGov = (gov2018["Estimate Cuban Population"] / gov2018["Estimate Total Population"]) / (gov2014["Estimate Cuban Population"] / gov2014["Estimate Total Population"])

changeOtherLatinXGov = (gov2018["Estimate Other Hispanic or Latino"] / gov2018["Estimate Total Population"]) / (gov2014["Estimate Other Hispanic or Latino"] / gov2014["Estimate Total Population"])

changeTotalLatinXGov = ((gov2018["Estimate Mexican Population"] + gov2018["Estimate Puerto Rican Population"] + gov2018["Estimate Cuban Population"] + gov2018["Estimate Other Hispanic or Latino"])
 / gov2018["Estimate Total Population"]) - ((gov2014["Estimate Mexican Population"] + gov2014["Estimate Puerto Rican Population"] + gov2014["Estimate Cuban Population"] + gov2014["Estimate Other Hispanic or Latino"])
 / gov2014["Estimate Total Population"])

#merge presidential election changes with demographic changes during those years

merged_presidential = pd.DataFrame({"change in democratic support" : changeDemPres, "change in republican support" : changeRepPres, "change in mexican population" : changeMexPres,
                                    "change in puerto rican population" : changePRPres, "change in cuban population" : changeCubanPres, "change in other LatinX population" : changeOtherLatinXPres,
                                    "change in total LatinX population" : changeTotalLatinXPres})

merged_presidential = merged_presidential.replace([np.inf, -np.inf], np.nan)    #replace inf values by NaN, occurs when starting value is zero

merged_presidential = merged_presidential.fillna(0)

#plot presidential election data for mexican population

plt.scatter(merged_presidential["change in mexican population"], merged_presidential["change in democratic support"], color = "blue", label = "democratic support")

plt.scatter(merged_presidential["change in mexican population"], merged_presidential["change in republican support"], color = "red", label = "republican support")

mexPop = merged_presidential["change in mexican population"].values.reshape(-1, 1)

demSupport = merged_presidential["change in democratic support"].values.reshape(-1, 1)

linregDem = LinearRegression()

linregDem.fit(mexPop, demSupport)

Y_predMexDem = linregDem.predict(mexPop)

plt.plot(mexPop, Y_predMexDem, color='cyan', label = "demcratic support trend")

repSupport = merged_presidential["change in republican support"].values.reshape(-1, 1)

linregRep = LinearRegression()

linregRep.fit(mexPop, repSupport)

Y_predMexRep = linregRep.predict(mexPop)

plt.plot(mexPop, Y_predMexRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in mexican population")

plt.ylabel("% point change in presidential political support")

plt.title("change in mexican population vs change in political support (2016 - 2020)")

plt.show()

#plot presidential election data for puetro rican population

plt.xlim(-2,3)  #has extreme x values that makes the distribution of plots not visible

plt.scatter(merged_presidential["change in puerto rican population"], merged_presidential["change in democratic support"], color = "blue", label = "democratic support")

plt.scatter(merged_presidential["change in puerto rican population"], merged_presidential["change in republican support"], color = "red", label = "republican support")

prPop = merged_presidential["change in puerto rican population"].values.reshape(-1, 1)

linregDem.fit(prPop, demSupport)

Y_predprDem = linregDem.predict(prPop)

plt.plot(prPop, Y_predprDem, color='cyan', label = "demcratic support trend")

linregRep.fit(prPop, repSupport)

Y_predprRep = linregRep.predict(prPop)

plt.plot(prPop, Y_predprRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in puerto rican population")

plt.ylabel("% point change in presidential political support")

plt.title("change in puetro rican population vs change in political support (2016 - 2020)")

plt.show()

#plot presidential election data for cuban population

plt.scatter(merged_presidential["change in cuban population"], merged_presidential["change in democratic support"], color = "blue", label = "democratic support")

plt.scatter(merged_presidential["change in cuban population"], merged_presidential["change in republican support"], color = "red", label = "republican support")

cubanPop = merged_presidential["change in cuban population"].values.reshape(-1, 1)

linregDem.fit(cubanPop, demSupport)

Y_predcubanDem = linregDem.predict(cubanPop)

plt.plot(cubanPop, Y_predcubanDem, color='cyan', label = "demcratic support trend")

linregRep.fit(cubanPop, repSupport)

Y_predcubanRep = linregRep.predict(cubanPop)

plt.plot(cubanPop, Y_predcubanRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in cuban population")

plt.ylabel("% point change in presidential political support")

plt.title("change in cuban population vs change in political support (2016 - 2020)")

plt.show()

#plot presidential election data for other LatinX populations

plt.scatter(merged_presidential["change in other LatinX population"], merged_presidential["change in democratic support"], color = "blue", label = "democratic support")

plt.scatter(merged_presidential["change in other LatinX population"], merged_presidential["change in republican support"], color = "red", label = "republican support")

otherPop = merged_presidential["change in other LatinX population"].values.reshape(-1, 1)

linregDem.fit(otherPop, demSupport)

Y_predotherDem = linregDem.predict(otherPop)

plt.plot(otherPop, Y_predotherDem, color='cyan', label = "demcratic support trend")

linregRep.fit(otherPop, repSupport)

Y_predotherRep = linregRep.predict(otherPop)

plt.plot(otherPop, Y_predotherRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in other LatinX population")

plt.ylabel("% point change in presidential political support")

plt.title("change in other LatinX population vs change in political support (2016 - 2020)")

plt.show()

#plot presidential election data for Total LatinX populations

plt.scatter(merged_presidential["change in total LatinX population"], merged_presidential["change in democratic support"], color = "blue", label = "democratic support")

plt.scatter(merged_presidential["change in total LatinX population"], merged_presidential["change in republican support"], color = "red", label = "republican support")

totalPop = merged_presidential["change in total LatinX population"].values.reshape(-1, 1)

linregDem.fit(totalPop, demSupport)

Y_predtotalDem = linregDem.predict(totalPop)

plt.plot(totalPop, Y_predtotalDem, color='cyan', label = "demcratic support trend")

linregRep.fit(totalPop, repSupport)

Y_predtotalRep = linregRep.predict(totalPop)

plt.plot(totalPop, Y_predtotalRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in total LatinX population")

plt.ylabel("% point change in presidential political support")

plt.title("change in total LatinX population vs change in political support (2016 - 2020)")

plt.show()

#merge governors election changes with demographic changes during those years

merged_governors = pd.DataFrame({"change in democratic support" : changeDemGov, "change in republican support" : changeRepGov, "change in mexican population" : changeMexGov,
                                    "change in puerto rican population" : changePRGov, "change in cuban population" : changeCubanGov, "change in other LatinX population" : changeOtherLatinXGov,
                                    "change in total LatinX population" : changeTotalLatinXGov})

merged_governors = merged_governors.replace([np.inf, -np.inf], np.nan)    #replace inf values by NaN, occurs when starting value is zero

merged_governors = merged_governors.fillna(0)

#plot governors election data for mexican population

plt.scatter(merged_governors["change in mexican population"], merged_governors["change in democratic support"], color = "blue", label = "democratic support")

plt.scatter(merged_governors["change in mexican population"], merged_governors["change in republican support"], color = "red", label = "republican support")

mexPop = merged_governors["change in mexican population"].values.reshape(-1, 1)

demSupport = merged_governors["change in democratic support"].values.reshape(-1, 1)

linregDem = LinearRegression()

linregDem.fit(mexPop, demSupport)

Y_predMexDem = linregDem.predict(mexPop)

plt.plot(mexPop, Y_predMexDem, color='cyan', label = "demcratic support trend")

repSupport = merged_governors["change in republican support"].values.reshape(-1, 1)

linregRep = LinearRegression()

linregRep.fit(mexPop, repSupport)

Y_predMexRep = linregRep.predict(mexPop)

plt.plot(mexPop, Y_predMexRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% change in mexican population")

plt.ylabel("% change in governors political support")

plt.title("change in mexican population vs change in political support (2014 - 2018)")

plt.show()

#plot governors election data for puetro rican population

plt.scatter(merged_governors["change in puerto rican population"], merged_governors["change in democratic support"], color = "blue", label = "democratic support")

plt.scatter(merged_governors["change in puerto rican population"], merged_governors["change in republican support"], color = "red", label = "republican support")

prPop = merged_governors["change in puerto rican population"].values.reshape(-1, 1)

linregDem.fit(prPop, demSupport)

Y_predprDem = linregDem.predict(prPop)

plt.plot(prPop, Y_predprDem, color='cyan', label = "demcratic support trend")

linregRep.fit(prPop, repSupport)

Y_predprRep = linregRep.predict(prPop)

plt.plot(prPop, Y_predprRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% change in puerto rican population")

plt.ylabel("% change in governors political support")

plt.title("change in puerto rican population vs change in political support (2014 - 2018)")

plt.show()

#plot governors election data for cuban population

plt.scatter(merged_governors["change in cuban population"], merged_governors["change in democratic support"], color = "blue", label = "democratic support")

plt.scatter(merged_governors["change in cuban population"], merged_governors["change in republican support"], color = "red", label = "republican support")

cubanPop = merged_governors["change in cuban population"].values.reshape(-1, 1)

linregDem.fit(cubanPop, demSupport)

Y_predcubanDem = linregDem.predict(cubanPop)

plt.plot(cubanPop, Y_predcubanDem, color='cyan', label = "demcratic support trend")

linregRep.fit(cubanPop, repSupport)

Y_predcubanRep = linregRep.predict(cubanPop)

plt.plot(cubanPop, Y_predcubanRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% change in cuban population")

plt.ylabel("% change in governors political support")

plt.title("change in cuban population vs change in political support (2014 - 2018)")

plt.show()

#plot governors election data for other LatinX populations

plt.scatter(merged_governors["change in other LatinX population"], merged_governors["change in democratic support"], color = "blue", label = "democratic support")

plt.scatter(merged_governors["change in other LatinX population"], merged_governors["change in republican support"], color = "red", label = "republican support")

otherPop = merged_governors["change in other LatinX population"].values.reshape(-1, 1)

linregDem.fit(otherPop, demSupport)

Y_predotherDem = linregDem.predict(otherPop)

plt.plot(otherPop, Y_predotherDem, color='cyan', label = "demcratic support trend")

linregRep.fit(otherPop, repSupport)

Y_predotherRep = linregRep.predict(otherPop)

plt.plot(otherPop, Y_predotherRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% change in other LatinX population")

plt.ylabel("% change in governors political support")

plt.title("change in other LatinX population vs change in political support (2014 - 2018)")

plt.show()

#plot governors election data for Total LatinX populations

plt.scatter(merged_governors["change in total LatinX population"], merged_governors["change in democratic support"], color = "blue", label = "democratic support")

plt.scatter(merged_governors["change in total LatinX population"], merged_governors["change in republican support"], color = "red", label = "republican support")

totalPop = merged_governors["change in total LatinX population"].values.reshape(-1, 1)

linregDem.fit(totalPop, demSupport)

Y_predtotalDem = linregDem.predict(totalPop)

plt.plot(totalPop, Y_predtotalDem, color='cyan', label = "demcratic support trend")

linregRep.fit(totalPop, repSupport)

Y_predtotalRep = linregRep.predict(totalPop)

plt.plot(totalPop, Y_predtotalRep, color='lightsalmon', label = "republican support trend")

plt.legend(loc="upper right")

plt.xlabel("% point change in total LatinX population")

plt.ylabel("% point change in governors political support")

plt.title("change in total LatinX population vs change in political support (2014 - 2018download disc)")

plt.show()
