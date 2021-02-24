import json
import pandas as pd
from src.commom import Info
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

bestTrains =  [
                  # 2.40
                'training-synth--stdDev_1-PL0_40-n_5-wallLoss_7',
                'tra.ining-synth--stdDev_1-PL0_50-n_4-wallLoss_7',
                'training-synth--stdDev_2-PL0_40-n_5-wallLoss_7',
                'training-synth--stdDev_2-PL0_50-n_4-wallLoss_7',
                'training-synth--stdDev_2-PL0_60-n_4-wallLoss_3',
                'training-synth--stdDev_3-PL0_40-n_5-wallLoss_7',
                'training-synth--stdDev_3-PL0_50-n_4-wallLoss_7',
                'training-synth--stdDev_3-PL0_60-n_4-wallLoss_3',
              ]

def checkHit(trains):
  hit = 0
  for bestTrain in bestTrains:
    hit += sum(trains == bestTrain)
  return hit

def loadJson(path):
  with open(path) as json_file:
    return json.load(json_file)

jsonPath = 'results/Pa.json'
paResultsPath = 'results/Results_Pa.csv'


resultsPa = pd.read_csv(paResultsPath)
resultsPa = resultsPa[["Files", "AVG error (m)"]].set_index("Files").T.to_dict("list")

jsonData = loadJson(jsonPath)

knn_erros = pd.read_csv("results/knn_real_errors_n_18.csv").squeeze()
pa_erros = pd.read_csv("results/Pa_best_synth_errors_wt_0.01.csv").squeeze()
model_erros = pd.read_csv("results/Propagation_model_errors.csv").squeeze()

avgByBasePa = pd.read_csv("results/Results_Pa.csv")

avgErrors = []
stdErrors = []
hits = []


"""
NaNs JSON

"training-synth--stdDev_1-PL0_60-n_6-wallLoss_5" ---> [10]
"training-synth--stdDev_1-PL0_60-n_6-wallLoss_7" ---> [10]
"training-synth--stdDev_2-PL0_60-n_6-wallLoss_7" ---> [10]
"training-synth--stdDev_3-PL0_60-n_6-wallLoss_5" ---> [10]
"training-synth--stdDev_3-PL0_60-n_6-wallLoss_7" ---> [10]
"""
for key, value in jsonData.items():
  df = pd.DataFrame(value)
  df = df.drop(columns="Real_base_8_beacons") #Regenerate Pa.json  and remove this line
  best_train = df.idxmin(axis=1)
  
  best_train_to_avg = best_train.apply(lambda row: resultsPa[row][0])
  hits.append(checkHit(best_train))
  avgErrors.append(best_train_to_avg.mean())
  stdErrors.append(best_train_to_avg.std())


#------ Bases
fig = go.Figure(data=go.Scatter(x=avgByBasePa["Files"], y=avgByBasePa["AVG error (m)"], 
                # error_y={"array":avgByBasePa["STD error (m)"]}
                ))
fig.layout = go.Layout(
    title={'text': "Avg error by training", 'x':.5, 'y': .9},
    xaxis={'title':'Training bases', 'showticklabels':True, 'type':'category'},
    yaxis={'title': "Avg error (m)"}
    )
fig.write_html("results/plots/figA.html")

fig = go.Figure(data=go.Scatter(x=avgByBasePa["Files"], y=avgByBasePa["Droped lines"]))
fig.layout = go.Layout(
    title={'text': "Test samples droped", 'x':.5, 'y': .9},
    xaxis={'title':'Training bases', 'showticklabels':True, 'type':'category'},
    yaxis={'title': "Droped samples"}
    )
fig.write_html("results/plots/figAA.html")

# ------ bases sorted

avgByBasePaSorted = avgByBasePa.sort_values("AVG error (m)")

fig = go.Figure(data=go.Scatter(x=avgByBasePaSorted["Files"], y=avgByBasePaSorted["AVG error (m)"],
                # error_y={"array":avgByBasePaSorted["STD error (m)"]}
                ))
fig.layout = go.Layout(
    title={'text': "Avg error by training (sorted)", 'x':.5, 'y': .9},
    xaxis={'title':'Training bases', 'showticklabels':True, 'type':'category'},
    yaxis={'title': "Avg error (m)"}
    )
fig.write_html("results/plots/figB.html")



#------ avg error
ticks = np.linspace(1,len(Info.labels), 20).astype(int)
fig = go.Figure(data=go.Scatter(x=ticks, y=avgErrors, error_y={'array': stdErrors}))
fig.layout = go.Layout(
    title={'text': "Evaluation of the avg error by amount of test samples", 'x':.5, 'y': .85},
    xaxis={'title':'Test Samples', 'showticklabels':True, 'type':'category'},
    yaxis={'title': "Avg error (m)"}
    )
fig.write_image("results/plots/fig1.pdf")

# ------- Hit error
ticks = np.linspace(1,len(Info.labels), 20).astype(int)
fig = go.Figure([go.Bar(x=ticks, y=hits)])
fig.layout = go.Layout(
    title={'text': "Evaluation of the hits by amount of test samples", 'x':.5, 'y': .85},
    xaxis={'title':'Test Samples', 'showticklabels':True, 'type':'category'},
    yaxis={'title': "Hits"}
    )
fig.write_image("results/plots/fig2.pdf")

# ----- Avg error solutions
colors = ["red", "green", "blue"]
solutionsName = [["knn_real_base"], ["PA_best_base"], ["Propagation_model"]]
solutionsAvgErrors = [[knn_erros.mean()], [pa_erros.mean()], [model_erros.mean()]]
stdErrorSolutions = [[knn_erros.std()], [pa_erros.std()], [model_erros.std()]]

fig=go.Figure()
for i, _ in enumerate(solutionsName):
  fig.add_traces(go.Bar(x=solutionsName[i], y=solutionsAvgErrors[i],
                        marker_color=colors[i], opacity=0.7,
                        error_y={'array': stdErrorSolutions[i]}))
fig.layout = go.Layout(
    title={'text': "Avg by Approaches", 'x':.5, 'y': .85},
    xaxis={'title':'Approaches', 'showticklabels':True, 'type':'category'},
    yaxis={'title': "Avg error (m)"},
    showlegend=False
    )
fig.write_image("results/plots/fig3.pdf")

#---- Cumulative
# fig, ax = plt.subplots()
# q = sns.kdeplot(knn_erros, cumulative=True, label=solutionsName[0], color="red", linestyle="--")
# sns.kdeplot(pa_erros, cumulative=True, label=solutionsName[1], color="green", linestyle="-")
# sns.kdeplot(model_erros, cumulative=True, label=solutionsName[2], color="blue", linestyle=":")

# ax.set_xlim(0,10)
# ax.set_xlabel("Error (m)")
# ax.set_ylabel("Density")
# ax.set_title("CDF comparison")
# ax.legend()

# fig.savefig("results/plots/fig4.pdf")

#---- Cumulative 2
solutions = [knn_erros, pa_erros, model_erros]
dashes = ['dash', 'dot', 'dashdot']

fig=go.Figure()
for i, solution in enumerate(solutions):
  _ , ax = plt.subplots()
  x, y = sns.kdeplot(solution, cumulative=True).get_lines()[0].get_data()
  fig.add_traces(go.Scatter(x=x, y=y, line={'dash':dashes[i]}, name=solutionsName[i][0], marker_color=colors[i]))

fig.layout = go.Layout(
    title={'text': "CDF comparison", 'x':.5, 'y': .85},
    xaxis={'title':'Error (m)', 'range':[0,10]},
    yaxis={'title': "Density", 'range':[0,1.1]},
    legend={'x':.7, 'y':.8}
    )

fig.write_image("results/plots/fig4.pdf")