import plotly.graph_objects as go


knn = [6.23,5.38,4.07,3.54,3.41,3.22,3.07,3.08,3.23,3.26,3.32,3.48,3.13,2.94,2.88,3.02,3.20,3.18,3.37,3.54,3.70,3.72,3.82,4.10,3.11,3.20,3.43,3.56,3.71,3.74,3.98,4.11,4.02,4.00,4.03,4.03,6.31,5.42,4.04,3.54,3.40,3.20,3.08,3.09,3.29,3.29,3.34,3.40,3.14,2.98,2.92,3.07,3.18,3.20,3.37,3.49,3.71,3.69,3.72,3.93,3.19,3.27,3.49,3.61,3.74,3.74,3.77,4.09,4.05,4.05,4.06,4.06,6.06,5.39,4.01,3.52,3.44,3.26,3.12,3.12,3.30,3.30,3.36,3.47,3.14,2.95,2.91,3.08,3.27,3.32,3.41,3.48,3.67,3.66,3.72,3.85,3.26,3.27,3.43,3.58,3.58,3.78,3.85,3.91,4.14,4.07,4.04,4.07]
pa =  [2.75,2.70,2.55,2.46,2.47,2.44,2.41,2.40,2.42,2.42,2.43,2.44,2.52,2.47,2.42,2.40,2.42,2.41,2.43,2.45,2.52,2.52,2.53,2.53,2.42,2.40,2.42,2.48,2.54,2.54,2.53,2.57,2.79,2.83,2.81,2.84,2.75,2.69,2.55,2.46,2.47,2.44,2.41,2.40,2.42,2.42,2.42,2.44,2.52,2.47,2.42,2.40,2.41,2.41,2.42,2.45,2.52,2.52,2.52,2.52,2.41,2.40,2.42,2.46,2.53,2.53,2.52,2.55,2.77,2.78,2.80,2.79,2.74,2.68,2.55,2.46,2.47,2.44,2.41,2.40,2.42,2.41,2.42,2.44,2.52,2.47,2.42,2.40,2.41,2.41,2.42,2.44,2.51,2.51,2.51,2.51,2.41,2.40,2.41,2.44,2.51,2.51,2.51,2.52,2.75,2.75,2.76,2.76]

fig = go.Figure()

# fig.add_trace(go.Box(y=knn))
# fig.add_trace(go.Box(y=pa))

fig.add_trace(go.Violin(x=[""]*len(knn), y=knn, side='negative', pointpos=0.9, name="Knn"))
fig.add_trace(go.Violin(x=[""]*len(pa), y=pa, side='positive', pointpos=-0.9, name="Pa"))


fig.update_traces(meanline_visible=True,
                  points='all', # show all points
                  jitter=0.5,  # add some jitter on points for better visibility
                  scalemode='width' #scale violin plot area with total count
)

fig.update_layout(violingap=0, violingroupgap=0, violinmode='overlay')

fig.layout = go.Layout(
    # title={'text': "Error distribution of the synthetic bases", 'x':.5, 'y': .8},
    xaxis={'title':'Synthetic bases', 'range':[-.4,.4]},
    yaxis={'title': "Avg error (m)"},
    legend={'x':.83, 'y':.98},
    # margin={"l":0, "r":0, "b":0, "t":0, "pad":0}
    margin={"pad":0, "t":20, "b":20, "l":30, "r":20},
    autosize=False,
    width=500,
    height=350,
    )
# fig.show()
fig.write_image("results/plots/comparison_syn_bases.pdf")