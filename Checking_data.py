with open("TilePickle_5.pkl", "rb") as f:
    tile = pickle.load(f)

# Flatten matrix
reshaped_tile = tile.reshape(tile.shape[0], (tile.shape[1] * tile.shape[2]))

# Transform to dataframe
df = pd.DataFrame(reshaped_tile.T, columns=['AggIndex1', 'AggIndex2', 'AggIndex3',
                                            'AggIndex4', 'AggIndex5', 'EdgeDensity1',
                                            'EdgeDensity2', 'EdgeDensity3', 'EdgeDensity4',
                                            'PatchDensity1', 'PatchDensity2', 'PatchDensity3',
                                            'PatchDensity4', 'LandcoverPercentage1', 'LandcoverPercentage2',
                                            'LandcoverPercentage3', 'LandcoverPercentage4', 'ShannonDiversity ',
                                            'current_deforestationDistance ', 'current_degradationDistance',
                                            'future_deforestation', 'RawSarVisionClasses', 'SarvisionBasemap',
                                            'scaledPopDensity', 'scaledASTER', 'RoadsDistance', 'UrbanicityDistance',
                                            'WaterwaysDistance', 'CoastlineDistance', 'MillDistance',
                                            'PalmOilConcession ',
                                            'gradientASTER', 'LogRoadDistance', 'Vegetype', 'CurrentMonth', 'y center',
                                            'x center', 'time', 'size '])


print(df.future_deforestation.value_counts())