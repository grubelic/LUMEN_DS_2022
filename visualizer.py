import os

import folium
from folium import plugins as fplugins
from folium.map import Icon
import pandas as pd
import webbrowser
from random import choice, seed
# https://nyctomachia.wordpress.com/2018/12/08/how-to-install-cartopy-and-shapely-for-python-on-windows/comment-page-1/
from cartopy import crs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
# https://coderzcolumn.com/tutorials/data-science/cartopy-basic-maps-scatter-map-bubble-map-and-connection-map

popup_template = """
<a href="file:///ABS_PATH/0.jpg" target="_blank">0</a>
<a href="file:///ABS_PATH/90.jpg" target="_blank">90</a>
<a href="file:///ABS_PATH/180.jpg" target="_blank">180</a>
<a href="file:///ABS_PATH/270.jpg" target="_blank">270</a>
"""
popup_template = popup_template.replace(
    "ABS_PATH",
    os.path.abspath(os.path.join(
        '..', "dataset", "data", "UUID"
    )).replace("\\", '/')
)

markercluster_template = """
function (cluster) {
 var childCount = cluster.getChildCount();
 let cntCorrect = 0;
 cluster.getAllChildMarkers().forEach(it => {
    if (it.options.icon.options.markerColor === "white") {
        cntCorrect += 1;
    }
 });
 const g = Math.floor(cntCorrect * 255 / childCount);
 const fg = (cntCorrect / childCount > 0.5) ? 'black' : 'white';
 return new L.DivIcon({ 
    html: '<div style="background-color: rgb(' + g + ', ' + g + ', ' + g + ');color: '+fg+'; font-family:"Arial Black"><span><b>' + childCount + '</b></span></div>', 
    
    className: 'marker-cluster', 
    iconSize: new L.Point(40, 40) });
 }
"""


# html: '<div style="background-color: rgb(' + (255 - g) + ', ' + g + ', 0);color: white"><span><b>' + childCount + '</b></span></div>',


def plot_df(df: pd.DataFrame):
    m = folium.Map()
    m.fit_bounds([
        (df.latitude.min(), df.longitude.min()),
        (df.latitude.max(), df.longitude.max())])
    popups = list(df.uuid.apply(lambda x: popup_template.replace("UUID", x)))
    icons = list(df.val.apply(lambda x: Icon(
        color="white" if x else "black",
        icon="ok" if x else "remove",
        icon_color="black" if x else "white"
    )))
    marker_cluster = fplugins.MarkerCluster(
        locations=df[["latitude", "longitude"]],
        popups=popups,
        icons=icons,
        icon_create_function=markercluster_template,
        options={"maxClusterRadius": "20", "singleMarkerMode": "true"}
    )
    marker_cluster.add_to(m)
    m.save("map.html")
    webbrowser.open("map.html")


if __name__ == "__main__":
    df = pd.read_csv("../dataset/data.csv")
    print(df.info())
    seed(42)
    if "val" not in df.columns:  # dodaj random column
        df["val"] = [choice((True, False)) for it in range(len(df.uuid))]
    plot_df(df)

