import pandas as pd, numpy as np, streamlit as st, plotly.express as px
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import joblib
from streamlit_folium import st_folium
import folium
from shapely.geometry import Point, shape
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

save_path = r"C:\Users\TEXNO\Desktop\Jupiter\Bina_az_saved_models"

districts = ['ABSHERON', 'BINAGADI', 'NARIMANOV', 'NASIMI', 'NIZAMI', 'SEBAIL', 'KHATAI', 'YASAMAL']

linear_models = {}
scalers = {}
polys = {}
rf_models = {}
xgb_models = {}

for district in districts:
    linear_models[district] = joblib.load(os.path.join(save_path, f"{district}_linear.pkl"))
    scalers[district] = joblib.load(os.path.join(save_path, f"{district}_scaler.pkl"))
    polys[district] = joblib.load(os.path.join(save_path, f"{district}_poly.pkl"))

    rf_models[district] = joblib.load(os.path.join(save_path, f"{district}_rf.pkl"))
    xgb_models[district] = joblib.load(os.path.join(save_path, f"{district}_xgb.pkl"))

st.set_page_config(page_title='Houses Price Prediction', page_icon='🏢', layout="wide")
st.sidebar.image(r"C:\Users\TEXNO\Desktop\Streamlit\Bina_az_project\houses_logo.png", use_container_width=True)

st.sidebar.markdown("<h1 style='text-align: left;'> ", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='text-align: left;'>🌐 Bina.az Dashboard</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Overview", "Visualization", "Price Prediction", "About Model"])
st.sidebar.markdown("<h1 style='text-align: left;'> ", unsafe_allow_html=True)
st.sidebar.markdown("----")
st.sidebar.markdown("#### 📂 Data Source")
st.sidebar.markdown("Bina.az")

if page == "Overview": 
    banner_image = Image.open(r"C:\Users\TEXNO\Desktop\Streamlit\Bina_az_project\city_banner_photo.jpg")
    resized_image = banner_image.resize((1200, 300))
    st.image(resized_image, use_container_width=True)
 
    st.title("🏢 Bina.az Houses Price Prediction Dashboard")
    
    st.markdown("""
    Welcome to the Bina.az Real Estate Dashboard!  

    This project focuses on analyzing and predicting house prices in different districts of Baku.  
    The data was collected from Bina.az and includes the following information for each listing:  
    - Price in Azerbaijani Manat (₼)
    - Area in square meters
    - Number of rooms
    - Floor information
    - Renovation status
    - Mortgage availability
    - Title deed availability
    - Location (latitude & longitude)
    - Seller type (owner, agency, etc.)
    
    **Main features of this dashboard:**
    1. Explore key metrics and sample listings per district.
    2. Visualize price distributions, area vs price, and other trends.
    3. Predict house prices using trained machine learning models (Linear Regression, Random Forest, XGBoost).
    4. Understand the models and their performance.
    
    Use the sidebar to navigate through different pages of the dashboard.
    """)

    district = st.selectbox("Choose a district", districts)

    df_path = os.path.join(save_path, f"{district}_data.pkl") 

    if os.path.exists(df_path):
        df = joblib.load(df_path)

        total_listings = df.shape[0]
        avg_price = df['prices'].mean()
        avg_area = df['area_sqm'].mean()
        mc_rooms = df['room_count'].mode()[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Listings", total_listings)
        col2.metric("Average Price (₼)", f"{avg_price:,.2f}")
        col3.metric("Average Area (sqm)", f"{avg_area:.2f}")
        col4.metric("Most Common Room Count:", f"{mc_rooms}")

        st.subheader(f"Sample Listings in {district}")
        st.dataframe(df[['prices', 'area_sqm', 'room_count', 'floor_number', 'total_floor']].head(10))

        st.subheader("Top 5 Most Expensive Listings")
        st.dataframe(df.sort_values(by='prices', ascending=False).head(5)[['prices', 'area_sqm', 'room_count', 'floor_number']])
    
    else:
        st.warning(f"No data found for {district}. Make sure the saved data file exists.")

if page == "Visualization":
    st.title("📉 Housing Market Visualizations 📈")

    district = st.selectbox("Choose a district", districts)
    df_path = os.path.join(save_path, f"{district}_data.pkl")

    df = joblib.load(df_path)

    st.sidebar.subheader("🔎 Filters")
    min_price, max_price = int(df['prices'].min()), int(df['prices'].max())
    price_range = st.sidebar.slider("Select Price Range (₼)", min_price, max_price, (min_price, max_price))
    room_filter = st.sidebar.multiselect("Select Room Count", sorted(df['room_count'].unique()), default=sorted(df['room_count'].unique()))

    df_filtered = df[(df['prices'] >= price_range[0]) & (df['prices'] <= price_range[1])]
    df_filtered = df_filtered[df_filtered['room_count'].isin(room_filter)]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price Distribution")
        fig = px.histogram(df_filtered, x="prices", nbins=50,
                title="House Price Distribution",
                labels={"prices": "Price (₼)"})
        fig.update_traces(marker_line_color="black", marker_line_width=1.2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Area vs. Price")
        fig = px.scatter(df_filtered, x="area_sqm", y="prices", color="room_count",
                            title="Area vs. Price by Room Count",
                            labels={"area_sqm": "Area (sqm)", "prices": "Price (₼)"},
                            hover_data=["floor_number", "total_floor"])
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Price Distribution by Renovation Status")
        df_filtered["renovation_status_label"] = df_filtered["renovation_status"].map({0: "Not Renovated", 1: "Renovated"})

        fig = px.violin(df_filtered, x="renovation_status_label", y="prices", 
                        color="renovation_status_label", box=True, 
                        title="House Price by Renovation Status",
                        labels={"renovation_status_label": "Renovation Status", "prices": "Price (₼)"})
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Average Price per sqm by Room Count")

        df_filtered["price_per_sqm"] = df_filtered["prices"] / df_filtered["area_sqm"]

        avg_ppsqm = df_filtered.groupby("room_count")["price_per_sqm"].mean().reset_index()

        avg_ppsqm = avg_ppsqm.sort_values("room_count")

        fig = px.bar(
            avg_ppsqm,
            x="room_count",
            y="price_per_sqm",
            title="Average Price per sqm",
            labels={"room_count": "Room Count", "price_per_sqm": "₼/sqm"},
            category_orders={"room_count": sorted(avg_ppsqm["room_count"].unique())} 
        )

        fig.update_traces(marker_line_color="black", marker_line_width=1.5)

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Geographical Distribution of Listings by Area")

    selected_districts = st.multiselect("Select District(s) for Map", districts, default=[district])

    df_map = pd.concat([joblib.load(os.path.join(save_path, f"{d}_data.pkl")) for d in selected_districts])

    zoom_level = st.slider("Map Initial Zoom Level", 5, 15, 10)

    fig_map = px.scatter_mapbox(
        df_map,
        lat="latitude",
        lon="longitude",
        size="area_sqm",
        size_max=15,
        color="prices",
        hover_name="room_count",
        hover_data=["prices", "area_sqm", "floor_number", "seller_type", "renovation_status"],
        mapbox_style="open-street-map",
        zoom=zoom_level,
        height=600,
        color_continuous_scale=px.colors.cyclical.IceFire
    )

    st.plotly_chart(fig_map, use_container_width=True)

if page == "Price Prediction":
    
    absheron_boundary = {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "properties": {},
        "geometry": {
          "coordinates": [
            [
              [
                49.91227576001697,
                40.50702701193498
              ],
              [
                49.86141859744106,
                40.518234756765736
              ],
              [
                49.8544521282447,
                40.51927580369616
              ],
              [
                49.80182820226116,
                40.53180649236177
              ],
              [
                49.701283656737786,
                40.52903171228789
              ],
              [
                49.684210370979855,
                40.54305772262296
              ],
              [
                49.645116661740786,
                40.539241506238284
              ],
              [
                49.62718187606717,
                40.53475816339758
              ],
              [
                49.58761815830661,
                40.54326525870482
              ],
              [
                49.58599267298189,
                40.54176307149825
              ],
              [
                49.5611001893073,
                40.549769633187736
              ],
              [
                49.54254134157685,
                40.56067790425905
              ],
              [
                49.52927643805174,
                40.58212362177537
              ],
              [
                49.5176592742792,
                40.60023768525198
              ],
              [
                49.514443545527655,
                40.61466442668461
              ],
              [
                49.499473707395396,
                40.64513064584152
              ],
              [
                49.454537166906476,
                40.66264778063126
              ],
              [
                49.44206975658656,
                40.62216910669855
              ],
              [
                49.446877335170115,
                40.58186323425414
              ],
              [
                49.42464005860532,
                40.563392403292625
              ],
              [
                49.41876842877471,
                40.532312438524315
              ],
              [
                49.33698389503465,
                40.51455181478457
              ],
              [
                49.294428427528885,
                40.46461228527701
              ],
              [
                49.23326540296114,
                40.48207891568991
              ],
              [
                49.17888235476414,
                40.438738108003065
              ],
              [
                49.13997640319573,
                40.443879313433115
              ],
              [
                49.131319158989754,
                40.39043675497635
              ],
              [
                49.04801885771971,
                40.36979220829761
              ],
              [
                48.97597233352897,
                40.31024886087536
              ],
              [
                48.93125924825998,
                40.29564470762628
              ],
              [
                48.947705541087586,
                40.26049745827912
              ],
              [
                48.8497999818045,
                40.26565912802368
              ],
              [
                48.87498390457321,
                40.22019824740772
              ],
              [
                48.94740561264648,
                40.200779750431096
              ],
              [
                49.03384885052134,
                40.1442037967949
              ],
              [
                49.15500206402845,
                40.14164356599045
              ],
              [
                49.22566135890915,
                40.05147671639142
              ],
              [
                49.2896428811915,
                39.95857167673458
              ],
              [
                49.36702895272458,
                40.069528864000546
              ],
              [
                49.37056983253658,
                40.15993275013494
              ],
              [
                49.37831801788025,
                40.200177297059454
              ],
              [
                49.41242752837236,
                40.244430237752425
              ],
              [
                49.446887809710425,
                40.302345767675575
              ],
              [
                49.42149948837883,
                40.344563794303696
              ],
              [
                49.40264299881927,
                40.387534296810315
              ],
              [
                49.42850437437902,
                40.43233788631622
              ],
              [
                49.4647403969417,
                40.442973708599276
              ],
              [
                49.52260536660452,
                40.448418827712615
              ],
              [
                49.558609569178195,
                40.431218158991555
              ],
              [
                49.58761333724655,
                40.40014944559996
              ],
              [
                49.613001783394665,
                40.38655057989415
              ],
              [
                49.633869133711514,
                40.3718606196903
              ],
              [
                49.6594895898667,
                40.36444027098395
              ],
              [
                49.684210493898135,
                40.36480053877295
              ],
              [
                49.69876693590123,
                40.37338951500105
              ],
              [
                49.71114382090278,
                40.38452056412306
              ],
              [
                49.713142334539974,
                40.39171744594391
              ],
              [
                49.70148695817359,
                40.39950070376051
              ],
              [
                49.67711210623596,
                40.40683147672206
              ],
              [
                49.64344539269089,
                40.44718771490196
              ],
              [
                49.62117280223661,
                40.44795269652096
              ],
              [
                49.63564648436909,
                40.48014464738051
              ],
              [
                49.68023877132014,
                40.45249914171128
              ],
              [
                49.72769094693213,
                40.4771950774453
              ],
              [
                49.78724413989778,
                40.464271655462255
              ],
              [
                49.91227576001697,
                40.50702701193498
              ]
            ]
          ],
          "type": "Polygon"
        }
      }
    ]
  }
    
    narimanov_boundary = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              49.86589083785046,
              40.438893217926136
            ],
            [
              49.86310525678814,
              40.43817425829522
            ],
            [
              49.863337621017095,
              40.436416195955935
            ],
            [
              49.86482209123842,
              40.433716018431156
            ],
            [
              49.86749195190375,
              40.43220217022291
            ],
            [
              49.86741052005907,
              40.42994976129014
            ],
            [
              49.86652399289065,
              40.425841569516805
            ],
            [
              49.86963380279571,
              40.421203636852034
            ],
            [
              49.85823325812929,
              40.415751458905206
            ],
            [
              49.82483637338983,
              40.40782571625199
            ],
            [
              49.828609111028086,
              40.39791958803548
            ],
            [
              49.83249810932088,
              40.39878159750185
            ],
            [
              49.833513333986716,
              40.39668200467956
            ],
            [
              49.8426654510617,
              40.39878540093327
            ],
            [
              49.8451644220045,
              40.39290856980213
            ],
            [
              49.85002911834788,
              40.382207139395106
            ],
            [
              49.862060918465545,
              40.38976107568715
            ],
            [
              49.86664799526396,
              40.39079850249654
            ],
            [
              49.87139665194834,
              40.3890000806695
            ],
            [
              49.87417275033843,
              40.38891970454807
            ],
            [
              49.88238271977035,
              40.38966896699143
            ],
            [
              49.884981561909406,
              40.391593840965356
            ],
            [
              49.88638589718789,
              40.3946475798082
            ],
            [
              49.88551547453659,
              40.39830608304389
            ],
            [
              49.8818836632863,
              40.403226239548374
            ],
            [
              49.89062769334937,
              40.408264636666175
            ],
            [
              49.88910445389382,
              40.41049201140904
            ],
            [
              49.89061855704765,
              40.41177432313654
            ],
            [
              49.89049573587789,
              40.414737531946976
            ],
            [
              49.89573838605934,
              40.415814837607485
            ],
            [
              49.896148577185016,
              40.414711841353856
            ],
            [
              49.90701616837616,
              40.4197769395563
            ],
            [
              49.91644466088796,
              40.42446519126834
            ],
            [
              49.918369485620666,
              40.42251371176443
            ],
            [
              49.92108994548417,
              40.42544887546947
            ],
            [
              49.91139248726665,
              40.42839096424906
            ],
            [
              49.90457627501239,
              40.42783011510321
            ],
            [
              49.901685025702314,
              40.43043762440705
            ],
            [
              49.89692594991682,
              40.430885012743694
            ],
            [
              49.89443178941738,
              40.43368691447046
            ],
            [
              49.89156944754568,
              40.43509624156398
            ],
            [
              49.88853890648812,
              40.43468100225064
            ],
            [
              49.88457420209656,
              40.43578544188284
            ],
            [
              49.880151373642505,
              40.43464383212496
            ],
            [
              49.87543589968823,
              40.43528835979188
            ],
            [
              49.87177364452279,
              40.43621434373881
            ],
            [
              49.87009692820273,
              40.4377618611895
            ],
            [
              49.86589083785046,
              40.438893217926136
            ]
          ]
        ],
        "type": "Polygon"
      }
    }
  ]
}
    
    binagadi_boundary = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              49.891795612041165,
              40.42872419350252
            ],
            [
              49.894125049781735,
              40.432821864212826
            ],
            [
              49.89233120520342,
              40.434756065981915
            ],
            [
              49.885577498160046,
              40.43566793183257
            ],
            [
              49.87663866544284,
              40.43420296323457
            ],
            [
              49.87221216259104,
              40.436483925416326
            ],
            [
              49.863154303886006,
              40.442576800012574
            ],
            [
              49.84660440083064,
              40.44664057022203
            ],
            [
              49.841788459588884,
              40.45273862397755
            ],
            [
              49.84819789953539,
              40.465336954841604
            ],
            [
              49.85354338810865,
              40.47021418178386
            ],
            [
              49.84926486692635,
              40.477526126022866
            ],
            [
              49.83644203204335,
              40.48077381020789
            ],
            [
              49.82415869541671,
              40.48605452065638
            ],
            [
              49.81185183662333,
              40.484395308499444
            ],
            [
              49.802252029795085,
              40.48115505067028
            ],
            [
              49.787846561028516,
              40.47506286588214
            ],
            [
              49.777199215025036,
              40.466927385864125
            ],
            [
              49.75693116417358,
              40.45842045537802
            ],
            [
              49.74793721773855,
              40.44941049069061
            ],
            [
              49.73342512951291,
              40.45766414987952
            ],
            [
              49.71793339416297,
              40.44581770303142
            ],
            [
              49.74578997799125,
              40.440360353349035
            ],
            [
              49.74265303185231,
              40.43469762447609
            ],
            [
              49.74452110789119,
              40.42693353994977
            ],
            [
              49.747094176426344,
              40.42087917687812
            ],
            [
              49.749393741908136,
              40.41287333050278
            ],
            [
              49.75355818165818,
              40.404956460783126
            ],
            [
              49.76521361424477,
              40.4018591528521
            ],
            [
              49.77448130811155,
              40.40421492460791
            ],
            [
              49.77252076768471,
              40.40801341701089
            ],
            [
              49.77443702375899,
              40.414473444097666
            ],
            [
              49.777937762935665,
              40.418014654149886
            ],
            [
              49.781042407198186,
              40.41479255698434
            ],
            [
              49.780296905656684,
              40.40805713941771
            ],
            [
              49.78031214274495,
              40.40542845356782
            ],
            [
              49.79496071094857,
              40.40924873573769
            ],
            [
              49.81792186274572,
              40.42419533398893
            ],
            [
              49.81985100955751,
              40.41987803495929
            ],
            [
              49.81643007362328,
              40.41660820261676
            ],
            [
              49.82610793398108,
              40.41432747128013
            ],
            [
              49.83355104345824,
              40.411378805558876
            ],
            [
              49.8573081459714,
              40.41639405729309
            ],
            [
              49.87275854083004,
              40.423666704630314
            ],
            [
              49.88093083617886,
              40.42764237860274
            ],
            [
              49.891795612041165,
              40.42872419350252
            ]
          ]
        ],
        "type": "Polygon"
      }
    }
  ]
}
    nasimi_boundary = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              [
                49.80211349081593,
                40.41328751448135
              ],
              [
                49.802495370608035,
                40.41293923027574
              ],
              [
                49.80084386512975,
                40.41621233583657
              ],
              [
                49.802653647641606,
                40.41829309783918
              ],
              [
                49.80774232213351,
                40.416679792658286
              ],
              [
                49.81044919081593,
                40.41870031448135
              ],
              [
                49.81411469081593,
                40.42133131448134
              ],
              [
                49.817346190815925,
                40.41995271448135
              ],
              [
                49.824858990815926,
                40.41670731448134
              ],
              [
                49.82169189081593,
                40.41347251448135
              ],
              [
                49.822077290815926,
                40.412442214481345
              ],
              [
                49.82355179081593,
                40.40850081448134
              ],
              [
                49.824971990815925,
                40.40883891448134
              ],
              [
                49.82685369081593,
                40.402888214481344
              ],
              [
                49.82757469081593,
                40.40104501448135
              ],
              [
                49.828527520115856,
                40.398708160485484
              ],
              [
                49.83231956081344,
                40.39968491724059
              ],
              [
                49.832795738285796,
                40.39861715020902
              ],
              [
                49.83860172968926,
                40.39997634410614
              ],
              [
                49.84188425966517,
                40.40071246511772
              ],
              [
                49.84258480927171,
                40.39871588686998
              ],
              [
                49.845148977721216,
                40.39255637574003
              ],
              [
                49.84813620962336,
                40.38651561929452
              ],
              [
                49.848774612644576,
                40.384699458554564
              ],
              [
                49.85012724693558,
                40.38174655389321
              ],
              [
                49.85402630679948,
                40.385390422561954
              ],
              [
                49.85681148544808,
                40.38766677168567
              ],
              [
                49.85850302277619,
                40.38908077864579
              ],
              [
                49.86054252130231,
                40.39072212287358
              ],
              [
                49.861738560664286,
                40.39153718111156
              ],
              [
                49.86484159893463,
                40.391531658491594
              ],
              [
                49.866144816944946,
                40.39057883245711
              ],
              [
                49.86615739615343,
                40.38965494018775
              ],
              [
                49.86420784598841,
                40.389106036847195
              ],
              [
                49.86299723367222,
                40.388677177034204
              ],
              [
                49.86172618994221,
                40.38815754230793
              ],
              [
                49.860804405816474,
                40.387935461705005
              ],
              [
                49.86072829081593,
                40.38808431448135
              ],
              [
                49.860588790815925,
                40.387963714481344
              ],
              [
                49.860489690815925,
                40.38781511448135
              ],
              [
                49.85885999081593,
                40.38345361448135
              ],
              [
                49.858687490815925,
                40.38301111448135
              ],
              [
                49.85853299081593,
                40.38283461448135
              ],
              [
                49.85827549081593,
                40.382442314481345
              ],
              [
                49.85835269081593,
                40.38163811448135
              ],
              [
                49.85871319081593,
                40.38099081448134
              ],
              [
                49.85966599081593,
                40.38022591448134
              ],
              [
                49.86188039081593,
                40.37916671448134
              ],
              [
                49.86226249081593,
                40.37886371448135
              ],
              [
                49.86276659081593,
                40.37821811448135
              ],
              [
                49.86320129081593,
                40.377752114481346
              ],
              [
                49.86483479081593,
                40.37656501448134
              ],
              [
                49.86527999081593,
                40.375874314481344
              ],
              [
                49.865625790815926,
                40.375501014481344
              ],
              [
                49.866059190815925,
                40.375513214481344
              ],
              [
                49.86778854081592,
                40.374701538702496
              ],
              [
                49.86684277030477,
                40.37373888808242
              ],
              [
                49.86608710215246,
                40.373148841740765
              ],
              [
                49.86456986489673,
                40.37300476215542
              ],
              [
                49.86288749674543,
                40.37354007056636
              ],
              [
                49.86151814486274,
                40.37421568362156
              ],
              [
                49.857950979039664,
                40.37414181201486
              ],
              [
                49.85702959178373,
                40.373726662616505
              ],
              [
                49.85592138711974,
                40.3735041306906
              ],
              [
                49.85432208550374,
                40.375664594789555
              ],
              [
                49.85379192645871,
                40.377402896008626
              ],
              [
                49.851534976792045,
                40.3767352815155
              ],
              [
                49.84932909081593,
                40.37602541448135
              ],
              [
                49.843070990815924,
                40.37413951448135
              ],
              [
                49.83589379081593,
                40.37219991448135
              ],
              [
                49.83397474886544,
                40.377031068588295
              ],
              [
                49.83364003398498,
                40.377765807340324
              ],
              [
                49.833369444108264,
                40.378607486868354
              ],
              [
                49.833258090815924,
                40.37896221448135
              ],
              [
                49.83268009081593,
                40.380860714481344
              ],
              [
                49.832158190815925,
                40.382481114481344
              ],
              [
                49.83122879081593,
                40.38450691448135
              ],
              [
                49.830793890815926,
                40.38543481448134
              ],
              [
                49.82969679081593,
                40.38634381448134
              ],
              [
                49.82719779081593,
                40.38800861448134
              ],
              [
                49.824731190815925,
                40.390521814481346
              ],
              [
                49.822449890815925,
                40.39215531448134
              ],
              [
                49.81937239081593,
                40.39474111448135
              ],
              [
                49.81706249081593,
                40.39598571448135
              ],
              [
                49.81581069081593,
                40.396483414481345
              ],
              [
                49.81450579081593,
                40.39716881448135
              ],
              [
                49.814138290815926,
                40.397399614481344
              ],
              [
                49.81388629081593,
                40.39760721448135
              ],
              [
                49.81290289081593,
                40.39856931448134
              ],
              [
                49.81005329081593,
                40.40115771448134
              ],
              [
                49.80837029081593,
                40.403163714481344
              ],
              [
                49.807553290815925,
                40.40445861448135
              ],
              [
                49.80671259081593,
                40.405797614481344
              ],
              [
                49.80416579081593,
                40.41009991448134
              ],
              [
                49.80250329081593,
                40.41282561448135
              ],
              [
                49.80211349081593,
                40.41328751448135
              ]
            ]
          ]
        ],
        "type": "MultiPolygon"
      }
    }
  ]
}
    
    khatai_boundary = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              49.96679228577369,
              40.34102591813274
            ],
            [
              49.9769422996617,
              40.34357169432897
            ],
            [
              49.98776318813327,
              40.339005332159104
            ],
            [
              49.99603233436003,
              40.33810532012696
            ],
            [
              49.99965477769727,
              40.33805653858582
            ],
            [
              50.0025551052567,
              40.33830274073709
            ],
            [
              50.00330216719533,
              40.3386869360998
            ],
            [
              50.00368988165897,
              40.3389782235227
            ],
            [
              50.00390074368817,
              40.339367088336886
            ],
            [
              50.00523877392372,
              40.33921207050281
            ],
            [
              50.005506519708675,
              40.3393103069809
            ],
            [
              50.00571508755523,
              40.339356671456684
            ],
            [
              50.005897946774006,
              40.33939465251866
            ],
            [
              50.006244923964545,
              40.33967168779381
            ],
            [
              50.00637384782097,
              40.33984202239192
            ],
            [
              50.006339697286286,
              40.339868841016724
            ],
            [
              50.006474752350684,
              40.34017664088225
            ],
            [
              50.00919826775863,
              40.34240088681914
            ],
            [
              50.009081959113345,
              40.34368042083632
            ],
            [
              50.010050968455246,
              40.344809554455935
            ],
            [
              50.011414638880794,
              40.34533708660963
            ],
            [
              50.01317297039493,
              40.34488703606664
            ],
            [
              50.015313923765405,
              40.34662607801771
            ],
            [
              49.99832652562057,
              40.34900885502199
            ],
            [
              49.99215721587224,
              40.34752832187458
            ],
            [
              49.989601124646256,
              40.346986219437255
            ],
            [
              49.98818403762046,
              40.34604640147862
            ],
            [
              49.98589465327916,
              40.346206699055855
            ],
            [
              49.98245283703588,
              40.347682627402264
            ],
            [
              49.98115922070227,
              40.34872091089572
            ],
            [
              49.98060362464306,
              40.35112278595142
            ],
            [
              49.97963455158367,
              40.354373872957275
            ],
            [
              49.97880605248543,
              40.356495483154035
            ],
            [
              49.97801618874196,
              40.358587589725566
            ],
            [
              49.978127816647174,
              40.35912913783568
            ],
            [
              49.97934466404524,
              40.360611311787835
            ],
            [
              49.980524865402,
              40.361995838447456
            ],
            [
              49.977913028656786,
              40.36214892566687
            ],
            [
              49.973626937587056,
              40.36148554520392
            ],
            [
              49.97054128909906,
              40.36062625956734
            ],
            [
              49.96977352960235,
              40.364983013750475
            ],
            [
              49.969616108834174,
              40.37047221752536
            ],
            [
              49.96894718673738,
              40.37200109966838
            ],
            [
              49.96883639912065,
              40.37450395659482
            ],
            [
              49.96883640006769,
              40.379168619470164
            ],
            [
              49.97148636108267,
              40.38253951891343
            ],
            [
              49.971010010952625,
              40.38588835707779
            ],
            [
              49.97015878425893,
              40.38842176006898
            ],
            [
              49.969540272970846,
              40.39116979338314
            ],
            [
              49.968818676469226,
              40.39189604059703
            ],
            [
              49.96799401129252,
              40.393189619358935
            ],
            [
              49.967195100877774,
              40.39489721806362
            ],
            [
              49.96539110962297,
              40.395073863733785
            ],
            [
              49.9533013623749,
              40.39528470051448
            ],
            [
              49.93964158448625,
              40.39496623529092
            ],
            [
              49.932924576533026,
              40.394225058339856
            ],
            [
              49.92517397240573,
              40.3933250603981
            ],
            [
              49.917006295410204,
              40.39213386815541
            ],
            [
              49.91110557959439,
              40.39198278639117
            ],
            [
              49.905210745708644,
              40.39109025444978
            ],
            [
              49.89714749938105,
              40.38981653296568
            ],
            [
              49.894910303388116,
              40.3894649195758
            ],
            [
              49.89235350796665,
              40.389897674256645
            ],
            [
              49.89054244454414,
              40.39065498825656
            ],
            [
              49.88855382588417,
              40.390871362120976
            ],
            [
              49.8861390746537,
              40.39022223844211
            ],
            [
              49.8820197587944,
              40.389194453634786
            ],
            [
              49.87946296337418,
              40.38865350420062
            ],
            [
              49.8743848835816,
              40.38903216926073
            ],
            [
              49.86972538454344,
              40.389121055440086
            ],
            [
              49.861270540035036,
              40.38884488362632
            ],
            [
              49.857528133155625,
              40.38388307663297
            ],
            [
              49.85792472786318,
              40.38101294467165
            ],
            [
              49.85886669699923,
              40.38003102176319
            ],
            [
              49.861742160193415,
              40.378898022940575
            ],
            [
              49.862981602994694,
              40.377576148879996
            ],
            [
              49.86432020044984,
              40.37512115510245
            ],
            [
              49.86843521844497,
              40.37379917614845
            ],
            [
              49.87102892360616,
              40.373855831804775
            ],
            [
              49.87172691857067,
              40.37455220126411
            ],
            [
              49.87450342817724,
              40.37476189192058
            ],
            [
              49.87864001168592,
              40.37492376504339
            ],
            [
              49.87952444315803,
              40.37468723891638
            ],
            [
              49.87968169038234,
              40.374092249937966
            ],
            [
              49.88403094092425,
              40.37429565559124
            ],
            [
              49.884820556621136,
              40.37402982744991
            ],
            [
              49.885011314700364,
              40.37347070868472
            ],
            [
              49.8878952001844,
              40.37352563578742
            ],
            [
              49.88796659364864,
              40.37300212210411
            ],
            [
              49.88982025100295,
              40.373571363037726
            ],
            [
              49.89066152859296,
              40.374157415762674
            ],
            [
              49.89469670085858,
              40.37410827159678
            ],
            [
              49.90294254845014,
              40.37227306036989
            ],
            [
              49.90268186134155,
              40.371767373973114
            ],
            [
              49.90360936022316,
              40.371615293454916
            ],
            [
              49.90546435798884,
              40.37037760057791
            ],
            [
              49.912398075014735,
              40.36863490315536
            ],
            [
              49.91442952319344,
              40.36801580513145
            ],
            [
              49.916417874075066,
              40.36753257084299
            ],
            [
              49.917268258949704,
              40.36834986442885
            ],
            [
              49.91811555419393,
              40.368337118078294
            ],
            [
              49.91907046677659,
              40.369308689838746
            ],
            [
              49.92024061403829,
              40.36821729989756
            ],
            [
              49.922491829006134,
              40.36820966914573
            ],
            [
              49.922922298364185,
              40.36720496942762
            ],
            [
              49.92699799104931,
              40.36374318093615
            ],
            [
              49.929021612816726,
              40.365014474707394
            ],
            [
              49.93082906687937,
              40.36402085451985
            ],
            [
              49.930474385527695,
              40.36354747444181
            ],
            [
              49.93415948350285,
              40.36205527264029
            ],
            [
              49.937356690010944,
              40.36097768567576
            ],
            [
              49.93727679340397,
              40.360590633753546
            ],
            [
              49.93898919325853,
              40.35959659533274
            ],
            [
              49.94153206931203,
              40.35858402000383
            ],
            [
              49.945648422357515,
              40.35568213837145
            ],
            [
              49.94716606502716,
              40.354144477884425
            ],
            [
              49.94550670859643,
              40.35339732696286
            ],
            [
              49.94774475274271,
              40.35235415227747
            ],
            [
              49.94719478017147,
              40.35213483591431
            ],
            [
              49.94997335817131,
              40.350116013646684
            ],
            [
              49.95154756664719,
              40.34871627228174
            ],
            [
              49.95119976337642,
              40.34838163895731
            ],
            [
              49.95272960957575,
              40.34600570705612
            ],
            [
              49.95403855583878,
              40.34622878411339
            ],
            [
              49.955562822474256,
              40.34514070062941
            ],
            [
              49.95594388913361,
              40.34428993352806
            ],
            [
              49.95522045610447,
              40.34314453079827
            ],
            [
              49.956972867693835,
              40.33921254621807
            ],
            [
              49.9595127559954,
              40.33801498309505
            ],
            [
              49.96109521665339,
              40.33945777232345
            ],
            [
              49.96679228577369,
              40.34102591813274
            ]
          ]
        ],
        "type": "Polygon"
      }
    },
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              [
                -75.7110705,
                4.8012584
              ],
              [
                -75.7109621,
                4.8012614
              ],
              [
                -75.7108773,
                4.8012637
              ],
              [
                -75.7108775,
                4.8026396
              ],
              [
                -75.7109637,
                4.8026391
              ],
              [
                -75.7110572,
                4.8026385
              ],
              [
                -75.7110705,
                4.8012584
              ]
            ]
          ]
        ],
        "type": "MultiPolygon"
      }
    }
  ]
}
    
    nizami_boundary = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [49.8854319, 40.3923366],
                            [49.8858824, 40.3920516],
                            [49.886134, 40.3918069],
                            [49.8867729, 40.3913509],
                            [49.8874568, 40.3912614],
                            [49.8881145, 40.3911006],
                            [49.8883206, 40.3909832],
                            [49.8884508, 40.3908109],
                            [49.8892137, 40.390875],
                            [49.8897877, 40.390828],
                            [49.8903403, 40.390689],
                            [49.8908686, 40.3905195],
                            [49.8923092, 40.3898402],
                            [49.8926496, 40.3897105],
                            [49.8932854, 40.389547],
                            [49.8937198, 40.3894776],
                            [49.8941154, 40.3894415],
                            [49.8947347, 40.3894417],
                            [49.8953265, 40.3895001],
                            [49.8972718, 40.3897913],
                            [49.902539, 40.3906237],
                            [49.9062123, 40.3912193],
                            [49.9087107, 40.3916267],
                            [49.910359, 40.3918751],
                            [49.9127916, 40.3921219],
                            [49.9141918, 40.392259],
                            [49.9145915, 40.3922712],
                            [49.9149417, 40.3922639],
                            [49.915777, 40.3921834],
                            [49.9160385, 40.3921425],
                            [49.9163765, 40.3921088],
                            [49.9166608, 40.3921047],
                            [49.9169759, 40.3921241],
                            [49.9172221, 40.3921587],
                            [49.9187395, 40.3925286],
                            [49.9191995, 40.3926206],
                            [49.9197334, 40.3926883],
                            [49.9323301, 40.3941339],
                            [49.9349893, 40.3944686],
                            [49.9370778, 40.3947011],
                            [49.9384323, 40.3948053],
                            [49.9403179, 40.3950076],
                            [49.9415832, 40.3951133],
                            [49.9429465, 40.3951644],
                            [49.9446867, 40.3951819],
                            [49.949491, 40.395219],
                            [49.9541835, 40.3951648],
                            [49.9649244, 40.3952133],
                            [49.9655333, 40.3952118],
                            [49.9661314, 40.3951771],
                            [49.9667323, 40.3950862],
                            [49.9671483, 40.3949197],
                            [49.9680412, 40.3981452],
                            [49.9682558, 40.3996567],
                            [49.9688405, 40.4007597],
                            [49.9690699, 40.401976],
                            [49.9693331, 40.4021104],
                            [49.9680814, 40.4040319],
                            [49.967148, 40.4057598],
                            [49.966784, 40.4065361],
                            [49.9670112, 40.4066186],
                            [49.9664667, 40.4073549],
                            [49.9654877, 40.4080054],
                            [49.96497, 40.4084547],
                            [49.9644714, 40.4091755],
                            [49.9642311, 40.4093748],
                            [49.9640165, 40.4095116],
                            [49.9648051, 40.4100344],
                            [49.964793, 40.4102652],
                            [49.9643987, 40.4103602],
                            [49.9638019, 40.4107227],
                            [49.9629181, 40.4111352],
                            [49.960413, 40.412332],
                            [49.9588573, 40.412381],
                            [49.9556118, 40.4126066],
                            [49.9523108, 40.4187264],
                            [49.9495581, 40.419653],
                            [49.9483216, 40.4203575],
                            [49.9460881, 40.4212577],
                            [49.9450091, 40.4220258],
                            [49.9421042, 40.4236491],
                            [49.9410813, 40.423687],
                            [49.940114, 40.4236185],
                            [49.9393657, 40.4236205],
                            [49.9384156, 40.4239309],
                            [49.9365695, 40.4245326],
                            [49.9353336, 40.4264143],
                            [49.9252476, 40.4249012],
                            [49.9246444, 40.4247548],
                            [49.9234775, 40.4244106],
                            [49.9222186, 40.4233303],
                            [49.9214188, 40.420153],
                            [49.9209778, 40.4202237],
                            [49.9207123, 40.4203003],
                            [49.9203797, 40.4204565],
                            [49.9201652, 40.4205978],
                            [49.9187873, 40.4222765],
                            [49.9172643, 40.4241759],
                            [49.9167212, 40.4248276],
                            [49.914633, 40.4238209],
                            [49.9113886, 40.4220232],
                            [49.9075091, 40.4200476],
                            [49.9046847, 40.418634],
                            [49.9035555, 40.4181174],
                            [49.901991, 40.417446],
                            [49.8997065, 40.4166307],
                            [49.897998, 40.4161223],
                            [49.8965274, 40.4157395],
                            [49.8960296, 40.4156032],
                            [49.8904304, 40.4147154],
                            [49.8911746, 40.4119487],
                            [49.889533, 40.4107191],
                            [49.8901087, 40.409361],
                            [49.8909269, 40.4081743],
                            [49.8879161, 40.406351],
                            [49.881893, 40.4027479],
                            [49.8828219, 40.4014336],
                            [49.8854128, 40.3979792],
                            [49.8858288, 40.397274],
                            [49.8860005, 40.3969308],
                            [49.8861104, 40.3966846],
                            [49.8861626, 40.3965456],
                            [49.8862767, 40.3961433],
                            [49.886376, 40.3957491],
                            [49.8864327, 40.3949212],
                            [49.8863786, 40.3940944],
                            [49.8859968, 40.3933675],
                            [49.8857295, 40.3928177],
                            [49.8854319, 40.3923366]
                        ]
                    ]
                ]
            }
        }
    ]
}

    sebail_boundary = {
  "type": "Feature",
  "properties": {
    "admin_level": "7",
    "boundary": "administrative",
    "name": "Səbail rayonu"
  },
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [49.85572539974004, 40.37296135765382],
        [49.85374980492702, 40.37591868875009],
        [49.833562600901104, 40.37045290444425],
        [49.82992671260712, 40.365243084345565],
        [49.82391677774331, 40.3659135292441],
        [49.81508569170563, 40.3624262883458],
        [49.81185018495546, 40.358251923233354],
        [49.80894591586511, 40.35530780987523],
        [49.798991558884325, 40.354087276413026],
        [49.797371709522935, 40.348935496323605],
        [49.79327034750787, 40.33590806834951],
        [49.79692704100856, 40.30819788019397],
        [49.79371347855198, 40.30592859768558],
        [49.80768033626569, 40.30442725974109],
        [49.81278762640903, 40.30208703121741],
        [49.81940739069324, 40.3051803885466],
        [49.82792917576029, 40.30376994814205],
        [49.82206171019294, 40.3126447629333],
        [49.834682251545274, 40.30983086610644],
        [49.843933196744985, 40.33531986379222],
        [49.838099693912994, 40.33770541938841],
        [49.84521325830673, 40.3424495434669],
        [49.85873446461565, 40.34513782687128],
        [49.846564065084664, 40.34549689992491],
        [49.843440059996624, 40.34888700431563],
        [49.83936539854, 40.34787913527276],
        [49.836266576730196, 40.35160243355878],
        [49.835438969362, 40.35667993022358],
        [49.83641191680127, 40.36312170746996],
        [49.841761896543034, 40.367530955475985],
        [49.85572539974004, 40.37296135765382]
      ]
    ]
  }
}

    yasamal_boundary = {
  "type": "Feature",
  "properties": {
    "admin_level": "7",
    "boundary": "administrative",
    "name": "Yasamal rayonu"
  },
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [49.7881922, 40.3810192],
        [49.7870364, 40.3838737],
        [49.7862639, 40.3871491],
        [49.7860265, 40.3884827],
        [49.7860064, 40.389836],
        [49.7860157, 40.3914081],
        [49.7860976, 40.3927301],
        [49.7863966, 40.3938799],
        [49.786528, 40.3942599],
        [49.7869846, 40.3951384],
        [49.7873971, 40.3958328],
        [49.7882232, 40.3969747],
        [49.7886416, 40.3976243],
        [49.7895804, 40.3995668],
        [49.7899988, 40.4003532],
        [49.7902749, 40.4007819],
        [49.7918308, 40.4029616],
        [49.7930671, 40.4049429],
        [49.7943711, 40.4070991],
        [49.7966891, 40.4108473],
        [49.7972971, 40.411754],
        [49.7978456, 40.4124668],
        [49.7999571, 40.4144789],
        [49.8002154, 40.4147642],
        [49.8005144, 40.4152482],
        [49.8010286, 40.4152251],
        [49.8015135, 40.4143864],
        [49.802307, 40.4129055],
        [49.8026968, 40.4124436],
        [49.8043593, 40.4097179],
        [49.8069061, 40.4054156],
        [49.8077468, 40.4040766],
        [49.8085638, 40.4027817],
        [49.8102468, 40.4007757],
        [49.8130964, 40.3981873],
        [49.8140798, 40.3972252],
        [49.8143318, 40.3970176],
        [49.8146993, 40.3967868],
        [49.8160042, 40.3961014],
        [49.817256, 40.3956037],
        [49.8195659, 40.3943591],
        [49.8226434, 40.3917733],
        [49.8249247, 40.3901398],
        [49.8273913, 40.3876266],
        [49.8298903, 40.3859618],
        [49.8309874, 40.3850528],
        [49.8314223, 40.3841249],
        [49.8323517, 40.3820991],
        [49.8328736, 40.3804787],
        [49.8334516, 40.3785802],
        [49.8344354, 40.3788034],
        [49.8345922, 40.3783952],
        [49.8334096, 40.3780651],
        [49.8360873, 40.3718179],
        [49.8309035, 40.3704564],
        [49.8258872, 40.3684942],
        [49.8248208, 40.3688831],
        [49.8248437, 40.3678542],
        [49.8253379, 40.3662324],
        [49.8252617, 40.365002],
        [49.8247681, 40.3637391],
        [49.8240761, 40.3636397],
        [49.8229791, 40.3635232],
        [49.8215522, 40.3622056],
        [49.8199772, 40.3632425],
        [49.8168786, 40.3610005],
        [49.815344, 40.3626217],
        [49.8143427, 40.3593121],
        [49.814161, 40.3589744],
        [49.8138742, 40.3587433],
        [49.812133, 40.357747],
        [49.8108822, 40.357005],
        [49.8104991, 40.3560366],
        [49.8104212, 40.3557211],
        [49.8103514, 40.355625],
        [49.8101849, 40.3554416],
        [49.8098289, 40.3551636],
        [49.8097324, 40.3550952],
        [49.8095173, 40.3549975],
        [49.8092136, 40.3548975],
        [49.8080716, 40.3547185],
        [49.8060198, 40.3544195],
        [49.8020616, 40.3539219],
        [49.800957, 40.3537782],
        [49.8003924, 40.3537282],
        [49.7999379, 40.3537315],
        [49.7996734, 40.3539241],
        [49.7986785, 40.355019],
        [49.7984251, 40.3554227],
        [49.798032, 40.3562555],
        [49.7977316, 40.3571406],
        [49.7962706, 40.3572227],
        [49.7961539, 40.3590195],
        [49.7960767, 40.3595912],
        [49.7956985, 40.3609891],
        [49.7955644, 40.3614796],
        [49.7954893, 40.3618679],
        [49.7954569, 40.3621788],
        [49.7953472, 40.3635663],
        [49.7951514, 40.3643429],
        [49.7949144, 40.3651457],
        [49.7946578, 40.3658041],
        [49.7943033, 40.3665164],
        [49.7939175, 40.3671693],
        [49.7936118, 40.3676393],
        [49.7932979, 40.3681113],
        [49.7929332, 40.3686815],
        [49.7923699, 40.369501],
        [49.7918417, 40.3703605],
        [49.7915759, 40.3708926],
        [49.7913277, 40.3714471],
        [49.791128, 40.3721064],
        [49.7909271, 40.3729716],
        [49.7906855, 40.37416],
        [49.7904655, 40.3752512],
        [49.7901222, 40.3766202],
        [49.789682, 40.3778148],
        [49.7881922, 40.3810192]
      ]
    ]
  }
}

st.title("💰 House Price Prediction")

district_map_settings = {
  'ABSHERON': {'location': [40.43, 49.57], 'zoom': 9},
  'BINAGADI': {'location': [40.47, 49.82], 'zoom': 12},
  'NARIMANOV': {'location': [40.40, 49.86], 'zoom': 12},
  'NASIMI': {'location': [40.39, 49.83], 'zoom': 13},
  'NIZAMI': {'location': [40.41, 49.92], 'zoom': 13},
  'SEBAIL': {'location': [40.33, 49.82], 'zoom': 12},
  'KHATAI': {'location': [40.38, 49.95], 'zoom': 12},
  'YASAMAL': {'location': [40.38, 49.81], 'zoom': 13},
}

district_bounds = {
    "ABSHERON": absheron_boundary,
    "NARIMANOV": narimanov_boundary,
    "KHATAI": khatai_boundary,
    "NASIMI": nasimi_boundary,
    "YASAMAL": yasamal_boundary,
    "BINAGADI": binagadi_boundary,
    "SEBAIL": sebail_boundary,
    "NIZAMI": nizami_boundary
}

def get_geometry(boundary):
    if "features" in boundary: 
        return shape(boundary["features"][0]["geometry"])
    elif "geometry" in boundary: 
        return shape(boundary["geometry"])
    else:  
        return shape(boundary)

district = st.selectbox("Choose a district", districts)
map_settings = district_map_settings.get(district, {'location': [40.40, 49.85], 'zoom': 12})

if "clicked_lat_lng" not in st.session_state:
    st.session_state.clicked_lat_lng = map_settings['location']
    st.session_state.current_district = district
elif st.session_state.get('current_district') != district:
    st.session_state.clicked_lat_lng = map_settings['location']
    st.session_state.current_district = district

m = folium.Map(location=map_settings['location'], zoom_start=map_settings['zoom'])

district_data = {
    "ABSHERON": (absheron_boundary, "blue"),
    "BINAGADI": (binagadi_boundary, "red"),
    "NASIMI": (nasimi_boundary, "green"),
    "NARIMANOV": (narimanov_boundary, "blue"),
    "KHATAI": (khatai_boundary, "red"),
    "SEBAIL": (sebail_boundary, "green"),
    "YASAMAL": (yasamal_boundary, "blue"),
    "NIZAMI": (nizami_boundary, "red")
}

if district in district_data:
    boundary, color = district_data[district]
    folium.GeoJson(boundary, style_function=lambda x: {"color": color, "weight": 2, "fillOpacity": 0.05}).add_to(m)

folium.Marker(
    st.session_state.clicked_lat_lng,
    tooltip="Current Location",
    icon=folium.Icon(color="red", icon="home")
).add_to(m)

map_data = st_folium(m, width=700, height=450, key="map")

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    point = Point(lon, lat)

    boundary = district_bounds.get(district)

    if boundary:
        polygon = get_geometry(boundary)
        if polygon.contains(point):
            new_location = [lat, lon]
            if st.session_state.clicked_lat_lng != new_location:
                st.session_state.clicked_lat_lng = new_location
                st.success(f"✅ Location updated inside {district.title()} boundary!")
                st.rerun()
        else:
            st.warning(f"❌ Selected location is outside {district.title()} boundary. Please click inside the highlighted area.")
    
latitude = st.session_state.clicked_lat_lng[0]
longitude = st.session_state.clicked_lat_lng[1]
st.write(f"Selected Location: Latitude {latitude:.4f}, Longitude {longitude:.4f}")

with st.form("prediction_form"):
    st.subheader("Enter House Details")

    seller_type = st.radio("Seller is Owner?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")

    st.markdown("### 🏠 Property Info")
    prop1, prop2, prop3 = st.columns(3)
    with prop1:
        area_sqm = st.number_input("Area (sqm)", min_value=20, max_value=1000, value=100)
    with prop2:
        room_count = st.number_input("Room Count", min_value=1, max_value=10, value=3)
    with prop3:
        renovation_status = st.radio("Renovated?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")

    prop4, prop5 = st.columns(2)
    with prop4:
        title_deed = st.radio("Title Deed", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    with prop5:
        mortgage_availability = st.radio("Mortgage Availability", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")

    st.markdown("### 🏢 Floor Info")
    fl1, fl2 = st.columns(2)
    with fl1:
        floor_number = st.number_input("Floor Number", min_value=1, max_value=30, value=5)
    with fl2:
        total_floor = st.number_input("Total Floors", min_value=1, max_value=30, value=10)

    submit = st.form_submit_button("🔮 Predict Price")

if submit:
    point = Point(longitude, latitude)

    boundary = district_bounds.get(district)
    if boundary:
        polygon = get_geometry(boundary)
        if not polygon.contains(point):
            st.warning(f"❌ Selected location is outside {district.title()} boundary.")
            st.stop()

    floor_ratio = floor_number / total_floor if total_floor > 0 else 0
    is_highest_floor = 1 if floor_number == total_floor else 0
    room_size = area_sqm / room_count if room_count > 0 else 0

    features = pd.DataFrame([[seller_type, latitude, longitude, area_sqm, room_count,
                                title_deed, renovation_status, mortgage_availability,
                                floor_number, total_floor, floor_ratio,
                                is_highest_floor, room_size]],
                            columns=['seller_type','latitude','longitude','area_sqm','room_count',
                                        'title_deed','renovation_status','mortgage_availability',
                                        'floor_number','total_floor','floor_ratio',
                                        'is_highest_floor','room_size'])

    scaler = scalers[district]
    poly = polys[district]
    features_scaled = scaler.transform(features)
    features_poly = poly.transform(features_scaled)
    linear_pred = np.exp(linear_models[district].predict(features_poly)[0])

    rf_pred = rf_models[district].predict(features)[0]
    xgb_pred = xgb_models[district].predict(features)[0]

    st.subheader("Predicted Price (₼)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", f"{linear_pred:,.0f} ₼")
    col2.metric("Random Forest", f"{rf_pred:,.0f} ₼")
    col3.metric("XGBoost", f"{xgb_pred:,.0f} ₼")

if page == "About Model":
    st.title("🤖 About the Machine Learning Models")

    st.markdown("""
    This section provides an overview of the models used to predict house prices in Baku.  

    **Models Used:**
    1. **Linear Regression**  
       - Captures linear relationships between house features (area, room count, floor, etc.) and price.
       - Works best for simple trends and provides interpretable coefficients.

    2. **Random Forest Regressor**  
       - An ensemble of decision trees to capture complex non-linear relationships.
       - Robust to outliers and handles categorical and numerical features effectively.

    3. **XGBoost Regressor**  
       - Gradient boosting model that iteratively improves predictions.
       - Often provides the best performance in structured data like this dataset.
    
    **Features Used in the Models:**  
    - Seller type (owner or agency)
    - Latitude and longitude
    - Area (sqm)
    - Room count
    - Renovation status
    - Mortgage availability
    - Title deed availability
    - Floor information (floor number, total floors)
    - Derived features: floor ratio, is highest floor, room size

    **Model Evaluation:**  
    Each model was trained and validated per district. Performance metrics include:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² Score

    **How to Use:**  
    Use the **Price Prediction** tab to enter house details and get predicted prices from all three models.  
    The predictions help buyers and sellers understand the approximate market value of properties in Baku.
    """)

    selected_district = st.selectbox("Select a district", districts)
    df_path = os.path.join(save_path, f"{selected_district}_data.pkl")

    if os.path.exists(df_path):
        df = joblib.load(df_path)

        def show_model_performance(model_name, y_test, y_pred):
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.markdown(f"<h3 style='text-align: center;'>{model_name}</h3>", unsafe_allow_html=True)
            st.markdown(f"**R-squared (R²):** {r2:.3f}")
            st.markdown(f"**Mean Absolute Error (MAE):** {round(mae)}")
            st.markdown(f"**Root Mean Squared Error (RMSE):** {round(rmse)}")

        def plot_actual_vs_predicted(y_test, y_pred, model_name):
            n = min(40, len(y_test))
            indices = np.arange(n)
            plt.figure(figsize=(15,6))
            bar_width = 0.35
            plt.bar(indices, y_test[:n], width=bar_width, label="Actual Prices")
            plt.bar(indices + bar_width, y_pred[:n], width=bar_width, label="Predicted Prices")
            plt.xlabel("Sample Index")
            plt.ylabel("Price")
            plt.title(f"Actual vs Predicted Prices ({model_name})")
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()

        X_lr = df.drop(['prices', 'url', 'images', 'location'], axis=1)
        y_lr = np.log1p(df['prices'])
        X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
            X_lr, y_lr, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_lr)
        X_test_scaled = scaler.transform(X_test_lr)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)
        lr_model = LinearRegression().fit(X_train_poly, y_train_lr)
        y_pred_lr = np.expm1(lr_model.predict(X_test_poly))
        y_test_real = np.expm1(y_test_lr)

        show_model_performance("Linear Regression", y_test_real, y_pred_lr)
        plot_actual_vs_predicted(y_test_real, y_pred_lr, "Linear Regression")

        X_rf = df.drop(['prices', 'url', 'images', 'location'], axis=1)
        y_rf = df['prices']
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
            X_rf, y_rf, test_size=0.2, random_state=42
        )
        rf_model = rf_models[selected_district]
        y_pred_rf = rf_model.predict(X_test_rf)

        show_model_performance("Random Forest", y_test_rf, y_pred_rf)
        plot_actual_vs_predicted(y_test_rf, y_pred_rf, "Random Forest")

        # --- XGBoost ---
        xgb_model = xgb_models[selected_district]
        y_pred_xgb = xgb_model.predict(X_test_rf)

        show_model_performance("XGBoost", y_test_rf, y_pred_xgb)
        plot_actual_vs_predicted(y_test_rf, y_pred_xgb, "XGBoost")

    else:
        st.warning(f"No saved data found for {selected_district}.")
