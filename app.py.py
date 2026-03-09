
import streamlit as st
import requests
import unicodedata
import re
import streamlit.components.v1 as components

API_KEY = "703d35b679aa8f01b02eff474186a6b0"


# ===== XỬ LÝ TEXT =====
def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')


def clean_city_name(city):
    city = city.replace("hom nay", "").strip()
    return city


def extract_city(user_input):
    text = remove_accents(user_input.lower())

    patterns = [
        r"thoi tiet (.+)",
        r"toi muon biet thoi tiet (.+)",
        r"weather in (.+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return clean_city_name(match.group(1).strip())

    return None


# ===== LẤY THỜI TIẾT =====
def get_weather(city):

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric&lang=vi"

    data = requests.get(url).json()

    if data.get("cod") != 200:
        return None, None, None

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    lat = data["coord"]["lat"]
    lon = data["coord"]["lon"]

    result = f"🌡 {data['name']}: {temp}°C - {desc}"

    return result, lat, lon


# ===== HIỂN THỊ BẢN ĐỒ =====
def show_weather_map(lat, lon):

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>

    <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>

    <style>

    #map {{
        width:100%;
        height:520px;
    }}

    .legend {{
        background:white;
        padding:12px;
        font-size:14px;
        border-radius:8px;
        box-shadow:0 0 10px rgba(0,0,0,0.2);
    }}

    </style>

    </head>

    <body>

    <div id="map"></div>

    <script>

    var map = L.map('map').setView([{lat}, {lon}], 4);

    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '© OpenStreetMap'
    }}).addTo(map);


    var tempLayer = L.tileLayer(
    "https://tile.openweathermap.org/map/temp_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
    {{opacity:0.8}}
    ).addTo(map);

    var rainLayer = L.tileLayer(
    "https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
    {{opacity:0.8}}
    );

    var cloudLayer = L.tileLayer(
    "https://tile.openweathermap.org/map/clouds_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
    {{opacity:0.8}}
    );

    var windLayer = L.tileLayer(
    "https://tile.openweathermap.org/map/wind_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
    {{opacity:0.8}}
    );

    var radarLayer = L.tileLayer(
    "https://tile.openweathermap.org/map/precipitation/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
    {{opacity:0.9}}
    );

    var overlays = {{
        "🌡 Nhiệt độ": tempLayer,
        "🌧 Mưa": rainLayer,
        "☁ Mây": cloudLayer,
        "💨 Gió": windLayer,
        "📡 Radar mưa": radarLayer
    }};

    L.control.layers(null, overlays).addTo(map);

    var legend = L.control({{position: 'bottomright'}});

    legend.onAdd = function(map) {{
        var div = L.DomUtil.create('div', 'legend');
        div.id = "legendBox";
        div.innerHTML = "<b>🌡 Nhiệt độ</b><br>Màu sắc thể hiện nhiệt độ khu vực.";
        return div;
    }};

    legend.addTo(map);

    function updateLegend(text){{
        document.getElementById("legendBox").innerHTML = text;
    }}

    map.on('overlayadd', function(e){{

        if(e.name === "🌡 Nhiệt độ"){{
            updateLegend("<b>🌡 Nhiệt độ</b><br>Màu sắc thể hiện nhiệt độ từng khu vực.");
        }}

        if(e.name === "🌧 Mưa"){{
            updateLegend("<b>🌧 Lượng mưa</b>");
        }}

        if(e.name === "☁ Mây"){{
            updateLegend("<b>☁ Mây</b>");
        }}

        if(e.name === "💨 Gió"){{
            updateLegend("<b>💨 Gió</b>");
        }}

        if(e.name === "📡 Radar mưa"){{
            updateLegend("<b>📡 Radar mưa realtime</b>");
        }}

    }});

    L.marker([{lat}, {lon}])
    .addTo(map)
    .bindPopup("Thành phố bạn tìm")
    .openPopup();

    </script>

    </body>
    </html>
    """

    components.html(html, height=540)


# ===== UI =====
st.set_page_config(page_title="Weather Map AI", page_icon="🌦")

st.title("🌦 Chatbot Thời Tiết + Bản đồ")

st.write("Hỏi thời tiết bất kỳ thành phố nào.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input(
    "Bạn muốn biết thời tiết ở đâu?",
    placeholder="ví dụ: thời tiết hà nội"
)

if user_input:

    text = remove_accents(user_input.lower())

    if "thoi tiet" in text or "weather" in text:

        city = extract_city(user_input)

        if city:

            result, lat, lon = get_weather(city)

            if result:

                st.success(result)

                st.session_state.history.append(result)

                st.subheader("🗺 Bản đồ thời tiết")

                show_weather_map(lat, lon)

            else:
                st.error("Không tìm thấy thành phố")

        else:
            st.warning("Bạn muốn xem thời tiết ở đâu?")

    else:
        st.info("Hiện tại mình chỉ hỗ trợ hỏi thời tiết")


if st.session_state.history:

    st.divider()
    st.subheader("🕘 Lịch sử")

    for item in reversed(st.session_state.history):
        st.write(item)
